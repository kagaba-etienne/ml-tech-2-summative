"""Policy-gradient methods on SkyBlood with TorchRL: REINFORCE (Monte Carlo), PPO and A2C."""

from pathlib import Path

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import ExplorationType
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import MLP, MaskedCategorical, OneHotCategorical, ProbabilisticActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, ReinforceLoss, ValueEstimators
from torchrl.objectives.value import GAE

from environment.custom_env import N_ACTIONS, OBS_DIM
from training import N_ENVS, ROOT, TOTAL_FRAMES, Run, make_batched_env, make_env, setup_run

HIDDEN = [64, 64]
MAX_EPISODE_STEPS = 250

R1 = dict(name="R1", lr=1e-3, gamma=0.99, episodes=8, baseline=True, entropy=0.0, adv_norm=True, note="Baseline")
P1 = dict(name="P1", lr=3e-4, fpb=2048, minibatch=256, epochs=4, clip=0.2, entropy=0.01, lmbda=0.95,
          gamma=0.99, value_coef=0.5, n_envs=N_ENVS, note="Baseline (verified)")
A1 = dict(name="A1", lr=7e-4, steps=16, gamma=0.99, lmbda=1.0, entropy=0.01, value_coef=0.5, n_envs=N_ENVS,
          note="Baseline (n-step)")

CONFIGS = {
    "reinforce": [
        R1,
        {**R1, "name": "R2", "lr": 3e-4, "note": "Slow lr"},
        {**R1, "name": "R3", "lr": 3e-3, "note": "Fast lr"},
        {**R1, "name": "R4", "gamma": 0.95, "note": "Short horizon"},
        {**R1, "name": "R5", "episodes": 2, "note": "Few episodes per update"},
        {**R1, "name": "R6", "episodes": 32, "note": "Many episodes per update"},
        {**R1, "name": "R7", "baseline": False, "note": "No baseline (pure MC)"},
        {**R1, "name": "R8", "entropy": 0.01, "note": "Small entropy bonus"},
        {**R1, "name": "R9", "entropy": 0.05, "note": "Large entropy bonus"},
        {**R1, "name": "R10", "adv_norm": False, "note": "No advantage normalisation"},
    ],
    "ppo": [
        P1,
        {**P1, "name": "P2", "lr": 1e-4, "note": "Slow lr"},
        {**P1, "name": "P3", "lr": 1e-3, "note": "Fast lr"},
        {**P1, "name": "P4", "fpb": 512, "minibatch": 128, "note": "Short rollout"},
        {**P1, "name": "P5", "fpb": 8192, "minibatch": 512, "note": "Long rollout"},
        {**P1, "name": "P6", "epochs": 10, "note": "Many epochs"},
        {**P1, "name": "P7", "clip": 0.1, "note": "Tight clip"},
        {**P1, "name": "P8", "clip": 0.3, "note": "Loose clip"},
        {**P1, "name": "P9", "entropy": 0.05, "note": "Large entropy bonus"},
        {**P1, "name": "P10", "lmbda": 0.80, "note": "Low GAE lambda"},
    ],
    "a2c": [
        A1,
        {**A1, "name": "A2", "lr": 2.5e-4, "note": "Slow lr"},
        {**A1, "name": "A3", "lr": 2e-3, "note": "Fast lr"},
        {**A1, "name": "A4", "steps": 5, "note": "Short rollout"},
        {**A1, "name": "A5", "steps": 64, "note": "Long rollout"},
        {**A1, "name": "A6", "lmbda": 0.95, "note": "GAE lambda 0.95"},
        {**A1, "name": "A7", "entropy": 0.0, "note": "No entropy bonus"},
        {**A1, "name": "A8", "entropy": 0.05, "note": "Large entropy bonus"},
        {**A1, "name": "A9", "value_coef": 1.0, "note": "Large value coefficient"},
        {**A1, "name": "A10", "gamma": 0.95, "note": "Short horizon"},
    ],
}


def get_config(algo: str, name: str) -> dict:
    return next(c for c in CONFIGS[algo] if c["name"] == name)


def build_actor_critic(spec, masked: bool = False):
    mlp = MLP(in_features=OBS_DIM, out_features=N_ACTIONS, num_cells=HIDDEN)
    logits = TensorDictModule(mlp, in_keys=["observation"], out_keys=["logits"])
    if masked:
        actor = ProbabilisticActor(logits, in_keys={"logits": "logits", "mask": "action_mask"},
                                   distribution_class=MaskedCategorical, return_log_prob=True, spec=spec)
    else:
        actor = ProbabilisticActor(logits, in_keys=["logits"], distribution_class=OneHotCategorical,
                                   return_log_prob=True, spec=spec)
    critic = ValueOperator(MLP(in_features=OBS_DIM, out_features=1, num_cells=HIDDEN), in_keys=["observation"])
    return actor, critic, mlp


def checkpoint_path(algo: str, name: str, models_root=None):
    return Path(models_root or ROOT / "models") / "pg" / f"{algo}_{name}.pt"


def _optimise(opt, params, loss):
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 0.5)
    opt.step()


def _loss_sum(out):
    # with entropy_coeff == 0 TorchRL drops the entropy terms from the output, so sum what is there
    return sum(v for k, v in out.items() if k.startswith("loss_"))


@torch.no_grad()
def _entropy(out, actor, td) -> float:
    """Policy entropy for logging, also when the loss did not compute it (entropy_coeff == 0)."""
    if "entropy" in out.keys():
        return out["entropy"].mean().item()
    return actor.get_dist(td).entropy().mean().item()


def _mean_rows(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}


def train(algo: str, config: str, seed: int = 0, device: str = "cpu", frames: int | None = None,
          masked: bool = False, logs_root=None, models_root=None):
    setup_run(seed)
    cfg = {**get_config(algo, config), "seed": seed, "masked": masked, "hidden": HIDDEN}
    total = frames or TOTAL_FRAMES
    spec = make_env(masked).action_spec
    actor, critic, mlp = build_actor_critic(spec, masked)
    actor, critic = actor.to(device), critic.to(device)
    run = Run(algo, cfg, total, mlp, actor, checkpoint_path(algo, config, models_root), device, logs_root)
    fn = {"reinforce": _reinforce, "ppo": _ppo, "a2c": _a2c}[algo]
    frames_seen = fn(cfg, total, actor, critic, run, masked, seed, device)
    run.finish(frames_seen)
    return run


# ---------- PPO ----------

def _ppo(cfg, total, actor, critic, run, masked, seed, device):
    fpb = min(cfg["fpb"], total)  # only differs for short smoke runs
    minibatch = min(cfg["minibatch"], fpb)
    env = make_batched_env(masked, seed, cfg["n_envs"])
    collector = Collector(env, actor, frames_per_batch=fpb, total_frames=total,
                          policy_device=device, env_device="cpu", storing_device=device, auto_register_policy_transforms=False)
    gae = GAE(gamma=cfg["gamma"], lmbda=cfg["lmbda"], value_network=critic, average_gae=False)
    loss = ClipPPOLoss(actor, critic, clip_epsilon=cfg["clip"], entropy_coeff=cfg["entropy"],
                       critic_coeff=cfg["value_coef"], normalize_advantage=True)
    rb = ReplayBuffer(storage=LazyTensorStorage(fpb, device=device), sampler=SamplerWithoutReplacement(),
                      batch_size=minibatch)
    opt = torch.optim.Adam(loss.parameters(), lr=cfg["lr"])

    frames_seen = 0
    for data in collector:
        frames_seen += data.numel()
        with torch.no_grad():
            gae(data)
        rb.extend(data.reshape(-1))
        rows = []
        for _ in range(cfg["epochs"]):
            for _ in range(max(1, data.numel() // minibatch)):
                batch = rb.sample()
                out = loss(batch)
                _optimise(opt, loss.parameters(), _loss_sum(out))
                rows.append({"entropy": _entropy(out, actor, batch),
                             **{k: out[k].item() for k in ("loss_objective", "loss_critic", "clip_fraction", "kl_approx")}})
        run.log_train(frames_seen, {**_mean_rows(rows), "lr": cfg["lr"]})
        run.step(frames_seen, data)
    collector.shutdown()
    return frames_seen


# ---------- A2C ----------

def _a2c(cfg, total, actor, critic, run, masked, seed, device):
    fpb = cfg["n_envs"] * cfg["steps"]
    env = make_batched_env(masked, seed, cfg["n_envs"])
    collector = Collector(env, actor, frames_per_batch=fpb, total_frames=total,
                          policy_device=device, env_device="cpu", storing_device=device, auto_register_policy_transforms=False)
    loss = A2CLoss(actor, critic, entropy_coeff=cfg["entropy"], critic_coeff=cfg["value_coef"])
    loss.make_value_estimator(ValueEstimators.GAE, gamma=cfg["gamma"], lmbda=cfg["lmbda"])
    opt = torch.optim.Adam(loss.parameters(), lr=cfg["lr"])

    frames_seen = 0
    for data in collector:
        frames_seen += data.numel()
        out = loss(data)
        _optimise(opt, loss.parameters(), _loss_sum(out))
        run.log_train(frames_seen, {
            "entropy": _entropy(out, actor, data), "loss_objective": out["loss_objective"].item(),
            "loss_critic": out["loss_critic"].item(), "lr": cfg["lr"],
        })
        run.step(frames_seen, data)
    collector.shutdown()
    return frames_seen


# ---------- REINFORCE ----------

def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """G_t = r_t + gamma * G_{t+1}, over one complete episode laid out along dim 0."""
    out = torch.zeros_like(rewards)
    g = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def _reinforce(cfg, total, actor, critic, run, masked, seed, device):
    env = make_env(masked).to(device)
    env.set_seed(seed)
    loss = ReinforceLoss(actor, critic)
    loss.make_value_estimator(ValueEstimators.TD1, gamma=cfg["gamma"])
    params = list(actor.parameters()) + (list(critic.parameters()) if cfg["baseline"] else [])
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    frames_seen = 0
    while frames_seen < total:
        episodes = []
        with torch.no_grad():
            for _ in range(cfg["episodes"]):
                with set_exploration_type(ExplorationType.RANDOM):
                    ep = env.rollout(MAX_EPISODE_STEPS, policy=actor, break_when_any_done=True)
                # The clock is in the observation, so the shift end is a true terminal state:
                # marking it terminated makes TD1 return the exact Monte Carlo return.
                ep["next", "terminated"] = ep["next", "done"].clone()
                if cfg["baseline"]:
                    loss.value_estimator(ep, params=loss.critic_network_params.detach(),
                                         target_params=loss.target_critic_network_params)
                else:
                    g = discounted_returns(ep["next", "reward"], cfg["gamma"])
                    ep["advantage"] = g
                    ep["value_target"] = g
                episodes.append(ep)
        td = torch.cat(episodes)
        frames_seen += td.shape[0]
        if cfg["adv_norm"]:
            adv = td["advantage"]
            td["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        out = loss(td)
        entropy = actor.get_dist(td).entropy().mean()
        objective = out["loss_actor"] - cfg["entropy"] * entropy
        total_loss = objective + out["loss_value"] if cfg["baseline"] else objective
        _optimise(opt, params, total_loss)
        run.log_train(frames_seen, {
            "entropy": entropy.item(), "loss_objective": out["loss_actor"].item(),
            "loss_critic": out["loss_value"].item() if cfg["baseline"] else float("nan"), "lr": cfg["lr"],
        })
        run.step(frames_seen, td)
    return frames_seen
