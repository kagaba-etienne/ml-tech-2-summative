"""DQN (double DQN, soft or hard target) on SkyBlood with TorchRL."""

from pathlib import Path

import torch
from tensordict.nn import TensorDictSequential
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.modules import MLP, EGreedyModule, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

from environment.custom_env import N_ACTIONS, OBS_DIM
from training import N_ENVS, ROOT, TOTAL_FRAMES, Run, make_batched_env, make_env, setup_run

BASE = dict(
    lr=5e-4, gamma=0.99, buffer=100_000, batch=128, eps_frac=0.5, target="soft", tau=0.005,
    hard_every=2000, double=True, n_envs=N_ENVS, fpb=32, init_random_frames=10_000,
    eps_init=1.0, eps_end=0.05, hidden=[128, 128],
)

CONFIGS = [
    {**BASE, "name": "D1", "note": "Baseline"},
    {**BASE, "name": "D2", "lr": 1e-4, "note": "Slow lr"},
    {**BASE, "name": "D3", "lr": 1e-3, "note": "Fast lr"},
    {**BASE, "name": "D4", "gamma": 0.95, "note": "Short horizon"},
    {**BASE, "name": "D5", "buffer": 10_000, "note": "Small replay"},
    {**BASE, "name": "D6", "batch": 32, "note": "Small batch"},
    {**BASE, "name": "D7", "eps_frac": 0.1, "note": "Little exploration"},
    {**BASE, "name": "D8", "eps_frac": 0.8, "note": "Long exploration"},
    {**BASE, "name": "D9", "target": "hard", "note": "Hard target every 2,000 updates"},
    {**BASE, "name": "D10", "double": False, "note": "Overestimation (no double Q)"},
]


def get_config(name: str) -> dict:
    return next(c for c in CONFIGS if c["name"] == name)


def build_qnet(spec, hidden, masked: bool = False):
    mlp = MLP(in_features=OBS_DIM, out_features=N_ACTIONS, num_cells=hidden)
    qnet = QValueActor(mlp, in_keys=["observation"], spec=spec, action_mask_key="action_mask" if masked else None)
    return qnet, mlp


def checkpoint_path(name: str, models_root=None):
    return Path(models_root or ROOT / "models") / "dqn" / f"{name}.pt"


def train(config: str, seed: int = 0, device: str = "cpu", frames: int | None = None,
          masked: bool = False, logs_root=None, models_root=None):
    setup_run(seed)
    cfg = {**get_config(config), "seed": seed, "masked": masked}
    total = frames or TOTAL_FRAMES
    init_random = min(cfg["init_random_frames"], total // 2)  # only differs for short smoke runs

    env = make_batched_env(masked, seed, cfg["n_envs"])
    spec = make_env(masked).action_spec
    qnet, mlp = build_qnet(spec, cfg["hidden"], masked)
    qnet = qnet.to(device)
    eg = EGreedyModule(spec, eps_init=cfg["eps_init"], eps_end=cfg["eps_end"],
                       annealing_num_steps=int(cfg["eps_frac"] * total),
                       action_mask_key="action_mask" if masked else None).to(device)
    policy = TensorDictSequential(qnet, eg)

    collector = Collector(env, policy, frames_per_batch=cfg["fpb"], total_frames=total,
                          init_random_frames=init_random, policy_device=device, env_device="cpu",
                          storing_device=device, auto_register_policy_transforms=False)
    rb = ReplayBuffer(storage=LazyTensorStorage(cfg["buffer"], device=device), batch_size=cfg["batch"])
    loss = DQNLoss(qnet, action_space=spec, delay_value=True, double_dqn=cfg["double"])
    loss.make_value_estimator(gamma=cfg["gamma"])
    if cfg["target"] == "soft":
        updater = SoftUpdate(loss, tau=cfg["tau"])
    else:
        updater = HardUpdate(loss, value_network_update_interval=cfg["hard_every"])
    opt = torch.optim.Adam(loss.parameters(), lr=cfg["lr"])

    run = Run("dqn", cfg, total, mlp, qnet, checkpoint_path(config, models_root), device, logs_root)
    updates_per_batch = cfg["fpb"] // 8
    frames_seen = 0
    for data in collector:
        frames_seen += data.numel()
        rb.extend(data.reshape(-1))
        if frames_seen >= init_random:
            losses = []
            for _ in range(updates_per_batch):
                out = loss(rb.sample().to(device))
                opt.zero_grad()
                out["loss"].backward()
                opt.step()
                updater.step()
                losses.append(out["loss"].item())
            run.log_train(frames_seen, {
                "loss": sum(losses) / len(losses),
                "mean_q": data["chosen_action_value"].mean().item(),
                "epsilon": eg.eps.item(),
            })
        eg.step(data.numel())
        run.step(frames_seen, data)
    collector.shutdown()
    run.finish(frames_seen)
    return run
