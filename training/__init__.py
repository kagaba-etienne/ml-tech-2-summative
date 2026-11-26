"""Helpers shared by the DQN and policy-gradient scripts: env construction, masked-greedy
evaluation, per-run logging and checkpoints."""

import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torchrl.data import Unbounded
from torchrl.envs import ActionMask, DTypeCastTransform, ExplorationType, GymWrapper, RewardSum, SerialEnv, StepCounter, TransformedEnv
from torchrl.envs.gym_like import default_info_dict_reader
from torchrl.envs.utils import set_exploration_type

from environment.custom_env import P_ORDER, SkyBloodEnv

TOTAL_FRAMES = 1_000_000
N_ENVS = 8
EVAL_SEEDS = list(range(10000, 10020))
EVAL_EVERY = 100_000
PROGRESS_EVERY = 50_000
COUNTERS = ["ep_deliveries", "ep_emergencies", "ep_expired", "ep_forced"]
ROOT = Path(__file__).resolve().parent.parent


def _gym_env(masked: bool, p_order: float = P_ORDER):
    """Penalty mode: one-hot actions, flat observation. Masked mode: categorical actions plus an action_mask key."""
    env = GymWrapper(SkyBloodEnv(dict_obs=masked, p_order=p_order), categorical_action_encoding=masked)
    spec = {"ep_stats": Unbounded(shape=(len(COUNTERS),), dtype=torch.int64)}
    env.set_info_dict_reader(default_info_dict_reader(["ep_stats"], spec=spec))
    return env


def _with_transforms(env, masked: bool):
    env = TransformedEnv(env)
    if masked:
        # gymnasium's MultiBinary is int8; TorchRL's masking ops need bool
        env.append_transform(DTypeCastTransform(torch.int8, torch.bool, in_keys=["action_mask"]))
        env.append_transform(ActionMask())
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    return env


def make_env(masked: bool = False, p_order: float = P_ORDER):
    return _with_transforms(_gym_env(masked, p_order), masked)


def make_batched_env(masked: bool, seed: int, n_envs: int = N_ENVS):
    # transforms wrap the batch rather than each sub-env: one call per step instead of n_envs
    env = _with_transforms(SerialEnv(n_envs, lambda: _gym_env(masked)), masked)
    env.set_seed(seed)
    return env


def setup_run(seed: int):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------- masked-greedy evaluation (used by training, eval, and play) ----------

@torch.no_grad()
def net_scores(net, obs, device="cpu") -> np.ndarray:
    """Raw logits (policy gradient) or Q-values (DQN) for one flat observation."""
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    return net(x).cpu().numpy()


def masked_argmax(scores: np.ndarray, mask: np.ndarray) -> int:
    return int(np.argmax(np.where(mask, scores, -np.inf)))


def evaluate(net, device="cpu", seeds=EVAL_SEEDS, p_order: float = P_ORDER) -> dict:
    """Greedy with invalid-action masking on held-out seeds."""
    returns, counts = [], {k: [] for k in COUNTERS}
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for seed in seeds:
            env = SkyBloodEnv(p_order=p_order)
            obs, info = env.reset(seed=seed)
            total, done = 0.0, False
            while not done:
                obs, r, term, trunc, info = env.step(masked_argmax(net_scores(net, obs, device), env.action_masks()))
                total += r
                done = term or trunc
            returns.append(total)
            for k in COUNTERS:
                counts[k].append(int(info[k]))
    out = {"returns": np.array(returns)}
    out.update({k: float(np.mean(v)) for k, v in counts.items()})
    return out


# ---------- logging and checkpoints ----------

class CsvLog:
    def __init__(self, path: Path):
        self.f = open(path, "w", newline="")
        self.w = None

    def row(self, d: dict):
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(d))
            self.w.writeheader()
        self.w.writerow(d)
        self.f.flush()

    def close(self):
        self.f.close()


class Run:
    """Writes logs/<algo>/<config>/*, runs the periodic evaluations, and saves final and best checkpoints."""

    def __init__(self, algo, cfg, total, net, module, ckpt_path, device, logs_root=None):
        self.algo, self.cfg, self.total = algo, cfg, total
        self.net, self.module, self.ckpt_path, self.device = net, module, Path(ckpt_path), device
        self.dir = Path(logs_root or ROOT / "logs") / algo / cfg["name"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps({"algo": algo, "total_frames": total, **cfg}, indent=2))
        self.episodes = CsvLog(self.dir / "episodes.csv")
        self.train = CsvLog(self.dir / "train.csv")
        self.evals = CsvLog(self.dir / "eval.csv")
        self.recent = deque(maxlen=50)
        self.best = -np.inf
        self.last_eval = None
        self.next_eval = min(EVAL_EVERY, total)
        self.next_print = min(PROGRESS_EVERY, total)
        self.t0 = time.time()

    def log_episodes(self, frames: int, td):
        done = td["next", "done"].squeeze(-1)
        if not done.any():
            return
        nxt = td["next"][done]
        returns = nxt["episode_reward"].reshape(-1).tolist()
        lengths = nxt["step_count"].reshape(-1).tolist()
        stats = nxt["ep_stats"].reshape(-1, len(COUNTERS)).tolist()
        for ret, length, (deliv, emerg, expired, forced) in zip(returns, lengths, stats):
            self.recent.append(ret)
            self.episodes.row({
                "frames": frames, "return": round(ret, 4), "length": int(length), "deliveries": deliv,
                "emergencies": emerg, "expired": expired, "forced_landings": forced,
            })

    def log_train(self, frames: int, row: dict):
        self.train.row({"frames": frames, **{k: round(float(v), 6) for k, v in row.items()}})

    def step(self, frames: int, td):
        """Call once per batch, after the gradient updates."""
        self.log_episodes(frames, td)
        if frames >= self.next_eval and frames < self.total:
            self._evaluate(frames)
            self.next_eval += EVAL_EVERY
        if frames >= self.next_print:
            self._progress(frames)
            self.next_print += PROGRESS_EVERY

    def finish(self, frames: int):
        self._evaluate(frames)
        self._save(self.ckpt_path)
        self._progress(frames)
        for log in (self.episodes, self.train, self.evals):
            log.close()
        best = self.ckpt_path.with_name(self.ckpt_path.stem + "_best.pt")
        print(f"[{self.algo} {self.cfg['name']}] done: final eval {self.last_eval:.1f}, best eval {self.best:.1f}"
              f"\n  logs   {self.dir}\n  model  {self.ckpt_path}\n  best   {best}")

    def _evaluate(self, frames: int):
        self.module.eval()
        res = evaluate(self.net, self.device)
        self.module.train()
        mean, std = float(res["returns"].mean()), float(res["returns"].std())
        self.last_eval = mean
        self.evals.row({
            "frames": frames, "eval_mean": round(mean, 4), "eval_std": round(std, 4),
            **{k: res[k] for k in COUNTERS},
        })
        if mean > self.best:
            self.best = mean
            self._save(self.ckpt_path.with_name(self.ckpt_path.stem + "_best.pt"), frames=frames, eval_mean=mean)

    def _save(self, path: Path, **extra):
        torch.save({"state_dict": self.module.state_dict(), "algo": self.algo, "config": self.cfg, **extra}, path)

    def _progress(self, frames: int):
        fps = frames / max(time.time() - self.t0, 1e-9)
        avg = np.mean(self.recent) if self.recent else float("nan")
        ev = f" | eval {self.last_eval:.1f}" if self.last_eval is not None else ""
        print(f"[{self.algo} {self.cfg['name']}] {frames:,}/{self.total:,} frames"
              f" | return(last {len(self.recent)}) {avg:.1f}{ev} | {fps:.0f} fps", flush=True)
