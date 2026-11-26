"""Episode traces for the three.js viewer: model loading, a step-by-step trace generator, JSON export,
a verbose terminal log, and the FastAPI server (static viewer + websocket)."""

import asyncio
import csv
import json
from pathlib import Path

import numpy as np

from environment.custom_env import (
    ACTION_NAMES, FAC, FACILITIES, HUB_LATLON, MAX_RANGE, P_ORDER, STORM_R, T, SkyBloodEnv, heuristic_action,
    km_to_latlon,
)

ROOT = Path(__file__).resolve().parent.parent
VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
SHORT = ["Kabgayi", "Ruhango", "Gitwe", "Muhororo", "Nyanza", "Kibuye"]
ACTION_LABELS = ACTION_NAMES[:3] + [f"FLY_DROP:{n}" for n in SHORT] + ACTION_NAMES[-1:]


# ---------- models ----------

def best_config(algo: str, logs_root=None) -> str:
    """Config whose final masked-greedy eval (last row of eval.csv) is highest."""
    best, best_mean = None, -np.inf
    for path in sorted(Path(logs_root or ROOT / "logs").glob(f"{algo}/*/eval.csv")):
        rows = list(csv.DictReader(open(path)))
        if rows and float(rows[-1]["eval_mean"]) > best_mean:
            best, best_mean = path.parent.name, float(rows[-1]["eval_mean"])
    if best is None:
        raise FileNotFoundError(f"no eval.csv under logs/{algo}/: train a config first")
    return best


def checkpoint_file(algo: str, config: str, best: bool = False, models_root=None) -> Path:
    from training.dqn_training import checkpoint_path as dqn_path
    from training.pg_training import checkpoint_path as pg_path

    path = dqn_path(config, models_root) if algo == "dqn" else pg_path(algo, config, models_root)
    return path.with_name(path.stem + "_best.pt") if best else path


def load_net(algo: str, config: str, best: bool = False, device: str = "cpu", models_root=None):
    """Rebuild the TorchRL modules, load the weights, and return the inner MLP (obs -> logits or Q)."""
    import torch

    from training import make_env

    ckpt = torch.load(checkpoint_file(algo, config, best, models_root), map_location=device, weights_only=True)
    masked = ckpt["config"].get("masked", False)
    spec = make_env(masked).action_spec
    if algo == "dqn":
        from training.dqn_training import build_qnet
        module, mlp = build_qnet(spec, ckpt["config"]["hidden"], masked)
    else:
        from training.pg_training import build_actor_critic
        module, _, mlp = build_actor_critic(spec, masked)
    module.load_state_dict(ckpt["state_dict"])
    return mlp.to(device).eval()


def load_model(algo: str, config: str = "best", device: str = "cpu") -> dict:
    """A policy for the trace: {'label', 'kind', 'act'}; act(env, obs) -> (action, top, mask_changed)."""
    if algo == "heuristic":
        return {"label": "Greedy heuristic", "kind": None, "act": lambda env, obs: (heuristic_action(env), [], False)}
    from training import masked_argmax, net_scores

    config = best_config(algo) if config == "best" else config
    net = load_net(algo, config, device=device)
    kind = "q" if algo == "dqn" else "prob"

    def act(env, obs):
        scores = net_scores(net, obs, device)
        mask = env.action_masks()
        a = masked_argmax(scores, mask)
        if kind == "prob":
            e = np.exp(scores - scores.max())
            values = e / e.sum()
        else:
            values = scores
        top = [[ACTION_LABELS[i], float(values[i]), bool(mask[i])] for i in np.argsort(-values)[:4]]
        return a, top, a != int(np.argmax(scores))

    return {"label": f"{algo.upper()} {config}", "kind": kind, "act": act, "config": config}


def print_eval(algo: str, config: str, device: str = "cpu"):
    from training import COUNTERS, EVAL_SEEDS, evaluate

    for best in (False, True):
        res = evaluate(load_net(algo, config, best, device), device)
        r = res["returns"]
        tag = "best-eval checkpoint" if best else "final checkpoint"
        print(f"{algo} {config} ({tag}), greedy with invalid-action masking, seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}:")
        print(f"  return {r.mean():.1f} ± {r.std():.1f}  (min {r.min():.1f}, max {r.max():.1f})")
        print("  per episode: " + ", ".join(f"{k.removeprefix('ep_')} {res[k]:.2f}" for k in COUNTERS))


# ---------- traces ----------

def _book(env) -> dict:
    return {"t": int(env.t), "urg": env.urg.tolist(), "left": env.left.tolist(), "storm": env.storm.tolist()}


def episode(model: dict, seed: int, p_order: float = P_ORDER):
    """Yields ('init', meta), then ('step', step, new_ticks) once per decision."""
    env = SkyBloodEnv(p_order=p_order)
    env.tick_log = []
    obs, _ = env.reset(seed=seed)
    meta = {
        "hub": {"lat": HUB_LATLON[0], "lon": HUB_LATLON[1], "name": "Zipline Muhanga"},
        "facilities": [
            {"k": k, "name": n, "short": SHORT[k], "x": x, "y": y, "dist": float(np.hypot(x, y)),
             "lat": km_to_latlon(x, y)[0], "lon": km_to_latlon(x, y)[1]}
            for k, (n, x, y) in enumerate(FACILITIES)
        ],
        "actions": ACTION_LABELS, "storm_r": STORM_R, "max_range": MAX_RANGE, "T": T, "tick_minutes": 3,
        "policy": model["label"], "kind": model["kind"], "seed": seed, "p_order": p_order, "start": _book(env),
    }
    yield "init", meta

    cum, i, done = 0.0, 0, False
    while not done:
        pos0, b0, packs0, t0 = env.pos.tolist(), float(env.b), int(env.packs), int(env.t)
        a, top, changed = model["act"](env, obs)
        env.tick_log.clear()
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        cum += r
        ticks = list(env.tick_log)
        if ticks:
            ticks[-1] = {**ticks[-1], "urg": env.urg.tolist(), "left": env.left.tolist()}
        target = FAC[a - 3].tolist() if 3 <= a <= 8 else [0.0, 0.0] if a == 9 else pos0
        step = {
            "i": i, "action": a, "name": ACTION_LABELS[a], "t0": t0, "t1": int(env.t),
            "pos0": pos0, "pos1": env.pos.tolist(), "target": target,
            "b0": b0, "b1": float(env.b), "packs0": packs0, "packs1": int(env.packs),
            "dist": info["dist_km"], "storm": bool(info["storm_route"]), "energy": info["energy_km"],
            "ticks": info["flight_ticks"],
            "reward": float(r), "cum": float(cum), "breakdown": info["breakdown"],
            "delivery": env.last_delivery, "new_orders": info["new_orders"], "n_expired": info["n_expired"],
            "forced": info["breakdown"]["forced_landing"] < 0, "forced_total": int(env.forced),
            "top": top, "mask_changed": bool(changed), "done": done, "terminated": bool(term),
        }
        yield "step", step, ticks
        i += 1


def export_json(path, model: dict, seed: int, p_order: float = P_ORDER):
    trace = {"meta": None, "steps": []}
    for item in episode(model, seed, p_order):
        if item[0] == "init":
            trace["meta"] = item[1]
        else:
            trace["steps"].append({"step": item[1], "ticks": item[2]})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(trace, default=lambda x: x.item()))
    return trace


# ---------- verbose terminal log ----------

def clock(t: float) -> str:
    minutes = 6 * 60 + int(round(t * 3))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _orders(urg, left) -> str:
    out = [f"{SHORT[k]}:{'EMERG' if u == 2 else 'routine'} {left[k]}t" for k, u in enumerate(urg) if u]
    return ", ".join(out) or "none"


def format_step(step: dict, book: dict) -> str:
    bd = {k: v for k, v in step["breakdown"].items() if abs(v) > 1e-9}
    main = max(bd, key=lambda k: abs(bd[k])) if bd else None
    comp = f"{main} {bd[main]:+.2f}" if main else ""
    storm = " STORM" if step["storm"] else ""
    flag = " [mask]" if step["mask_changed"] else ""
    return (f"{clock(step['t0'])} t={step['t0']:3d}  {step['name']:<17s}{flag} r={step['reward']:+7.2f} "
            f"({comp:<20s}) R={step['cum']:+8.2f}  bat={step['b1']:5.1f}km  cargo={step['packs1']}/2  "
            f"orders: {_orders(book['urg'], book['left'])}{storm}")


class Summary:
    def __init__(self):
        self.deliveries = self.emergencies = self.wasted = self.invalid = self.expired = self.masked = 0
        self.ret, self.forced, self.steps, self.t = 0.0, 0, 0, 0

    def add(self, step: dict):
        self.steps += 1
        self.ret, self.t, self.forced = step["cum"], step["t1"], step["forced_total"]
        if step["delivery"]:
            self.deliveries += 1
            self.emergencies += step["delivery"]["emergency"]
        self.wasted += step["breakdown"]["wasted"] < 0
        self.invalid += step["breakdown"]["invalid"] < 0
        self.expired += step["n_expired"]
        self.masked += step["mask_changed"]

    def text(self, label: str, seed: int) -> str:
        return (f"\n=== Episode summary: {label}, seed {seed} ===\n"
                f"  return {self.ret:+.2f} over {self.steps} decisions, shift ended {clock(self.t)} (t={self.t})\n"
                f"  deliveries {self.deliveries} (emergencies {self.emergencies}), expired {self.expired}, "
                f"wasted {self.wasted}, invalid {self.invalid}, forced landings {self.forced}\n"
                f"  decisions changed by invalid-action masking: {self.masked}")


def run_terminal(model: dict, seed: int, speed: float | None = None, p_order: float = P_ORDER):
    """Runs one episode in the terminal; with speed set, paces it in simulated ticks per second."""
    import time

    summary = Summary()
    for item in episode(model, seed, p_order):
        if item[0] == "init":
            print(f"SkyBlood | policy {model['label']} | seed {seed} | p_order {p_order} | "
                  "greedy with invalid-action masking")
            continue
        step, ticks = item[1], item[2]
        print(format_step(step, ticks[-1] if ticks else {"urg": [0] * 6, "left": [0] * 6}))
        summary.add(step)
        if speed:
            time.sleep((step["t1"] - step["t0"]) / speed)
    print(summary.text(model["label"], seed))
    return summary


# ---------- server ----------

def make_app(device: str = "cpu", p_order: float = P_ORDER):
    """p_order is the demand used when a client does not send one (play.py passes its --p-order)."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles

    default_p_order = p_order
    app = FastAPI(title="SkyBlood viewer")

    @app.middleware("http")
    async def revalidate(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.websocket("/ws")
    async def ws(websocket: WebSocket, algo: str = "heuristic", config: str = "best", seed: int = 10003,
                 speed: float = 4.0, p_order: float | None = None):
        await websocket.accept()
        model = load_model(algo, config, device)
        summary = Summary()
        try:
            await _stream(websocket, model, seed, speed, summary, p_order if p_order is not None else default_p_order)
        except WebSocketDisconnect:
            print("viewer disconnected", flush=True)

    app.mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer")
    return app


async def _stream(websocket, model: dict, seed: int, speed: float, summary: Summary, p_order: float = P_ORDER):
    for item in episode(model, seed, p_order):
        if item[0] == "init":
            print(f"\nviewer connected | policy {model['label']} | seed {seed} | p_order {p_order} | "
                  f"speed {speed} ticks/s")
            await websocket.send_json({"type": "init", "meta": item[1]})
            continue
        step, ticks = item[1], item[2]
        await websocket.send_json({"type": "step", "step": step, "ticks": ticks})
        print(format_step(step, ticks[-1] if ticks else {"urg": [0] * 6, "left": [0] * 6}), flush=True)
        summary.add(step)
        await asyncio.sleep((step["t1"] - step["t0"]) / max(speed, 1e-3))
    await websocket.send_json({"type": "end"})
    print(summary.text(model["label"], seed), flush=True)
    await websocket.close()
