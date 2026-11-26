import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import webbrowser  # noqa: E402
from urllib.parse import urlencode  # noqa: E402

import numpy as np  # noqa: E402

from environment.custom_env import P_ORDER  # noqa: E402
from environment.rendering import VIEWER_DIR, episode, export_json, load_model, make_app, run_terminal  # noqa: E402

HOST, PORT = "127.0.0.1", 8765
DEMO_SPEED = 1.33  # 240 ticks in about 3 minutes
HELD_OUT = range(10000, 10020)


def representative_seed(model: dict, p_order: float = P_ORDER) -> int:
    """Held-out seed whose return is closest to the policy's median return."""
    returns = {s: episode_return(model, s, p_order) for s in HELD_OUT}
    median = np.median(list(returns.values()))
    return min(returns, key=lambda s: abs(returns[s] - median))


def episode_return(model: dict, seed: int, p_order: float = P_ORDER) -> float:
    cum = 0.0
    for item in episode(model, seed, p_order):
        if item[0] == "step":
            cum = item[1]["cum"]
    return cum


def serve(device: str, p_order: float):
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(make_app(device, p_order), host=HOST, port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise SystemExit(f"port {PORT} is busy: stop the other play.py first")
        time.sleep(0.05)


def main():
    p = argparse.ArgumentParser(description="Run a trained SkyBlood agent (masked greedy) in the 3D viewer")
    p.add_argument("--algo", choices=["dqn", "reinforce", "ppo", "a2c", "heuristic"], default="ppo")
    p.add_argument("--config", default="best", help="e.g. P1, or 'best' (highest final eval in logs/)")
    p.add_argument("--seed", type=int, default=10003)
    p.add_argument("--speed", type=float, default=4.0, help="simulated ticks per second (1 tick = 3 min)")
    p.add_argument("--demo", action="store_true", help="representative held-out seed, ~3-minute shift")
    p.add_argument("--no-browser", action="store_true", help="terminal only: no server, no browser")
    p.add_argument("--export", metavar="NAME", help="write the trace to environment/viewer/traces/NAME")
    p.add_argument("--p-order", type=float, default=P_ORDER,
                   help=f"order rate per tick per free hospital (training: {P_ORDER}; 0.02 doubles demand)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    model = load_model(args.algo, args.config, args.device)
    config = model.get("config", "-")
    if args.demo:
        args.seed, args.speed = representative_seed(model, args.p_order), DEMO_SPEED
        print(f"demo: representative held-out seed {args.seed} (closest to the median return), "
              f"{args.speed} ticks/s ≈ {240 / args.speed / 60:.1f} min per shift")

    if args.export:
        path = VIEWER_DIR / "traces" / args.export
        export_json(path, model, args.seed, args.p_order)
        print(f"trace written to {path}\n  replay: http://{HOST}:{PORT}/?ep=traces/{args.export}")

    if args.no_browser:
        run_terminal(model, args.seed, args.speed if args.demo else None, args.p_order)
        return

    serve(args.device, args.p_order)
    query = urlencode({"mode": "ws", "algo": args.algo, "config": config, "seed": args.seed, "speed": args.speed,
                       "p_order": args.p_order})
    url = f"http://{HOST}:{PORT}/?{query}"
    print(f"viewer: {url}\n(the per-step log prints here once the page connects; Ctrl+C to quit)", flush=True)
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
