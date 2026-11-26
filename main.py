import os

# One compute thread per run (set before numpy/torch load), so several runs can share the CPU.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]


def main():
    p = argparse.ArgumentParser(description="SkyBlood: RL blood-delivery dispatcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train one config (one run per command)")
    t.add_argument("--algo", choices=ALGOS, required=True)
    t.add_argument("--config", required=True, help="e.g. D1, R3, P1, A10")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu", help="cpu or cuda")
    t.add_argument("--frames", type=int, default=None, help="smoke tests only; default 1,000,000")
    t.add_argument("--masked", action="store_true", help="train with action masking (use for all four algorithms)")

    e = sub.add_parser("eval", help="masked-greedy evaluation on held-out seeds 10000-10019")
    e.add_argument("--algo", choices=ALGOS, required=True)
    e.add_argument("--config", required=True)
    e.add_argument("--device", default="cpu")

    sub.add_parser("analyze", help="figures and summary tables from logs/ into assets/")

    b = sub.add_parser("baselines", help="heuristic / hold / random / masked-random returns")
    b.add_argument("--first-seed", type=int, default=10000)
    b.add_argument("--n-seeds", type=int, default=20)
    b.add_argument("--p-order", type=float, default=None)

    args = p.parse_args()
    if args.cmd == "train":
        if args.algo == "dqn":
            from training.dqn_training import train
            train(args.config, args.seed, args.device, args.frames, args.masked)
        else:
            from training.pg_training import train
            train(args.algo, args.config, args.seed, args.device, args.frames, args.masked)
    elif args.cmd == "eval":
        from environment.rendering import print_eval
        print_eval(args.algo, args.config, args.device)
    elif args.cmd == "analyze":
        from training.analysis import analyze
        analyze()
    elif args.cmd == "baselines":
        from environment.custom_env import P_ORDER, print_baselines
        print_baselines(args.first_seed, args.n_seeds, args.p_order or P_ORDER)


if __name__ == "__main__":
    main()
