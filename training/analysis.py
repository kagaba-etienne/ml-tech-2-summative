"""`main.py analyze`: figures and summary tables from logs/ into assets/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from environment.custom_env import baseline_returns  # noqa: E402
from training import EVAL_SEEDS, ROOT, TOTAL_FRAMES, evaluate  # noqa: E402

ALGOS = ["dqn", "reinforce", "ppo", "a2c"]
NAMES = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C"}
PREFIX = {"dqn": "D", "reinforce": "R", "ppo": "P", "a2c": "A"}
COLOR = {"dqn": "#2a78d6", "reinforce": "#eb6834", "ppo": "#1baf7a", "a2c": "#eda100"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SMOOTH = 100  # episodes in the rolling mean
HARD_P_ORDER = 0.02

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "lines.linewidth": 1.6,
})


# ---------- loading ----------

def load_runs(logs_root: Path) -> dict:
    """{algo: {config: {"cfg", "episodes", "train", "eval"}}}, skipping and reporting missing runs."""
    from training.dqn_training import CONFIGS as DQN
    from training.pg_training import CONFIGS as PG

    expected = {"dqn": [c["name"] for c in DQN], **{a: [c["name"] for c in PG[a]] for a in PG}}
    runs, missing = {}, []
    for algo in ALGOS:
        runs[algo] = {}
        for name in expected[algo]:
            d = logs_root / algo / name
            files = [d / f for f in ("config.json", "episodes.csv", "train.csv", "eval.csv")]
            if not all(f.exists() and f.stat().st_size for f in files):
                missing.append(f"{algo}/{name}")
                continue
            runs[algo][name] = {
                "cfg": json.loads(files[0].read_text()),
                "episodes": pd.read_csv(files[1]),
                "train": pd.read_csv(files[2]),
                "eval": pd.read_csv(files[3]),
            }
    if missing:
        print(f"missing runs (skipped): {', '.join(missing)}")
    return runs


def smoothed(ep: pd.DataFrame) -> pd.DataFrame:
    s = ep[["frames", "return"]].copy()
    s["smooth"] = s["return"].rolling(SMOOTH, min_periods=SMOOTH).mean()
    return s.dropna()


def thin(df: pd.DataFrame, n: int = 600) -> pd.DataFrame:
    return df.iloc[:: max(1, len(df) // n)]


def k_frames(ax):
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x else "0"))


def ref_line(ax, y, label, style="--"):
    ax.axhline(y, color=MUTED, linestyle=style, linewidth=1.1, zorder=1)
    ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=MUTED, annotation_clip=False)


# ---------- metrics ----------

def run_metrics(run: dict, target: float) -> dict:
    s = smoothed(run["episodes"])
    hit = s[s["smooth"] >= target]
    last = run["episodes"][run["episodes"]["frames"] >= 0.9 * run["cfg"]["total_frames"]]
    final = run["eval"].iloc[-1]
    ev_hit = run["eval"][run["eval"]["eval_mean"] >= target]
    return {
        "final_eval_mean": float(final["eval_mean"]), "final_eval_std": float(final["eval_std"]),
        "best_eval_during_training": float(run["eval"]["eval_mean"].max()),
        "frames_to_90pct": int(hit["frames"].iloc[0]) if len(hit) else None,
        "frames_to_90pct_greedy_eval": int(ev_hit["frames"].iloc[0]) if len(ev_hit) else None,
        "plateau_return": float(last["return"].mean()) if len(last) else float("nan"),
        "deliveries": float(final["ep_deliveries"]), "emergencies": float(final["ep_emergencies"]),
        "expired": float(final["ep_expired"]), "forced_landings": float(final["ep_forced"]),
    }


def best_config(algo_runs: dict, metrics: dict) -> str | None:
    if not algo_runs:
        return None
    return max(algo_runs, key=lambda n: metrics[n]["final_eval_mean"])


# ---------- figures ----------

def fig_rewards_grid(algo, algo_runs, heuristic, out):
    base = f"{PREFIX[algo]}1"
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        name = f"{PREFIX[algo]}{i + 1}"
        if name in algo_runs and name != base and base in algo_runs:
            b = thin(smoothed(algo_runs[base]["episodes"]))
            ax.plot(b["frames"], b["smooth"], color="#a8a7a1", linewidth=1.3, label=f"{base} (baseline)")
        if name in algo_runs:
            s = thin(smoothed(algo_runs[name]["episodes"]))
            ax.plot(s["frames"], s["smooth"], color=COLOR[algo], label=name)
            ax.set_title(f"{name}: {algo_runs[name]['cfg']['note']}", loc="left", fontsize=10)
        else:
            ax.set_title(f"{name}: missing", loc="left", fontsize=10, color=MUTED)
        ax.axhline(heuristic, color=MUTED, linestyle="--", linewidth=1.0)
        k_frames(ax)
    for ax in axes[1]:
        ax.set_xlabel("Training frames (decisions)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"Episode return ({SMOOTH}-ep mean)")
    handles = [plt.Line2D([], [], color=COLOR[algo]), plt.Line2D([], [], color="#a8a7a1"),
               plt.Line2D([], [], color=MUTED, linestyle="--", linewidth=1.0)]
    fig.legend(handles, ["this config", f"{base} baseline config", f"greedy heuristic ({heuristic:.1f})"],
               loc="upper right", ncol=3, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle(f"{NAMES[algo]}: training return per hyperparameter configuration (1M frames each)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_rewards_comparison(runs, metrics, base_means, out):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ends = []
    for algo in ALGOS:
        best = best_config(runs[algo], metrics[algo])
        if not best:
            continue
        s = thin(smoothed(runs[algo][best]["episodes"]))
        ax.plot(s["frames"], s["smooth"], color=COLOR[algo], label=f"{NAMES[algo]} ({best})")
        ends.append([float(s["smooth"].iloc[-30:].mean()), f"{NAMES[algo]} {best}", float(s["frames"].iloc[-1])])
    ends.sort(reverse=True)
    for i in range(1, len(ends)):  # keep end labels at least 14 return units apart
        ends[i][0] = min(ends[i][0], ends[i - 1][0] - 14)
    for y, text, x in ends:
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ref_line(ax, base_means["heuristic"], f"heuristic {base_means['heuristic']:.0f}")
    ref_line(ax, base_means["masked_random"], f"masked random {base_means['masked_random']:.0f}", ":")
    ref_line(ax, base_means["random"], f"random {base_means['random']:.0f}", ":")
    ax.set_xlabel("Training frames (decisions)")
    ax.set_ylabel(f"Episode return ({SMOOTH}-episode rolling mean)")
    ax.set_title("Best configuration of each algorithm vs baselines (training return, includes exploration)",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower right", ncol=2, bbox_to_anchor=(0.9, 0.1))
    k_frames(ax)
    ax.set_xlim(0, TOTAL_FRAMES * 1.12)
    fig.savefig(out)
    plt.close(fig)


def fig_dqn_objective(dqn_runs, best, out):
    picks = [n for n in [best, "D2", "D9", "D10"] if n in dqn_runs]
    picks = list(dict.fromkeys(picks))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for color, name in zip(palette, picks):
        t = dqn_runs[name]["train"]
        roll = max(1, len(t) // 200)
        sm = t.rolling(roll * 5, min_periods=1).mean().iloc[::roll]
        label = f"{name}: {dqn_runs[name]['cfg']['note']}" + (" (best)" if name == best else "")
        a1.plot(sm["frames"], sm["loss"], color=color, label=label)
        a2.plot(sm["frames"], sm["mean_q"], color=color, label=label)
    a1.set_yscale("log")
    a1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a1.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_yticks([5, 10, 20, 30])
    a1.set_title("TD loss (smooth L1, log scale)", loc="left")
    a1.set_ylabel("Loss")
    a2.set_title("Mean Q of the chosen action", loc="left")
    a2.set_ylabel("Q-value")
    for ax in (a1, a2):
        ax.set_xlabel("Training frames (decisions)")
        k_frames(ax)
    a2.legend(loc="lower right", fontsize=9)
    fig.suptitle("DQN objective: best configuration and contrasting settings", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _entropy_curve(run):
    t = run["train"][["frames", "entropy"]]
    roll = max(1, len(t) // 300)
    return t.rolling(roll * 3, min_periods=1).mean().iloc[::roll]


def fig_pg_entropy(runs, metrics, out):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for algo in ["reinforce", "ppo", "a2c"]:
        best = best_config(runs[algo], metrics[algo])
        if not best:
            continue
        e = _entropy_curve(runs[algo][best])
        ax.plot(e["frames"], e["entropy"], color=COLOR[algo], label=f"{NAMES[algo]} ({best})")
    ax.axhline(np.log(10), color=MUTED, linestyle=":", linewidth=1.0)
    ax.annotate("uniform over 10 actions (ln 10)", (0.01, np.log(10)), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xlabel("Training frames (decisions)")
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_title("Policy entropy of the best REINFORCE, PPO and A2C configurations", loc="left", fontweight="bold")
    ax.legend()
    k_frames(ax)
    fig.savefig(out)
    plt.close(fig)


def fig_pg_entropy_grid(algo, algo_runs, out):
    base = f"{PREFIX[algo]}1"
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        name = f"{PREFIX[algo]}{i + 1}"
        if name in algo_runs and name != base and base in algo_runs:
            b = _entropy_curve(algo_runs[base])
            ax.plot(b["frames"], b["entropy"], color="#a8a7a1", linewidth=1.3)
        if name in algo_runs:
            e = _entropy_curve(algo_runs[name])
            ax.plot(e["frames"], e["entropy"], color=COLOR[algo])
            ax.set_title(f"{name}: {algo_runs[name]['cfg']['note']}", loc="left", fontsize=10)
        else:
            ax.set_title(f"{name}: missing", loc="left", fontsize=10, color=MUTED)
        k_frames(ax)
    for ax in axes[1]:
        ax.set_xlabel("Training frames (decisions)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Policy entropy (nats)")
    fig.legend([plt.Line2D([], [], color=COLOR[algo]), plt.Line2D([], [], color="#a8a7a1")],
               ["this config", f"{base} baseline config"], loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.02))
    fig.suptitle(f"{NAMES[algo]}: policy entropy per configuration", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_convergence(rows, target, out):
    df = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 9.5), sharey=True)
    y = np.arange(len(df))[::-1]
    colors = [COLOR[a] for a in df["algo"]]
    reached = df["frames_to_90pct"].notna()
    a1.barh(y[reached], df.loc[reached, "frames_to_90pct"], color=np.array(colors)[reached], height=0.7)
    for yi in y[~reached]:
        a1.text(TOTAL_FRAMES * 0.01, yi, "not reached", va="center", fontsize=8.5, color=MUTED)
    a1.set_xlim(0, TOTAL_FRAMES)
    a1.set_xlabel(f"Frames until the {SMOOTH}-episode training return first reaches {target:.0f} (90% of heuristic)")
    a1.set_title("Speed: frames to 90% of the heuristic (training return, with exploration)", loc="left")
    k_frames(a1)
    a2.barh(y, df["plateau_return"], color=colors, height=0.7)
    a2.axvline(target / 0.9, color=MUTED, linestyle="--", linewidth=1.0)
    a2.set_xlabel("Mean training return over the last 10% of frames")
    a2.set_title("Plateau return (dashed: heuristic)", loc="left")
    a1.set_yticks(y)
    a1.set_yticklabels(df["config"])
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[a]) for a in ALGOS]
    fig.legend(handles, [NAMES[a] for a in ALGOS], loc="upper right", ncol=4, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Convergence of all 40 runs", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_generalization(gen, out):
    panels = [("Held-out seeds 10000–10019 (training conditions, p_order = 0.01)", "normal"),
              (f"Harder variant: twice the order rate (p_order = {HARD_P_ORDER}), same seeds", "hard")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=False)
    for ax, (title, key) in zip(axes, panels):
        labels, data, colors = zip(*[(lab, vals[key], col) for lab, vals, col in gen])
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.4},
                        whiskerprops={"color": MUTED}, capprops={"color": MUTED},
                        flierprops={"markeredgecolor": MUTED, "markersize": 4})
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_edgecolor(MUTED)
            patch.set_alpha(0.85)
        for i, d in enumerate(data, start=1):
            ax.annotate(f"{np.mean(d):.0f}", (i, np.max(d)), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Episode return (masked greedy)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Generalisation: best model of each algorithm vs baselines (numbers = mean)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------- tables ----------

HYPER = {
    "dqn": ["lr", "gamma", "buffer", "batch", "eps_frac", "target", "double"],
    "reinforce": ["lr", "gamma", "episodes", "baseline", "entropy", "adv_norm"],
    "ppo": ["lr", "fpb", "minibatch", "epochs", "clip", "entropy", "lmbda"],
    "a2c": ["lr", "steps", "gamma", "lmbda", "entropy", "value_coef"],
}


def fmt_frames(f):
    return "not reached" if f is None or pd.isna(f) else f"{int(f):,}"


def write_summary(algo, algo_runs, metrics, assets):
    rows = []
    for name, run in algo_runs.items():
        m = metrics[name]
        rows.append({"config": name, **{h: run["cfg"][h] for h in HYPER[algo]}, "note": run["cfg"]["note"], **m})
    df = pd.DataFrame(rows)
    df.to_csv(assets / f"summary_{algo}.csv", index=False)
    head = ["Run", *HYPER[algo], "Note", "Final eval (mean ± std)", "Frames to 90% (training)",
            "Frames to 90% (greedy eval)", "Plateau return", "Best eval during training"]
    lines = [f"# {NAMES[algo]}: hyperparameters and results", "",
             "Final eval = final checkpoint, greedy with invalid-action masking on held-out seeds 10000–10019. "
             "Frames to 90% (training) = first frame where the 100-episode mean training return (sampled actions, "
             "so it includes exploration) reaches 90% of the heuristic; (greedy eval) = first 100k-frame evaluation "
             "at or above that level. "
             "Plateau = mean training return over the last 10% of frames. "
             "Best eval during training is selected on the same seeds (optimistic).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["config"], *[str(r[h]) for h in HYPER[algo]], r["note"],
                 f"{r['final_eval_mean']:.1f} ± {r['final_eval_std']:.1f}", fmt_frames(r["frames_to_90pct"]),
                 fmt_frames(r["frames_to_90pct_greedy_eval"]),
                 f"{r['plateau_return']:.1f}", f"{r['best_eval_during_training']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    (assets / f"summary_{algo}.md").write_text("\n".join(lines) + "\n")


def write_summary_best(runs, metrics, gen, base_means, assets):
    lines = ["# Best configuration of each algorithm", "",
             "Best = highest final eval within the algorithm's 10-config grid. Same 1M-frame budget, training seed 0.",
             "", "| Algorithm | Config | Change from baseline | Final eval (mean ± std) | Hard variant (p_order 0.02) "
             "| Frames to 90% (training) | Frames to 90% (greedy eval) | Plateau | Deliveries | Emergencies | Expired "
             "| Forced landings |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    gen_by = {lab: vals for lab, vals, _ in gen}
    for algo in ALGOS:
        best = best_config(runs[algo], metrics[algo])
        if not best:
            continue
        m, hard = metrics[algo][best], gen_by[f"{NAMES[algo]} {best}"]["hard"]
        lines.append(f"| {NAMES[algo]} | {best} | {runs[algo][best]['cfg']['note']} | "
                     f"{m['final_eval_mean']:.1f} ± {m['final_eval_std']:.1f} | {np.mean(hard):.1f} ± {np.std(hard):.1f} | "
                     f"{fmt_frames(m['frames_to_90pct'])} | {fmt_frames(m['frames_to_90pct_greedy_eval'])} | "
                     f"{m['plateau_return']:.1f} | {m['deliveries']:.2f} | "
                     f"{m['emergencies']:.2f} | {m['expired']:.2f} | {m['forced_landings']:.2f} |")
    lines += ["", "Baselines on the same held-out seeds:", ""]
    lines += [f"- {k}: {v:.1f}" for k, v in base_means.items()]
    (assets / "summary_best.md").write_text("\n".join(lines) + "\n")


# ---------- paired load test and behaviour evidence ----------

STRESS_SEEDS = list(range(20000, 20200))  # never used in training or model selection
STRESS_LOADS = [0.01, 0.02]


def stress_test(best: dict, assets: Path):
    """Paired comparison with the heuristic on the same 200 unseen seeds, at the nominal and double order rate."""
    from environment.rendering import load_net

    rows = []
    for p in STRESS_LOADS:
        h = baseline_returns("heuristic", STRESS_SEEDS, p)
        for algo, cfg in best.items():
            r = evaluate(load_net(algo, cfg), seeds=STRESS_SEEDS, p_order=p)["returns"]
            d = r - h
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"p_order": p, "algo": algo, "config": cfg, "agent_mean": r.mean(), "heuristic_mean": h.mean(),
                         "diff_mean": d.mean(), "ci_low": d.mean() - 1.96 * se, "ci_high": d.mean() + 1.96 * se,
                         "agent_better_pct": 100 * float(np.mean(d > 0))})
    df = pd.DataFrame(rows)
    df.round(2).to_csv(assets / "stress_test.csv", index=False)
    lines = ["# Paired comparison with the greedy heuristic", "",
             f"Same {len(STRESS_SEEDS)} unseen seeds ({STRESS_SEEDS[0]}–{STRESS_SEEDS[-1]}) for agent and heuristic; "
             "difference = agent − heuristic per seed; 95% CI = mean ± 1.96 standard errors. "
             f"p_order {STRESS_LOADS[1]} doubles the order rate and was never seen in training.", "",
             "| Order rate | Algorithm | Config | Agent mean | Heuristic mean | Difference (95% CI) | Agent better on |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['p_order']} | {NAMES[r['algo']]} | {r['config']} | {r['agent_mean']:.1f} | "
                     f"{r['heuristic_mean']:.1f} | {r['diff_mean']:+.1f} ({r['ci_low']:+.1f} to {r['ci_high']:+.1f}) | "
                     f"{r['agent_better_pct']:.0f}% of seeds |")
    (assets / "stress_test.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, p in zip(axes, STRESS_LOADS):
        sub = df[df["p_order"] == p].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        for yi, (_, r) in zip(y, sub.iterrows()):
            ax.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=COLOR[r["algo"]], linewidth=2.5)
            ax.plot(r["diff_mean"], yi, "o", color=COLOR[r["algo"]], markersize=8, markeredgecolor="white")
            ax.annotate(f"{r['diff_mean']:+.1f}", (r["ci_high"], yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK,
                        bbox={"boxstyle": "square,pad=0.15", "facecolor": "white", "edgecolor": "none"})
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{NAMES[a]} {c}" for a, c in zip(sub["algo"], sub["config"])])
        ax.set_xlabel("Return difference vs heuristic (agent − heuristic)")
        load = "nominal order rate" if p == STRESS_LOADS[0] else "double order rate (unseen in training)"
        ax.set_title(f"p_order = {p}: {load}", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        lo, hi = min(df["ci_low"].min(), -5), df["ci_high"].max()
        ax.set_xlim(lo - 5, hi + 18)
    fig.suptitle(f"Paired comparison with the greedy heuristic on {len(STRESS_SEEDS)} unseen seeds (mean and 95% CI)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(assets / "stress_test.png")
    plt.close(fig)


def behaviour(models: list, assets: Path):
    """Dispatcher behaviour of each policy over the held-out seeds, from full episode traces."""
    from environment.rendering import episode

    rows = []
    for label, model in models:
        c = dict(eps=0, sorties=0, two_unit=0, multi_drop=0, swaps=0, launch_battery=0.0, flights=0, storm=0,
                 choice=0, emerg_first=0, holds=0, invalid=0, masked=0, deliveries=0, emergencies=0, expired=0, wasted=0)
        for seed in EVAL_SEEDS:
            c["eps"] += 1
            book = None
            for item in episode(model, seed):
                if item[0] == "init":
                    book = item[1]["start"]
                    continue
                st, ticks = item[1], item[2]
                a, hub = st["action"], st["pos0"] == [0.0, 0.0]
                if 3 <= a <= 8 and st["packs0"] > 0:
                    urg = book["urg"]
                    if 2 in urg and 1 in urg:
                        c["choice"] += 1
                        c["emerg_first"] += urg[a - 3] == 2
                    if hub:
                        c["sorties"] += 1
                        c["two_unit"] += st["packs0"] == 2
                        c["launch_battery"] += st["b0"]
                    else:
                        c["multi_drop"] += 1
                if (3 <= a <= 9) and st["dist"] > 0:
                    c["flights"] += 1
                    c["storm"] += st["storm"]
                c["swaps"] += a == 2 and st["breakdown"]["invalid"] == 0
                c["holds"] += a == 0
                c["invalid"] += st["breakdown"]["invalid"] < 0
                c["masked"] += st["mask_changed"]
                c["deliveries"] += st["delivery"] is not None
                c["emergencies"] += bool(st["delivery"] and st["delivery"]["emergency"])
                c["expired"] += st["n_expired"]
                c["wasted"] += st["breakdown"]["wasted"] < 0
                if ticks:
                    book = ticks[-1]
        n = c["eps"]
        rows.append({
            "policy": label, "deliveries": c["deliveries"] / n, "emergencies": c["emergencies"] / n,
            "expired": c["expired"] / n, "wasted_units": c["wasted"] / n, "sorties": c["sorties"] / n,
            "two_unit_sortie_pct": 100 * c["two_unit"] / max(c["sorties"], 1),
            "fly_on_drops": c["multi_drop"] / n, "swaps": c["swaps"] / n,
            "battery_at_launch_km": c["launch_battery"] / max(c["sorties"], 1),
            "storm_flight_pct": 100 * c["storm"] / max(c["flights"], 1),
            "emergency_first_pct": 100 * c["emerg_first"] / max(c["choice"], 1), "holds": c["holds"] / n,
            "invalid_actions": c["invalid"] / n, "mask_changed": c["masked"] / n,
        })
    df = pd.DataFrame(rows)
    df.round(2).to_csv(assets / "behaviour.csv", index=False)
    head = ["Policy", "Deliveries", "Emergencies", "Expired", "Wasted units", "Sorties from hub", "2-unit sorties",
            "Fly-on drops", "Battery swaps", "Battery at launch", "Flights through storm",
            "Emergency chosen first*", "HOLDs", "Invalid actions", "Masking changed choice"]
    lines = ["# Dispatcher behaviour per episode (held-out seeds 10000–10019, masked greedy)", "",
             "*When an emergency and a routine order were both open at a FLY_DROP decision, the share of drops "
             "that went to the emergency. Fly-on drops = a second drop made without returning to the hub. "
             "Counts are per episode; battery at launch is in km (max 160).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['deliveries']:.1f} | {r['emergencies']:.1f} | {r['expired']:.2f} | "
            f"{r['wasted_units']:.1f} | {r['sorties']:.1f} | "
            f"{r['two_unit_sortie_pct']:.0f}% | {r['fly_on_drops']:.1f} | {r['swaps']:.1f} | "
            f"{r['battery_at_launch_km']:.0f} | {r['storm_flight_pct']:.0f}% | {r['emergency_first_pct']:.0f}% | "
            f"{r['holds']:.1f} | {r['invalid_actions']:.1f} | {r['mask_changed']:.1f} |")
    (assets / "behaviour.md").write_text("\n".join(lines) + "\n")


# ---------- entry point ----------

def analyze(logs_root=None, assets=None):
    logs_root = Path(logs_root or ROOT / "logs")
    assets = Path(assets or ROOT / "assets")
    assets.mkdir(parents=True, exist_ok=True)
    runs = load_runs(logs_root)
    if not any(runs.values()):
        print("no runs found in logs/: nothing to analyze")
        return

    print("baselines on held-out seeds ...")
    base = {n: baseline_returns(n, EVAL_SEEDS) for n in ("heuristic", "random", "masked_random", "hold")}
    base_means = {n: float(v.mean()) for n, v in base.items()}
    target = 0.9 * base_means["heuristic"]
    metrics = {a: {n: run_metrics(r, target) for n, r in runs[a].items()} for a in ALGOS}

    for algo in ALGOS:
        if runs[algo]:
            fig_rewards_grid(algo, runs[algo], base_means["heuristic"], assets / f"rewards_{algo}.png")
            write_summary(algo, runs[algo], metrics[algo], assets)
            if algo != "dqn":
                fig_pg_entropy_grid(algo, runs[algo], assets / f"pg_entropy_{algo}.png")
    fig_rewards_comparison(runs, metrics, base_means, assets / "rewards_comparison.png")
    if runs["dqn"]:
        fig_dqn_objective(runs["dqn"], best_config(runs["dqn"], metrics["dqn"]), assets / "dqn_objective.png")
    fig_pg_entropy(runs, metrics, assets / "pg_entropy.png")

    conv = [{"algo": a, "config": n, "frames_to_90pct": metrics[a][n]["frames_to_90pct"],
             "frames_to_90pct_greedy_eval": fmt_frames(metrics[a][n]["frames_to_90pct_greedy_eval"]),
             "plateau_return": metrics[a][n]["plateau_return"], "final_eval_mean": metrics[a][n]["final_eval_mean"]}
            for a in ALGOS for n in runs[a]]
    pd.DataFrame(conv).assign(frames_to_90pct=lambda d: d["frames_to_90pct"].map(fmt_frames)).to_csv(
        assets / "convergence.csv", index=False)
    fig_convergence(conv, target, assets / "convergence.png")

    print("generalisation: evaluating best models ...")
    from environment.rendering import load_net

    gen = []
    for algo in ALGOS:
        best = best_config(runs[algo], metrics[algo])
        if best:
            net = load_net(algo, best)
            gen.append((f"{NAMES[algo]} {best}", {
                "normal": evaluate(net)["returns"], "hard": evaluate(net, p_order=HARD_P_ORDER)["returns"]},
                COLOR[algo]))
    for name, label in [("heuristic", "Greedy heuristic"), ("masked_random", "Masked random"), ("random", "Random")]:
        gen.append((label, {"normal": base[name], "hard": baseline_returns(name, EVAL_SEEDS, HARD_P_ORDER)},
                    "#c3c2b7"))
    fig_generalization(gen, assets / "generalization.png")
    write_summary_best(runs, metrics, gen, base_means, assets)

    from environment.rendering import load_model

    best = {a: best_config(runs[a], metrics[a]) for a in ALGOS if runs[a]}
    print(f"paired load test on {len(STRESS_SEEDS)} unseen seeds ...")
    stress_test(best, assets)
    print("behaviour evidence from traces ...")
    behaviour([("Greedy heuristic", load_model("heuristic"))]
              + [(f"{NAMES[a]} {c}", load_model(a, c)) for a, c in best.items()], assets)
    print(f"wrote figures and tables to {assets}")
