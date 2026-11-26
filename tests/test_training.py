import csv
import json

import pytest

from environment.rendering import episode, export_json, load_model
from training import EVAL_SEEDS, evaluate


def _rows(path):
    return list(csv.DictReader(open(path)))


@pytest.mark.parametrize("algo,config", [("dqn", "D1"), ("reinforce", "R7"), ("ppo", "P1"), ("a2c", "A1"), ("a2c", "A7")])
def test_smoke_training_writes_logs_and_checkpoint(tmp_path, algo, config):
    if algo == "dqn":
        from training.dqn_training import train
        run = train(config, frames=300, logs_root=tmp_path / "logs", models_root=tmp_path / "models")
    else:
        from training.pg_training import train
        run = train(algo, config, frames=300, logs_root=tmp_path / "logs", models_root=tmp_path / "models")
    d = tmp_path / "logs" / algo / config
    assert json.loads((d / "config.json").read_text())["name"] == config
    assert _rows(d / "train.csv")
    ev = _rows(d / "eval.csv")
    assert len(ev) == 1 and set(ev[0]) >= {"frames", "eval_mean", "eval_std"}
    assert run.ckpt_path.exists() and run.ckpt_path.with_name(run.ckpt_path.stem + "_best.pt").exists()
    r = evaluate(run.net, seeds=EVAL_SEEDS[:2])["returns"]
    assert len(r) == 2


def test_trace_export(tmp_path):
    trace = export_json(tmp_path / "t.json", load_model("heuristic"), seed=10003)
    assert trace["meta"]["seed"] == 10003 and len(trace["meta"]["facilities"]) == 6
    steps = trace["steps"]
    assert steps[-1]["step"]["t1"] == 240 and all(s["ticks"] for s in steps)
    for s in steps:
        if s["step"]["delivery"]:
            k = s["step"]["delivery"]["k"]
            assert s["ticks"][-1]["urg"][k] == 0
    assert json.loads((tmp_path / "t.json").read_text())["steps"][0]["step"]["name"] == steps[0]["step"]["name"]


def test_episode_generator_shape():
    items = list(episode(load_model("heuristic"), 10000))
    assert items[0][0] == "init" and all(kind == "step" for kind, *_ in items[1:])


@pytest.mark.parametrize("algo,config", [("dqn", "D1"), ("ppo", "P1")])
def test_smoke_masked_mode(tmp_path, algo, config):
    from environment.rendering import load_net

    if algo == "dqn":
        from training.dqn_training import train
        train(config, frames=200, masked=True, logs_root=tmp_path / "logs", models_root=tmp_path / "models")
    else:
        from training.pg_training import train
        train(algo, config, frames=200, masked=True, logs_root=tmp_path / "logs", models_root=tmp_path / "models")
    net = load_net(algo, config, models_root=tmp_path / "models")
    assert len(evaluate(net, seeds=EVAL_SEEDS[:1])["returns"]) == 1


def test_analyze_skips_missing_logs(tmp_path, capsys):
    from training.analysis import analyze

    analyze(logs_root=tmp_path / "logs", assets=tmp_path / "assets")
    out = capsys.readouterr().out
    assert "missing runs" in out and "nothing to analyze" in out


def test_episode_p_order():
    meta = next(episode(load_model("heuristic"), 10000, p_order=0.02))[1]
    assert meta["p_order"] == 0.02
