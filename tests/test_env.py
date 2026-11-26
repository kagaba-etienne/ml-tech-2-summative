import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.custom_env import (
    FAC, HOLD, LOAD, MAX_RANGE, N_ACTIONS, OBS_DIM, RETURN, SWAP, T,
    SkyBloodEnv, baseline_returns,
)


def fresh(seed=0, **kw):
    env = SkyBloodEnv(**kw)
    env.reset(seed=seed)
    return env


def quiet(env):
    """Remove randomness that would disturb a hand-built scenario."""
    env.p_order = 0.0
    env.urg[:] = 0
    env.left[:] = 0
    env.storm = np.array([60.0, 60.0])
    env.storm_drift = np.zeros(2)
    return env


def test_spaces_and_shapes():
    env = SkyBloodEnv()
    obs, info = env.reset(seed=0)
    assert env.action_space.n == N_ACTIONS == 10
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert info["action_mask"].shape == (10,)
    denv = SkyBloodEnv(dict_obs=True)
    dobs, _ = denv.reset(seed=0)
    assert dobs["observation"].shape == (OBS_DIM,)
    assert dobs["action_mask"].shape == (10,)


def test_reset_start_state():
    env = fresh(3)
    assert env.at_hub() and env.b == MAX_RANGE and env.packs == 0 and env.t == 0 and env.forced == 0
    assert (env.urg > 0).sum() == 2
    assert 15 <= np.linalg.norm(env.storm) <= 45


def test_obs_in_bounds_random_play():
    env = fresh(1)
    rng = np.random.default_rng(0)
    for _ in range(2000):
        obs, _, term, trunc, info = env.step(int(rng.integers(N_ACTIONS)))
        assert env.observation_space.contains(obs)
        for key in ("delivery", "waiting", "expired", "wasted", "invalid", "forced_landing"):
            assert key in info["breakdown"]
        if term or trunc:
            env.reset()


def test_determinism():
    def roll(seed):
        env = fresh(seed)
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(100):
            obs, r, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
            out.append((obs.copy(), r))
            if term or trunc:
                break
        return out

    a, b = roll(7), roll(7)
    assert all(np.array_equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))


def test_masks_and_preconditions():
    env = quiet(fresh())
    m = env.action_masks()
    assert m[HOLD] and m[LOAD] and not m[SWAP] and not m[RETURN] and not m[3:9].any()
    _, r, *_ = env.step(SWAP)
    assert r == pytest.approx(-0.5) and env.t == 2
    _, r, *_ = env.step(3)
    assert r == pytest.approx(-0.5) and env.t == 3
    _, r, *_ = env.step(RETURN)
    assert r == pytest.approx(-0.5) and env.t == 4
    env.step(LOAD)
    env.step(LOAD)
    assert env.packs == 2 and not env.action_masks()[LOAD]
    _, r, *_ = env.step(LOAD)
    assert r == pytest.approx(-0.5) and env.packs == 2
    env.step(3)
    m = env.action_masks()
    assert not m[3] and m[4] and m[RETURN] and not m[LOAD] and not m[SWAP]


def test_masked_same_facility_is_zero_km_drop():
    env = quiet(fresh())
    env.step(LOAD); env.step(LOAD)
    env.step(3)
    b, t = env.b, env.t
    _, r, _, _, info = env.step(3)
    assert info["dist_km"] == 0 and info["flight_ticks"] == 1 and env.t == t + 1
    assert env.b == b and r == pytest.approx(-2.0) and env.packs == 0


def test_delivery_reward_bounds():
    for urg, lo, hi in ((1, 10, 15), (2, 20, 25)):
        env = quiet(fresh())
        env.step(LOAD)
        env.urg[0], env.left[0] = urg, {1: 80, 2: 20}[urg]
        _, r, _, _, info = env.step(3)
        gain = info["breakdown"]["delivery"]
        assert lo <= gain <= hi
        assert env.urg[0] == 0 and env.left[0] == 0


def test_wasted_unit():
    env = quiet(fresh())
    env.step(LOAD)
    _, r, _, _, info = env.step(3)
    assert info["breakdown"]["wasted"] == -2.0


def test_expiry_penalties():
    env = quiet(fresh())
    env.urg[0], env.left[0] = 1, 1
    env.urg[1], env.left[1] = 2, 1
    _, r, _, _, info = env.step(HOLD)
    assert info["breakdown"]["expired"] == pytest.approx(-15.0)
    assert info["n_expired"] == 2 and env.urg[:2].sum() == 0


def test_waiting_cost():
    env = quiet(fresh())
    env.urg[0], env.left[0] = 1, 80
    env.urg[1], env.left[1] = 2, 20
    _, r, *_ = env.step(HOLD)
    assert r == pytest.approx(-2 * (0.03 + 0.06))


def test_storm_multipliers():
    env = quiet(fresh())
    env.step(LOAD)
    env.storm = FAC[4] / 2  # on the route to Nyanza
    d = float(np.linalg.norm(FAC[4]))
    b0 = env.b
    _, _, _, _, info = env.step(3 + 4)
    assert info["storm_route"]
    assert info["energy_km"] == pytest.approx(1.5 * d)
    assert info["flight_ticks"] == int(np.ceil(1.3 * d / 5))
    assert env.b == pytest.approx(b0 - 1.5 * d)


def test_forced_landing_not_terminal_until_third():
    env = quiet(fresh())
    for n in (1, 2, 3):
        env.step(LOAD)
        env.b = 5.0
        _, r, term, _, info = env.step(3 + 5)
        assert env.forced == n and env.packs == 0 and env.at_hub() and env.b == MAX_RANGE
        assert info["breakdown"]["forced_landing"] == pytest.approx(-22.0)
        assert term == (n == 3)


def test_loiter_dry_forced_landing():
    env = quiet(fresh())
    env.step(LOAD)
    env.step(3)
    env.b = 5.0
    _, r, *_ = env.step(HOLD)
    assert env.forced == 1 and env.at_hub() and r == pytest.approx(-20.0)


def test_truncation_at_T():
    env = quiet(fresh())
    trunc = False
    while not trunc:
        _, _, term, trunc, _ = env.step(HOLD)
        assert not term
    assert env.t == T


def test_gymnasium_check_env():
    check_env(SkyBloodEnv(), skip_render_check=True)
    check_env(SkyBloodEnv(dict_obs=True), skip_render_check=True)


def test_torchrl_specs_default_and_masked():
    from torchrl.envs import ActionMask, GymWrapper, TransformedEnv
    from torchrl.envs.utils import check_env_specs

    check_env_specs(GymWrapper(SkyBloodEnv()))
    menv = TransformedEnv(GymWrapper(SkyBloodEnv(dict_obs=True), categorical_action_encoding=True), ActionMask())
    check_env_specs(menv)
    td = menv.reset()
    menv.rand_step(td)


def test_heuristic_beats_random():
    seeds = range(20)
    assert baseline_returns("heuristic", seeds).mean() > baseline_returns("random", seeds).mean() + 100
