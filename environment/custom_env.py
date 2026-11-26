"""SkyBlood: a blood-delivery drone dispatcher at Zipline Muhanga, as a semi-MDP."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

HUB_LATLON = (-2.0792548, 29.7803631)
FACILITIES = [
    ("Kabgayi Hospital", -3.3, -2.7),
    ("Ruhango Provincial Hospital (Kinazi)", 14.0, -12.8),
    ("Gitwe Hospital", -10.1, -18.6),
    ("Muhororo Hospital", -17.3, 16.0),
    ("Nyanza Hospital", -3.4, -30.3),
    ("Kibuye Referral Hospital", -48.3, 1.3),
]
FAC = np.array([[x, y] for _, x, y in FACILITIES], dtype=np.float64)
N_FAC = len(FACILITIES)

SPEED = 5.0  # km per tick
T = 240
MAX_RANGE = 160.0
CAP = 2
SCALE = 60.0
DEADLINE = {1: 80, 2: 20}
STORM_R = 12.0
P_ORDER = 0.01
P_EMERG = 1 / 3

HOLD, LOAD, SWAP, RETURN = 0, 1, 2, 9
ACTION_NAMES = ["HOLD", "LOAD_UNIT", "SWAP_BATTERY"] + [f"FLY_DROP:{k}" for k in range(N_FAC)] + ["RETURN_TO_HUB"]
N_ACTIONS = len(ACTION_NAMES)
OBS_DIM = 38
BREAKDOWN_KEYS = ("delivery", "waiting", "expired", "wasted", "invalid", "forced_landing")


def km_to_latlon(x, y):
    lat = HUB_LATLON[0] + y / 110.574
    lon = HUB_LATLON[1] + x / (111.320 * math.cos(math.radians(HUB_LATLON[0])))
    return lat, lon


def segment_distances(p, a, ends):
    """Minimum distance from point p to each segment a->ends[i] (the start point when a segment is degenerate)."""
    ab = ends - a
    denom = (ab * ab).sum(axis=1)
    safe = np.where(denom == 0.0, 1.0, denom)
    u = np.where(denom == 0.0, 0.0, np.clip((ab @ (p - a)) / safe, 0.0, 1.0))
    closest = a + u[:, None] * ab
    return np.hypot(closest[:, 0] - p[0], closest[:, 1] - p[1])


class SkyBloodEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dict_obs: bool = False, p_order: float = P_ORDER):
        self.dict_obs = dict_obs
        self.p_order = p_order
        self.action_space = spaces.Discrete(N_ACTIONS)
        box = spaces.Box(-1.0, 1.0, (OBS_DIM,), np.float32)
        if dict_obs:
            self.observation_space = spaces.Dict(
                {"observation": box, "action_mask": spaces.MultiBinary(int(N_ACTIONS))}
            )
        else:
            self.observation_space = box
        self.rng = np.random.default_rng()
        self.tick_log = None  # the renderer sets a list here to record per-tick snapshots
        self._reset_state()

    def _reset_state(self):
        self.pos = np.zeros(2)
        self.b = MAX_RANGE
        self.packs = 0
        self.t = 0
        self.forced = 0
        self.urg = np.zeros(N_FAC, dtype=np.int64)
        self.left = np.zeros(N_FAC, dtype=np.int64)
        self.storm = np.zeros(2)
        self.storm_drift = np.zeros(2)
        self.ep_deliveries = 0
        self.ep_emergencies = 0
        self.ep_expired = 0

    # ---------- helpers ----------

    def at_hub(self) -> bool:
        return self.pos[0] == 0.0 and self.pos[1] == 0.0

    def _open_order(self, k):
        self.urg[k] = 2 if self.rng.random() < P_EMERG else 1
        self.left[k] = DEADLINE[int(self.urg[k])]

    def _storm_routes(self, targets) -> np.ndarray:
        return segment_distances(self.storm, self.pos, targets) <= STORM_R

    def action_masks(self) -> np.ndarray:
        hub = self.at_hub()
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[HOLD] = True
        m[LOAD] = hub and self.packs < CAP
        m[SWAP] = hub and self.b < MAX_RANGE
        m[3:3 + N_FAC] = self.packs > 0 and ~(FAC == self.pos).all(axis=1)
        m[RETURN] = not hub
        return m

    def _obs(self):
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0:2] = self.pos / SCALE
        o[2] = self.b / MAX_RANGE
        o[3] = self.packs / CAP
        per = o[4:28].reshape(N_FAC, 4)
        per[:, 0:2] = (FAC - self.pos) / SCALE
        per[:, 2] = self.urg / 2
        per[:, 3] = self.left / DEADLINE[1]
        o[28:34] = self._storm_routes(FAC)
        o[34:36] = (self.storm - self.pos) / SCALE
        o[36] = float(self.at_hub())
        o[37] = self.t / T
        o = np.clip(o, -1.0, 1.0)
        if self.dict_obs:
            return {"observation": o, "action_mask": self.action_masks().astype(np.int8)}
        return o

    # ---------- dynamics ----------

    def _tick(self, n: int) -> float:
        r = 0.0
        for _ in range(n):
            if self.t >= T:
                break
            wait = -0.03 * float(self.urg.sum())
            self._rb["waiting"] += wait
            r += wait

            open_ = self.urg > 0
            self.left = np.where(open_, self.left - 1, 0)

            exp = open_ & (self.left <= 0)
            if exp.any():
                pen = -5.0 * np.count_nonzero(exp & (self.urg == 1)) - 10.0 * np.count_nonzero(exp & (self.urg == 2))
                self._rb["expired"] += pen
                r += pen
                self._n_expired += int(np.count_nonzero(exp))
                self.urg[exp] = 0

            new = (self.urg == 0) & (self.rng.random(N_FAC) < self.p_order)
            for k in np.flatnonzero(new):
                self._open_order(k)
                self._new_orders.append(int(k))

            self.storm = np.minimum(np.maximum(self.storm + self.storm_drift + self.rng.normal(0, 1.0, size=2), -60), 60)
            self.t += 1
            if self.tick_log is not None:
                self.tick_log.append(
                    {"t": self.t, "urg": self.urg.tolist(), "left": self.left.tolist(), "storm": self.storm.tolist()}
                )
        return r

    def _fly(self, target) -> tuple[bool, float]:
        d = math.hypot(target[0] - self.pos[0], target[1] - self.pos[1])
        storm = bool(self._storm_routes(target[None])[0])
        energy = d * (1.5 if storm else 1.0)
        ticks = max(1, math.ceil(d * (1.3 if storm else 1.0) / SPEED))
        self._flight = {"dist_km": d, "storm_route": storm, "energy_km": energy, "flight_ticks": ticks}
        if energy > self.b:
            self.forced += 1
            lost = self.packs
            self.packs = 0
            pen = -(20.0 + 2.0 * lost)
            self._rb["forced_landing"] += pen
            r = pen + self._tick(ticks + 20)
            self.pos = np.zeros(2)
            self.b = MAX_RANGE
            return False, r
        self.b -= energy
        r = self._tick(ticks)
        self.pos = np.array(target, dtype=np.float64)
        return True, r

    def _invalid(self, ticks: int) -> float:
        self._rb["invalid"] += -0.5
        return -0.5 + self._tick(ticks)

    def step(self, action):
        a = int(action)
        self._rb = dict.fromkeys(BREAKDOWN_KEYS, 0.0)
        self._n_expired = 0
        self._new_orders = []
        self._flight = {"dist_km": 0.0, "storm_route": False, "energy_km": 0.0, "flight_ticks": 0}
        self.last_delivery = None
        hub = self.at_hub()

        if a == HOLD:
            if not hub:
                self.b -= 10.0
            r = self._tick(2)
        elif a == LOAD:
            if hub and self.packs < CAP:
                self.packs += 1
                r = self._tick(1)
            else:
                r = self._invalid(1)
        elif a == SWAP:
            if hub and self.b < MAX_RANGE:
                self.b = MAX_RANGE
                r = self._tick(2)
            else:
                r = self._invalid(2)
        elif a == RETURN:
            r = self._fly(np.zeros(2))[1] if not hub else self._invalid(1)
        else:
            k = a - 3
            if self.packs == 0:
                r = self._invalid(1)
            else:
                ok, r = self._fly(FAC[k])
                if ok:
                    r += self._drop(k)

        if self.b <= 0 and not self.at_hub():
            self.forced += 1
            self.packs = 0
            self._rb["forced_landing"] += -20.0
            self.pos = np.zeros(2)
            self.b = MAX_RANGE
            r += -20.0 + self._tick(20)

        self.ep_expired += self._n_expired
        terminated = self.forced >= 3
        truncated = self.t >= T and not terminated
        return self._obs(), float(r), terminated, truncated, self._info()

    def _drop(self, k) -> float:
        self.packs -= 1
        u = int(self.urg[k])
        if u == 0:
            self._rb["wasted"] += -2.0
            return -2.0
        gain = (20.0 if u == 2 else 10.0) + 5.0 * self.left[k] / DEADLINE[u]
        self.urg[k] = 0
        self.left[k] = 0
        self._rb["delivery"] += gain
        self.ep_deliveries += 1
        self.ep_emergencies += int(u == 2)
        self.last_delivery = {"k": int(k), "gain": float(gain), "emergency": u == 2}
        return float(gain)

    def _info(self):
        return {
            "breakdown": dict(self._rb),
            "n_expired": self._n_expired,
            "new_orders": list(self._new_orders),
            **self._flight,
            "action_mask": self.action_masks(),
            **self._counters(),
        }

    def _counters(self):
        c = {"ep_deliveries": self.ep_deliveries, "ep_emergencies": self.ep_emergencies,
             "ep_expired": self.ep_expired, "ep_forced": self.forced}
        # one packed array: TorchRL's info-dict reader costs per key, so training reads only this
        return {**c, "ep_stats": np.array(list(c.values()), dtype=np.int64)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()
        if self.tick_log is not None:
            self.tick_log.clear()
        for k in self.rng.choice(N_FAC, 2, replace=False):
            self._open_order(k)
        ang = self.rng.uniform(0, 2 * math.pi)
        dist = self.rng.uniform(15, 45)
        self.storm = np.array([dist * math.cos(ang), dist * math.sin(ang)])
        self.storm_drift = self.rng.normal(0, 0.8, size=2)
        return self._obs(), {"action_mask": self.action_masks(), **self._counters()}


# ---------- baselines ----------

def heuristic_action(env) -> int:
    e = env.unwrapped
    pending = np.flatnonzero(e.urg > 0)
    if e.packs > 0 and len(pending):
        k = min(pending, key=lambda k: (-e.urg[k], e.left[k]))
        return 3 + int(k)
    if not e.at_hub():
        return RETURN
    if e.b < MAX_RANGE:
        return SWAP
    if len(pending) and e.packs < min(CAP, len(pending)):
        return LOAD
    return HOLD


def random_action(env, rng) -> int:
    return int(rng.integers(N_ACTIONS))


def masked_random_action(env, rng) -> int:
    return int(rng.choice(np.flatnonzero(env.unwrapped.action_masks())))


def hold_action(env, rng=None) -> int:
    return HOLD


BASELINES = {
    "heuristic": lambda env, rng: heuristic_action(env),
    "hold": hold_action,
    "random": random_action,
    "masked_random": masked_random_action,
}


def run_episode(policy, seed: int, p_order: float = P_ORDER) -> float:
    env = SkyBloodEnv(p_order=p_order)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1_000_000)
    total, done = 0.0, False
    while not done:
        _, r, term, trunc, _ = env.step(policy(env, rng))
        total += r
        done = term or trunc
    return total


def baseline_returns(name: str, seeds, p_order: float = P_ORDER) -> np.ndarray:
    return np.array([run_episode(BASELINES[name], s, p_order) for s in seeds])


def print_baselines(first_seed: int = 10000, n_seeds: int = 20, p_order: float = P_ORDER):
    seeds = range(first_seed, first_seed + n_seeds)
    print(f"Baselines on seeds {first_seed}-{first_seed + n_seeds - 1} (p_order={p_order})")
    for name in BASELINES:
        ret = baseline_returns(name, seeds, p_order)
        print(f"  {name:14s} {ret.mean():8.1f} ± {ret.std():5.1f}")
