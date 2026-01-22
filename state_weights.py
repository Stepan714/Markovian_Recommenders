from __future__ import annotations
from typing import Literal
import math


def make_state_weights(
    map_user_state: dict,
    *,
    decay_mode: Literal["time", "order_exp", "order_linear"] = "time",

    # time
    half_life_hours: float = 24 * 0.9,
    time_unit: Literal["s", "ms"] = "s",

    # order_exp
    gamma: float = 0.85,

    # order_linear
    min_w: float = 0.05,

    normalize: bool = True,
) -> tuple[list[str], list[float]]:
    states, ts_list = [], []
    i = 0
    while True:
        k = f"state_{i}"
        if k not in map_user_state:
            break
        v = map_user_state[k]
        if v is None:
            break
        states.append(v)
        ts_list.append(map_user_state.get(f"state_ts_{i}", None))
        i += 1

    if not states:
        return [], []

    if decay_mode == "time":
        last_ts = map_user_state.get("last_timestamp")
        if last_ts is None or half_life_hours <= 0:
            raw = [1.0] * len(states)
        else:
            scale = 1.0 if time_unit == "s" else 1e-3
            kdec = math.log(2.0) / (half_life_hours * 3600.0)
            raw = []
            for ts in ts_list:
                if ts is None:
                    raw.append(1.0)
                else:
                    age_sec = max(0.0, (float(last_ts) - float(ts)) * scale)
                    raw.append(math.exp(-kdec * age_sec))

    elif decay_mode == "order_exp":
        raw = [float(gamma) ** i for i in range(len(states))]

    elif decay_mode == "order_linear":
        n = len(states)
        if n == 1:
            raw = [1.0]
        else:
            raw = [1.0 - (1.0 - float(min_w)) * (i / (n - 1)) for i in range(n)]

    else:
        raise ValueError(f"Unknown decay_mode={decay_mode}")

    if not normalize:
        return states, raw

    s = sum(raw)
    weights = [x / s for x in raw] if s > 0 else raw
    return states, weights
