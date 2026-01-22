from __future__ import annotations

from typing import Literal
import math
import numpy as np
import polars as pl

from download_dataset import YambdaDataset
from markov_chain_polars import MarkovChainPolars

from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from state_weights import make_state_weights


# ===================== CONFIG =====================
DATASET_MODE = "flat"
DATASET_SIZE = "50m"
INTERACTION = "multi_event"

N = 10
DEPTH = 40

DECAY_MODE: Literal["time", "order_exp", "order_linear"] = "time"
HALF_LIFE_HOURS = 24 * 0.9
TIME_UNIT: Literal["s", "ms"] = "s"
GAMMA = 0.85
MIN_W = 0.05

COMBINE: Literal["sum", "max"] = "sum"
ONLY_LISTEN = True

BATCH_SIZE = 256
N_JOBS = -1
BACKEND = "threading"
SHOW_PROGRESS = True
REVERSE_STATES = True
# ==================================================

def train_val_test_polars(
    df: pl.DataFrame,
    test_timestamp: int,
    val_timestamp: int,
    train_timestamp: int,
) -> tuple[pl.DataFrame, pl.DataFrame | None, pl.DataFrame]:
    train = df.filter(pl.col("timestamp") <= train_timestamp)
    users = train["uid"].unique()
    validation = df.filter(
        (pl.col("timestamp") > train_timestamp)
        & (pl.col("timestamp") <= val_timestamp)
        & (pl.col("uid").is_in(users))
    )
    test = df.filter(
        (pl.col("timestamp") > test_timestamp)
        & (pl.col("uid").is_in(users))
    )
    return train, validation, test


def ndcg_score(y_true, y_pred, k=10):
    y_pred = list(map(int, y_pred[:k]))
    y_true = set(map(int, y_true))
    rels = [1 if item in y_true else 0 for item in y_pred]
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(rels))
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def build_result_from_mh(mh: MarkovChainPolars) -> pl.DataFrame:
    Map = pl.DataFrame({
        "node_id": list(mh.map_name_nodes.keys()),
        "node_name": list(mh.map_name_nodes.values()),
    }).with_columns(
        node_id=pl.col("node_id").cast(pl.Utf8),
        node_id_str=pl.format("id_node_{}", pl.col("node_id")),
    )

    result = (
        mh.P
        .join(
            Map.select(["node_id_str", "node_name"]).rename({"node_name": "from_node_name"}),
            left_on="from_node_id", right_on="node_id_str", how="left"
        )
        .join(
            Map.select(["node_id_str", "node_name"]).rename({"node_name": "to_node_name"}),
            left_on="to_node_id", right_on="node_id_str", how="left"
        )
    )

    def split_event_item(col: str, prefix: str):
        return [
            pl.when(pl.col(col).str.contains("_"))
              .then(pl.col(col).str.split_exact("_", 1).struct.field("field_0"))
              .otherwise(pl.col(col))
              .alias(f"{prefix}_event_type"),

            pl.when(pl.col(col).str.contains("_"))
              .then(pl.col(col).str.split_exact("_", 1).struct.field("field_1"))
              .otherwise(pl.lit(""))
              .alias(f"{prefix}_item_id"),
        ]

    result = result.with_columns(
        split_event_item("from_node_name", "from") +
        split_event_item("to_node_name", "to")
    )

    result = result.with_columns(
        from_item_id=pl.when(pl.col("from_event_type") == "START")
                       .then(pl.lit(""))
                       .otherwise(pl.col("from_item_id"))
    )

    result = result.with_columns(
        p=pl.col("Probability").str.replace("%", "").cast(pl.Float64) / 100.0
    )

    result = result.with_columns([
        pl.col("from_event_type").cast(pl.Utf8),
        pl.col("from_item_id").cast(pl.Utf8),
        pl.col("to_event_type").cast(pl.Utf8),
        pl.col("to_item_id").cast(pl.Utf8),
        pl.concat_str([pl.col("from_event_type"), pl.lit("_"), pl.col("from_item_id")]).alias("key"),
        pl.concat_str([pl.col("to_event_type"), pl.lit("_"), pl.col("to_item_id")]).alias("key_to"),
    ])

    result = (
        result
        .select(["key", "key_to", "to_event_type", "to_item_id", "p"])
        .with_columns(
            to_item_id=pl.col("to_item_id").cast(pl.Int64, strict=False)
        )
    )
    return result


def recomendation_system_weighted(
    N: int,
    map_user_state: dict,
    result_lazy: pl.LazyFrame,
) -> dict:
    states, weights = make_state_weights(
        map_user_state,
        decay_mode=DECAY_MODE,
        half_life_hours=HALF_LIFE_HOURS,
        time_unit=TIME_UNIT,
        gamma=GAMMA,
        min_w=MIN_W,
        normalize=True,
    )
    if not states:
        return {}

    wdf = pl.DataFrame({"key": states, "w": list(weights)})

    filt = pl.lit(True)
    if ONLY_LISTEN:
        filt = filt & (pl.col("to_event_type") == "listen")

    scored = (
        result_lazy
        .filter(pl.col("key").is_in(states))
        .filter(filt)
        .join(wdf.lazy(), on="key", how="inner")
        .with_columns((pl.col("p") * pl.col("w")).alias("weighted_p"))
    )

    agg_expr = (
        pl.col("weighted_p").sum().alias("score")
        if COMBINE == "sum" else
        pl.col("weighted_p").max().alias("score")
    )

    topN_df = (
        scored
        .group_by(["key_to", "to_event_type", "to_item_id"])
        .agg(agg_expr)
        .sort("score", descending=True)
        .limit(N)
        .collect()
    )

    sample = {}
    for i, row in enumerate(topN_df.to_dicts()):
        sample[f"{i}"] = f"{row['to_event_type']}_{row['to_item_id']}"
        sample[f"p_{i}"] = row["score"]
    return sample


def evaluate_once_batched(
    result_df: pl.DataFrame,
    test_lazy: pl.LazyFrame,
    result_lazy: pl.LazyFrame,
    *,
    N: int,
    batch_size: int = 256,
    n_jobs: int = -1,
    backend: str = "threading",
    show_progress: bool = True,
    reverse_states: bool = True,
):
    state_cols = sorted(
        [c for c in result_df.columns if c.startswith("state_") and not c.startswith("state_ts_")],
        key=lambda x: int(x.split("_")[1])
    )
    ts_cols = sorted(
        [c for c in result_df.columns if c.startswith("state_ts_")],
        key=lambda x: int(x.split("_")[2])
    )

    targets_lazy = result_df.lazy().select(["uid", "last_ts"])
    future_likes_df = (
        test_lazy
        .join(targets_lazy, on="uid", how="inner")
        .filter((pl.col("event_type") == "like") & (pl.col("timestamp") > pl.col("last_ts")))
        .group_by("uid")
        .agg(pl.col("item_id").cast(pl.Int64).implode().alias("likes_list"))
        .collect()
    )

    def _to_flat_set(seq):
        acc = set()
        for x in seq:
            if isinstance(x, (list, tuple)):
                acc.update(x)
            else:
                acc.add(x)
        return {int(v) for v in acc if v is not None}

    future_likes_map = {r["uid"]: _to_flat_set(r["likes_list"]) for r in future_likes_df.to_dicts()}

    select_cols = ["uid", "last_ts"] + state_cols + ts_cols
    rows = result_df.select(select_cols).to_dicts()

    def chunked(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    batches = list(chunked(rows, batch_size))

    def extract_rec_items(res_dict):
        rec_items = []
        for k, val in res_dict.items():
            if str(k).startswith("p_"):
                continue
            parsed = None
            for src in (val, k):
                try:
                    parsed = int(str(src).rsplit("_", 1)[-1])
                    break
                except Exception:
                    continue
            if parsed is not None:
                rec_items.append(parsed)
        return rec_items[:N]

    def eval_batch(batch_rows):
        r_sum = 0.0
        d_sum = 0.0
        n_sum = 0

        for row in batch_rows:
            uid = int(row["uid"])
            last_ts = row["last_ts"]

            ordered_cols = list(reversed(state_cols)) if reverse_states else state_cols
            ordered_ts_cols = list(reversed(ts_cols)) if reverse_states else ts_cols

            ordered_states, ordered_ts = [], []
            for sc, tc in zip(ordered_cols, ordered_ts_cols):
                s_val = row[sc]
                if s_val is None:
                    continue
                ordered_states.append(s_val)
                ordered_ts.append(row.get(tc))

            if not ordered_states:
                continue

            map_user_state = {f"state_{i}": s for i, s in enumerate(ordered_states)}
            for i, ts in enumerate(ordered_ts):
                map_user_state[f"state_ts_{i}"] = ts
            map_user_state["last_timestamp"] = last_ts

            res = recomendation_system_weighted(N, map_user_state, result_lazy)
            rec_items = extract_rec_items(res)

            like_set = future_likes_map.get(uid, set())
            if not like_set:
                continue

            inter = like_set.intersection(rec_items)
            r_sum += len(inter) / float(len(like_set))
            d_sum += ndcg_score(like_set, rec_items, k=N)
            n_sum += 1

        return r_sum, d_sum, n_sum

    if show_progress:
        ctx = tqdm_joblib(tqdm(total=len(batches), desc=f"Batches like {batch_size} users"))
    else:
        class _Dummy:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        ctx = _Dummy()

    with ctx:
        out = Parallel(n_jobs=n_jobs, backend=backend, batch_size=1)(
            delayed(eval_batch)(b) for b in batches
        )

    total_recall = sum(r for r, _, _ in out)
    total_ndcg = sum(d for _, d, _ in out)
    total_n = sum(n for _, _, n in out)

    return {
        "avg_recall": (total_recall / float(total_n)) if total_n > 0 else 0.0,
        "avg_ndcg": (total_ndcg / float(total_n)) if total_n > 0 else 0.0,
        "n_users": int(total_n),
    }


def main():
    print("Loading dataset...")
    dataset = YambdaDataset(DATASET_MODE, DATASET_SIZE)
    events = dataset.interaction(INTERACTION)
    df = pl.from_pandas(events.to_pandas())

    HOUR_SECONDS = 60 * 60
    DAY_SECONDS = 24 * HOUR_SECONDS

    train, val, test = train_val_test_polars(
        df,
        26000000 - DAY_SECONDS,
        26000000 - DAY_SECONDS,
        26000000 - DAY_SECONDS
    )

    print("Building Markov chain...")
    mh = MarkovChainPolars(
        name="Yambda_Markov",
        data=train,
        struct=("event_type", "item_id"),
        time_column="timestamp",
        users_id_column="uid",
    )
    mh.preprocessing_data()
    mh.build_markov_chain_optimized()

    result = build_result_from_mh(mh)

    test_lazy = test.lazy()
    train_lazy = train.lazy()
    result_lazy = result.lazy()

    users_with_like = (
        test_lazy
        .filter(pl.col("event_type") == "like")
        .select("uid")
        .unique()
    )

    train_for_users = (
        train_lazy
        .join(users_with_like, on="uid", how="inner")
        .with_columns(chain=pl.concat_str([pl.col("event_type"), pl.lit("_"), pl.col("item_id")]))
    )

    last_chains = (
        train_for_users
        .group_by("uid")
        .agg([
            pl.col("chain").sort_by(pl.col("timestamp")).tail(DEPTH).alias("last_chain"),
            pl.col("timestamp").sort_by(pl.col("timestamp")).tail(DEPTH).alias("last_ts_list"),
        ])
        .with_columns(last_ts=pl.col("last_ts_list").list.get(-1))
    )

    last_chain_states = last_chains.with_columns(
        [pl.col("last_chain").list.slice(i, 1).list.first().alias(f"state_{i}") for i in range(DEPTH)]
        +
        [pl.col("last_ts_list").list.slice(i, 1).list.first().alias(f"state_ts_{i}") for i in range(DEPTH)]
    ).drop(["last_chain", "last_ts_list"])

    result_df = last_chain_states.collect()

    print(f"Prepared {result_df.height} user histories.")
    print("Evaluating...")

    metrics = evaluate_once_batched(
        result_df=result_df,
        test_lazy=test_lazy,
        result_lazy=result_lazy,
        N=N,
        batch_size=BATCH_SIZE,
        n_jobs=N_JOBS,
        backend=BACKEND,
        show_progress=SHOW_PROGRESS,
        reverse_states=REVERSE_STATES,
    )

    print("\nRESULTS")
    print("=" * 60)
    print(f"DECAY_MODE = {DECAY_MODE}")
    if DECAY_MODE == "time":
        print(f"half_life_hours = {HALF_LIFE_HOURS}")
    elif DECAY_MODE == "order_exp":
        print(f"gamma = {GAMMA}")
    else:
        print(f"min_w = {MIN_W}")

    print(f"N={N}, DEPTH={DEPTH}")
    print(f"avg_recall = {metrics['avg_recall']:.6f}")
    print(f"avg_ndcg   = {metrics['avg_ndcg']:.6f}")
    print(f"n_users    = {metrics['n_users']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
