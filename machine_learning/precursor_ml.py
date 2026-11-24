"""4단계 진입 전 센서 변화량을 혼동행렬 평가에 제공한다."""
import numpy as np
import pandas as pd
from config import DATA_PATH, SENSOR_COLUMNS

HORIZON = 120


def features():
    data = pd.read_csv(DATA_PATH)
    parts = []
    for exp, g in data.groupby("exp_id"):
        event = g.loc[g.tr_stage.eq(4), "rel_time"].min()
        g = g[g.tr_stage.isin([1, 2, 3]) & (g.rel_time < event)].sort_values("rel_time")
        baseline = g.loc[g.rel_time < 60, SENSOR_COLUMNS].median()
        # Right-closed windows: all measurements are available at the decision time.
        bins = np.ceil(g.rel_time / 10).astype(int) * 10
        levels = g[SENSOR_COLUMNS].groupby(bins).median()
        delta = levels - baseline
        trend = levels - levels.reindex(levels.index - 30).set_axis(levels.index)
        frame = delta.join(trend.add_suffix("_change"))
        frame = frame[(frame.index >= 60) & (frame.index < event)].copy()
        frame["lead"] = event - frame.index
        frame["exp"] = exp
        frame["method"] = g.tr_method.iloc[0]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)
