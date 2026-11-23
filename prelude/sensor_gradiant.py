import os
import sys
import logging
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "..", "output", "battery_labels.csv")

SENSORS = {
    "meas_v": "전압 (V)",
    "meas_t1": "배터리 온도 1 (℃)",
    "meas_t2": "배터리 온도 2 (℃)",
    "meas_p": "압력",
    "thermal_t": "열화상 온도 (℃)",
    "meas_co": "일산화탄소 CO (ppm)",
    "meas_co2": "이산화탄소 CO2 (ppm)",
}
UNITS = {
    "meas_v": "V",
    "meas_t1": "℃",
    "meas_t2": "℃",
    "meas_p": "값",
    "thermal_t": "℃",
    "meas_co": "ppm",
    "meas_co2": "ppm",
}
METHODS = {
    1: {
        "name": "가열",
        "file": "heating",
        "color": "#eb6834",
        "mean_color": "#a63c12",
    },
    2: {
        "name": "과충전",
        "file": "overcharge",
        "color": "#2a78d6",
        "mean_color": "#123f7a",
    },
}
CALM_COLOR = "#9a9a9a"
END_STAGE = 4
BASELINE_SECONDS = 60
BIN_SIZE = 1
EDGE = 3
CALM_END = 50
MAX_GAP = 2
WHISKER = 1.5
NOTE = "실험 시작 0% ~ 4단계 도달 100%, 주변·열화상 온도 급상승 이후 제외"


def load_sensor_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype={"exp_id": str}, encoding="utf-8-sig")
    logger.info(
        f"센서 데이터 {len(df)}행, 실험 {df['exp_id'].nunique()}개를 불러왔습니다."
    )
    return df


def first_time(df: pd.DataFrame, condition: pd.Series) -> pd.Series:
    return df["rel_time"].where(condition).groupby(df["exp_id"]).transform("min")


def add_progress_axis(
    df: pd.DataFrame, ambient_rise: float, thermal_limit: float
) -> pd.DataFrame:
    stage_time = first_time(df, df["tr_stage"] >= END_STAGE)
    baseline = (
        df["atmp_t"]
        .where(df["rel_time"] < BASELINE_SECONDS)
        .groupby(df["exp_id"])
        .transform("median")
    )
    ambient_time = first_time(df, df["atmp_t"] > baseline + ambient_rise)
    thermal_time = first_time(df, df["thermal_t"] > thermal_limit)
    cut_time = pd.concat([ambient_time, thermal_time], axis=1).min(axis=1)
    end_time = stage_time.mask(cut_time < stage_time, cut_time)

    keep = df["rel_time"] <= end_time
    keep &= ~(df["rel_time"] >= cut_time)
    result = df.assign(time_axis=df["rel_time"] / end_time * 100)[keep]
    logger.info(
        f"4단계 도달 또는 주변 온도 {ambient_rise:g}℃ 상승·열화상 온도 {thermal_limit:g}℃ 초과 이후 "
        f"{len(df) - len(result)}행을 제외하고 {len(result)}행을 사용합니다."
    )
    return result


def experiment_bins(data: pd.DataFrame) -> pd.DataFrame:
    time_bin = ((data["time_axis"] // BIN_SIZE) * BIN_SIZE).rename("time_bin")
    return data.groupby([time_bin, "exp_id"])[list(SENSORS)].mean()


def sensor_mean(per_experiment: pd.DataFrame, min_experiments: int) -> pd.DataFrame:
    grouped = per_experiment.groupby("time_bin")
    mean = grouped.mean()
    return mean[grouped.size() >= min_experiments]


def smooth_gradient(frame: pd.DataFrame, smooth: int) -> pd.DataFrame:
    smoothed = frame.rolling(smooth, center=True, min_periods=1).mean()
    gradient = np.gradient(
        smoothed.to_numpy(dtype=float),
        smoothed.index.to_numpy(dtype=float),
        axis=0,
    )
    return pd.DataFrame(gradient, index=frame.index, columns=frame.columns)


def calm_part(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return frame[(frame.index >= EDGE) & (frame.index < CALM_END)]


def robust_z(gradient: pd.Series) -> pd.Series:
    calm = calm_part(gradient)
    median = calm.median()
    mad = (calm - median).abs().median() * 1.4826
    if mad == 0:
        return gradient * 0
    z = (gradient - median) / mad
    return z.mask(z.index < EDGE, 0)


def find_spike(z: pd.Series, threshold: float) -> dict | None:
    hits = z.index[z.abs() >= threshold]
    if len(hits) == 0:
        return None

    groups = []
    start = previous = hits[0]
    for value in hits[1:]:
        if value - previous > (MAX_GAP + 1) * BIN_SIZE:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))

    start, end = max(groups, key=lambda group: z.loc[group[0] : group[1]].abs().max())
    segment = z.loc[start:end]
    peak = segment.abs().idxmax()
    return {"start": start, "end": end, "peak": peak, "z": segment.loc[peak]}


def count_outside(calm: pd.Series, spike: pd.Series, sign: float) -> int:
    q1, q3 = calm.quantile([0.25, 0.75])
    if sign > 0:
        return int((spike > q3 + WHISKER * (q3 - q1)).sum())
    return int((spike < q1 - WHISKER * (q3 - q1)).sum())


def draw_box(
    ax: plt.Axes, values: pd.Series, position: int, color: str, rng: np.random.Generator
) -> None:
    box = ax.boxplot(
        [values],
        positions=[position],
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": color},
        capprops={"color": color},
        boxprops={"edgecolor": color},
    )
    box["boxes"][0].set_facecolor(color)
    box["boxes"][0].set_alpha(0.25)
    jitter = rng.uniform(-0.15, 0.15, len(values))
    ax.scatter(
        position + jitter,
        values,
        s=10,
        color=color,
        alpha=0.6,
        linewidths=0,
        zorder=3,
    )


def plot_spike_boxplot(
    data: pd.DataFrame,
    method: int,
    z_threshold: float,
    smooth: int,
    min_experiments: int,
    output_path: str,
) -> plt.Figure:
    info = METHODS[method]
    per_experiment = experiment_bins(data[data["tr_method"] == method])
    mean_gradient = smooth_gradient(
        sensor_mean(per_experiment, min_experiments), smooth
    )
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    for ax, (column, label) in zip(axes.flat, SENSORS.items()):
        exp_gradient = smooth_gradient(per_experiment[column].unstack("exp_id"), smooth)
        calm = calm_part(exp_gradient).mean().dropna()
        spike_range = find_spike(robust_z(mean_gradient[column]), z_threshold)

        draw_box(ax, calm, 1, CALM_COLOR, rng)
        ticks = [f"안정 구간\n{EDGE}~{CALM_END - BIN_SIZE:g}%"]
        if spike_range is None:
            ax.text(
                2,
                calm.median(),
                "급변 구간 없음",
                ha="center",
                va="center",
                fontsize=12,
                color="#666666",
            )
            title = f"{label}\n평균선 미분값 z≥{z_threshold:g} 구간 없음"
            logger.info(f"{info['name']} {label}: 급변 구간이 없습니다.")
        else:
            start, end, peak = (
                spike_range["start"],
                spike_range["end"],
                spike_range["peak"],
            )
            spike = exp_gradient.loc[start:end].mean().dropna()
            outside = count_outside(calm, spike, spike_range["z"])
            draw_box(ax, spike, 2, info["color"], rng)
            ticks.append(f"급변 구간\n{start:g}~{end:g}%")
            direction = "상승" if spike_range["z"] > 0 else "하강"
            title = (
                f"{label}\n급변 {start:g}~{end:g}% (정점 {peak:g}%, {direction}) · "
                f"안정 구간 범위 밖 {outside}/{len(spike)}개"
            )
            logger.info(
                f"{info['name']} {label}: 급변 구간 {start:g}~{end:g}% "
                f"(정점 {peak:g}%, z={spike_range['z']:.0f}), "
                f"안정 구간 범위를 벗어난 실험 {outside}/{len(spike)}개"
            )

        ax.set_xticks([1, 2], ticks + [""] * (2 - len(ticks)))
        ax.set_xlim(0.4, 2.6)
        ax.axhline(0, color="#bbbbbb", linewidth=0.8, zorder=0)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(f"기울기 ({UNITS[column]} / 진행률 1%)", fontsize=11)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    note_ax = axes.flat[-1]
    note_ax.axis("off")
    note_ax.scatter([], [], s=30, color=CALM_COLOR, label="안정 구간의 실험별 기울기")
    note_ax.scatter(
        [], [], s=30, color=info["color"], label="급변 구간의 실험별 기울기"
    )
    note_ax.legend(loc="center", frameon=False, fontsize=13)

    fig.suptitle(
        f"{info['name']} 실험의 안정 구간 대비 급변 구간 기울기 ({NOTE})", fontsize=17
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    logger.info(f"{info['name']} 실험 boxplot을 저장했습니다: {output_path}")
    return fig


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="의미 있는 7종 센서(전압, 배터리 온도 1·2, 압력, 열화상 온도, CO, CO2)의 실험 평균선을 미분해 급변 구간을 찾고, "
        "안정 구간과 급변 구간의 실험별 기울기를 boxplot으로 비교하는 스크립트입니다. "
        "가로축은 prelude.py의 progress _no_ambient 버전과 같으며, 가열/과충전별로 2장을 저장합니다."
    )
    arg_parser.add_argument(
        "--input_path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="preprocessing.py로 만든 csv 파일 경로를 지정합니다. (기본값: ../output/battery_labels.csv)",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default=BASE_DIR,
        help="그래프 이미지를 저장할 폴더를 지정합니다. 폴더가 없으면 생성합니다. (기본값: prelude 폴더)",
    )
    arg_parser.add_argument(
        "--ambient_rise",
        type=float,
        default=5.0,
        help="실험 시작 후 60초 동안의 주변 온도 중앙값보다 이 값(℃) 이상 오른 시점부터 제외합니다. (기본값: 5)",
    )
    arg_parser.add_argument(
        "--thermal_limit",
        type=float,
        default=200.0,
        help="열화상 온도가 이 값(℃)을 넘은 시점부터 제외합니다. (기본값: 200)",
    )
    arg_parser.add_argument(
        "--z_threshold",
        type=float,
        default=5.0,
        help="평균선 미분값이 안정 구간 대비 이 강건 z 이상이면 급변 구간으로 봅니다. (기본값: 5)",
    )
    arg_parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="미분하기 전에 1%% 구간 값을 몇 칸 이동평균할지 지정합니다. (기본값: 3)",
    )
    arg_parser.add_argument(
        "--min_experiments",
        type=int,
        default=5,
        help="평균선을 계산할 최소 실험 수를 지정합니다. 이보다 적은 실험만 남은 구간은 제외합니다. (기본값: 5)",
    )
    arg_parser.add_argument(
        "--show",
        action="store_true",
        help="그래프를 모두 저장한 뒤 창으로 띄웁니다.",
    )
    return arg_parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    if not os.path.isfile(args.input_path):
        logger.error(f"입력 파일이 존재하지 않습니다: {args.input_path}")
        return 1
    os.makedirs(args.output_dir, exist_ok=True)

    data = add_progress_axis(
        load_sensor_data(args.input_path), args.ambient_rise, args.thermal_limit
    )
    for method, info in METHODS.items():
        output_path = os.path.join(args.output_dir, f"{info['file']}_spike_boxplot.png")
        fig = plot_spike_boxplot(
            data,
            method,
            args.z_threshold,
            args.smooth,
            args.min_experiments,
            output_path,
        )
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
