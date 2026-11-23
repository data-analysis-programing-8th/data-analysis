import os
import sys
import logging
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "..", "output", "battery_labels.csv")

SENSORS = {
    "meas_v": "전압 (V)",
    "meas_a": "전류 (A)",
    "meas_t1": "배터리 온도 1 (℃)",
    "meas_t2": "배터리 온도 2 (℃)",
    "meas_p": "압력",
    "atmp_t": "주변 온도 (℃)",
    "thermal_t": "열화상 온도 (℃)",
    "meas_ch4": "메탄 CH4 (ppm)",
    "meas_co": "일산화탄소 CO (ppm)",
    "meas_co2": "이산화탄소 CO2 (ppm)",
    "meas_hcn": "시안화수소 HCN (ppm)",
    "meas_hcl": "염화수소 HCl (ppm)",
    "meas_hf": "불화수소 HF (ppm)",
    "meas_n2o": "아산화질소 N2O (ppm)",
    "meas_no": "일산화질소 NO (ppm)",
    "meas_no2": "이산화질소 NO2 (ppm)",
    "meas_so2": "이산화황 SO2 (ppm)",
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
LOW_COLOR = "#f7f7f7"
MISSING_COLOR = "#d9d9d9"
DARK_TEXT = "#222222"
LIGHT_TEXT_LEVEL = 0.6
END_STAGE = 4
BASELINE_SECONDS = 60
BIN_COUNT = 20
BIN_SIZE = 100 / BIN_COUNT
NOTE = "실험 시작 0% ~ 4단계 도달 100%, 주변·열화상 온도 급상승 이후 제외"
XLABEL = "4단계 도달 시점 대비 상대 시간 (%), 주변·열화상 온도가 먼저 급상승하면 그 시점을 100%로 봄"


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


def bin_mean(data: pd.DataFrame, name: str) -> pd.DataFrame:
    time_bin = (
        (data["time_axis"] // BIN_SIZE)
        .clip(upper=BIN_COUNT - 1)
        .astype(int)
        .rename("time_bin")
    )
    per_experiment = data.groupby([time_bin, "exp_id"])[list(SENSORS)].mean()
    grouped = per_experiment.groupby("time_bin")
    counts = grouped.size().reindex(range(BIN_COUNT), fill_value=0)
    logger.info(
        f"{name} 실험: 구간별 실험 수 최소 {counts.min()}개, 최대 {counts.max()}개"
    )
    return grouped.mean().reindex(range(BIN_COUNT)).T


def normalize_rows(mean: pd.DataFrame) -> pd.DataFrame:
    low = mean.min(axis=1)
    span = mean.max(axis=1) - low
    return mean.sub(low, axis=0).div(span.where(span > 0), axis=0)


def format_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def plot_heatmap(data: pd.DataFrame, method: int, output_path: str) -> plt.Figure:
    info = METHODS[method]
    method_data = data[data["tr_method"] == method]
    mean = bin_mean(method_data, info["name"])
    scaled = normalize_rows(mean)
    span = mean.max(axis=1) - mean.min(axis=1)
    cmap = LinearSegmentedColormap.from_list(
        info["file"], [LOW_COLOR, info["color"], info["mean_color"]]
    )
    cmap.set_bad(MISSING_COLOR)

    x_edges = [index * BIN_SIZE for index in range(BIN_COUNT + 1)]
    y_edges = list(range(len(SENSORS) + 1))

    fig, ax = plt.subplots(figsize=(22, 12))
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        scaled.to_numpy(dtype=float),
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolors="white",
        linewidth=1.5,
    )
    for row, column in enumerate(SENSORS):
        for index in range(BIN_COUNT):
            level = scaled.loc[column, index]
            text_color = (
                "white"
                if pd.notna(level) and level > LIGHT_TEXT_LEVEL
                else DARK_TEXT
            )
            ax.text(
                (index + 0.5) * BIN_SIZE,
                row + 0.5,
                format_value(mean.loc[column, index]),
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    ax.set_xlim(0, 100)
    ax.set_ylim(len(SENSORS), 0)
    ax.set_xticks(x_edges, [f"{edge:g}" for edge in x_edges])
    ax.set_yticks(
        [row + 0.5 for row in range(len(SENSORS))],
        [
            f"{label}\n변화폭 {format_value(span[column])}"
            for column, label in SENSORS.items()
        ],
        fontsize=10,
    )
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    ax.set_xlabel(XLABEL, fontsize=13)
    ax.set_title(
        f"칸 숫자: 실제 평균값 · 색: 센서별로 20칸 평균의 최솟값 0 ~ 최댓값 1 · "
        f"회색: 값이 일정해 정규화할 수 없음 · 실험 {method_data['exp_id'].nunique()}개의 실험별 평균을 다시 평균",
        loc="left",
        fontsize=12,
        color="#555555",
    )

    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.02, pad=0.01)
    colorbar.set_label("센서별 정규화 값", fontsize=12)
    colorbar.outline.set_visible(False)

    fig.suptitle(
        f"{info['name']} 실험의 센서별 평균 히트맵 ({NOTE})", fontsize=17
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    logger.info(f"{info['name']} 실험 히트맵을 저장했습니다: {output_path}")
    return fig


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="17종 센서의 실험 평균을 진행률 5% 구간 20칸으로 나눠 17x20 히트맵으로 표현하는 스크립트입니다. "
        "가로축은 sensor_visualization.py의 progress _no_ambient 버전과 같으며, 가열/과충전별로 2장을 저장합니다. "
        "색은 센서별로 0~1 정규화한 값이고, 칸 숫자는 실제 평균값입니다."
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
        help="히트맵 이미지를 저장할 폴더를 지정합니다. 폴더가 없으면 생성합니다. (기본값: prelude 폴더)",
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
        "--show",
        action="store_true",
        help="히트맵을 모두 저장한 뒤 창으로 띄웁니다.",
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
        output_path = os.path.join(
            args.output_dir, f"{info['file']}_progress_no_ambient_heatmap.png"
        )
        fig = plot_heatmap(data, method, output_path)
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
