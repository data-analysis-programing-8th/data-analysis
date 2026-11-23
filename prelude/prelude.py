import os
import sys
import logging
import argparse

import matplotlib.pyplot as plt
import pandas as pd

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
ALIGNS = {
    "start": {
        "xlabel": "실험 시작 후 상대 시간 (초)",
        "note": "실험 시작 ~ 4단계 도달",
        "unit": "초",
        "bin_size": 10,
    },
    "runaway": {
        "xlabel": "4단계 도달까지 남은 시간 (초)",
        "note": "4단계 도달 시점 = 0초",
        "unit": "초",
        "bin_size": 10,
    },
    "progress": {
        "xlabel": "4단계 도달 시점 대비 상대 시간 (%)",
        "note": "실험 시작 0% ~ 4단계 도달 100%",
        "unit": "%",
        "bin_size": 1,
    },
}
NO_AMBIENT_ALIGN = "progress"
CUT_NOTE = "주변·열화상 온도 급상승 이후 제외"
CUT_XLABEL = "주변·열화상 온도가 먼저 급상승하면 그 시점을 100%로 봄"
END_STAGE = 4
BASELINE_SECONDS = 60


def load_sensor_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, dtype={"exp_id": str}, encoding="utf-8-sig")
    logger.info(
        f"센서 데이터 {len(df)}행, 실험 {df['exp_id'].nunique()}개를 불러왔습니다."
    )
    return df


def first_time(df: pd.DataFrame, condition: pd.Series) -> pd.Series:
    return df["rel_time"].where(condition).groupby(df["exp_id"]).transform("min")


def first_stage_time(df: pd.DataFrame, stage: int) -> pd.Series:
    return first_time(df, df["tr_stage"] >= stage)


def count_experiments(df: pd.DataFrame, mask: pd.Series) -> int:
    return df.loc[mask, "exp_id"].nunique()


def add_end_time(
    df: pd.DataFrame, use_cut: bool, ambient_rise: float, thermal_limit: float
) -> pd.DataFrame:
    stage_time = first_stage_time(df, END_STAGE)
    if not use_cut:
        return df.assign(cut_time=float("nan"), end_time=stage_time)

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

    logger.info(
        f"주변 온도가 {ambient_rise:g}℃ 이상 오른 실험 {count_experiments(df, ambient_time.notna())}개, "
        f"열화상 온도가 {thermal_limit:g}℃를 넘은 실험 {count_experiments(df, thermal_time.notna())}개는 "
        f"먼저 일어난 시점부터 제외합니다. "
        f"이 중 4단계 라벨보다 먼저 일어난 실험: {count_experiments(df, cut_time < stage_time)}개"
    )
    return df.assign(cut_time=cut_time, end_time=end_time)


def describe_axis(align: str, use_cut: bool) -> tuple[str, str]:
    axis = ALIGNS[align]
    if not use_cut:
        return axis["note"], axis["xlabel"]

    return f"{axis['note']}, {CUT_NOTE}", f"{axis['xlabel']}, {CUT_XLABEL}"


def add_time_axis(df: pd.DataFrame, align: str) -> pd.DataFrame:
    if align == "start":
        return df.assign(time_axis=df["rel_time"])
    if align == "runaway":
        return df.assign(time_axis=df["rel_time"] - df["end_time"])
    return df.assign(time_axis=df["rel_time"] / df["end_time"] * 100)


def select_range(df: pd.DataFrame, note: str) -> pd.DataFrame:
    keep = df["rel_time"] <= df["end_time"]
    keep &= ~(df["rel_time"] >= df["cut_time"])

    result = df[keep]
    logger.info(
        f"{note} 기준으로 {len(df) - len(result)}행을 제외하고 {len(result)}행을 사용합니다."
    )
    return result


def sensor_mean(
    data: pd.DataFrame, bin_size: float, min_experiments: int
) -> pd.DataFrame:
    time_bin = ((data["time_axis"] // bin_size) * bin_size).rename("time_bin")
    per_experiment = data.groupby([time_bin, "exp_id"])[list(SENSORS)].mean()
    grouped = per_experiment.groupby("time_bin")
    mean = grouped.mean()
    counts = grouped.size()
    return mean[counts >= min_experiments]


def plot_sensors(
    df: pd.DataFrame,
    method: int,
    align: str,
    note: str,
    xlabel: str,
    min_experiments: int,
    output_path: str,
) -> plt.Figure:
    info = METHODS[method]
    bin_size = ALIGNS[align]["bin_size"]
    data = df[df["tr_method"] == method]
    mean = sensor_mean(data, bin_size, min_experiments)
    experiments = data.groupby("exp_id")

    fig, axes = plt.subplots(3, 6, figsize=(26, 11), sharex=True)
    for ax, (column, label) in zip(axes.flat, SENSORS.items()):
        for _, experiment in experiments:
            ax.plot(
                experiment["time_axis"],
                experiment[column],
                color=info["color"],
                alpha=0.08,
                linewidth=0.5,
            )
        ax.plot(mean.index, mean[column], color=info["mean_color"], linewidth=2)
        ax.set_title(label, loc="left", fontsize=12)
        ax.grid(color="#e5e5e5", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    if align == "progress":
        axes[0, 0].set_xlim(0, 100)

    legend_ax = axes.flat[-1]
    legend_ax.plot(
        [],
        [],
        color=info["color"],
        alpha=0.35,
        linewidth=1.5,
        label=f"개별 실험 (n={experiments.ngroups})",
    )
    legend_ax.plot(
        [],
        [],
        color=info["mean_color"],
        linewidth=2,
        label=f"실험 평균 ({bin_size:g}{ALIGNS[align]['unit']} 구간, 실험 {min_experiments}개 이상)",
    )
    legend_ax.legend(loc="center", frameon=False, fontsize=13)
    legend_ax.axis("off")
    axes[-2, -1].tick_params(labelbottom=True)

    fig.suptitle(f"{info['name']} 실험의 센서 데이터와 평균 ({note})", fontsize=18)
    fig.supxlabel(xlabel, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    logger.info(f"{info['name']} 실험 그래프를 저장했습니다: {output_path}")
    return fig


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="가열/과충전 실험별로 17종 센서 데이터를 연한 색으로 모두 그리고, 실험 평균을 선으로 표현하는 스크립트입니다. "
        "가로축 3종(start, runaway, progress)의 일반 버전과 progress의 _no_ambient 버전을 가열/과충전별로, 총 8장 한 번에 저장합니다. "
        "일반 버전은 4단계 도달까지의 데이터를, _no_ambient 버전은 여기서 주변·열화상 온도가 급상승한 이후를 뺀 데이터를 그립니다."
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
        help="실험 시작 후 60초 동안의 주변 온도 중앙값보다 이 값(℃) 이상 오른 시점부터 열폭주가 이미 진행된 것으로 보고 제외합니다. (기본값: 5)",
    )
    arg_parser.add_argument(
        "--thermal_limit",
        type=float,
        default=200.0,
        help="열화상 온도가 이 값(℃)을 넘은 시점부터 열폭주가 이미 진행된 것으로 보고 제외합니다. (기본값: 200)",
    )
    arg_parser.add_argument(
        "--min_experiments",
        type=int,
        default=5,
        help="평균선을 그릴 최소 실험 수를 지정합니다. 이보다 적은 실험만 남은 구간은 평균선을 생략합니다. (기본값: 5)",
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

    df = load_sensor_data(args.input_path)

    for use_cut in (False, True):
        cut_df = add_end_time(df, use_cut, args.ambient_rise, args.thermal_limit)
        suffix = "_no_ambient" if use_cut else ""
        aligns = [NO_AMBIENT_ALIGN] if use_cut else list(ALIGNS)

        for align in aligns:
            note, xlabel = describe_axis(align, use_cut)
            data = select_range(add_time_axis(cut_df, align), note)

            for method, info in METHODS.items():
                output_path = os.path.join(
                    args.output_dir, f"{info['file']}_{align}{suffix}.png"
                )
                fig = plot_sensors(
                    data,
                    method,
                    align,
                    note,
                    xlabel,
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
