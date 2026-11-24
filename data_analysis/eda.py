import argparse
import sys
from pathlib import Path
import math
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import DATA_PATH, GRAPH_PATH, SENSOR_COLUMNS, RANDOM_STATE

sns.set_theme(style='whitegrid')
from matplotlib import font_manager
FONT_CANDIDATES = ['Malgun Gothic', 'NanumGothic', 'NanumSquare', 'AppleGothic', 'Noto Sans CJK KR']
available_fonts = {font.name for font in font_manager.fontManager.ttflist}
for font_name in FONT_CANDIDATES:
    if font_name in available_fonts:
        plt.rcParams['font.family'] = font_name
        break
else:
    print('한글 글꼴을 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.')
plt.rcParams['axes.unicode_minus'] = False

SENSOR_LABELS = {
    'meas_v': '전압', 'meas_a': '전류',
    'meas_t1': '배터리 온도 1', 'meas_t2': '배터리 온도 2',
    'meas_p': '압력', 'atmp_t': '주위 온도', 'thermal_t': '열화상 온도',
    'meas_ch4': '메탄(CH4)', 'meas_co': '일산화탄소(CO)',
    'meas_co2': '이산화탄소(CO2)', 'meas_hcn': '시안화수소(HCN)',
    'meas_hcl': '염화수소(HCl)', 'meas_hf': '불화수소(HF)',
    'meas_n2o': '아산화질소(N2O)', 'meas_no': '일산화질소(NO)',
    'meas_no2': '이산화질소(NO2)', 'meas_so2': '이산화황(SO2)',
}

def sensor_label(column: str) -> str:
    return f'{SENSOR_LABELS.get(column, column)}\n({column})'

CATEGORY_LABELS = {'tr_method': '실험 방식', 'bat_class': '배터리 유형'}

PLOT_SAMPLE = 30000  # 그림에만 적용하는 표본 상한; CSV 통계는 전체 행 기준


def save_plot(filename: str, output_dir: Path) -> None:
    plt.savefig(output_dir / filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f'[저장] {output_dir / filename}')


def load_data(data_path: Path) -> pd.DataFrame:  # 전처리 CSV를 읽고 필수 컬럼을 검증하며 센서를 숫자로 변환
    if not data_path.is_file():
        raise FileNotFoundError(f'데이터 파일이 없습니다: {data_path}')
    df = pd.read_csv(data_path, low_memory=False)
    required = ['exp_id', 'tr_stage', *SENSOR_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'필수 컬럼이 없습니다: {missing}')
    if df.empty:
        raise ValueError('데이터 파일에 행이 없습니다.')
    for c in SENSOR_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    df['tr_stage'] = pd.to_numeric(df['tr_stage'], errors='coerce')
    if 'rel_time' in df:
        df['rel_time'] = pd.to_numeric(df['rel_time'], errors='coerce')
    return df


def basic_statistics(df: pd.DataFrame, output_dir: Path) -> None:  #전체 데이터 구조, 센서 기초 통계, 자료형을 CSV로 저장
    print(f'\n[데이터 개요] {len(df):,}행 / {len(df.columns)}열 / 실험 {df.exp_id.nunique():,}개')
    stats = df[SENSOR_COLUMNS].describe().T
    stats.to_csv(output_dir / 'sensor_statistics.csv', encoding='utf-8-sig')
    schema = pd.DataFrame({'dtype': df.dtypes.astype(str), 'missing_count': df.isna().sum(), 'missing_percent': df.isna().mean() * 100})
    schema.to_csv(output_dir / 'data_schema.csv', encoding='utf-8-sig')
    print(stats[['count', 'mean', 'std', 'min', '50%', 'max']].round(3).to_string())




def missing_ratio(df: pd.DataFrame, output_dir: Path) -> None:  #센서별 결측치 비율 분석

    ratio = (
        df[SENSOR_COLUMNS]
        .isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    ratio.rename("missing_percent").to_csv(output_dir / "missing_ratio.csv",encoding="utf-8-sig")

    exp_missing = (df.groupby("exp_id")[SENSOR_COLUMNS].agg(lambda s: s.isna().mean() * 100))
    exp_missing.to_csv(output_dir / "missing_by_experiment.csv",encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 6))

    x = np.arange(len(ratio))
    bars = ax.bar(x, ratio.values, color="steelblue")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [sensor_label(c) for c in ratio.index],
        rotation=45,
        ha="right"
    )

    ax.set_ylim(0, 100)
    ax.set_title("센서별 결측치 비율")
    ax.set_xlabel("센서")
    ax.set_ylabel("결측치 비율 (%)")

    if (ratio == 0).all():
        ax.text(
            0.5, 0.55,
            "전체 센서 결측치 없음 (0%)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold"
        )
    else:
        for bar, value in zip(bars, ratio.values):
            ax.annotate(
                f"{value:.2f}%",
                (bar.get_x() + bar.get_width() / 2,
                 bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8
            )

    fig.tight_layout()
    save_plot("missing_ratio.png", output_dir)

def correlation_analysis(df: pd.DataFrame, output_dir: Path) -> None:  #센서 간 피어슨 상관관계 분석

    corr = df[SENSOR_COLUMNS].corr()

    corr.to_csv(output_dir / "sensor_correlation.csv",encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(17, 13))

    sns.heatmap(
        corr.rename(index=sensor_label,columns=sensor_label),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax
    )

    ax.set_title("센서 간 상관관계 분석")
    ax.tick_params(axis="x", labelrotation=45)
    ax.tick_params(axis="y", labelrotation=0)

    fig.tight_layout()
    save_plot("correlation.png", output_dir)


def stage_distribution(df: pd.DataFrame, output_dir: Path) -> None:  #열폭주 단계별 데이터 분포 분석

    counts = (df["tr_stage"].value_counts(dropna=False).sort_index())

    counts.to_csv(output_dir / "stage_counts.csv",encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(counts.index.astype(str),counts.values,color="steelblue")

    ax.set(
        title="열폭주 단계별 데이터 분포",
        xlabel="열폭주 단계",
        ylabel="데이터 개수"
    )
    fig.tight_layout()
    save_plot("stage_distribution.png", output_dir)

def experiment_distribution(df: pd.DataFrame, output_dir: Path) -> None:  #실험별 데이터 개수 분석

    counts = (df.groupby("exp_id").size().sort_values(ascending=False))

    counts.to_csv(output_dir / "experiment_counts.csv",encoding="utf-8-sig" )

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(range(len(counts)),counts.values,color="steelblue")

    ax.set(
        title="실험별 데이터 개수",
        xlabel="실험 (데이터 개수 내림차순)",
        ylabel="데이터 개수"
    )

    fig.tight_layout()
    save_plot("experiment_distribution.png", output_dir)


def sensor_histograms(df: pd.DataFrame, output_dir: Path) -> None:  #센서별 측정값 분포 히스토그램

    cols = list(SENSOR_COLUMNS)
    nrows = math.ceil(len(cols) / 4)

    fig, axes = plt.subplots(nrows, 4,figsize=(18, 3.5 * nrows),squeeze=False)

    for ax, c in zip(axes.flat, cols):
        values = df[c].dropna()

        if len(values) > PLOT_SAMPLE:
            values = values.sample(PLOT_SAMPLE,random_state=RANDOM_STATE)

        if not values.empty:
            sns.histplot(
                values,
                bins=40,
                ax=ax,
                color="steelblue"
            )
        else:
            ax.text(
                0.5, 0.5, "데이터 없음",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

        ax.set(
            title=sensor_label(c),
            xlabel="측정값",
            ylabel="빈도"
        )

    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("센서별 측정값 분포",fontsize=16)
    fig.tight_layout()

    save_plot("sensor_histograms.png", output_dir)

def sensor_boxplots(df: pd.DataFrame, output_dir: Path) -> None:  #센서별 측정값 범위 및 잠재적 이상치 분석

    cols = list(SENSOR_COLUMNS)
    nrows = math.ceil(len(cols) / 4)

    fig, axes = plt.subplots(nrows, 4,figsize=(18, 3.5 * nrows),squeeze=False)

    for ax, c in zip(axes.flat, cols):
        values = df[c].dropna()

        if len(values) > PLOT_SAMPLE:
            values = values.sample(PLOT_SAMPLE,random_state=RANDOM_STATE)

        if not values.empty:
            ax.boxplot(values.to_numpy(),flierprops={"markersize": 1.5})
        else:
            ax.text(
                0.5, 0.5, "데이터 없음",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

        ax.set(title=sensor_label(c),ylabel="측정값")

    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("센서별 이상치 및 분포",fontsize=16)
    fig.tight_layout()
    save_plot("sensor_boxplots.png", output_dir)

def outlier_summary(df: pd.DataFrame, output_dir: Path) -> None:  #전체 데이터의 IQR 바깥 관측값 개수를 기록.
    records = []
    for c in SENSOR_COLUMNS:
        v = df[c].dropna()
        if v.empty:
            records.append({'sensor': c, 'lower': np.nan, 'upper': np.nan, 'outlier_count': 0, 'outlier_percent': np.nan})
            continue
        q1, q3 = v.quantile([.25, .75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((v < lower) | (v > upper)).sum())
        records.append({'sensor': c, 'lower': lower, 'upper': upper, 'outlier_count': count, 'outlier_percent': count / len(v) * 100})
    pd.DataFrame(records).set_index('sensor').to_csv(output_dir / 'iqr_outlier_summary.csv', encoding='utf-8-sig')

def category_distributions(df: pd.DataFrame,output_dir: Path) -> None:  #실험 방식 및 배터리 유형 분포 분석

    def display_label(column, value):
        if pd.isna(value):
            return "결측"
        try:
            number = int(float(value))
        except (ValueError, TypeError, OverflowError):
            return str(value)

        if column == "tr_method":
            return {1: "가열",2: "과충전"}.get(number, f"기타 방식 {number}")
        if column == "bat_class":
            return f"배터리 유형 {number}"
        return str(value)

    for column, filename in [("tr_method", "method_distribution.png"),("bat_class", "battery_class_distribution.png")]:
        if column not in df.columns:
            continue

        labels = df[column].map(
            lambda value: display_label(column, value)
        )
        counts = labels.value_counts()
        summary = (
            df.assign(display_category=labels)
            .groupby("display_category")
            .agg(
                row_count=("exp_id", "size"),
                experiment_count=("exp_id", "nunique")
            )
        )
        summary.to_csv(output_dir / f"{column}_counts.csv",encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(
            counts.index,
            counts.values,
            color="steelblue",
            width=0.6
        )

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:,}",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom"
            )

        ax.set_ylim(0, counts.max() * 1.15)
        ax.set_title(f"{CATEGORY_LABELS.get(column, column)}별 데이터 분포")
        ax.set_xlabel(CATEGORY_LABELS.get(column, column))
        ax.set_ylabel("데이터 행 수")

        fig.tight_layout()
        save_plot(filename, output_dir)


def stage_sensor_patterns(df: pd.DataFrame,output_dir: Path) -> None:  #열폭주 단계별 센서 중앙값 비교

    groups = {
        "전기_압력": [
            "meas_v", "meas_a", "meas_p"
        ],
        "온도": [
            "meas_t1", "meas_t2",
            "atmp_t", "thermal_t"
        ],
        "가스": [
            "meas_ch4", "meas_co", "meas_co2",
            "meas_hcn", "meas_hcl", "meas_hf",
            "meas_n2o", "meas_no",
            "meas_no2", "meas_so2"
        ]
    }

    med = (
        df.groupby("tr_stage")[SENSOR_COLUMNS].median().sort_index())

    med.to_csv(output_dir / "stage_sensor_medians.csv",encoding="utf-8-sig")

    for group_name, columns in groups.items():
        columns = [c for c in columns if c in med.columns]

        if not columns:
            continue

        subset = med[columns]
        scale = subset.std(ddof=0).replace(0, np.nan)
        scaled = (subset - subset.mean()) / scale
        fig, ax = plt.subplots(figsize=(max(8, len(columns) * 1.5), 6))

        sns.heatmap(
            scaled.rename(columns=sensor_label),
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".1f",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "센서별 상대 표준화 점수"}
        )

        ax.set_title(f"열폭주 단계별 {group_name} 센서 패턴")
        ax.set_xlabel("센서")
        ax.set_ylabel("열폭주 단계")
        ax.tick_params(axis="x", labelrotation=35)

        fig.tight_layout()

        save_plot(f"stage_sensor_{group_name}.png",output_dir)


def method_sensor_patterns(df: pd.DataFrame, output_dir: Path) -> None:  #가열과 과충전 실험의 센서 중앙값 비교

    if "tr_method" not in df.columns:
        return

    med = df.groupby("tr_method")[SENSOR_COLUMNS].median()

    med.to_csv(output_dir / "method_sensor_medians.csv",encoding="utf-8-sig")

    if med.empty:
        return

    method_names = {
        1: "가열",
        2: "과충전"
    }

    groups = {
        "전기_압력": [
            "meas_v", "meas_a", "meas_p"
        ],
        "온도": [
            "meas_t1", "meas_t2",
            "atmp_t", "thermal_t"
        ],
        "가스": [
            "meas_ch4", "meas_co", "meas_co2",
            "meas_hcn", "meas_hcl", "meas_hf",
            "meas_n2o", "meas_no",
            "meas_no2", "meas_so2"
        ]
    }

    for group_name, columns in groups.items():
        columns = [c for c in columns if c in med.columns]

        if not columns:
            continue

        ncols = 2
        nrows = math.ceil(len(columns) / ncols)

        fig, axes = plt.subplots(nrows, ncols,figsize=(12, 3.5 * nrows),squeeze=False)

        for ax, column in zip(axes.flat, columns):
            values = med[column].dropna()

            names = [method_names.get(int(m), str(m))for m in values.index]

            ax.bar(names, values.values,color=["steelblue", "coral"][:len(values)])

            ax.set_title(sensor_label(column))
            ax.set_ylabel("중앙값")
            ax.tick_params(axis="x", labelrotation=0)

        for ax in axes.flat[len(columns):]:
            ax.set_visible(False)

        fig.suptitle(f"실험 방식별 {group_name} 센서 중앙값 비교",fontsize=15)
        fig.tight_layout()

        save_plot(f"method_sensor_{group_name}.png",output_dir)

def time_quality(df: pd.DataFrame, output_dir: Path) -> None:  #실험별 경과시간 범위, 시간 역행/중복, 관측 간격을 점검
    if 'rel_time' not in df:
        print('[건너뜀] rel_time 컬럼 없음')
        return
    records = []
    for exp_id, g in df.groupby('exp_id', sort=False):
        times = g['rel_time'].dropna()
        delta = times.diff().dropna()
        positive = delta[delta > 0]
        records.append({'exp_id': exp_id, 'rows': len(g), 'time_min_s': times.min(), 'time_max_s': times.max(), 'duplicate_or_backward_steps': int((delta <= 0).sum()), 'median_positive_interval_s': positive.median() if not positive.empty else np.nan})
    pd.DataFrame(records).to_csv(output_dir / 'experiment_time_quality.csv', index=False, encoding='utf-8-sig')


def stage_arrival_boxplot(df: pd.DataFrame, output_dir: Path) -> None:
    stages = [1, 2, 3, 4]
    rows = df[df.tr_stage.isin(stages) & df.tr_method.isin([1, 2])]
    first = rows.groupby(["tr_method", "exp_id", "tr_stage"]).rel_time.min().dropna()
    groups = first.groupby(level=["tr_method", "tr_stage"])
    q1 = groups.transform("quantile", q=0.25)
    q3 = groups.transform("quantile", q=0.75)
    margin = 1.5 * (q3 - q1)
    first = first[first.between(q1 - margin, q3 + margin)]
    first.rename("first_arrival_seconds").to_csv(output_dir / "stage_arrival_by_experiment.csv")
    first.groupby(level=["tr_method", "tr_stage"]).agg(["mean", "count"]).to_csv(
        output_dir / "stage_arrival_summary.csv"
    )
    fig, ax = plt.subplots(figsize=(14, 7.5))
    methods = [(1, "과열(가열)", "steelblue", -0.20), (2, "과충전", "coral", 0.20)]
    for method, name, color, offset in methods:
        subset = first[first.index.get_level_values("tr_method") == method]
        table = subset.unstack("tr_stage").reindex(columns=stages)
        jitter = np.linspace(0.09, 0.21, len(table))
        x = np.array(stages) + offset + jitter[:, None]
        values = table.to_numpy()
        ax.plot(x.T, values.T, color=color, linewidth=0.6, alpha=0.12, zorder=1)
        ax.scatter(x.ravel(), values.ravel(), s=16, facecolor=color, edgecolor="white",
                   linewidth=0.55, alpha=0.95, zorder=3)
        ax.fill_between([], [], [], color=color, edgecolor="black",
                        label=f"{name} · {len(table)}회")
        for stage in stages:
            values = table[stage].dropna()
            if values.empty:
                continue
            x = stage + offset
            ax.boxplot(values, positions=[x], widths=0.15, manage_ticks=False, showfliers=False,
                       patch_artist=True, boxprops={"facecolor": color, "alpha": 0.85},
                       medianprops={"color": "#53391e"}, zorder=2)
            mean = values.mean()
            ax.plot(x, mean, "kD", markersize=5, zorder=4)
            ax.annotate(f"{mean:,.1f}초", (x, mean), xytext=(0, 7), ha="center",
                        textcoords="offset points", fontsize=10, fontweight="bold", zorder=5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1))
    ax.plot([], [], "kD", label="평균")
    ax.plot([], [], "o-", color="gray", linewidth=0.7, markersize=4, label="개별 실험 (점·연결선)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    ax.grid(False, axis="x")
    for boundary in (1.5, 2.5, 3.5):
        ax.axvline(boundary, color="#dedede", linewidth=0.8, zorder=0)
    ax.set_xticks(stages, [f"{stage}단계" for stage in stages])
    ax.set(xlabel="열폭주 단계", ylabel="실험 시작 후 경과 시간 (초)",
           title="과열·과충전 실험의 단계별 최초 도달 시간 분포 (이상치 제외)", xlim=(0.5, stages[-1] + 0.5))
    fig.tight_layout()
    save_plot("stage_arrival_1_to_4_boxplot.png", output_dir)


def main(data_path=None, output_dir=None) -> None:  #전체 EDA를 실행.
    data_path = Path(data_path) if data_path is not None else Path(DATA_PATH)
    output_dir = Path(output_dir) if output_dir is not None else Path(GRAPH_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[데이터 읽기] {data_path}')
    df = load_data(data_path)
    for fn in [basic_statistics, missing_ratio, correlation_analysis, stage_distribution, experiment_distribution, sensor_histograms, sensor_boxplots, outlier_summary, category_distributions, stage_sensor_patterns, method_sensor_patterns, time_quality, stage_arrival_boxplot]:
        print(f'\n[분석] {fn.__name__}')
        fn(df, output_dir)
    print(f'\n[완료] EDA 결과: {output_dir}')


def cli():
    parser = argparse.ArgumentParser(description="배터리 데이터 EDA 및 그래프 저장")
    parser.add_argument("--data", default=str(DATA_PATH), help="전처리 CSV 경로")
    parser.add_argument("--output", default=str(GRAPH_PATH), help="그래프 및 통계 CSV 저장 폴더")
    args = parser.parse_args()

    print("===== 배터리 열폭주 데이터 탐색적 분석(EDA) =====")
    try:
        main(data_path=args.data, output_dir=args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli())
