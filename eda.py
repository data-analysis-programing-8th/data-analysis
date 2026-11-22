from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")  # 터미널에서 실행할 때 그래프 창을 띄우지 않고 파일로 저장
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATA_PATH, GRAPH_PATH, SENSOR_COLUMNS

sns.set_theme(style="whitegrid")


def save_plot(filename: str, output_dir: Path) -> None:  # Matplotlib로 그린 그래프를 PNG로 저장하고, 메모리에서 닫음
    plt.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[저장] {output_dir / filename}")


def load_data(data_path: Path) -> pd.DataFrame: # csv 파일을 읽고 필수 컬럼 확인 후, 센서값을 숫자로 변환해 반환

    if not data_path.is_file():
        raise FileNotFoundError(
            f"CSV 파일이 없습니다: {data_path}\n"
            "전처리된 battery_labels.csv 파일을 output 폴더에 넣어 주세요."
        )
    df = pd.read_csv(data_path, low_memory=False)
    required = ["exp_id", "tr_stage", *SENSOR_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")
    if df.empty:
        raise ValueError("CSV에 데이터가 없습니다.")
    for column in SENSOR_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def basic_statistics(df: pd.DataFrame, output_dir: Path) -> None:  # 전체 행/열 수, 센서별 통계량, 결측치 개수를 출력 및 csv로 저장

    print("\n===== Basic Statistics =====")
    print(f"데이터 크기: {df.shape[0]:,}행 × {df.shape[1]}열")
    print("\n[센서별 기초 통계]")
    stats = df[SENSOR_COLUMNS].describe().T
    print(stats.to_string())
    stats.to_csv(output_dir / "sensor_statistics.csv", encoding="utf-8-sig")

    missing = df.isna().sum().rename("missing_count")
    missing.to_csv(output_dir / "missing_counts.csv", encoding="utf-8-sig")
    print("\n[결측치 개수]")
    print(missing.to_string())


def correlation_analysis(df: pd.DataFrame, output_dir: Path) -> None:  # 17개 센서 간 상관계수 계산 후 csv와 히트맵 저장
    corr = df[SENSOR_COLUMNS].corr()
    corr.to_csv(output_dir / "sensor_correlation.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(17, 13))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Sensor Correlation Matrix")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    save_plot("correlation.png", output_dir)


def stage_distribution(df: pd.DataFrame, output_dir: Path) -> None:  # tr_stage별 데이터 개수 집계, csv와 막대그래프 저장
    counts = df["tr_stage"].value_counts(dropna=False).sort_index()
    counts.rename("count").to_csv(output_dir / "stage_counts.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.countplot(
        data=df,
        x="tr_stage",
        order=sorted(df["tr_stage"].dropna().unique()),
        color="steelblue",
        ax=ax,
    )
    ax.set(title="Stage Distribution", xlabel="Stage", ylabel="Sample Count")
    save_plot("stage_distribution.png", output_dir)


def experiment_distribution(df: pd.DataFrame, output_dir: Path) -> None:  # exp_id별 데이터 개수, 실험별 샘플 수 분포 확인
    counts = df.groupby("exp_id", dropna=False).size().sort_values(ascending=False)
    counts.rename("count").to_csv(output_dir / "experiment_counts.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(max(14, min(24, len(counts) * 0.20)), 6))
    ax.bar(range(len(counts)), counts.to_numpy())
    ax.set(title="Samples per Experiment", xlabel="Experiment ID", ylabel="Sample Count")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([str(x) for x in counts.index], rotation=90, fontsize=6)
    save_plot("experiment_distribution.png", output_dir)


def missing_ratio(df: pd.DataFrame, output_dir: Path) -> None:
    """각 센서의 결측치 비율(%)을 구해 csv와 막대그래프로 저장.
    모델 학습 전에 결측치 처리가 필요한지 파악할 수 있다."""
    ratio = (df[SENSOR_COLUMNS].isna().mean() * 100).sort_values(ascending=False)
    ratio.rename("missing_percent").to_csv(output_dir / "missing_ratio.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.barplot(x=ratio.index, y=ratio.values, color="steelblue", ax=ax)
    ax.set(title="Missing Value Ratio by Sensor", xlabel="Sensor", ylabel="Missing (%)")
    ax.tick_params(axis="x", labelrotation=45)
    save_plot("missing_ratio.png", output_dir)


def sensor_histograms(df: pd.DataFrame, output_dir: Path) -> None: #센서별 값의 분포를 히스토그램으로 시각화, 값의 집중,치우침,분포를 살펴본다.
    
    columns = SENSOR_COLUMNS
    ncols = 4
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows))
    for ax, column in zip(axes.flat, columns):
        values = df[column].dropna()
        if not values.empty:
            sns.histplot(values, bins=40, ax=ax, color="steelblue")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=column, xlabel="Value", ylabel="Count")
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Sensor Value Distributions", fontsize=16)
    fig.tight_layout()
    save_plot("sensor_histograms.png", output_dir)


def sensor_boxplots(df: pd.DataFrame, output_dir: Path) -> None: #센서별 중앙값,사분위,이상치를 boxplot으로 그림.센서별 단위가 달라 각각 개별 축 사용. 
    
    columns = SENSOR_COLUMNS
    ncols = 4
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.2 * nrows))
    for ax, column in zip(axes.flat, columns):
        values = df[column].dropna()
        if not values.empty:
            sns.boxplot(y=values, ax=ax, color="lightblue", fliersize=1)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set(title=column, xlabel="", ylabel="Value")
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    fig.suptitle("Sensor Boxplots", fontsize=16)
    fig.tight_layout()
    save_plot("sensor_boxplots.png", output_dir)


def category_distributions(df: pd.DataFrame, output_dir: Path) -> None:  #tr_method, bat_class 컬럼별 샘플 수를 비교. 컬럼이 없으면 넘김.
    
    for column, filename, title in [
        ("tr_method", "method_distribution.png", "Experiment Method Distribution"),
        ("bat_class", "battery_class_distribution.png", "Battery Class Distribution"),
    ]:
        if column not in df.columns:
            print(f"[건너뜀] {column}: 컬럼 없음")
            continue
        counts = df[column].value_counts(dropna=False).sort_index()
        counts.rename("count").to_csv(output_dir / f"{column}_counts.csv", encoding="utf-8-sig")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([str(v) for v in counts.index], counts.to_numpy())
        ax.set(title=title, xlabel=column, ylabel="Sample Count")
        save_plot(filename, output_dir)


def main(data_path=None, output_dir=None) -> None:  #EDA 전체 실행을 담당하는 메인 함수. 데이터 로딩 -> 통계 -> 그래프 저장순으로 실행
    data_path = Path(data_path) if data_path is not None else Path(DATA_PATH)
    output_dir = Path(output_dir) if output_dir is not None else Path(GRAPH_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[데이터 읽기] {data_path}")
    df = load_data(data_path)
    basic_statistics(df, output_dir)
    correlation_analysis(df, output_dir)
    stage_distribution(df, output_dir)
    experiment_distribution(df, output_dir)
    missing_ratio(df, output_dir)
    sensor_histograms(df, output_dir)
    sensor_boxplots(df, output_dir)
    category_distributions(df, output_dir)
    print(f"\n[완료] EDA 결과 저장 위치: {output_dir}")


if __name__ == "__main__":
    main()