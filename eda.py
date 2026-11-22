
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PATH, GRAPH_PATH, SENSOR_COLUMNS


# 기초 통계 분석
def basic_statistics(df):
    print("\n===== Basic Statistics =====")
    print("데이터 크기:", df.shape)

    print("\n[데이터 정보]")
    df.info()

    print("\n[센서별 기초 통계]")
    print(df[SENSOR_COLUMNS].describe())

    print("\n[컬럼별 결측치 개수]")
    print(df.isnull().sum())


# 센서 상관관계 분석
def correlation_analysis(df):
    print("\n===== Correlation Analysis =====")

    plt.figure(figsize=(20, 12))

    sns.heatmap(
        df[SENSOR_COLUMNS].corr(),
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        center=0,
        vmin=-1,
        vmax=1
    )

    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.title("Sensor Correlation Matrix")
    plt.tight_layout()

    plt.savefig(
        GRAPH_PATH / "correlation.png",
        dpi=300
    )

    plt.show()
    plt.close()


# 단계별 데이터 분포
def stage_distribution(df):
    print("\n===== Stage Distribution =====")

    print(df["tr_stage"].value_counts().sort_index())

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        x="tr_stage",
        order=sorted(df["tr_stage"].dropna().unique())
    )

    plt.title("Stage Distribution")
    plt.xlabel("Stage")
    plt.ylabel("Count")
    plt.tight_layout()

    plt.savefig(
        GRAPH_PATH / "stage_distribution.png",
        dpi=300
    )

    plt.show()
    plt.close()


# 실험 ID별 데이터 분포
def experiment_distribution(df):
    print("\n===== Experiment Distribution =====")

    counts = df.groupby("exp_id").size().sort_index()

    print(counts)

    plt.figure(figsize=(18, 7))

    counts.plot(kind="bar")

    plt.title("Samples per Experiment")
    plt.xlabel("Experiment ID")
    plt.ylabel("Sample Count")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()

    plt.savefig(
        GRAPH_PATH / "experiment_distribution.png",
        dpi=300
    )

    plt.show()
    plt.close()


# 센서별 결측치 비율
def missing_ratio(df):
    print("\n===== Missing Value Ratio =====")

    missing = df[SENSOR_COLUMNS].isna().mean() * 100

    print(missing.sort_values(ascending=False))

    plt.figure(figsize=(14, 6))

    sns.barplot(
        x=missing.index,
        y=missing.values
    )

    plt.title("Missing Value Ratio by Sensor")
    plt.xlabel("Sensor")
    plt.ylabel("Missing Ratio (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(
        GRAPH_PATH / "missing_ratio.png",
        dpi=300
    )

    plt.show()
    plt.close()


# 전체 EDA 실행
def main():
    GRAPH_PATH.mkdir(parents=True, exist_ok=True)

    print("\n===== Loading Dataset =====")
    df = pd.read_csv(DATA_PATH)

    print("데이터 로딩 완료:", DATA_PATH)

    basic_statistics(df)
    correlation_analysis(df)
    stage_distribution(df)
    experiment_distribution(df)
    missing_ratio(df)

    print("\n===== EDA Completed =====")
    print("그래프 저장 위치:", GRAPH_PATH)


if __name__ == "__main__":
    main()
