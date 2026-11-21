import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATA_PATH, GRAPH_PATH, SENSOR_COLUMNS


# 기초 통계 분석
def basic_statistics(df):
    print("데이터 크기:", df.shape)
    print(df[SENSOR_COLUMNS].describe())
    print(df.isnull().sum())


# 상관관계 분석
def correlation_analysis(df):
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
    plt.yticks(rotation=0)

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
    plt.figure(figsize=(10, 6))

    sns.countplot(data=df, x="tr_stage")

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


def main():
    GRAPH_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    basic_statistics(df)
    correlation_analysis(df)
    stage_distribution(df)


if __name__ == "__main__":
    main()