import argparse
import sys

import eda
from config import DATA_PATH, GRAPH_PATH


def main():
    parser = argparse.ArgumentParser(description="배터리 데이터 EDA 및 그래프 저장")
    parser.add_argument("--data", default=str(DATA_PATH), help="전처리 CSV 경로")
    parser.add_argument("--output", default=str(GRAPH_PATH), help="그래프 및 통계 CSV 저장 폴더")
    args = parser.parse_args()

    print("===== 배터리 열폭주 데이터 탐색적 분석(EDA) =====")
    try:
        eda.main(data_path=args.data, output_dir=args.output)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
