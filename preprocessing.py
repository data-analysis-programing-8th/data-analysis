import os
import sys
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

import orjson
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

SPLITS = ("Training", "Validation")
LABEL_DIR = "02.라벨링데이터"
COLUMN_ORDER = [
    'exp_id', 'split', 'data_id', 'test_day', 'test_time', 'rel_time',
    'meas_v', 'meas_a', 'meas_t1', 'meas_t2', 'meas_p',
    'meas_ch4', 'meas_co', 'meas_co2', 'meas_hcn', 'meas_hcl', 'meas_hf',
    'meas_n2o', 'meas_no', 'meas_no2', 'meas_so2',
    'indu_t', 'atmp_t', 'thermal_t',
    'imgdata_id', 'tr_stage', 'tr_method', 'soc_status', 'bat_class',
]


class LabelRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data_id: str = Field(pattern=r"^\d+$", description="실험 내 프레임 번호 (예: 00001)")
    test_day: str = Field(pattern=r"^\d{6}$", description="측정 날짜 YYMMDD")
    test_time: str = Field(pattern=r"^\d{9}$", description="측정 시각 HHMMSSmmm")

    meas_v: float
    meas_a: float
    meas_t1: float
    meas_t2: float
    meas_p: float
    meas_ch4: float
    meas_co: float
    meas_co2: float
    meas_hcn: float
    meas_hcl: float
    meas_hf: float
    meas_n2o: float
    meas_no: float
    meas_no2: float
    meas_so2: float
    indu_t: float | None = Field(default=None, description="히터 온도 (과충전 실험에는 없음)")
    atmp_t: float
    thermal_t: float

    imgdata_id: str = Field(pattern=r"^\d{8}_\d{6}(_\d+)?\.png$")
    tr_stage: int = Field(ge=1, le=6)
    tr_method: int = Field(ge=1, le=2)
    soc_status: int = Field(ge=1, le=2)
    bat_class: int = Field(ge=1, le=3)


COLUMNS = [*LabelRecord.model_fields, "exp_id", "split"]


def find_label_files(input_path: str) -> list[tuple[str, str, str]]:
    files = []
    for split in SPLITS:
        label_root = os.path.join(input_path, split, LABEL_DIR)
        if not os.path.isdir(label_root):
            logger.warning(f"{split} 라벨링 데이터 폴더가 없어 건너뜁니다: {label_root}")
            continue

        exp_dirs = sorted((e for e in os.scandir(label_root) if e.is_dir()), key=lambda e: e.name)
        for exp_dir in exp_dirs:
            exp_id = exp_dir.name.rsplit("_", 1)[-1]
            for entry in sorted(os.scandir(exp_dir.path), key=lambda e: e.name):
                if entry.is_file() and entry.name.endswith(".json"):
                    files.append((split, exp_id, entry.path))

        logger.info(f"{split}: 실험 {len(exp_dirs)}개 폴더를 찾았습니다.")
    return files


def _read_bytes(path: str) -> bytes | OSError:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        return e


def load_labels(files: list[tuple[str, str, str]], workers: int = 16) -> pd.DataFrame:
    rows = []
    failures = 0
    with logging_redirect_tqdm(), ThreadPoolExecutor(max_workers=workers) as executor:
        blobs = executor.map(_read_bytes, (path for _, _, path in files))
        for (split, exp_id, path), blob in tqdm(zip(files, blobs), total=len(files), desc="라벨링 JSON 로드", unit="file"):
            try:
                if isinstance(blob, OSError):
                    raise blob
                record = LabelRecord.model_validate(orjson.loads(blob))
            except (OSError, orjson.JSONDecodeError, ValidationError) as e:
                failures += 1
                logger.warning(f"읽기/검증 실패로 건너뜁니다: {path}\n{e}")
                continue

            row = record.model_dump()
            row["exp_id"] = exp_id
            row["split"] = split
            rows.append(row)

    if failures:
        logger.warning(f"전체 {len(files)}개 중 {failures}개 파일을 건너뛰었습니다.")
    logger.info(f"라벨링 데이터 {len(rows)}개를 불러왔습니다.")
    return pd.DataFrame(rows, columns=COLUMNS)


def add_relative_time(df: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(df["test_day"] + df["test_time"], format="%y%m%d%H%M%S%f")
    start = timestamp.groupby(df["exp_id"]).transform("min")

    result = df.assign(rel_time=(timestamp - start).dt.total_seconds())
    return result.sort_values(["exp_id", "data_id"], ignore_index=True)


def parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(description='배터리 열폭주 제어 멀티모달 데이터를 pandas.DataFrame에서 처리할 수 있도록, csv 파일로 변환하는 스크립트입니다.')
    arg_parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='Training, Validation 폴더가 있는 압축 해제된 데이터셋 경로를 지정합니다. (예: ./3.개방데이터_압축해제/1.데이터)'
    )
    arg_parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='출력 csv 파일 경로를 지정합니다. 상위 폴더가 없으면 생성합니다. (예: ./output/battery_labels.csv)'
    )
    arg_parser.add_argument(
        '--workers',
        type=int,
        default=16,
        help='라벨링 파일을 병렬로 읽을 스레드 수를 지정합니다. (기본값: 16)'
    )
    return arg_parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    logger.info("배터리 열폭주 제어 멀티모달 데이터를 변환하는 스크립트를 시작합니다.")

    if not os.path.isdir(args.input_path):
        logger.error(f"입력 경로가 존재하지 않습니다: {args.input_path}")
        return 1

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    files = find_label_files(args.input_path)
    if not files:
        logger.error(f"라벨링 JSON 파일을 찾지 못했습니다: {args.input_path}")
        return 1

    df = load_labels(files, workers=args.workers)
    df = add_relative_time(df)[COLUMN_ORDER]

    df.to_csv(args.output_path, index=False, encoding='utf-8-sig')
    logger.info(f"{len(df)}행 x {len(df.columns)}열을 저장했습니다: {args.output_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
