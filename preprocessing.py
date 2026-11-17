import os
import logging
import argparse

from pydantic import BaseModel, ConfigDict, Field


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


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='배터리 열폭주 제어 멀티모달 데이터를 pandas.DataFrame에서 처리할 수 있도록, csv 파일로 변환하는 스크립트입니다.')
    arg_parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help='입력 데이터셋의 경로를 지정합니다. (예: ./data/데이터/)'
    )
    arg_parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='출력 데이터셋의 경로를 지정합니다. (예: ./data/output_dataset.csv)'
    )
    args = arg_parser.parse_args()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logging.basicConfig(level=logging.INFO, handlers=[stream_handler])
    logger = logging.getLogger(__name__)
    logger.addHandler(stream_handler)

    logger.info("배터리 열폭주 제어 멀티모달 데이터를 변환하는 스크립트를 시작합니다.")

    if not os.path.exists(args.input_path) or not os.path.exists(args.output_path):
        logger.error(f"입력 또는 출력 경로가 존재하지 않습니다")
        exit(255)

    input_path = args.input_path
    root_dir = os.listdir(path=f'{input_path}')

    if "Training" in root_dir:
        train_dir = os.path.join(input_path, "Training")

    if "Validation" in root_dir:
        val_dir = os.path.join(input_path, "Validation")
