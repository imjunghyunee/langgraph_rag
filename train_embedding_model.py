# --------------------------------------------------------------
# 임베딩 모델 파인튜닝 스크립트
# - 질의-응답 쌍 데이터를 사용하여 임베딩 모델을 파인튜닝합니다
# - MultipleNegativesRankingLoss를 이용하여 학습합니다
# --------------------------------------------------------------

import os
import json
import torch
import logging
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# 로깅 설정 - 학습 진행 상황을 콘솔에 출력합니다
logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)


def main():
    # 1. 데이터 파일 경로 설정
    data_path = "my/pair/data.json"
    output_path = "finetuned/model/path"

    # 2. 모델 로드 - jinaai/jina-embeddings-v3 모델을 불러옵니다
    # trust_remote_code=True는 온라인에서 모델 코드를 신뢰하고 실행하도록 설정합니다
    logging.info(f"모델을 로드합니다: jinaai/jina-embeddings-v3")
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)

    # GPU 사용 가능 여부 확인 및 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"사용 중인 장치: {device}")
    model.to(device)  # 모델을 GPU로 이동 (가능한 경우)

    # 3. 학습 데이터 로드
    logging.info(f"데이터를 로드합니다: {data_path}")
    train_examples = load_and_prepare_data(data_path)
    logging.info(f"총 {len(train_examples)}개의 학습 예제가 준비되었습니다")

    # 4. 데이터 로더 설정
    # batch_size=32는 한 번에 32개의 예제를 처리합니다
    # 각 배치에서는 32개의 질문-문서 쌍이 있고,
    # 각 질문에 대해 1개의 양성(positive) 문서와 31개의 음성(negative) 문서가 자동으로 생성됩니다
    batch_size = 32
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    logging.info(f"배치 크기: {batch_size}, 총 배치 수: {len(train_dataloader)}")

    # 5. 손실 함수 설정 - MultipleNegativesRankingLoss
    # 이 손실 함수는 질문과 관련 문서의 임베딩을 가깝게, 관련 없는 문서의 임베딩을 멀게 학습시킵니다
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 6. 학습 설정
    # warmup_steps는 학습률이 점진적으로 증가하는 단계 수입니다
    # 전체 단계의 10%를 워밍업으로 사용하는 것이 일반적입니다
    num_epochs = 3  # 전체 데이터셋을 3번 반복합니다
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
    logging.info(f"총 에폭: {num_epochs}, 워밍업 단계: {warmup_steps}")

    # 7. 모델 학습 시작
    logging.info("모델 학습을 시작합니다...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": 2e-5},  # 학습률 설정
        output_path=output_path,  # 학습된 모델이 저장될 경로
        show_progress_bar=True,  # 진행 상태 표시줄 표시
    )

    logging.info(f"모델 학습이 완료되었습니다. 모델이 {output_path}에 저장되었습니다.")


def load_and_prepare_data(data_path: str) -> List[InputExample]:
    """
    JSON 파일에서 쌍 데이터를 로드하고 SentenceTransformer의 학습 형식으로 변환합니다.

    Args:
        data_path: 쌍 데이터가 저장된 JSON 파일 경로

    Returns:
        InputExample 객체 리스트 - SentenceTransformer 학습에 사용됩니다
    """
    # JSON 파일 로드
    with open(data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    # 각 쌍을 InputExample 형식으로 변환
    # InputExample은 SentenceTransformer가 학습에 사용하는 데이터 형식입니다
    # texts 리스트의 첫 번째 항목은 질문(쿼리), 두 번째 항목은 관련 문서(포지티브 샘플)입니다
    examples = []
    for question, text in pairs:
        example = InputExample(texts=[question, text])
        examples.append(example)

    return examples


if __name__ == "__main__":
    main()
