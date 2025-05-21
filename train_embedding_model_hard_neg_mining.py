# -------------------------------------------------------------------
# train_embedding_model.py
#  - Positive-aware Hard-Negative Mining(NV-Retriever) + Triplet Loss
#  - 학습 데이터: [{"query": "...", "passage": "..."}] or [[query, passage], ...]
#  - 출력: 파인튜닝된 SentenceTransformer 모델
# -------------------------------------------------------------------

from __future__ import annotations

import json, logging, argparse
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    util,
)
from torch.utils.data import DataLoader

# --------------------- 하이퍼 파라미터 기본값 ----------------------
STUDENT_MODEL_NAME = "jinaai/jina-embeddings-v3"
TEACHER_MODEL_NAME = "intfloat/e5-mistral-7b-instruct"
BATCH_SIZE = 16  # Triplet 예제 하나당 3개의 문장을 포함하므로 16이 안전
EPOCHS = 3
LR = 2e-5
TOP_K = 20  # mining 시 후보 개수
PERC_THRESHOLD = 0.95  # pos_score * 0.95 보다 높은 neg 제외
WARMUP_RATIO = 0.1  # 총 스텝의 10 %

# ------------------------- 로깅 설정 -------------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ------------------- Hard-Negative Mining 함수 ---------------------
def mine_hard_negatives(
    teacher: SentenceTransformer,
    queries: List[str],
    positives: List[str],
    top_k: int = TOP_K,
    perc_threshold: float = PERC_THRESHOLD,
    batch_size: int = 256,
    device: str = "cpu",
) -> List[Tuple[str, str, str]]:
    """
    Positive-aware TopK-PercPos 방식으로 (query, positive, hard_negative) triplet 리스트 반환
    """
    logging.info("🪨 Hard-negative mining 시작 …")
    teacher.to(device)
    teacher.eval()

    # 1) 전체 positive 문단 임베딩 한 번에 계산
    all_pass_emb = teacher.encode(
        positives,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )  # shape=(N, D)

    triplets: List[Tuple[str, str, str]] = []

    # 2) 각 쿼리마다 hardest negative 1개 선택
    for q, pos in tqdm(
        list(zip(queries, positives)), desc="mining", total=len(queries)
    ):
        q_emb = teacher.encode(q, convert_to_tensor=True, normalize_embeddings=True)
        # cosine scores = dot product (정규화된 상태)
        scores = (all_pass_emb @ q_emb).cpu().numpy()  # (N,)

        pos_emb = teacher.encode(pos, convert_to_tensor=True, normalize_embeddings=True)
        pos_score = float(util.cos_sim(q_emb, pos_emb)[0])

        # top-k 후보 인덱스 (자기 자신 포함 가능)
        top_idx = scores.argsort()[::-1][: top_k + 1]

        neg_idx = None
        for idx in top_idx:
            cand, score = positives[idx], scores[idx]
            if cand == pos:  # positive 자체는 제외
                continue
            # pos_score * α 보다 높은 candidate 는 제외 (false-negative 가능)
            if score < pos_score * perc_threshold:
                neg_idx = idx
                break

        # 예외: 조건을 만족하는 neg가 없으면 최저 점수 passage 사용
        if neg_idx is None:
            neg_idx = scores.argmin()

        triplets.append((q, pos, positives[neg_idx]))

    logging.info(f"✅ Hard-negatives mined: {len(triplets)}")
    return triplets


# --------------------- 데이터 로드 & 전처리 ------------------------
def load_pairs(data_path: Path) -> List[Tuple[str, str]]:
    """
    다양한 JSON 형태([q, p] or {"query": q, "passage": p}) 지원
    """
    logging.info(f"데이터 로드: {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    pairs: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, list):  # [query, passage]
            query, passage = item
        elif isinstance(item, dict):  # {"query": ..., "passage": ...}
            query = item["query"]
            passage = item["passage"]
        else:
            raise ValueError("지원하지 않는 데이터 포맷")
        pairs.append((query.strip(), passage.strip()))
    logging.info(f"총 쌍 개수: {len(pairs)}")
    return pairs


def build_dataloader(
    triplets: List[Tuple[str, str, str]],
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    (q, pos, neg) → InputExample(texts=[q, pos, neg])
    """
    examples = [InputExample(texts=list(tpl)) for tpl in triplets]
    return DataLoader(examples, shuffle=True, batch_size=batch_size)


# -------------------------- 학습 루틴 -----------------------------
def train(
    student: SentenceTransformer,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    warmup_ratio: float,
    output_dir: Path,
):
    """
    TripletLoss 로 파인튜닝
    """
    total_steps = len(dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    logging.info(
        f"학습 시작  | epochs={epochs}  total_steps={total_steps}  warmup={warmup_steps}"
    )

    loss_fn = losses.TripletLoss(
        student,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        margin=0.05,
    )

    student.fit(
        train_objectives=[(dataloader, loss_fn)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    logging.info(f"모델 저장 완료 → {output_dir}")


# ------------------------------ main -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="쌍 데이터 JSON 경로")
    parser.add_argument("--output", type=Path, required=True, help="모델 저장 폴더")
    parser.add_argument("--student", default=STUDENT_MODEL_NAME)
    parser.add_argument("--teacher", default=TEACHER_MODEL_NAME)
    args = parser.parse_args()

    # 1) 모델 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")

    logging.info(f"📚 Student 모델 로드: {args.student}")
    student = SentenceTransformer(args.student, trust_remote_code=True).to(device)

    logging.info(f"🎓 Teacher 모델 로드: {args.teacher}")
    teacher = SentenceTransformer(args.teacher).to(device)

    # 2) 데이터 로드
    pairs = load_pairs(args.data)
    queries, positives = zip(*pairs)

    # 3) Hard-negative mining
    triplets = mine_hard_negatives(
        teacher,
        list(queries),
        list(positives),
        top_k=TOP_K,
        perc_threshold=PERC_THRESHOLD,
        device=device,
    )

    # 4) DataLoader
    dataloader = build_dataloader(triplets, batch_size=BATCH_SIZE)

    # 5) 학습
    args.output.mkdir(parents=True, exist_ok=True)
    train(
        student=student,
        dataloader=dataloader,
        epochs=EPOCHS,
        lr=LR,
        warmup_ratio=WARMUP_RATIO,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
