# --------------------------------------------------------------
# Embedding model fine-tuning with MultipleNegativesRankingLoss
# Target model  : BAAI/bge-m3  (trainable, ~560 M params)
# Environment   : CPU-only
# --------------------------------------------------------------
import os, json, torch, logging
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader

# ---- 1. Hugging Face Hub download options --------------------
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # show tqdm bars
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster, robust dl

# ---- 2. Logging ---------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# ---- 3. Data loader -----------------------------------------
def load_pairs(path: str) -> List[InputExample]:
    with open(path, encoding="utf-8") as f:
        pairs = json.load(f)  # [[question, answer], ...]
    return [InputExample(texts=[q, a]) for q, a in pairs]


# ---- 4. Training pipeline -----------------------------------
def main():
    data_path = "./pair_data.json"
    output_path = "./ckpt"
    model_name = "BAAI/bge-m3"  # ← jina-v3 대체 모델

    logging.info(f"Load model: {model_name}")
    model = SentenceTransformer(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logging.info(f"Device in use: {device}")

    train_examples = load_pairs(data_path)
    logging.info(f"{len(train_examples)} training examples loaded")

    loader = DataLoader(train_examples, batch_size=16, shuffle=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    logging.info("Start fine-tuning …")
    model.fit(
        train_objectives=[(loader, loss_fn)],
        epochs=1,  # CPU 기준 예시
        warmup_steps=int(len(loader) * 0.1),
        optimizer_params={"lr": 2e-5},
        output_path=output_path,
        show_progress_bar=True,
    )
    logging.info(f"Finished. Model saved to {output_path}")


if __name__ == "__main__":
    main()
