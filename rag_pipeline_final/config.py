"""
공통으로 사용하는 설정 값을 한 곳에서 관리
"""

from pathlib import Path
import os, openai

# ----- 벡터 DB 및 임베딩 설정 -----
EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"
VECTOR_DB_PATH: Path = Path("./db/faiss")

REMOTE_LLM_URL: str = "http://localhost:8000/v1/chat/completions"
REMOTE_LLM_MODEL: str = "agent:Llama-4-Scout-17B-16E-Instruct"

# # ----- openAI API 키 (환경 변수로 설정 권장) -----
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL: str = "gpt-4o"

# ----- 검색 파라미터 -----
TOP_K: int = 5
SIM_THRESHOLD: float = 0.65  # 관련성 판단 임계치(코사인 유사도 기준)
RERANK: bool = True  # 검색 결과 재정렬 여부
