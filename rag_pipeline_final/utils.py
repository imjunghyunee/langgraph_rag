"""
보조 함수 모음
"""

from __future__ import annotations
from typing import List, Tuple
from langchain.schema import BaseRetriever, Document
import openai
from rag_pipeline import config


# 검색 문서를 LLM 입력용 문자열로 변환
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        [f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)]
    )


# messages(List[Tuple[str,str]]) → langchain memory 호환 포맷
def messages_to_history(messages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return messages


# 단일 프롬프트 헬퍼
def build_llm_prompt(question: str, context: str) -> str:
    return (
        "You are an AI assistant for semiconductor industry. Please reason step by step, and put your final answer within \boxed{}.\n\n"
        f"[Context]\n{context}\n\n"
        f"[Question]\n{question}\n\n"
        "Based on the provided Context, answer precisely to the Quetion. You must ansewr in Korean."
    )


def chat_with_gpt(
    messages: list[dict[str, str]],
    max_tokens: int = 1000,
    temperature: float = 0.6,
) -> str:
    """
    GPT‑4o에게 Chat Completion 요청 후 첫 번째 응답 텍스트 반환
    """
    resp = openai.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
