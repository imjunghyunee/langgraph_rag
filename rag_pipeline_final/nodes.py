from __future__ import annotations

import json
import requests
from typing import List
from rag_pipeline.graph_state import GraphState
from rag_pipeline import retrievers, config


def node_retrieve(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context = retrievers.vectordb_retrieve(query)
    return {"context": context}


def node_retrieve_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context = retrievers.vectordb_hybrid_retrieve(query, weights=hybrid_weights)
    return {"context": context}


def node_retrieve_summary(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context, explanation = retrievers.summary_retrieve(query)
    return {"context": context, "explanation": explanation}


def node_retrieve_summary_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context, explanation = retrievers.summary_hybrid_retrieve(
        query, weights=hybrid_weights
    )
    return {"context": context, "explanation": explanation}


def node_retrieve_hyde(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context, explanation = retrievers.hyde_retrieve(query)
    return {"context": context, "explanation": explanation}


def node_retrieve_hyde_hybrid(state: GraphState) -> GraphState:
    query: str = state["question"][-1]

    # 하이브리드 가중치 확인
    hybrid_weights = state.get("hybrid_weights", [0.5, 0.5])

    context, explanation = retrievers.hyde_hybrid_retrieve(
        query, weights=hybrid_weights
    )
    return {"context": context, "explanation": explanation}


def node_relevance_check(state: GraphState) -> GraphState:
    with open("score/path.json", "r", encoding="utf-8") as f:
        scores: List[float] = json.load(f)

    context_docs = state["context"]  # List[Document]
    contents = [d.page_content for d in context_docs]

    filtered_scores: List[float] = []
    filtered_context: List[str] = []

    for i, score in enumerate(scores):
        if score >= config.SIM_THRESHOLD:
            filtered_scores.append(score)
            filtered_context.append(contents[i])

    return {"context": filtered_context, "score": filtered_scores}


def node_llm_answer(state: GraphState) -> GraphState:
    query: str = state["question"][-1]
    context: str | List[str] = state["context"]

    # 컨텍스트가 리스트면 문자열로 합침
    if isinstance(context, list):
        context_str = "\n\n---\n\n".join(context)
    else:
        context_str = context

    payload = {
        "model": config.REMOTE_LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an AI assistant."},
            {
                "role": "user",
                "content": f"[Question]\n{query}\n\n[Context]\n{context_str}",
            },
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    res = requests.post(config.REMOTE_LLM_URL, json=payload, timeout=60).json()
    answer = res["choices"][0]["message"]["content"].strip()
    if not isinstance(answer, str):
        answer = str(answer)

    return {"answer": answer, "messages": [("assistant", answer)]}
