"""
FastAPI 메인 애플리케이션
파이토케미컬 연구 플랫폼 백엔드
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Genspark LLM 프록시 설정 ─────────────────────────────────────────────────
# 환경변수 OPENAI_API_KEY / OPENAI_BASE_URL 이 주입돼 있으면 그것을 우선 사용
# (Genspark 프록시: https://www.genspark.ai/api/llm_proxy/v1)
_ENV_OPENAI_KEY  = os.environ.get("GSK_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
_ENV_OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "")

def resolve_openai_key(ui_key: str = "") -> tuple[str, str]:
    """(api_key, base_url) 반환 — 환경변수 우선, 없으면 UI 입력값 사용"""
    key  = _ENV_OPENAI_KEY  or ui_key
    base = _ENV_OPENAI_BASE or ""
    return key, base

# 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent))

from pubmed_crawler import (
    crawl_phytochemical_papers, load_all_papers, get_stats,
    PHYTOCHEMICAL_CATEGORIES, HEALTH_CONDITIONS, ABSTRACTS_DIR
)
from graph_builder import (
    build_graph_from_papers, load_graph, search_graph, get_graph_stats, save_graph
)
from rag_chatbot import RAGChatbot

# ─── 앱 초기화 ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PhytoGraph Research Platform",
    description="파이토케미컬 건강 상관성 연구 플랫폼",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# 전역 상태
crawl_progress = {}
build_progress = {}
index_progress = {}
chatbot_instance: Optional[RAGChatbot] = None


# ─── Request Models ───────────────────────────────────────────────────────────
class CrawlRequest(BaseModel):
    phytochemical: str
    health_condition: Optional[str] = None
    max_results: int = 50
    meta_analysis_only: bool = False
    pub_type: Optional[str] = None


class BuildGraphRequest(BaseModel):
    use_all_papers: bool = True
    pmids: Optional[list] = None
    openai_api_key: Optional[str] = ""  # UI 입력 (없으면 환경변수 사용)


class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[list] = []
    openai_api_key: Optional[str] = ""  # UI 입력 (없으면 환경변수 사용)


class IndexRequest(BaseModel):
    openai_api_key: Optional[str] = ""  # UI 입력 (없으면 환경변수 사용)


# ─── 유틸리티 ─────────────────────────────────────────────────────────────────
def get_or_create_chatbot(api_key: str = "") -> RAGChatbot:
    global chatbot_instance
    key, base = resolve_openai_key(api_key)
    if chatbot_instance is None:
        chatbot_instance = RAGChatbot(key, base_url=base)
    return chatbot_instance


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """메인 페이지 서빙"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Frontend not found</h1>")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "PhytoGraph Research Platform"}


# ─── PubMed 크롤링 ────────────────────────────────────────────────────────────

@app.get("/api/categories")
async def get_categories():
    """파이토케미컬 카테고리 및 건강 조건 목록"""
    return {
        "phytochemical_categories": PHYTOCHEMICAL_CATEGORIES,
        "health_conditions": HEALTH_CONDITIONS
    }


@app.post("/api/crawl")
async def crawl_papers(request: CrawlRequest, background_tasks: BackgroundTasks):
    """PubMed 논문 크롤링 시작 (비동기)"""
    task_id = f"crawl_{request.phytochemical}_{id(request)}"
    crawl_progress[task_id] = {"status": "started", "step": "initializing"}
    
    def update_progress(data):
        crawl_progress[task_id] = {**data, "status": "running"}
    
    def do_crawl():
        try:
            result = crawl_phytochemical_papers(
                phytochemical=request.phytochemical,
                health_condition=request.health_condition,
                max_results=request.max_results,
                meta_analysis_only=request.meta_analysis_only,
                pub_type=request.pub_type,
                progress_callback=update_progress
            )
            if result["total"] == 0:
                crawl_progress[task_id] = {
                    "status": "complete",
                    "result": {
                        "total": 0, "meta_count": 0, "new_count": 0,
                        "query": result.get("query", ""),
                        "papers": [],
                        "warning": "검색 결과가 없습니다. 검색어를 확인하거나 잠시 후 다시 시도하세요. (NCBI API 일시 차단 가능)"
                    }
                }
            else:
                crawl_progress[task_id] = {
                    "status": "complete",
                    "result": {
                        "total": result["total"],
                        "meta_count": result["meta_count"],
                        "new_count": result["new_count"],
                        "query": result["query"],
                        "papers": result["papers"][:10]
                    }
                }
        except Exception as e:
            crawl_progress[task_id] = {"status": "error", "message": str(e)}
    
    background_tasks.add_task(do_crawl)
    return {"task_id": task_id, "message": "크롤링 시작됨"}


@app.get("/api/crawl/status/{task_id}")
async def get_crawl_status(task_id: str):
    """크롤링 작업 상태 확인"""
    if task_id not in crawl_progress:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return crawl_progress[task_id]


@app.get("/api/papers")
async def get_papers(
    page: int = 1,
    limit: int = 20,
    meta_only: bool = False,
    search: Optional[str] = None,
    year: Optional[str] = None,
    journal: Optional[str] = None,
    pub_type: Optional[str] = None,
    sort_by: str = "year"
):
    """저장된 논문 목록 조회 — 다중 필터 + 정렬 지원"""
    papers = load_all_papers()
    
    if meta_only:
        papers = [p for p in papers if p.get("is_meta_analysis")]
    
    if search:
        search_lower = search.lower()
        papers = [p for p in papers if 
                  search_lower in p.get("title", "").lower() or
                  search_lower in p.get("abstract", "").lower()]
    
    if year:
        papers = [p for p in papers if p.get("year") == year]
    
    if journal:
        j_lower = journal.lower()
        papers = [p for p in papers if j_lower in p.get("journal", "").lower()]
    
    if pub_type == "meta":
        papers = [p for p in papers if p.get("is_meta_analysis")]
    elif pub_type == "sr":
        papers = [p for p in papers if p.get("is_systematic_review")]

    # 정렬
    if sort_by == "year":
        papers.sort(key=lambda x: x.get("year", "0"), reverse=True)
    elif sort_by == "title":
        papers.sort(key=lambda x: x.get("title", "").lower())
    elif sort_by == "journal":
        papers.sort(key=lambda x: x.get("journal", "").lower())
    
    total = len(papers)
    start = (page - 1) * limit
    paginated = papers[start:start + limit]
    
    return {
        "papers": paginated,
        "total": total,
        "page": page,
        "total_pages": (total + limit - 1) // limit
    }


@app.get("/api/papers/stats")
async def get_paper_stats():
    """논문 통계"""
    return get_stats()


# ─── Graph DB ────────────────────────────────────────────────────────────────

@app.post("/api/graph/build")
async def build_graph(request: BuildGraphRequest, background_tasks: BackgroundTasks):
    """LLM으로 그래프 DB 구축 (비동기)"""
    task_id = f"graph_{id(request)}"
    build_progress[task_id] = {"status": "started"}
    
    def update_progress(data):
        build_progress[task_id] = {**data, "status": "running"}
    
    def do_build():
        from openai import OpenAI
        try:
            key, base = resolve_openai_key(request.openai_api_key or "")
            client = OpenAI(
                api_key=key,
                base_url=base or None  # None이면 기본 OpenAI 엔드포인트
            )
            print(f"[GraphBuild] base_url={base!r}, key_prefix={key[:8]}...")
            
            if request.use_all_papers:
                papers = load_all_papers()
            else:
                papers = [p for p in load_all_papers() 
                         if p.get('pmid') in (request.pmids or [])]
            
            if not papers:
                build_progress[task_id] = {
                    "status": "error", 
                    "message": "처리할 논문이 없습니다. 먼저 크롤링을 실행하세요."
                }
                return
            
            # 메모리/비용 절약: 최대 30개 논문
            papers = papers[:30]
            
            result = build_graph_from_papers(papers, client, update_progress)
            
            build_progress[task_id] = {
                "status": "complete",
                "stats": result.get("stats", {}),
                "message": f"그래프 구축 완료"
            }
        except Exception as e:
            build_progress[task_id] = {"status": "error", "message": str(e)}
    
    background_tasks.add_task(do_build)
    return {"task_id": task_id, "message": "그래프 구축 시작됨"}


@app.get("/api/graph/build/status/{task_id}")
async def get_build_status(task_id: str):
    """그래프 구축 상태"""
    if task_id not in build_progress:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return build_progress[task_id]


@app.get("/api/graph")
async def get_graph(
    search: Optional[str] = None,
    node_type: Optional[str] = None,
    relation_type: Optional[str] = None,
    focus_node: Optional[str] = None,
    phytochemical: Optional[str] = None,
    food_source_view: bool = False
):
    """그래프 데이터 반환 — 다중 필터 지원"""
    if focus_node:
        # 특정 노드 중심 서브그래프 (1-hop 이웃)
        data = load_graph()
        target = next((n for n in data["nodes"] if n["id"] == focus_node), None)
        if target:
            connected_edges = [e for e in data["edges"]
                               if e["source"] == focus_node or e["target"] == focus_node]
            neighbor_ids = set()
            for e in connected_edges:
                neighbor_ids.add(e["source"])
                neighbor_ids.add(e["target"])
            neighbor_ids.add(focus_node)
            data = {
                "nodes": [n for n in data["nodes"] if n["id"] in neighbor_ids],
                "edges": connected_edges
            }
    elif phytochemical:
        # 특정 파이토케미컬 중심 서브그래프
        data = load_graph()
        phyto_lower = phytochemical.lower()
        phyto_nodes = [n for n in data["nodes"]
                       if n.get("type") == "Phytochemical"
                       and phyto_lower in n.get("name", "").lower()]
        phyto_ids = {n["id"] for n in phyto_nodes}
        connected_edges = [e for e in data["edges"]
                           if e["source"] in phyto_ids or e["target"] in phyto_ids]
        neighbor_ids = phyto_ids.copy()
        for e in connected_edges:
            neighbor_ids.add(e["source"])
            neighbor_ids.add(e["target"])
        data = {
            "nodes": [n for n in data["nodes"] if n["id"] in neighbor_ids],
            "edges": connected_edges
        }
    elif search:
        data = search_graph(search)
    else:
        data = load_graph()

    # 식품 소스 전용 뷰: Phytochemical ↔ FoodSource 관계만
    if food_source_view:
        allowed_types = {"Phytochemical", "FoodSource"}
        fs_nodes = [n for n in data.get("nodes", []) if n.get("type") in allowed_types]
        fs_ids = {n["id"] for n in fs_nodes}
        fs_edges = [e for e in data.get("edges", [])
                    if e["source"] in fs_ids and e["target"] in fs_ids]
        data = {"nodes": fs_nodes, "edges": fs_edges}

    # 노드 타입 필터
    if node_type:
        filtered_nodes = [n for n in data.get("nodes", []) if n.get("type") == node_type]
        node_ids = {n["id"] for n in filtered_nodes}
        filtered_edges = [e for e in data.get("edges", [])
                          if e["source"] in node_ids and e["target"] in node_ids]
        data = {"nodes": filtered_nodes, "edges": filtered_edges}

    # 관계 타입 필터 (엣지만, 연결된 노드는 유지)
    if relation_type:
        filtered_edges = [e for e in data.get("edges", []) if e.get("type") == relation_type]
        connected_ids = set()
        for e in filtered_edges:
            connected_ids.add(e["source"])
            connected_ids.add(e["target"])
        filtered_nodes = [n for n in data.get("nodes", []) if n["id"] in connected_ids]
        data = {"nodes": filtered_nodes, "edges": filtered_edges}

    return data


@app.get("/api/graph/node/{node_id}")
async def get_node_detail(node_id: str):
    """특정 노드 상세 정보 + 연결 노드 목록"""
    data = load_graph()
    node = next((n for n in data["nodes"] if n["id"] == node_id), None)
    if not node:
        raise HTTPException(404, "노드를 찾을 수 없습니다")

    # 연결된 엣지 + 이웃 노드
    out_edges = [e for e in data["edges"] if e["source"] == node_id]
    in_edges  = [e for e in data["edges"] if e["target"] == node_id]
    neighbor_ids = {e["target"] for e in out_edges} | {e["source"] for e in in_edges}
    neighbors = [n for n in data["nodes"] if n["id"] in neighbor_ids]

    return {
        "node": node,
        "out_edges": out_edges,
        "in_edges": in_edges,
        "neighbors": neighbors,
        "degree": len(neighbor_ids)
    }


@app.get("/api/graph/phytochemicals")
async def get_phytochemical_nodes():
    """그래프 내 파이토케미컬 노드 목록 (필터 드롭다운용)"""
    data = load_graph()
    phytos = [{"id": n["id"], "name": n["name"]}
              for n in data.get("nodes", []) if n.get("type") == "Phytochemical"]
    phytos.sort(key=lambda x: x["name"])
    return {"phytochemicals": phytos}


@app.get("/api/graph/relation-types")
async def get_relation_types():
    """그래프 내 관계 타입 목록 + 카운트"""
    data = load_graph()
    counts: dict = {}
    for e in data.get("edges", []):
        t = e.get("type", "UNKNOWN")
        counts[t] = counts.get(t, 0) + 1
    return {"relation_types": [{"type": k, "count": v} for k, v in sorted(counts.items())]}


@app.get("/api/papers/years")
async def get_paper_years():
    """논문 연도 목록 (필터 드롭다운용)"""
    papers = load_all_papers()
    years = sorted({p.get("year", "") for p in papers if p.get("year")}, reverse=True)
    return {"years": years}


@app.get("/api/papers/journals")
async def get_paper_journals():
    """논문 저널 목록 (상위 30개)"""
    papers = load_all_papers()
    counts: dict = {}
    for p in papers:
        j = p.get("journal", "").strip()
        if j:
            counts[j] = counts.get(j, 0) + 1
    sorted_j = sorted(counts.items(), key=lambda x: -x[1])[:30]
    return {"journals": [j for j, _ in sorted_j]}


@app.get("/api/graph/stats")
async def get_graph_statistics():
    """그래프 통계"""
    return get_graph_stats()


@app.delete("/api/graph")
async def clear_graph():
    """그래프 초기화"""
    save_graph({"nodes": [], "edges": [], "stats": {}})
    return {"message": "그래프가 초기화되었습니다"}


# ─── RAG 챗봇 ────────────────────────────────────────────────────────────────

@app.post("/api/rag/index")
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """문서 인덱싱 (비동기)"""
    task_id = f"index_{id(request)}"
    index_progress[task_id] = {"status": "started"}
    
    def update_progress(data):
        index_progress[task_id] = {**data, "status": "running"}
    
    def do_index():
        try:
            chatbot = get_or_create_chatbot(request.openai_api_key or "")
            count = chatbot.index_from_files(update_progress)
            index_progress[task_id] = {
                "status": "complete",
                "indexed_count": count,
                "total_indexed": chatbot.get_collection_stats()["indexed_documents"]
            }
        except Exception as e:
            index_progress[task_id] = {"status": "error", "message": str(e)}
    
    background_tasks.add_task(do_index)
    return {"task_id": task_id, "message": "인덱싱 시작됨"}


@app.get("/api/rag/index/status/{task_id}")
async def get_index_status(task_id: str):
    """인덱싱 상태"""
    if task_id not in index_progress:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return index_progress[task_id]


@app.post("/api/rag/chat")
async def chat(request: ChatRequest):
    """RAG 챗봇 질의응답"""
    try:
        chatbot = get_or_create_chatbot(request.openai_api_key)
        result = chatbot.chat(request.message, request.chat_history)
        return result
    except Exception as e:
        raise HTTPException(500, f"챗봇 오류: {str(e)}")


@app.get("/api/rag/stats")
async def get_rag_stats():
    """RAG 시스템 통계"""
    if chatbot_instance:
        return chatbot_instance.get_collection_stats()
    return {"indexed_documents": 0, "status": "not_initialized"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=False)
