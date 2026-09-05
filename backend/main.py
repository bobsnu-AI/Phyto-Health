"""
FastAPI 메인 애플리케이션 — 멀티유저 버전
각 사용자는 독립된 data/users/<user_id>/ 디렉토리와 RAG/Graph 인스턴스를 가짐
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Genspark LLM 프록시 설정 ─────────────────────────────────────────────────
_ENV_OPENAI_KEY  = os.environ.get("GSK_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")
_ENV_OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "")

def resolve_openai_key(ui_key: str = "") -> tuple[str, str]:
    key  = _ENV_OPENAI_KEY  or ui_key
    base = _ENV_OPENAI_BASE or ""
    return key, base

# 모듈 임포트
import sys
sys.path.insert(0, str(Path(__file__).parent))

from auth import (
    RegisterRequest, LoginRequest,
    register_user, login_user, get_user_info,
    get_current_user, get_user_data_dir,
    ROOT_DATA_DIR
)
from pubmed_crawler import (
    crawl_phytochemical_papers,
    PHYTOCHEMICAL_CATEGORIES, HEALTH_CONDITIONS
)
from graph_builder import (
    build_graph_from_papers, search_graph, get_graph_stats,
    save_graph, find_subgraph_for_entities,
    load_graph_from_dir, get_graph_stats_from_dir
)
from rag_chatbot import RAGChatbot

# ─── 앱 초기화 ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PhytoGraph Research Platform",
    description="파이토케미컬 건강 상관성 연구 플랫폼 — 멀티유저",
    version="2.0.0"
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

# ─── 사용자별 인스턴스 컨테이너 ──────────────────────────────────────────────
# { user_id: RAGChatbot }
_chatbot_cache: dict[str, RAGChatbot] = {}
# { user_id: { task_id: progress_dict } }
_crawl_progress: dict[str, dict] = {}
_build_progress: dict[str, dict] = {}
_index_progress: dict[str, dict] = {}


def get_chatbot(user_id: str) -> RAGChatbot:
    """사용자별 RAGChatbot 인스턴스 반환 (없으면 생성)"""
    if user_id not in _chatbot_cache:
        key, base = resolve_openai_key("")
        user_dir = get_user_data_dir(user_id)
        _chatbot_cache[user_id] = RAGChatbot(
            openai_api_key=key,
            base_url=base,
            data_dir=str(user_dir)
        )
    return _chatbot_cache[user_id]


def load_all_papers_for_user(user_id: str) -> list:
    """사용자별 abstracts 디렉토리에서 모든 논문 로드"""
    user_dir = get_user_data_dir(user_id)
    abstracts_dir = user_dir / "abstracts"
    papers = []
    for f in abstracts_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                papers.append(json.load(fp))
        except Exception:
            pass
    return papers


def get_stats_for_user(user_id: str) -> dict:
    """사용자별 논문 통계"""
    papers = load_all_papers_for_user(user_id)
    meta = [p for p in papers if p.get("is_meta_analysis")]
    sr   = [p for p in papers if p.get("is_systematic_review")]
    years = {}
    for p in papers:
        y = p.get("year", "")
        if y:
            years[y] = years.get(y, 0) + 1
    return {
        "total_papers": len(papers),
        "meta_analysis": len(meta),
        "systematic_review": len(sr),
        "papers_by_year": years
    }


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
    openai_api_key: Optional[str] = ""


class ChatRequest(BaseModel):
    message: str
    chat_history: Optional[list] = []
    openai_api_key: Optional[str] = ""


class IndexRequest(BaseModel):
    openai_api_key: Optional[str] = ""


# ─── 인증 API ────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
async def register(request: RegisterRequest):
    """회원가입"""
    return register_user(request.username, request.password, request.email or "")


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """로그인 → JWT 토큰 반환"""
    return login_user(request.username, request.password)


@app.get("/api/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    """현재 로그인된 사용자 정보"""
    return get_user_info(current_user["user_id"])


# ─── 공개 API ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Frontend not found</h1>")


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "PhytoGraph Research Platform v2.0 (Multi-user)"}


@app.get("/api/categories")
async def get_categories():
    return {
        "phytochemical_categories": PHYTOCHEMICAL_CATEGORIES,
        "health_conditions": HEALTH_CONDITIONS
    }


# ─── PubMed 크롤링 (사용자별) ─────────────────────────────────────────────────

@app.post("/api/crawl")
async def crawl_papers(
    request: CrawlRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """PubMed 논문 크롤링 — 사용자 자신의 abstracts 디렉토리에 저장"""
    user_id  = current_user["user_id"]
    task_id  = f"crawl_{request.phytochemical}_{id(request)}"
    user_dir = get_user_data_dir(user_id)
    abstracts_dir = user_dir / "abstracts"

    _crawl_progress.setdefault(user_id, {})
    _crawl_progress[user_id][task_id] = {"status": "started", "step": "initializing"}

    def update_progress(data):
        _crawl_progress[user_id][task_id] = {**data, "status": "running"}

    def do_crawl():
        try:
            result = crawl_phytochemical_papers(
                phytochemical=request.phytochemical,
                health_condition=request.health_condition,
                max_results=request.max_results,
                meta_analysis_only=request.meta_analysis_only,
                pub_type=request.pub_type,
                progress_callback=update_progress,
                abstracts_dir=abstracts_dir      # ← 사용자별 경로
            )
            if result["total"] == 0:
                _crawl_progress[user_id][task_id] = {
                    "status": "complete",
                    "result": {
                        "total": 0, "meta_count": 0, "new_count": 0,
                        "query": result.get("query", ""),
                        "papers": [],
                        "warning": "검색 결과가 없습니다."
                    }
                }
            else:
                _crawl_progress[user_id][task_id] = {
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
            _crawl_progress[user_id][task_id] = {"status": "error", "message": str(e)}

    background_tasks.add_task(do_crawl)
    return {"task_id": task_id, "message": "크롤링 시작됨"}


@app.get("/api/crawl/status/{task_id}")
async def get_crawl_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    user_tasks = _crawl_progress.get(user_id, {})
    if task_id not in user_tasks:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return user_tasks[task_id]


@app.get("/api/papers")
async def get_papers(
    page: int = 1,
    limit: int = 20,
    meta_only: bool = False,
    search: Optional[str] = None,
    year: Optional[str] = None,
    journal: Optional[str] = None,
    pub_type: Optional[str] = None,
    sort_by: str = "year",
    current_user: dict = Depends(get_current_user)
):
    """사용자별 논문 목록"""
    papers = load_all_papers_for_user(current_user["user_id"])

    if meta_only:
        papers = [p for p in papers if p.get("is_meta_analysis")]
    if search:
        sl = search.lower()
        papers = [p for p in papers if sl in p.get("title","").lower() or sl in p.get("abstract","").lower()]
    if year:
        papers = [p for p in papers if p.get("year") == year]
    if journal:
        jl = journal.lower()
        papers = [p for p in papers if jl in p.get("journal","").lower()]
    if pub_type == "meta":
        papers = [p for p in papers if p.get("is_meta_analysis")]
    elif pub_type == "sr":
        papers = [p for p in papers if p.get("is_systematic_review")]

    if sort_by == "year":
        papers.sort(key=lambda x: x.get("year","0"), reverse=True)
    elif sort_by == "title":
        papers.sort(key=lambda x: x.get("title","").lower())
    elif sort_by == "journal":
        papers.sort(key=lambda x: x.get("journal","").lower())

    total = len(papers)
    start = (page - 1) * limit
    return {
        "papers": papers[start:start+limit],
        "total": total,
        "page": page,
        "total_pages": (total + limit - 1) // limit
    }


@app.get("/api/papers/stats")
async def get_paper_stats(current_user: dict = Depends(get_current_user)):
    return get_stats_for_user(current_user["user_id"])


@app.get("/api/papers/years")
async def get_paper_years(current_user: dict = Depends(get_current_user)):
    papers = load_all_papers_for_user(current_user["user_id"])
    years = sorted({p.get("year","") for p in papers if p.get("year")}, reverse=True)
    return {"years": years}


@app.get("/api/papers/journals")
async def get_paper_journals(current_user: dict = Depends(get_current_user)):
    papers = load_all_papers_for_user(current_user["user_id"])
    counts: dict = {}
    for p in papers:
        j = p.get("journal","").strip()
        if j:
            counts[j] = counts.get(j,0) + 1
    sorted_j = sorted(counts.items(), key=lambda x: -x[1])[:30]
    return {"journals": [j for j,_ in sorted_j]}


@app.get("/api/papers/filter-options")
async def get_filter_options(current_user: dict = Depends(get_current_user)):
    """삭제 필터용: 사용자 논문의 고유 phytochemical / health_condition 목록 반환"""
    papers = load_all_papers_for_user(current_user["user_id"])
    phytos = sorted({p.get("phytochemical","") for p in papers if p.get("phytochemical")})
    healths = sorted({p.get("health_condition","") for p in papers if p.get("health_condition")})
    return {"phytochemicals": phytos, "health_conditions": healths}


@app.delete("/api/papers/{pmid}")
async def delete_paper(pmid: str, current_user: dict = Depends(get_current_user)):
    """단일 논문 삭제 (PMID 기준)"""
    user_id = current_user["user_id"]
    user_dir = get_user_data_dir(user_id)
    abstracts_dir = user_dir / "abstracts"

    deleted = 0
    for ext in [".json", ".txt"]:
        f = abstracts_dir / f"{pmid}{ext}"
        if f.exists():
            f.unlink()
            deleted += 1

    if deleted == 0:
        raise HTTPException(404, f"논문 {pmid}을(를) 찾을 수 없습니다.")

    # ChromaDB에서도 제거
    try:
        chatbot = get_chatbot(user_id)
        chatbot.collection.delete(ids=[pmid])
    except Exception:
        pass

    return {"deleted": 1, "pmid": pmid}


class BulkDeleteRequest(BaseModel):
    phytochemical: Optional[str] = None
    health_condition: Optional[str] = None
    pmids: Optional[list] = None  # 개별 복수 삭제용


@app.delete("/api/papers")
async def delete_papers_bulk(
    request: BulkDeleteRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    일괄 삭제:
    - pmids 목록 지정 → 해당 논문들만 삭제
    - phytochemical 지정 → 해당 소재 논문 전체 삭제
    - health_condition 지정 → 해당 기능성 논문 전체 삭제
    - 둘 다 지정 → AND 조건으로 삭제
    """
    user_id = current_user["user_id"]
    user_dir = get_user_data_dir(user_id)
    abstracts_dir = user_dir / "abstracts"

    papers = load_all_papers_for_user(user_id)
    deleted_pmids = []

    if request.pmids:
        # 직접 pmid 목록 삭제
        target_pmids = set(str(p) for p in request.pmids)
        targets = [p for p in papers if str(p.get("pmid","")) in target_pmids]
    else:
        # 필터 조건으로 삭제
        targets = papers
        if request.phytochemical:
            targets = [p for p in targets if p.get("phytochemical","").lower() == request.phytochemical.lower()]
        if request.health_condition:
            targets = [p for p in targets if p.get("health_condition","").lower() == request.health_condition.lower()]

    if not targets:
        return {"deleted": 0, "message": "삭제할 논문이 없습니다."}

    for paper in targets:
        pmid = str(paper.get("pmid",""))
        if not pmid:
            continue
        for ext in [".json", ".txt"]:
            f = abstracts_dir / f"{pmid}{ext}"
            if f.exists():
                f.unlink()
        deleted_pmids.append(pmid)

    # ChromaDB에서도 일괄 제거
    if deleted_pmids:
        try:
            chatbot = get_chatbot(user_id)
            chatbot.collection.delete(ids=deleted_pmids)
        except Exception:
            pass

    return {
        "deleted": len(deleted_pmids),
        "pmids": deleted_pmids,
        "message": f"{len(deleted_pmids)}편이 삭제되었습니다."
    }


# ─── Graph DB (사용자별) ──────────────────────────────────────────────────────

@app.post("/api/graph/build")
async def build_graph(
    request: BuildGraphRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """사용자 전용 그래프 구축"""
    user_id  = current_user["user_id"]
    task_id  = f"graph_{id(request)}"
    user_dir = get_user_data_dir(user_id)

    _build_progress.setdefault(user_id, {})
    _build_progress[user_id][task_id] = {"status": "started"}

    def update_progress(data):
        _build_progress[user_id][task_id] = {**data, "status": "running"}

    def do_build():
        from openai import OpenAI
        try:
            key, base = resolve_openai_key(request.openai_api_key or "")
            client = OpenAI(api_key=key, base_url=base or None)

            if request.use_all_papers:
                papers = load_all_papers_for_user(user_id)
            else:
                all_p = load_all_papers_for_user(user_id)
                papers = [p for p in all_p if p.get("pmid") in (request.pmids or [])]

            if not papers:
                _build_progress[user_id][task_id] = {
                    "status": "error",
                    "message": "처리할 논문이 없습니다."
                }
                return

            papers = papers[:30]
            result = build_graph_from_papers(papers, client, update_progress, graph_dir=user_dir/"graph")

            _build_progress[user_id][task_id] = {
                "status": "complete",
                "stats": result.get("stats", {}),
                "message": "그래프 구축 완료"
            }
        except Exception as e:
            _build_progress[user_id][task_id] = {"status": "error", "message": str(e)}

    background_tasks.add_task(do_build)
    return {"task_id": task_id, "message": "그래프 구축 시작됨"}


@app.get("/api/graph/build/status/{task_id}")
async def get_build_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    user_tasks = _build_progress.get(user_id, {})
    if task_id not in user_tasks:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return user_tasks[task_id]


@app.get("/api/graph")
async def get_graph(
    search: Optional[str] = None,
    node_type: Optional[str] = None,
    relation_type: Optional[str] = None,
    focus_node: Optional[str] = None,
    phytochemical: Optional[str] = None,
    food_source_view: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """사용자별 그래프 데이터"""
    user_dir = get_user_data_dir(current_user["user_id"])
    data = load_graph_from_dir(user_dir / "graph")

    if focus_node:
        connected_edges = [e for e in data["edges"] if e["source"]==focus_node or e["target"]==focus_node]
        neighbor_ids = {e["source"] for e in connected_edges} | {e["target"] for e in connected_edges}
        neighbor_ids.add(focus_node)
        data = {"nodes": [n for n in data["nodes"] if n["id"] in neighbor_ids], "edges": connected_edges}
    elif phytochemical:
        pl = phytochemical.lower()
        phyto_ids = {n["id"] for n in data["nodes"] if n.get("type")=="Phytochemical" and pl in n.get("name","").lower()}
        connected_edges = [e for e in data["edges"] if e["source"] in phyto_ids or e["target"] in phyto_ids]
        neighbor_ids = phyto_ids.copy()
        for e in connected_edges:
            neighbor_ids.add(e["source"]); neighbor_ids.add(e["target"])
        data = {"nodes": [n for n in data["nodes"] if n["id"] in neighbor_ids], "edges": connected_edges}
    elif search:
        sl = search.lower()
        matched = {n["id"] for n in data["nodes"] if sl in n.get("name","").lower()}
        edges = [e for e in data["edges"] if e["source"] in matched or e["target"] in matched]
        for e in edges:
            matched.add(e["source"]); matched.add(e["target"])
        data = {"nodes": [n for n in data["nodes"] if n["id"] in matched], "edges": edges}

    if food_source_view:
        allowed = {"Phytochemical","FoodSource"}
        fs_nodes = [n for n in data.get("nodes",[]) if n.get("type") in allowed]
        fs_ids = {n["id"] for n in fs_nodes}
        data = {"nodes": fs_nodes, "edges": [e for e in data.get("edges",[]) if e["source"] in fs_ids and e["target"] in fs_ids]}

    if node_type:
        fn = [n for n in data.get("nodes",[]) if n.get("type")==node_type]
        fids = {n["id"] for n in fn}
        data = {"nodes": fn, "edges": [e for e in data.get("edges",[]) if e["source"] in fids and e["target"] in fids]}

    if relation_type:
        fe = [e for e in data.get("edges",[]) if e.get("type")==relation_type]
        cids = {e["source"] for e in fe} | {e["target"] for e in fe}
        data = {"nodes": [n for n in data.get("nodes",[]) if n["id"] in cids], "edges": fe}

    return data


@app.get("/api/graph/node/{node_id}")
async def get_node_detail(
    node_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_dir = get_user_data_dir(current_user["user_id"])
    data = load_graph_from_dir(user_dir / "graph")
    node = next((n for n in data["nodes"] if n["id"]==node_id), None)
    if not node:
        raise HTTPException(404, "노드를 찾을 수 없습니다")
    out_edges = [e for e in data["edges"] if e["source"]==node_id]
    in_edges  = [e for e in data["edges"] if e["target"]==node_id]
    neighbor_ids = {e["target"] for e in out_edges} | {e["source"] for e in in_edges}
    return {
        "node": node,
        "out_edges": out_edges,
        "in_edges": in_edges,
        "neighbors": [n for n in data["nodes"] if n["id"] in neighbor_ids],
        "degree": len(neighbor_ids)
    }


@app.get("/api/graph/phytochemicals")
async def get_phytochemical_nodes(current_user: dict = Depends(get_current_user)):
    user_dir = get_user_data_dir(current_user["user_id"])
    data = load_graph_from_dir(user_dir / "graph")
    phytos = sorted(
        [{"id": n["id"], "name": n["name"]} for n in data.get("nodes",[]) if n.get("type")=="Phytochemical"],
        key=lambda x: x["name"]
    )
    return {"phytochemicals": phytos}


@app.get("/api/graph/relation-types")
async def get_relation_types(current_user: dict = Depends(get_current_user)):
    user_dir = get_user_data_dir(current_user["user_id"])
    data = load_graph_from_dir(user_dir / "graph")
    counts: dict = {}
    for e in data.get("edges",[]):
        t = e.get("type","UNKNOWN")
        counts[t] = counts.get(t,0) + 1
    return {"relation_types": [{"type": k, "count": v} for k,v in sorted(counts.items())]}


@app.get("/api/graph/stats")
async def get_graph_statistics(current_user: dict = Depends(get_current_user)):
    user_dir = get_user_data_dir(current_user["user_id"])
    return get_graph_stats_from_dir(user_dir / "graph")


@app.delete("/api/graph")
async def clear_graph(current_user: dict = Depends(get_current_user)):
    user_dir = get_user_data_dir(current_user["user_id"])
    graph_file = user_dir / "graph" / "phytochemical_graph.json"
    with open(graph_file, "w") as f:
        json.dump({"nodes":[],"edges":[],"stats":{}}, f)
    return {"message": "그래프가 초기화되었습니다"}


@app.get("/api/graph/path")
async def get_graph_path(
    entities: str = Query(...),
    current_user: dict = Depends(get_current_user)
):
    user_dir = get_user_data_dir(current_user["user_id"])
    entity_list = [e.strip() for e in entities.split(",") if e.strip()]
    result = find_subgraph_for_entities(entity_list, max_hops=2, graph_dir=user_dir/"graph")
    return result


# ─── RAG 챗봇 (사용자별) ──────────────────────────────────────────────────────

@app.post("/api/rag/index")
async def index_documents(
    request: IndexRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    task_id = f"index_{id(request)}"

    _index_progress.setdefault(user_id, {})
    _index_progress[user_id][task_id] = {"status": "started"}

    def update_progress(data):
        _index_progress[user_id][task_id] = {**data, "status": "running"}

    def do_index():
        try:
            chatbot = get_chatbot(user_id)
            count = chatbot.index_from_files(update_progress)
            _index_progress[user_id][task_id] = {
                "status": "complete",
                "indexed_count": count,
                "total_indexed": chatbot.get_collection_stats()["indexed_documents"]
            }
        except Exception as e:
            _index_progress[user_id][task_id] = {"status": "error", "message": str(e)}

    background_tasks.add_task(do_index)
    return {"task_id": task_id, "message": "인덱싱 시작됨"}


@app.get("/api/rag/index/status/{task_id}")
async def get_index_status(
    task_id: str,
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["user_id"]
    user_tasks = _index_progress.get(user_id, {})
    if task_id not in user_tasks:
        raise HTTPException(404, "작업을 찾을 수 없습니다")
    return user_tasks[task_id]


@app.post("/api/rag/chat")
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        chatbot = get_chatbot(current_user["user_id"])
        result  = chatbot.chat(request.message, request.chat_history)
        return result
    except Exception as e:
        raise HTTPException(500, f"챗봇 오류: {str(e)}")


@app.get("/api/rag/stats")
async def get_rag_stats(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    if user_id in _chatbot_cache:
        return _chatbot_cache[user_id].get_collection_stats()
    return {"indexed_documents": 0, "status": "not_initialized"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=False)
