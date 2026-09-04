"""
RAG (Retrieval Augmented Generation) 챗봇 모듈
ChromaDB 벡터스토어 + OpenAI GPT 기반
파이토케미컬 건강 상관성 전문 Q&A
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from graph_builder import find_subgraph_for_entities, subgraph_to_context_text

# DATA_DIR: 환경변수 DATA_DIR → Railway Volume(/data) → 앱 내부 data/ 순으로 우선 사용
_env_data = os.environ.get("DATA_DIR", "")
DATA_DIR = Path(_env_data) if _env_data else Path(__file__).parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
ABSTRACTS_DIR = DATA_DIR / "abstracts"

COLLECTION_NAME = "phytochemical_papers"

SYSTEM_PROMPT = """당신은 파이토케미컬(식물 생리활성물질)과 건강 상관성 전문 AI 연구 어시스턴트입니다.
PubMed 논문 데이터베이스를 기반으로 정확하고 신뢰할 수 있는 정보를 제공합니다.

응답 원칙:
1. 제공된 논문 컨텍스트에 기반하여 답변하세요
2. 과학적 근거를 명확히 제시하세요 (PMID, 저자, 연도)
3. 메타분석 및 체계적 리뷰 논문을 우선적으로 인용하세요
4. 한국어로 답변하되, 전문용어는 영문 병기하세요
5. 불확실한 경우 "논문 근거가 부족합니다"라고 명시하세요
6. 의학적 조언보다는 연구 정보 제공에 집중하세요

응답 형식:
- 핵심 결론을 먼저 제시
- 관련 연구 근거 나열
- 메커니즘 설명 (가능한 경우)
- 인용 논문 목록
"""


class RAGChatbot:
    def __init__(self, openai_api_key: str, base_url: str = ""):
        # Genspark 프록시 우선, 없으면 공식 OpenAI
        _base = base_url or os.environ.get("OPENAI_BASE_URL", "") or None
        # API 키 우선순위: GSK_TOKEN > OPENAI_API_KEY > UI 입력값
        _key  = (os.environ.get("GSK_TOKEN", "") or
                 os.environ.get("OPENAI_API_KEY", "") or
                 openai_api_key)
        self.client = OpenAI(api_key=_key, base_url=_base)
        print(f"[RAGChatbot] base_url={_base!r}, key_prefix={_key[:8]}...")
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )
        self._init_collection()
    
    def _init_collection(self):
        """ChromaDB 컬렉션 초기화"""
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            print(f"✅ 기존 컬렉션 로드: {self.collection.count()}개 문서")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ 새 컬렉션 생성")
    
    def get_embedding(self, text: str) -> list:
        """ChromaDB 내장 임베딩 사용 (sentence-transformers all-MiniLM-L6-v2)
        OpenAI 임베딩 API 불필요 — 로컬에서 직접 실행
        """
        # ChromaDB query_texts 방식을 사용하므로 이 함수는 직접 호출 안 됨
        # retrieve()에서 query_texts=[query] 로 자동 임베딩
        raise NotImplementedError("get_embedding은 사용하지 않습니다. retrieve()를 직접 호출하세요.")
    
    def index_papers(self, papers: list, progress_callback=None) -> int:
        """논문 목록을 벡터DB에 인덱싱"""
        if not papers:
            return 0
        
        # 이미 인덱싱된 ID 확인
        existing_ids = set()
        try:
            existing = self.collection.get()
            existing_ids = set(existing.get('ids', []))
        except Exception:
            pass
        
        new_papers = [p for p in papers if p and p.get('pmid') and 
                      str(p['pmid']) not in existing_ids]
        
        if not new_papers:
            return 0
        
        batch_size = 50
        indexed_count = 0
        
        for i in range(0, len(new_papers), batch_size):
            batch = new_papers[i:i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for j, paper in enumerate(batch):
                if not paper.get('abstract'):
                    continue
                
                # 인덱싱 텍스트: 제목 + 초록
                doc_text = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}"
                
                documents.append(doc_text[:2000])
                metadatas.append({
                    "pmid": str(paper.get('pmid', '')),
                    "title": paper.get('title', '')[:500],
                    "journal": paper.get('journal', '')[:200],
                    "year": str(paper.get('year', '')),
                    "is_meta_analysis": str(paper.get('is_meta_analysis', False)),
                    "url": paper.get('url', ''),
                    "authors": ', '.join(paper.get('authors', [])[:3])
                })
                ids.append(str(paper['pmid']))
                indexed_count += 1
                
                if progress_callback:
                    progress_callback({
                        "step": "indexing",
                        "current": i + j + 1,
                        "total": len(new_papers),
                        "message": f"인덱싱 중: {paper.get('title', '')[:40]}..."
                    })
            
            if documents:
                # ✅ embeddings 파라미터 제거 — ChromaDB가 자체 임베딩 생성
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
        
        return indexed_count
    
    def index_from_files(self, progress_callback=None) -> int:
        """저장된 JSON 파일들을 인덱싱"""
        papers = []
        for json_file in ABSTRACTS_DIR.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    papers.append(json.load(f))
            except Exception:
                pass
        
        return self.index_papers(papers, progress_callback)
    
    def retrieve(self, query: str, n_results: int = 5) -> list:
        """관련 논문 검색"""
        if self.collection.count() == 0:
            return []
        
        try:
            # ✅ query_texts 사용 — ChromaDB 내장 임베딩으로 자동 변환
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            retrieved = []
            for i in range(len(results['ids'][0])):
                retrieved.append({
                    "pmid": results['metadatas'][0][i].get('pmid', ''),
                    "title": results['metadatas'][0][i].get('title', ''),
                    "journal": results['metadatas'][0][i].get('journal', ''),
                    "year": results['metadatas'][0][i].get('year', ''),
                    "is_meta_analysis": results['metadatas'][0][i].get('is_meta_analysis', 'False') == 'True',
                    "url": results['metadatas'][0][i].get('url', ''),
                    "authors": results['metadatas'][0][i].get('authors', ''),
                    "snippet": results['documents'][0][i][:500],
                    "relevance_score": 1 - results['distances'][0][i]
                })
            
            return retrieved
        except Exception as e:
            print(f"검색 오류: {e}")
            return []
    
    def chat(self, message: str, chat_history: list = None) -> dict:
        """
        진짜 GraphRAG Q&A — 벡터 검색 + 지식 그래프 관계를 함께 GPT에 전달
        
        흐름:
        1. 질문에서 엔티티 키워드 추출 (사전 기반)
        2. ChromaDB 벡터 검색 → 관련 논문 5편
        3. 지식 그래프에서 엔티티 관련 서브그래프 탐색 (GPT 호출 전!)
        4. [논문 컨텍스트 + 그래프 관계] 를 GPT에 함께 전달
        5. GPT가 두 소스를 모두 보고 답변 생성
        """
        if chat_history is None:
            chat_history = []

        # ── ① 질문에서 엔티티 키워드 추출 (GPT 호출 전) ──────────────────────
        query_entities = self._extract_entities_from_query(message)
        print(f"[GraphRAG] 질문 엔티티: {query_entities}")

        # ── ② ChromaDB 벡터 검색 ──────────────────────────────────────────────
        retrieved_papers = self.retrieve(message, n_results=5)

        # 논문 제목에서도 추가 엔티티 보강
        for paper in retrieved_papers[:3]:
            title_entities = self._extract_entities_from_query(paper.get("title", ""))
            query_entities = list(set(query_entities + title_entities))[:25]

        # ── ③ 지식 그래프 선제 조회 (GPT 호출 전!) ───────────────────────────
        graph_subgraph = {"nodes": [], "edges": [], "seed_ids": [], "paths": []}
        graph_context_text = ""
        if query_entities:
            graph_subgraph = find_subgraph_for_entities(query_entities, max_hops=2)
            if graph_subgraph.get("edges"):
                graph_context_text = subgraph_to_context_text(graph_subgraph, max_relations=25)
                print(f"[GraphRAG] 그래프 컨텍스트: {len(graph_subgraph['edges'])}개 관계 추출")

        # ── ④ 컨텍스트 구성 (논문 + 그래프 관계 통합) ────────────────────────
        context_parts = []

        # 4-A: 논문 컨텍스트
        if retrieved_papers:
            context_parts.append("=== 관련 논문 (벡터 유사도 검색) ===")
            for i, paper in enumerate(retrieved_papers, 1):
                meta_label = "🔬 [메타분석]" if paper['is_meta_analysis'] else "📄"
                context_parts.append(
                    f"{i}. {meta_label} {paper['title']}\n"
                    f"   저널: {paper['journal']} ({paper['year']}) | PMID: {paper['pmid']}\n"
                    f"   내용: {paper['snippet'][:300]}..."
                )

        # 4-B: 그래프 관계 컨텍스트 (핵심 추가!)
        if graph_context_text:
            context_parts.append("\n" + graph_context_text)

        context = "\n".join(context_parts) if context_parts else "관련 논문 및 그래프 데이터를 찾을 수 없습니다."

        # ── ⑤ GPT 메시지 구성 ─────────────────────────────────────────────────
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for hist in chat_history[-4:]:
            messages.append({"role": hist["role"], "content": hist["content"]})

        # 그래프 정보 유무에 따라 프롬프트 차별화
        if graph_context_text:
            user_message = f"""질문: {message}

{context}

위 정보를 참고하여 답변해주세요:
1. 지식 그래프의 관계 경로(→)를 활용해 메커니즘을 설명하세요
2. 논문 근거와 그래프 관계를 연결하여 설명하세요
3. 답변 마지막에 인용한 논문의 PMID를 명시해주세요"""
        else:
            user_message = f"""질문: {message}

{context}

위 논문들을 참고하여 답변해주세요. 답변 마지막에 인용한 논문의 PMID를 명시해주세요."""

        messages.append({"role": "user", "content": user_message})

        # ── ⑥ GPT 호출 ───────────────────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                temperature=0.3
            )

            answer = response.choices[0].message.content

            # 답변 후 그래프 경로 보강 (답변 텍스트로 추가 엔티티 보강)
            if not graph_subgraph.get("edges"):
                graph_subgraph = self._extract_graph_paths(message, answer, retrieved_papers)

            return {
                "answer": answer,
                "sources": retrieved_papers,
                "total_docs_indexed": self.collection.count(),
                "graph_paths": graph_subgraph,
                "graph_used": bool(graph_context_text),          # 그래프가 답변에 실제 사용됐는지
                "graph_relations_count": len(graph_subgraph.get("edges", []))
            }

        except Exception as e:
            return {
                "answer": f"오류가 발생했습니다: {str(e)}",
                "sources": [],
                "total_docs_indexed": self.collection.count(),
                "graph_paths": {"nodes": [], "edges": [], "seed_ids": [], "paths": []},
                "graph_used": False,
                "graph_relations_count": 0
            }

    def _extract_entities_from_query(self, text: str) -> list:
        """
        질문/제목 텍스트에서 파이토케미컬·건강조건 엔티티 키워드 추출
        GPT 호출 없이 사전(dictionary) 기반으로 빠르게 처리
        """
        PHYTO_KEYWORDS = [
            "curcumin", "quercetin", "resveratrol", "catechin", "epicatechin",
            "anthocyanin", "lycopene", "kaempferol", "apigenin", "luteolin",
            "sulforaphane", "allicin", "genistein", "daidzein", "berberine",
            "ellagic acid", "chlorogenic acid", "ferulic acid", "caffeic acid",
            "epigallocatechin", "egcg", "capsaicin", "naringenin", "hesperidin",
            "piperine", "gingerol", "curcuminoid", "polyphenol", "flavonoid",
            "isoflavone", "stilbene", "terpenoid", "carotenoid", "glucosinolate",
            "beta-carotene", "zeaxanthin", "lutein", "pterostilbene", "fisetin",
            "spermidine", "urolithin", "indole", "terpene", "saponin",
        ]
        HEALTH_KEYWORDS = [
            "cancer", "diabetes", "inflammation", "obesity", "cardiovascular",
            "hypertension", "alzheimer", "parkinson", "arthritis", "depression",
            "anxiety", "oxidative stress", "metabolic", "insulin", "cholesterol",
            "tumor", "apoptosis", "nf-kb", "antioxidant", "anti-inflammatory",
            "gut microbiome", "microbiota", "autophagy", "senescence", "aging",
            "liver", "kidney", "brain", "heart", "lung", "colon", "breast",
            "nrf2", "sirt1", "ampk", "mtor", "mapk", "vegf", "il-6", "tnf",
            "reactive oxygen", "mitochondria", "endothelial", "adipose",
        ]
        # 한국어 → 영어 간단 매핑
        KO_MAP = {
            "커큐민": "curcumin", "케르세틴": "quercetin", "레스베라트롤": "resveratrol",
            "안토시아닌": "anthocyanin", "설포라판": "sulforaphane", "베르베린": "berberine",
            "염증": "inflammation", "당뇨": "diabetes", "암": "cancer",
            "심혈관": "cardiovascular", "고혈압": "hypertension", "항산화": "antioxidant",
            "비만": "obesity", "알츠하이머": "alzheimer", "파킨슨": "parkinson",
            "장내미생물": "gut microbiome", "자가포식": "autophagy", "노화": "aging",
        }

        text_lower = text.lower()
        found = []

        # 영어 키워드 매칭
        for kw in PHYTO_KEYWORDS + HEALTH_KEYWORDS:
            if kw in text_lower:
                found.append(kw)

        # 한국어 → 영어 변환
        for ko, en in KO_MAP.items():
            if ko in text:
                found.append(en)

        # 영문 단어 추출 (4글자 이상, 불용어 제외)
        STOPWORDS = {
            "this", "that", "with", "from", "have", "been", "were", "also",
            "these", "their", "which", "such", "more", "than", "after",
            "study", "paper", "result", "effect", "effects", "showed",
            "found", "using", "used", "based", "high", "significant",
            "increase", "decrease", "level", "role", "type", "line",
        }
        words = re.findall(r'[A-Za-z][A-Za-z\-]{3,}', text)
        for w in words:
            wl = w.lower()
            if wl not in STOPWORDS and len(wl) >= 4:
                found.append(wl)

        # 중복 제거 + 최대 20개
        seen = set()
        result = []
        for f in found:
            if f not in seen:
                seen.add(f)
                result.append(f)
        return result[:20]
    
    def _extract_graph_paths(self, question: str, answer: str, papers: list) -> dict:
        """
        GraphRAG 핵심 — 질문+답변+논문제목에서 엔티티 추출 → 서브그래프 반환
        
        추출 전략:
        1. 논문 제목의 주요 단어 (파이토케미컬/건강조건 키워드)
        2. 답변에서 파이토케미컬/질환 전문용어 패턴 추출
        3. find_subgraph_for_entities() 호출
        """
        try:
            # ── 엔티티 후보 수집 ─────────────────────────────────────────────
            candidates = set()

            # 1) 질문에서 직접 단어 추출 (3글자 이상 영문/한글 복합어)
            question_words = re.findall(r'[A-Za-z][A-Za-z\-]{2,}', question)
            candidates.update(w.lower() for w in question_words if len(w) >= 4)

            # 2) 논문 제목에서 의미있는 명사구 추출
            for paper in papers[:5]:
                title = paper.get("title", "")
                # 영문 명사구: 대문자 시작 단어 또는 연속 단어
                title_words = re.findall(r'[A-Za-z][A-Za-z\-]{3,}', title)
                candidates.update(w.lower() for w in title_words)

            # 3) 답변에서 괄호 안 영문 전문용어 추출 (한국어 병기 패턴)
            answer_terms = re.findall(r'[A-Za-z][A-Za-z\-\s]{2,}(?=\s*[\)\]】])', answer)
            candidates.update(t.strip().lower() for t in answer_terms if len(t.strip()) >= 4)

            # 4) 파이토케미컬/건강조건 전형 키워드 우선 추출
            PHYTO_KEYWORDS = [
                "curcumin", "quercetin", "resveratrol", "catechin", "epicatechin",
                "anthocyanin", "lycopene", "kaempferol", "apigenin", "luteolin",
                "sulforaphane", "allicin", "genistein", "daidzein", "berberine",
                "ellagic acid", "chlorogenic acid", "ferulic acid", "caffeic acid",
                "epigallocatechin", "egcg", "capsaicin", "naringenin", "hesperidin",
                "piperine", "gingerol", "curcuminoid", "polyphenol", "flavonoid",
                "isoflavone", "stilbene", "terpenoid", "carotenoid", "glucosinolate"
            ]
            HEALTH_KEYWORDS = [
                "cancer", "diabetes", "inflammation", "obesity", "cardiovascular",
                "hypertension", "alzheimer", "parkinson", "arthritis", "depression",
                "anxiety", "oxidative stress", "metabolic", "insulin", "cholesterol",
                "tumor", "apoptosis", "nf-kb", "antioxidant", "anti-inflammatory"
            ]
            combined_text = (question + " " + answer + " " +
                             " ".join(p.get("title", "") for p in papers)).lower()
            for kw in PHYTO_KEYWORDS + HEALTH_KEYWORDS:
                if kw in combined_text:
                    candidates.add(kw)

            # ── 노이즈 제거 (불용어) ────────────────────────────────────────
            STOPWORDS = {
                "this", "that", "with", "from", "have", "been", "were", "also",
                "these", "their", "which", "such", "more", "than", "after",
                "through", "between", "against", "about", "into", "during",
                "study", "paper", "result", "analysis", "effect", "effects",
                "showed", "showed", "found", "using", "used", "based", "high",
                "significant", "significantly", "increase", "decrease", "level",
                "pmid", "journal", "abstract", "title", "author"
            }
            candidates = {c for c in candidates if c not in STOPWORDS and len(c) >= 4}

            if not candidates:
                return {"nodes": [], "edges": [], "seed_ids": [], "paths": []}

            # ── 그래프 경로 탐색 ────────────────────────────────────────────
            entity_list = sorted(candidates)[:20]  # 최대 20개 엔티티
            print(f"[GraphRAG] 엔티티 추출: {entity_list[:10]}...")
            subgraph = find_subgraph_for_entities(entity_list, max_hops=2)
            return subgraph

        except Exception as e:
            print(f"[GraphRAG] 경로 추출 오류: {e}")
            return {"nodes": [], "edges": [], "seed_ids": [], "paths": []}

    def get_collection_stats(self) -> dict:
        """벡터DB 통계"""
        try:
            count = self.collection.count()
            return {"indexed_documents": count, "status": "active"}
        except Exception:
            return {"indexed_documents": 0, "status": "error"}
