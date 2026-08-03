"""
RAG (Retrieval Augmented Generation) 챗봇 모듈
ChromaDB 벡터스토어 + OpenAI GPT 기반
파이토케미컬 건강 상관성 전문 Q&A
"""

import json
import os
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
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
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
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
        """OpenAI 임베딩 생성"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text[:8000]
        )
        return response.data[0].embedding
    
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
            embeddings = []
            metadatas = []
            ids = []
            
            for j, paper in enumerate(batch):
                if not paper.get('abstract'):
                    continue
                
                # 인덱싱 텍스트: 제목 + 초록
                doc_text = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}"
                
                try:
                    embedding = self.get_embedding(doc_text)
                except Exception as e:
                    print(f"임베딩 오류: {e}")
                    continue
                
                documents.append(doc_text[:2000])
                embeddings.append(embedding)
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
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
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
            query_embedding = self.get_embedding(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
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
        """RAG 기반 Q&A 응답 생성"""
        if chat_history is None:
            chat_history = []
        
        # 관련 논문 검색
        retrieved_papers = self.retrieve(message, n_results=5)
        
        # 컨텍스트 구성
        context_parts = []
        if retrieved_papers:
            context_parts.append("=== 관련 논문 ===")
            for i, paper in enumerate(retrieved_papers, 1):
                meta_label = "🔬 [메타분석]" if paper['is_meta_analysis'] else "📄"
                context_parts.append(
                    f"{i}. {meta_label} {paper['title']}\n"
                    f"   저널: {paper['journal']} ({paper['year']}) | PMID: {paper['pmid']}\n"
                    f"   내용: {paper['snippet'][:300]}..."
                )
        
        context = "\n".join(context_parts) if context_parts else "관련 논문을 찾을 수 없습니다."
        
        # 메시지 구성
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # 이전 대화 기록 추가
        for hist in chat_history[-4:]:  # 최근 4개만
            messages.append({"role": hist["role"], "content": hist["content"]})
        
        # 현재 질문 + 컨텍스트
        user_message = f"""질문: {message}

{context}

위 논문들을 참고하여 답변해주세요. 답변 마지막에 인용한 논문의 PMID를 명시해주세요."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=1500
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": retrieved_papers,
                "total_docs_indexed": self.collection.count()
            }
        
        except Exception as e:
            return {
                "answer": f"오류가 발생했습니다: {str(e)}",
                "sources": [],
                "total_docs_indexed": self.collection.count()
            }
    
    def get_collection_stats(self) -> dict:
        """벡터DB 통계"""
        try:
            count = self.collection.count()
            return {"indexed_documents": count, "status": "active"}
        except Exception:
            return {"indexed_documents": 0, "status": "error"}
