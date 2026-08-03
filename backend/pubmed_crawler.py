"""
PubMed 크롤링 모듈 - 파이토케미컬 & 건강 상관성 논문 수집
실시간 크롤링, 메타분석 논문 필터링 지원
"""

import os
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import math
import time
import re
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ABSTRACTS_DIR = DATA_DIR / "abstracts"
ABSTRACTS_DIR.mkdir(parents=True, exist_ok=True)

# PubMed API 기본 URL
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 파이토케미컬 기본 검색 카테고리
PHYTOCHEMICAL_CATEGORIES = {
    "Polyphenols": ["resveratrol", "quercetin", "curcumin", "catechins", "anthocyanins", "flavonoids"],
    "Carotenoids": ["lycopene", "beta-carotene", "lutein", "zeaxanthin", "astaxanthin"],
    "Glucosinolates": ["sulforaphane", "indole-3-carbinol", "sinigrin"],
    "Terpenoids": ["limonene", "perillyl alcohol", "betulinic acid", "ursolic acid"],
    "Alkaloids": ["berberine", "capsaicin", "piperine", "caffeine"],
    "Organosulfur": ["allicin", "diallyl sulfide", "S-allyl cysteine"],
    "Isoflavones": ["genistein", "daidzein", "formononetin"],
    "Stilbenes": ["resveratrol", "pterostilbene"],
}

HEALTH_CONDITIONS = [
    "cancer", "cardiovascular disease", "diabetes", "obesity",
    "inflammation", "oxidative stress", "gut microbiota",
    "cognitive function", "aging", "immune function",
    "hypertension", "liver disease", "metabolic syndrome"
]


def fetch_with_retry(url: str, params: dict = None, max_retries: int = 3) -> bytes:
    """재시도 로직이 포함된 HTTP GET 요청"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.content
            elif response.status_code == 429:  # Rate limit
                time.sleep(2 ** attempt)
            else:
                time.sleep(1)
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise e
    return None


def search_pubmed_ids(query: str, max_results: int = 200, pub_type: str = None) -> list:
    """
    PubMed 검색으로 PMID 목록 반환
    pub_type: 'Meta-Analysis', 'Systematic Review', 'Review', 'Clinical Trial' 등
    """
    search_query = query
    if pub_type:
        search_query = f"({query}) AND {pub_type}[pt]"

    params = {
        "db": "pubmed",
        "term": search_query,
        "retmax": max_results,
        "retmode": "xml",
        "sort": "relevance"
    }
    
    content = fetch_with_retry(ESEARCH_URL, params)
    if not content:
        return []
    
    root = ET.fromstring(content)
    ids = [id_elem.text for id_elem in root.findall('.//Id')]
    return ids


def fetch_paper_details(pmids: list) -> list:
    """PMID 목록으로 논문 상세 정보 수집 (100개씩 배치 처리)"""
    all_papers = []
    
    # 100개씩 배치 처리
    batch_size = 100
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract"
        }
        
        content = fetch_with_retry(EFETCH_URL, params)
        if not content:
            continue
        
        try:
            root = ET.fromstring(content)
            articles = root.findall('PubmedArticle')
            
            for article in articles:
                paper = parse_article(article)
                if paper:
                    all_papers.append(paper)
        except ET.ParseError:
            print(f"XML 파싱 오류 (배치 {i})")
        
        time.sleep(0.34)  # NCBI Rate limit 준수 (3 req/sec)
    
    return all_papers


def parse_article(article_elem) -> dict:
    """XML PubmedArticle 요소를 딕셔너리로 파싱"""
    try:
        pmid_elem = article_elem.find('.//PMID')
        if pmid_elem is None:
            return None
        pmid = pmid_elem.text

        # 제목
        title_elem = article_elem.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else ""
        if title:
            title = re.sub(r'<[^>]+>', '', title)

        # 초록
        abstract_elem = article_elem.find('.//Abstract')
        abstract = ""
        if abstract_elem is not None:
            abstract = ' '.join(t for t in abstract_elem.itertext()).strip()
            abstract = re.sub(r'<[^>]+>', '', abstract)

        # 저널
        journal_elem = article_elem.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else ""

        # 출판연도
        year_elem = article_elem.find('.//PubDate/Year')
        if year_elem is None:
            year_elem = article_elem.find('.//PubDate/MedlineDate')
        year = year_elem.text[:4] if year_elem is not None and year_elem.text else ""

        # 저자
        authors = []
        for author in article_elem.findall('.//Author'):
            lastname = author.find('LastName')
            firstname = author.find('ForeName')
            if lastname is not None:
                name = lastname.text
                if firstname is not None:
                    name = f"{firstname.text} {name}"
                authors.append(name)

        # MeSH 용어
        mesh_terms = []
        for mesh in article_elem.findall('.//MeshHeading/DescriptorName'):
            mesh_terms.append(mesh.text)

        # 논문 타입
        pub_types = []
        for pt in article_elem.findall('.//PublicationType'):
            pub_types.append(pt.text)

        # DOI
        doi = ""
        for id_elem in article_elem.findall('.//ArticleId'):
            if id_elem.get('IdType') == 'doi':
                doi = id_elem.text
                break

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": authors,
            "mesh_terms": mesh_terms,
            "pub_types": pub_types,
            "doi": doi,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "is_meta_analysis": any("Meta-Analysis" in pt for pt in pub_types),
            "is_systematic_review": any("Systematic Review" in pt for pt in pub_types),
            "crawled_at": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"파싱 오류: {e}")
        return None


def build_phytochemical_query(phytochemical: str, health_condition: str = None) -> str:
    """파이토케미컬 + 건강 조건 검색 쿼리 빌드"""
    base_query = f'"{phytochemical}"[Title/Abstract]'
    
    if health_condition:
        return f'({base_query}) AND ("{health_condition}"[Title/Abstract])'
    
    # 건강 조건 없으면 모든 건강 관련 검색
    health_query = " OR ".join([f'"{h}"[Title/Abstract]' for h in HEALTH_CONDITIONS[:5]])
    return f'({base_query}) AND ({health_query})'


def crawl_phytochemical_papers(
    phytochemical: str,
    health_condition: str = None,
    max_results: int = 50,
    meta_analysis_only: bool = False,
    pub_type: str = None,
    progress_callback=None
) -> dict:
    """
    특정 파이토케미컬 관련 논문 크롤링 메인 함수
    Returns: {papers: [], total: int, meta_count: int, new_count: int}
    """
    query = build_phytochemical_query(phytochemical, health_condition)
    
    if meta_analysis_only:
        pub_type = "Meta-Analysis"
    
    if progress_callback:
        progress_callback({"step": "searching", "message": f"PubMed 검색 중: {phytochemical}"})
    
    pmids = search_pubmed_ids(query, max_results, pub_type)
    
    if not pmids:
        return {"papers": [], "total": 0, "meta_count": 0, "new_count": 0}
    
    # 이미 저장된 PMID 확인
    existing_pmids = get_existing_pmids()
    new_pmids = [p for p in pmids if p not in existing_pmids]
    
    if progress_callback:
        progress_callback({
            "step": "fetching",
            "message": f"논문 데이터 수집 중: {len(pmids)}개 발견, {len(new_pmids)}개 신규"
        })
    
    papers = fetch_paper_details(pmids)
    
    # 로컬 저장
    saved_count = save_papers(papers)
    
    meta_count = sum(1 for p in papers if p.get("is_meta_analysis"))
    
    if progress_callback:
        progress_callback({
            "step": "complete",
            "message": f"완료: {len(papers)}개 수집, {meta_count}개 메타분석, {saved_count}개 저장"
        })
    
    return {
        "papers": papers,
        "total": len(papers),
        "meta_count": meta_count,
        "new_count": len(new_pmids),
        "query": query
    }


def get_existing_pmids() -> set:
    """이미 저장된 PMID 세트 반환"""
    pmids = set()
    
    # abstracts 폴더에서
    for f in ABSTRACTS_DIR.glob("*.json"):
        pmids.add(f.stem)
    
    # 메타데이터 CSV에서
    meta_file = DATA_DIR / "papers_metadata.csv"
    if meta_file.exists():
        try:
            df = pd.read_csv(meta_file)
            pmids.update(df['pmid'].astype(str).values)
        except Exception:
            pass
    
    return pmids


def save_papers(papers: list) -> int:
    """논문 데이터를 JSON으로 저장 + 메타데이터 CSV 업데이트"""
    saved = 0
    
    for paper in papers:
        if not paper:
            continue
        pmid = paper['pmid']
        
        # JSON 저장 (전체 데이터)
        json_path = ABSTRACTS_DIR / f"{pmid}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(paper, f, ensure_ascii=False, indent=2)
        
        # 텍스트 파일도 저장 (호환성)
        txt_path = ABSTRACTS_DIR / f"{pmid}.txt"
        content = f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        saved += 1
    
    # 메타데이터 CSV 업데이트
    update_metadata_csv(papers)
    
    return saved


def update_metadata_csv(papers: list):
    """메타데이터 CSV 업데이트"""
    meta_file = DATA_DIR / "papers_metadata.csv"
    
    new_rows = []
    for p in papers:
        if not p:
            continue
        new_rows.append({
            "pmid": p["pmid"],
            "title": p["title"],
            "journal": p["journal"],
            "year": p["year"],
            "is_meta_analysis": p["is_meta_analysis"],
            "is_systematic_review": p["is_systematic_review"],
            "mesh_terms": "|".join(p["mesh_terms"]),
            "url": p["url"],
            "crawled_at": p["crawled_at"]
        })
    
    if not new_rows:
        return
    
    new_df = pd.DataFrame(new_rows)
    
    if meta_file.exists():
        existing_df = pd.read_csv(meta_file)
        combined = pd.concat([existing_df, new_df]).drop_duplicates(subset='pmid')
        combined.to_csv(meta_file, index=False)
    else:
        new_df.to_csv(meta_file, index=False)


def load_all_papers() -> list:
    """저장된 모든 논문 로드"""
    papers = []
    for json_file in ABSTRACTS_DIR.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                papers.append(json.load(f))
        except Exception:
            pass
    return papers


def get_stats() -> dict:
    """크롤링 통계 반환"""
    papers = load_all_papers()
    
    total = len(papers)
    meta_count = sum(1 for p in papers if p.get("is_meta_analysis"))
    sr_count = sum(1 for p in papers if p.get("is_systematic_review"))
    
    years = [p.get("year", "") for p in papers if p.get("year")]
    year_dist = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1
    
    return {
        "total_papers": total,
        "meta_analysis_count": meta_count,
        "systematic_review_count": sr_count,
        "year_distribution": dict(sorted(year_dist.items(), reverse=True)[:10])
    }
