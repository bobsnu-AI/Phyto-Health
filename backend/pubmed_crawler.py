"""
PubMed 크롤링 모듈 - 파이토케미컬 & 건강 상관성 논문 수집
Europe PMC API (NCBI 차단 대안) + NCBI 폴백 지원
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

# DATA_DIR: 환경변수 DATA_DIR → Railway Volume(/data) → 앱 내부 data/ 순으로 우선 사용
_env_data = os.environ.get("DATA_DIR", "")
DATA_DIR = Path(_env_data) if _env_data else Path(__file__).parent.parent / "data"
ABSTRACTS_DIR = DATA_DIR / "abstracts"
ABSTRACTS_DIR.mkdir(parents=True, exist_ok=True)

# Europe PMC API (NCBI 대안 - 차단 없음)
EUROPEPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EUROPEPMC_ARTICLE = "https://www.ebi.ac.uk/europepmc/webservices/rest/article"

# NCBI (폴백용, API key 있을 때 우선 사용)
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

PHYTOCHEMICAL_CATEGORIES = {
    "Polyphenols":   ["resveratrol","quercetin","curcumin","catechins","anthocyanins","flavonoids"],
    "Carotenoids":   ["lycopene","beta-carotene","lutein","zeaxanthin","astaxanthin"],
    "Glucosinolates":["sulforaphane","indole-3-carbinol","sinigrin"],
    "Terpenoids":    ["limonene","perillyl alcohol","betulinic acid","ursolic acid"],
    "Alkaloids":     ["berberine","capsaicin","piperine","caffeine"],
    "Organosulfur":  ["allicin","diallyl sulfide","S-allyl cysteine"],
    "Isoflavones":   ["genistein","daidzein","formononetin"],
    "Stilbenes":     ["resveratrol","pterostilbene"],
}

HEALTH_CONDITIONS = [
    "cancer","cardiovascular disease","diabetes","obesity",
    "inflammation","oxidative stress","gut microbiota",
    "cognitive function","aging","immune function",
    "hypertension","liver disease","metabolic syndrome",
    "sarcopenia"
]

# 논문 타입 → Europe PMC pubType 매핑
PUBTYPE_MAP = {
    "Meta-Analysis":            "META-ANALYSIS",
    "Systematic Review":        "SYSTEMATIC_REVIEW",
    "Randomized Controlled Trial": "RANDOMIZED-CONTROLLED-TRIAL",
    "Review":                   "REVIEW",
    "Clinical Trial":           "CLINICAL-TRIAL",
}


# ──────────────────────────────────────────────────────────────────────────────
# Europe PMC 크롤링 (주 소스)
# ──────────────────────────────────────────────────────────────────────────────

def search_europepmc(query: str, max_results: int = 100,
                     pub_type: str = None, meta_only: bool = False) -> list:
    """
    Europe PMC 검색 → 논문 딕셔너리 리스트 반환 (resultType=lite)
    
    ⚠️ 핵심 주의사항:
    - SRC:MED, HAS_ABSTRACT:Y 필터를 쿼리에 추가하면 API가 {'version':'6.9'} 만 반환
    - 해결: 쿼리는 단순하게 유지하고, 결과를 클라이언트에서 pmid/abstract 유무로 필터링
    """
    # ✅ PUB_TYPE 필터: 단순 키워드 추가 대신 Europe PMC 공식 구문 사용
    # " meta-analysis" 키워드 방식 → PUB_TYPE:"meta-analysis" 로 정확히 511건 필터링
    epmc_query = query
    if meta_only or pub_type == "Meta-Analysis":
        epmc_query += ' AND PUB_TYPE:"meta-analysis"'
    elif pub_type == "Systematic Review":
        epmc_query += ' AND PUB_TYPE:"systematic review"'
    elif pub_type == "Review":
        epmc_query += ' AND PUB_TYPE:review'
    elif pub_type == "Randomized Controlled Trial":
        epmc_query += ' AND PUB_TYPE:"randomized controlled trial"'
    elif pub_type == "Clinical Trial":
        epmc_query += ' AND PUB_TYPE:"clinical trial"'

    print(f"[EPMC] 최종 쿼리: {epmc_query!r}")

    all_papers = []
    next_cursor = None  # 첫 요청은 cursorMark 없이

    while len(all_papers) < max_results:
        page_size = min(max_results - len(all_papers), 100)
        # 초록 enrichment 비용을 줄이기 위해 배치당 최대 25개 요청
        fetch_size = min(page_size, 25)
        # ✅ sort 파라미터 제거 — 'RELEVANCE' 값이 Europe PMC에서 {'version':'6.9'} 응답 유발
        params = {
            "query":      epmc_query,
            "format":     "json",
            "pageSize":   fetch_size,
            "resultType": "lite",   # lite: IP 차단 없음
        }
        if next_cursor:
            params["cursorMark"] = next_cursor

        try:
            resp = requests.get(EUROPEPMC_SEARCH, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[EPMC] 검색 오류: {e}")
            break

        # 응답 디버깅
        hit_count = data.get("hitCount", 0)
        print(f"[EPMC] hitCount={hit_count}, keys={list(data.keys())}")

        results = data.get("resultList", {}).get("result", [])
        if not results:
            print(f"[EPMC] 결과 없음 — 검색 종료")
            break

        # ① lite 파싱 (pmid 있는 것만 통과)
        lite_papers = []
        for r in results:
            paper = parse_europepmc_lite(r)
            if paper:  # parse_europepmc_lite가 pmid 없으면 None 반환
                lite_papers.append(paper)

        print(f"[EPMC] lite 파싱 완료: {len(lite_papers)}편 (pmid 있음)")

        # ② abstract 상세 조회 및 클라이언트 사이드 필터링
        if lite_papers:
            enriched = enrich_with_abstracts(lite_papers)
            # ✅ 클라이언트 사이드 필터링: abstract가 있는 논문만 보관
            valid = [p for p in enriched if p.get("abstract") and len(p["abstract"]) > 50]
            print(f"[EPMC] 초록 보강 후 유효 논문: {len(valid)}/{len(enriched)}편")
            all_papers.extend(valid)

        # ③ 다음 커서
        nc = data.get("nextCursorMark")
        if not nc or nc == next_cursor or len(all_papers) >= max_results:
            break
        next_cursor = nc
        time.sleep(0.3)

    print(f"[EPMC] 최종 수집: {len(all_papers[:max_results])}편")
    return all_papers[:max_results]


def parse_europepmc_lite(r: dict) -> dict:
    """Europe PMC lite 결과 파싱 (초록 제외) — pmid 없으면 None"""
    try:
        pmid = str(r.get("pmid", "") or "").strip()
        # ✅ pmid 없는 경우 → PubMed 소스 아님, 스킵
        if not pmid or not pmid.isdigit():
            return None
        
        pub_types_raw = r.get("pubTypeList", {})
        pub_types = []
        if pub_types_raw:
            pt = pub_types_raw.get("pubType", [])
            if isinstance(pt, str): pt = [pt]
            pub_types = [p.lower() for p in (pt or [])]

        return {
            "pmid":               pmid,
            "title":              re.sub(r'<[^>]+>', '', r.get("title", "") or "").strip().rstrip("."),
            "abstract":           "",   # 별도 조회 필요
            "journal":            r.get("journalTitle", "") or "",
            "year":               str(r.get("pubYear", "") or ""),
            "authors":            [],
            "mesh_terms":         [],
            "pub_types":          pub_types,
            "doi":                r.get("doi", "") or "",
            "url":                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "is_meta_analysis":   any("meta-analysis" in pt for pt in pub_types),
            "is_systematic_review": any("systematic" in pt for pt in pub_types),
            "crawled_at":         datetime.now().isoformat(),
            "source":             "europepmc"
        }
    except Exception as e:
        print(f"lite 파싱 오류: {e}")
        return None


def enrich_with_abstracts(papers: list) -> list:
    """
    Europe PMC article API로 초록 상세 조회
    
    전략:
    1. 개별 article API 시도 → abstractText 필드
    2. 실패 시 search API에서 abstract 직접 조회 (resultType=lite는 abstract 포함)
    """
    enriched = []
    for paper in papers:
        pmid = paper.get("pmid", "")
        if not pmid:
            enriched.append(paper)
            continue
        try:
            # 방법 1: article detail API
            resp = requests.get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/article/MED/{pmid}",
                params={"format": "json"},
                timeout=15
            )
            abstract = ""
            if resp.status_code == 200:
                data = resp.json()
                # result 키 또는 최상위에 직접 있을 수 있음
                article = data.get("result") or data
                if isinstance(article, dict):
                    raw = (article.get("abstractText") or
                           article.get("abstract") or
                           article.get("body") or "")
                    abstract = re.sub(r'<[^>]+>', '', str(raw)).strip()

                    # 저자 보강
                    author_list = article.get("authorList", {}).get("author", [])
                    if isinstance(author_list, dict): author_list = [author_list]
                    paper["authors"] = [
                        a.get("fullName") or f"{a.get('firstName','')} {a.get('lastName','')}".strip()
                        for a in (author_list or [])[:6] if a
                    ]

                    # MeSH 보강
                    mesh_terms = []
                    for mh in article.get("meshHeadingList", {}).get("meshHeading", []):
                        if isinstance(mh, dict):
                            mesh_terms.append(mh.get("descriptorName", ""))
                    paper["mesh_terms"] = [m for m in mesh_terms if m]

                    # ✅ pubType 보강 — lite 모드에서 누락된 pub_type 플래그 업데이트
                    # article API는 pubTypeList 딕셔너리가 아닌 pubType 세미콜론 구분 문자열로 반환
                    # 예: "meta-analysis; systematic review; journal article"
                    pub_type_str = article.get("pubType", "") or ""
                    if not pub_type_str:
                        # pubTypeList 구조도 폴백으로 시도
                        pt_raw = article.get("pubTypeList", {})
                        if isinstance(pt_raw, dict):
                            ptl = pt_raw.get("pubType", [])
                            if isinstance(ptl, str):
                                pub_type_str = ptl
                            elif isinstance(ptl, list):
                                pub_type_str = "; ".join(ptl)
                    if pub_type_str:
                        pub_types = [pt.strip().lower() for pt in pub_type_str.split(";") if pt.strip()]
                        paper["pub_types"] = pub_types
                        paper["is_meta_analysis"] = any("meta-analysis" in pt for pt in pub_types)
                        paper["is_systematic_review"] = any("systematic" in pt for pt in pub_types)
                        print(f"  → PMID:{pmid} pub_types={pub_types} | meta={paper['is_meta_analysis']}")

            # 방법 2: abstract 없으면 search API로 재조회 (pmid 직접 검색)
            if not abstract or len(abstract) < 50:
                try:
                    r2 = requests.get(
                        EUROPEPMC_SEARCH,
                        params={
                            "query": f"EXT_ID:{pmid} AND SRC:MED",
                            "format": "json",
                            "pageSize": 1,
                            "resultType": "core"
                        },
                        timeout=15
                    )
                    if r2.status_code == 200:
                        d2 = r2.json()
                        results2 = d2.get("resultList", {}).get("result", [])
                        if results2:
                            raw2 = results2[0].get("abstractText", "") or ""
                            ab2 = re.sub(r'<[^>]+>', '', raw2).strip()
                            if len(ab2) > len(abstract):
                                abstract = ab2
                except Exception:
                    pass

            if abstract:
                paper["abstract"] = abstract
                print(f"  ✓ PMID:{pmid} abstract {len(abstract)}chars")
            else:
                print(f"  ✗ PMID:{pmid} abstract 없음")

        except Exception as e:
            print(f"  ! PMID:{pmid} enrich 오류: {e}")
        enriched.append(paper)
        time.sleep(0.15)  # Rate limit
    return enriched


def parse_europepmc_result(r: dict) -> dict:
    """Europe PMC JSON 결과 → 표준 논문 딕셔너리"""
    try:
        pmid = str(r.get("pmid", "") or r.get("id", ""))
        if not pmid:
            return None

        pub_types = []
        if r.get("pubTypeList"):
            pt_list = r["pubTypeList"].get("pubType", [])
            if isinstance(pt_list, str):
                pt_list = [pt_list]
            pub_types = [pt.lower() for pt in pt_list]

        # 저자
        authors = []
        author_list = r.get("authorList", {}).get("author", [])
        if isinstance(author_list, dict):
            author_list = [author_list]
        for a in author_list[:6]:
            name = a.get("fullName") or f"{a.get('firstName','')} {a.get('lastName','')}".strip()
            if name:
                authors.append(name)

        # MeSH 용어
        mesh_terms = []
        for mh in r.get("meshHeadingList", {}).get("meshHeading", []):
            if isinstance(mh, dict):
                mesh_terms.append(mh.get("descriptorName", ""))

        # 초록
        abstract = r.get("abstractText", "") or ""
        abstract = re.sub(r'<[^>]+>', '', abstract).strip()

        # DOI
        doi = ""
        for dl in r.get("fullTextUrlList", {}).get("fullTextUrl", []):
            if isinstance(dl, dict) and dl.get("documentStyle") == "doi":
                doi = dl.get("url", "")
                break

        return {
            "pmid":               pmid,
            "title":              r.get("title", "").strip().rstrip("."),
            "abstract":           abstract,
            "journal":            r.get("journalTitle", "") or r.get("journal", {}).get("title", ""),
            "year":               str(r.get("pubYear", "") or ""),
            "authors":            authors,
            "mesh_terms":         [m for m in mesh_terms if m],
            "pub_types":          pub_types,
            "doi":                doi,
            "url":                f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "is_meta_analysis":   any("meta-analysis" in pt.lower() for pt in pub_types),
            "is_systematic_review": any("systematic" in pt.lower() for pt in pub_types),
            "crawled_at":         datetime.now().isoformat(),
            "source":             "europepmc"
        }
    except Exception as e:
        print(f"파싱 오류: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 메인 크롤링 함수
# ──────────────────────────────────────────────────────────────────────────────

def build_phytochemical_query(phytochemical: str, health_condition: str = None) -> str:
    """파이토케미컬 + 건강 조건 검색 쿼리 빌드 (Europe PMC 형식)"""
    # 단일 단어 건강 조건 목록 (공백 없는 것 우선)
    simple_conditions = ["cancer", "diabetes", "obesity", "inflammation", "aging", "hypertension"]
    
    if health_condition:
        # 공백 포함 조건은 따옴표로 감싸기
        if " " in health_condition:
            return f'{phytochemical} AND "{health_condition}"'
        return f'{phytochemical} AND {health_condition}'
    
    # 공백 없는 단순 조건만 OR로 묶기
    health_q = " OR ".join(simple_conditions)
    return f'{phytochemical} AND ({health_q})'


def crawl_phytochemical_papers(
    phytochemical: str,
    health_condition: str = None,
    max_results: int = 50,
    meta_analysis_only: bool = False,
    pub_type: str = None,
    progress_callback=None,
    abstracts_dir=None
) -> dict:
    """
    파이토케미컬 관련 논문 크롤링 메인 함수
    Europe PMC API 우선 사용
    """
    _abs_dir = Path(abstracts_dir) if abstracts_dir else ABSTRACTS_DIR
    _abs_dir.mkdir(parents=True, exist_ok=True)
    query = build_phytochemical_query(phytochemical, health_condition)

    if progress_callback:
        progress_callback({"step": "searching",
                           "message": f"Europe PMC 검색 중: {phytochemical}"})

    papers = search_europepmc(
        query=query,
        max_results=max_results,
        pub_type=pub_type,
        meta_only=meta_analysis_only
    )

    if not papers:
        return {"papers": [], "total": 0, "meta_count": 0,
                "new_count": 0, "query": query}

    existing_pmids = get_existing_pmids(_abs_dir)
    new_count = sum(1 for p in papers if p["pmid"] not in existing_pmids)

    if progress_callback:
        progress_callback({
            "step": "saving",
            "message": f"{len(papers)}편 수집, {new_count}편 신규 저장 중..."
        })

    saved = save_papers(papers, _abs_dir)
    meta_count = sum(1 for p in papers if p.get("is_meta_analysis"))

    if progress_callback:
        progress_callback({
            "step": "complete",
            "message": f"완료: {len(papers)}편 수집 | 메타분석 {meta_count}편 | {saved}편 저장"
        })

    return {
        "papers":     papers,
        "total":      len(papers),
        "meta_count": meta_count,
        "new_count":  new_count,
        "query":      query
    }


# ──────────────────────────────────────────────────────────────────────────────
# 저장 / 로드 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def get_existing_pmids(abs_dir=None) -> set:
    _dir = Path(abs_dir) if abs_dir else ABSTRACTS_DIR
    pmids = set()
    for f in _dir.glob("*.json"):
        pmids.add(f.stem)
    meta_file = DATA_DIR / "papers_metadata.csv"
    if meta_file.exists():
        try:
            df = pd.read_csv(meta_file)
            pmids.update(df["pmid"].astype(str).values)
        except Exception:
            pass
    return pmids


def save_papers(papers: list, abs_dir=None) -> int:
    _dir = Path(abs_dir) if abs_dir else ABSTRACTS_DIR
    _dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for paper in papers:
        if not paper:
            continue
        pmid = paper["pmid"]
        json_path = _dir / f"{pmid}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(paper, f, ensure_ascii=False, indent=2)
        txt_path = _dir / f"{pmid}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {paper['title']}\n\nAbstract: {paper['abstract']}")
        saved += 1
    update_metadata_csv(papers)
    return saved


def update_metadata_csv(papers: list):
    meta_file = DATA_DIR / "papers_metadata.csv"
    rows = []
    for p in papers:
        if not p:
            continue
        rows.append({
            "pmid":               p["pmid"],
            "title":              p["title"],
            "journal":            p["journal"],
            "year":               p["year"],
            "is_meta_analysis":   p["is_meta_analysis"],
            "is_systematic_review": p["is_systematic_review"],
            "mesh_terms":         "|".join(p["mesh_terms"]),
            "url":                p["url"],
            "crawled_at":         p["crawled_at"]
        })
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if meta_file.exists():
        existing = pd.read_csv(meta_file)
        combined = pd.concat([existing, new_df]).drop_duplicates(subset="pmid")
        combined.to_csv(meta_file, index=False)
    else:
        new_df.to_csv(meta_file, index=False)


def load_all_papers() -> list:
    papers = []
    for jf in ABSTRACTS_DIR.glob("*.json"):
        try:
            with open(jf, "r", encoding="utf-8") as f:
                papers.append(json.load(f))
        except Exception:
            pass
    return papers


def get_stats() -> dict:
    papers = load_all_papers()
    total  = len(papers)
    meta   = sum(1 for p in papers if p.get("is_meta_analysis"))
    sr     = sum(1 for p in papers if p.get("is_systematic_review"))
    years  = [p.get("year", "") for p in papers if p.get("year")]
    year_dist = {}
    for y in years:
        year_dist[y] = year_dist.get(y, 0) + 1
    return {
        "total_papers":          total,
        "meta_analysis_count":   meta,
        "systematic_review_count": sr,
        "year_distribution":     dict(sorted(year_dist.items(), reverse=True)[:10])
    }
