"""
LLM 기반 Graph Database 구축 모듈
파이토케미컬 - 건강 상관성 엔티티/관계 추출 → NetworkX 그래프
"""

import json
import os
import re
from pathlib import Path
from typing import Optional
import networkx as nx
import pandas as pd
from openai import OpenAI

# DATA_DIR: 환경변수 DATA_DIR → Railway Volume(/data) → 앱 내부 data/ 순으로 우선 사용
_env_data = os.environ.get("DATA_DIR", "")
DATA_DIR = Path(_env_data) if _env_data else Path(__file__).parent.parent / "data"
GRAPH_DIR = DATA_DIR / "graph"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_FILE = GRAPH_DIR / "phytochemical_graph.json"

# 노드 타입 정의
NODE_TYPES = {
    "Phytochemical": "#4CAF50",   # 초록
    "HealthCondition": "#F44336", # 빨강
    "Mechanism": "#2196F3",       # 파랑
    "FoodSource": "#FF9800",      # 주황
    "Biomarker": "#9C27B0",       # 보라
    "Study": "#607D8B",           # 회색
}

# 관계 타입
RELATION_TYPES = {
    "IMPROVES": "개선",
    "REDUCES": "감소",
    "INHIBITS": "억제",
    "ACTIVATES": "활성화",
    "FOUND_IN": "함유",
    "AFFECTS": "영향",
    "BIOMARKER_OF": "바이오마커",
    "ASSOCIATED_WITH": "연관",
    "PREVENTS": "예방",
    "INCREASES": "증가",
}

EXTRACTION_PROMPT = """
당신은 생의학 문헌에서 파이토케미컬과 건강 상관성 관계를 추출하는 전문가입니다.

다음 논문 초록에서 엔티티와 관계를 JSON 형식으로 추출하세요.

추출 규칙:
1. 엔티티 타입:
   - Phytochemical: 파이토케미컬/식물 화합물 (예: curcumin, quercetin, resveratrol)
   - HealthCondition: 질병/건강 상태 (예: cancer, diabetes, inflammation)
   - Mechanism: 작용 메커니즘 (예: antioxidant activity, NF-κB pathway)
   - FoodSource: 식품 소스 (예: turmeric, green tea, berries)
   - Biomarker: 바이오마커 (예: CRP, TNF-α, IL-6)

2. 관계 타입:
   - IMPROVES: 개선/치료 효과
   - REDUCES: 감소 효과
   - INHIBITS: 억제
   - ACTIVATES: 활성화
   - FOUND_IN: 함유 (파이토케미컬 → 식품)
   - AFFECTS: 영향
   - PREVENTS: 예방
   - INCREASES: 증가

JSON 형식:
{
  "entities": [
    {"id": "unique_id", "name": "entity_name", "type": "EntityType", "description": "설명"}
  ],
  "relations": [
    {"source": "entity_id1", "target": "entity_id2", "type": "RELATION_TYPE", 
     "evidence": "근거 문장", "confidence": 0.0-1.0}
  ]
}

중요: 반드시 유효한 JSON만 반환하세요. 추가 텍스트 없이.

논문 초록:
"""


def extract_graph_entities(paper: dict, openai_client: OpenAI) -> dict:
    """단일 논문에서 그래프 엔티티/관계 추출"""
    
    text = f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}"
    
    if len(text) < 100:
        return {"entities": [], "relations": []}
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a biomedical entity extraction expert. Return ONLY valid JSON, no explanation."},
                {"role": "user", "content": EXTRACTION_PROMPT + text[:3000]}
            ],
            temperature=0.1
            # ⚠️ max_tokens 제거 — Genspark 프록시에서 빈 응답 유발
        )
        
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        if not content:
            print(f"  ⚠ PMID:{paper.get('pmid')} 빈 응답 finish_reason={finish_reason}")
            return {"entities": [], "relations": []}
        content = content.strip()
        print(f"  → PMID:{paper.get('pmid')} {len(content)}chars finish={finish_reason} preview={content[:60]!r}")
        
        # JSON 추출 (마크다운 코드블록 제거)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # 빈 응답 처리
        if not content or content in ("", "null", "{}"):
            return {"entities": [], "relations": []}
        
        # { } 로 시작하지 않으면 JSON 블록 찾기
        if not content.startswith("{") and not content.startswith("["):
            start = content.find("{")
            if start >= 0:
                content = content[start:]
        
        result = json.loads(content)
        
        # PMID를 엔티티 ID에 추가 (고유성 보장)
        pmid = paper.get('pmid', 'unknown')
        for entity in result.get("entities", []):
            entity["id"] = f"{entity['id']}_{pmid}"
            entity["pmid"] = pmid
        
        for rel in result.get("relations", []):
            rel["source"] = f"{rel['source']}_{pmid}"
            rel["target"] = f"{rel['target']}_{pmid}"
            rel["pmid"] = pmid
        
        return result
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"엔티티 추출 오류 (PMID: {paper.get('pmid')}): {e}")
        return {"entities": [], "relations": []}


def build_graph_from_papers(papers: list, openai_client: OpenAI, progress_callback=None) -> dict:
    """
    다수 논문에서 그래프 구축
    중복 엔티티 병합, 정규화 처리
    """
    G = nx.DiGraph()
    
    all_entities = {}
    all_relations = []
    
    for i, paper in enumerate(papers):
        if not paper.get('abstract'):
            continue
        
        if progress_callback:
            progress_callback({
                "step": "extracting",
                "current": i + 1,
                "total": len(papers),
                "message": f"엔티티 추출 중: {paper.get('title', '')[:50]}..."
            })
        
        result = extract_graph_entities(paper, openai_client)
        
        # 엔티티 병합 (이름 정규화)
        for entity in result.get("entities", []):
            normalized_name = normalize_entity_name(entity['name'])
            entity_key = f"{entity['type']}:{normalized_name}"
            
            if entity_key not in all_entities:
                all_entities[entity_key] = {
                    "id": entity_key,
                    "name": normalized_name,
                    "type": entity['type'],
                    "description": entity.get('description', ''),
                    "pmids": [entity.get('pmid', '')],
                    "mention_count": 1,
                    "color": NODE_TYPES.get(entity['type'], "#999999")
                }
            else:
                all_entities[entity_key]["mention_count"] += 1
                pmid = entity.get('pmid', '')
                if pmid and pmid not in all_entities[entity_key]["pmids"]:
                    all_entities[entity_key]["pmids"].append(pmid)
        
        # 관계 수집
        for rel in result.get("relations", []):
            # 소스/타겟 노드 찾기
            src_entity = next(
                (e for e in result.get("entities", []) if e['id'] == rel['source']), None
            )
            tgt_entity = next(
                (e for e in result.get("entities", []) if e['id'] == rel['target']), None
            )
            
            if src_entity and tgt_entity:
                src_key = f"{src_entity['type']}:{normalize_entity_name(src_entity['name'])}"
                tgt_key = f"{tgt_entity['type']}:{normalize_entity_name(tgt_entity['name'])}"
                
                all_relations.append({
                    "source": src_key,
                    "target": tgt_key,
                    "type": rel.get('type', 'ASSOCIATED_WITH'),
                    "label": RELATION_TYPES.get(rel.get('type', ''), rel.get('type', '')),
                    "evidence": rel.get('evidence', ''),
                    "confidence": rel.get('confidence', 0.5),
                    "pmid": rel.get('pmid', '')
                })
    
    # NetworkX 그래프 구축
    for entity in all_entities.values():
        G.add_node(
            entity["id"],
            **entity
        )
    
    # 관계 집계 (중복 엣지 → weight 증가)
    edge_dict = {}
    for rel in all_relations:
        edge_key = (rel['source'], rel['target'], rel['type'])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = {**rel, "weight": 1, "pmids": [rel['pmid']]}
        else:
            edge_dict[edge_key]["weight"] += 1
            if rel['pmid'] not in edge_dict[edge_key]["pmids"]:
                edge_dict[edge_key]["pmids"].append(rel['pmid'])
    
    for edge_key, edge_data in edge_dict.items():
        if edge_data['source'] in G and edge_data['target'] in G:
            G.add_edge(edge_data['source'], edge_data['target'], **edge_data)
    
    # 그래프 저장
    graph_data = serialize_graph(G)
    save_graph(graph_data)
    
    if progress_callback:
        progress_callback({
            "step": "complete",
            "message": f"그래프 구축 완료: {G.number_of_nodes()}개 노드, {G.number_of_edges()}개 엣지"
        })
    
    return graph_data


def normalize_entity_name(name: str) -> str:
    """엔티티 이름 정규화 (소문자, 공백 정리)"""
    if not name:
        return ""
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized


def serialize_graph(G: nx.DiGraph) -> dict:
    """NetworkX 그래프를 JSON 직렬화 형태로 변환"""
    nodes = []
    for node_id, data in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "name": data.get("name", node_id),
            "type": data.get("type", "Unknown"),
            "color": data.get("color", "#999999"),
            "mention_count": data.get("mention_count", 1),
            "pmids": data.get("pmids", []),
            "description": data.get("description", ""),
            "size": min(5 + data.get("mention_count", 1) * 3, 30)
        })
    
    edges = []
    for src, tgt, data in G.edges(data=True):
        edges.append({
            "source": src,
            "target": tgt,
            "type": data.get("type", "ASSOCIATED_WITH"),
            "label": data.get("label", ""),
            "weight": data.get("weight", 1),
            "confidence": data.get("confidence", 0.5),
            "evidence": data.get("evidence", ""),
            "pmids": data.get("pmids", [])
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "node_types": {},
            "edge_types": {}
        }
    }


def save_graph(graph_data: dict):
    """그래프 데이터 JSON 저장"""
    # 통계 업데이트
    node_types = {}
    for node in graph_data["nodes"]:
        t = node["type"]
        node_types[t] = node_types.get(t, 0) + 1
    
    edge_types = {}
    for edge in graph_data["edges"]:
        t = edge["type"]
        edge_types[t] = edge_types.get(t, 0) + 1
    
    graph_data["stats"]["node_types"] = node_types
    graph_data["stats"]["edge_types"] = edge_types
    
    with open(GRAPH_FILE, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)


def load_graph() -> dict:
    """저장된 그래프 로드"""
    if GRAPH_FILE.exists():
        with open(GRAPH_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"nodes": [], "edges": [], "stats": {}}


def search_graph(query: str) -> dict:
    """그래프에서 특정 엔티티 검색"""
    graph_data = load_graph()
    query_lower = query.lower()
    
    matching_nodes = [
        n for n in graph_data["nodes"]
        if query_lower in n.get("name", "").lower()
    ]
    
    node_ids = {n["id"] for n in matching_nodes}
    
    # 연결된 엣지 찾기
    related_edges = [
        e for e in graph_data["edges"]
        if e["source"] in node_ids or e["target"] in node_ids
    ]
    
    # 이웃 노드도 포함
    neighbor_ids = set()
    for e in related_edges:
        neighbor_ids.add(e["source"])
        neighbor_ids.add(e["target"])
    
    all_node_ids = node_ids | neighbor_ids
    all_nodes = [n for n in graph_data["nodes"] if n["id"] in all_node_ids]
    
    return {
        "nodes": all_nodes,
        "edges": related_edges,
        "query": query
    }


def subgraph_to_context_text(subgraph: dict, max_relations: int = 30) -> str:
    """
    GraphRAG 핵심 — 서브그래프를 GPT가 읽을 수 있는 관계 텍스트로 변환
    
    출력 예시:
    === 지식 그래프 관계 (논문 기반 추출) ===
    • curcumin  →[INHIBITS]→  nf-kb pathway
      근거: "Curcumin inhibits NF-κB activation..." (논문 3편)
    • curcumin  →[REDUCES]→  inflammation
      근거: "Curcumin significantly reduced..." (논문 5편)
    • nf-kb pathway  →[AFFECTS]→  chronic inflammation
    ...
    """
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    seed_ids = set(subgraph.get("seed_ids", []))

    if not edges:
        return ""

    # 노드 ID → 이름 맵
    id_to_name = {n["id"]: n["name"] for n in nodes}
    id_to_type = {n["id"]: n["type"] for n in nodes}

    # 관계 타입 한국어 레이블
    REL_LABELS = {
        "IMPROVES":        "개선",
        "REDUCES":         "감소/억제",
        "INHIBITS":        "억제",
        "ACTIVATES":       "활성화",
        "FOUND_IN":        "함유됨",
        "AFFECTS":         "영향",
        "PREVENTS":        "예방",
        "INCREASES":       "증가",
        "ASSOCIATED_WITH": "연관",
        "BIOMARKER_OF":    "바이오마커",
    }

    # seed 노드 관련 엣지 우선 정렬 (관련성 높은 것 먼저)
    def edge_priority(e):
        src_is_seed = e["source"] in seed_ids
        tgt_is_seed = e["target"] in seed_ids
        weight = e.get("weight", 1)
        return (-(src_is_seed + tgt_is_seed), -weight)

    sorted_edges = sorted(edges, key=edge_priority)[:max_relations]

    lines = ["=== 지식 그래프 관계 (논문 기반 추출) ==="]
    for e in sorted_edges:
        src_name = id_to_name.get(e["source"], e["source"])
        tgt_name = id_to_name.get(e["target"], e["target"])
        src_type = id_to_type.get(e["source"], "")
        tgt_type = id_to_type.get(e["target"], "")
        rel_type = e.get("type", "ASSOCIATED_WITH")
        rel_kor  = REL_LABELS.get(rel_type, rel_type)
        weight   = e.get("weight", 1)
        evidence = e.get("evidence", "")

        # 관계 라인
        line = f"• [{src_type}] {src_name}  →[{rel_type}/{rel_kor}]→  [{tgt_type}] {tgt_name}"
        if weight > 1:
            line += f"  (논문 {weight}편에서 확인)"
        lines.append(line)

        # 근거 문장 (있는 경우만)
        if evidence and len(evidence) > 10:
            lines.append(f"  근거: \"{evidence[:120]}\"")

    lines.append(f"\n총 {len(sorted_edges)}개 관계 / {len(nodes)}개 엔티티 노드")
    return "\n".join(lines)


def get_graph_stats() -> dict:
    """그래프 통계 반환"""
    graph_data = load_graph()
    return graph_data.get("stats", {
        "total_nodes": len(graph_data.get("nodes", [])),
        "total_edges": len(graph_data.get("edges", []))
    })


def find_subgraph_for_entities(entity_names: list, max_hops: int = 2) -> dict:
    """
    GraphRAG 경로 탐색 — 엔티티 이름 목록으로 연결 서브그래프 반환
    
    흐름:
    1. entity_names 부분 일치로 seed 노드 탐색
    2. BFS max_hops 내 이웃 수집
    3. seed 간 최단경로 엣지도 포함
    4. {nodes, edges, seed_ids, paths} 반환
    """
    if not entity_names:
        return {"nodes": [], "edges": [], "seed_ids": [], "paths": []}

    graph_data = load_graph()
    all_nodes = graph_data.get("nodes", [])
    all_edges = graph_data.get("edges", [])

    if not all_nodes:
        return {"nodes": [], "edges": [], "seed_ids": [], "paths": []}

    # ── 1. seed 노드 탐색 (소문자 부분 일치) ─────────────────────────────────
    name_lower = [n.lower().strip() for n in entity_names if n]
    seed_nodes = []
    for node in all_nodes:
        node_name = node.get("name", "").lower()
        node_id   = node.get("id", "").lower()
        if any(e in node_name or e in node_id for e in name_lower):
            seed_nodes.append(node)

    if not seed_nodes:
        # 폴백: 단어 단위 토큰 포함 여부로 재탐색
        tokens = set()
        for e in name_lower:
            tokens.update(e.split())
        tokens = {t for t in tokens if len(t) > 3}  # 짧은 단어 제외
        for node in all_nodes:
            node_name = node.get("name", "").lower()
            if any(t in node_name for t in tokens):
                seed_nodes.append(node)

    seed_ids = {n["id"] for n in seed_nodes}

    # ── 2. NetworkX 그래프 구성 ───────────────────────────────────────────────
    G = nx.DiGraph()
    for node in all_nodes:
        G.add_node(node["id"])
    for edge in all_edges:
        G.add_edge(edge["source"], edge["target"],
                   type=edge.get("type", ""),
                   label=edge.get("label", ""),
                   evidence=edge.get("evidence", ""),
                   weight=edge.get("weight", 1))

    # ── 3. BFS로 max_hops 이내 이웃 수집 ─────────────────────────────────────
    visited = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(max_hops):
        next_frontier = set()
        for nid in frontier:
            if nid in G:
                for neighbor in list(G.successors(nid)) + list(G.predecessors(nid)):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
        frontier = next_frontier

    subgraph_node_ids = visited

    # ── 4. seed 간 최단경로 (undirected) 추가 ────────────────────────────────
    path_edges: list = []
    G_undirected = G.to_undirected()
    seed_list = list(seed_ids)
    paths_info = []
    for i in range(len(seed_list)):
        for j in range(i + 1, len(seed_list)):
            s, t = seed_list[i], seed_list[j]
            if s in G_undirected and t in G_undirected:
                try:
                    path = nx.shortest_path(G_undirected, s, t)
                    if len(path) <= max_hops + 2:  # 너무 먼 경로 제외
                        subgraph_node_ids.update(path)
                        paths_info.append({
                            "from": s,
                            "to": t,
                            "path": path,
                            "length": len(path) - 1
                        })
                except nx.NetworkXNoPath:
                    pass

    # ── 5. 최종 노드/엣지 필터링 ─────────────────────────────────────────────
    result_nodes = [n for n in all_nodes if n["id"] in subgraph_node_ids]
    result_edges = [e for e in all_edges
                    if e["source"] in subgraph_node_ids
                    and e["target"] in subgraph_node_ids]

    print(f"[GraphRAG] 엔티티={entity_names} → seed={len(seed_ids)}개, "
          f"서브그래프={len(result_nodes)}노드/{len(result_edges)}엣지")

    return {
        "nodes": result_nodes,
        "edges": result_edges,
        "seed_ids": list(seed_ids),
        "paths": paths_info
    }
