"""
인증 모듈 — JWT + SQLite 기반 멀티유저 관리
각 사용자는 독립된 data/<user_id>/ 디렉토리를 가짐
"""

import os
import sqlite3
import hashlib
import secrets
import time
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import jwt
from fastapi import HTTPException, Depends, Header
from pydantic import BaseModel

# ─── 설정 ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET", "phytograph-secret-change-in-production-2024")
ALGORITHM  = "HS256"
TOKEN_EXPIRE_DAYS = 30

# 루트 data 디렉토리 (Railway Volume or 앱 내부)
_env_data = os.environ.get("DATA_DIR", "")
ROOT_DATA_DIR = Path(_env_data) if _env_data else Path(__file__).parent.parent / "data"
ROOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# users.db 경로: USERS_DB_DIR 환경변수 우선 → DATA_DIR → 앱 내부 data/
_env_users_db_dir = os.environ.get("USERS_DB_DIR", "")
_users_db_base = Path(_env_users_db_dir) if _env_users_db_dir else ROOT_DATA_DIR
_users_db_base.mkdir(parents=True, exist_ok=True)

# 사용자 DB 경로 (계정 정보만 저장 — Volume에 보관해야 재배포 후에도 유지)
USERS_DB_PATH = _users_db_base / "users.db"

# Seed 데이터 경로 (신규 사용자에게 복사할 초기 데이터)
SEED_DATA_DIR = Path(__file__).parent.parent / "data"


# ─── 데이터 모델 ──────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = ""


class LoginRequest(BaseModel):
    username: str
    password: str


class UserInfo(BaseModel):
    user_id: str
    username: str
    email: str
    created_at: str


# ─── SQLite 사용자 DB ─────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(str(USERS_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """users 테이블 초기화"""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    TEXT PRIMARY KEY,
            username   TEXT UNIQUE NOT NULL,
            email      TEXT DEFAULT '',
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print(f"✅ 사용자 DB 초기화: {USERS_DB_PATH}")


# ─── 비밀번호 해싱 ────────────────────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = "phytograph_salt_v1"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


# ─── JWT ─────────────────────────────────────────────────────────────────────
def create_token(user_id: str, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "토큰이 만료되었습니다. 다시 로그인하세요.")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "유효하지 않은 토큰입니다.")


# ─── 인증 의존성 (FastAPI Depends) ────────────────────────────────────────────
def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Authorization: Bearer <token> 헤더에서 사용자 정보 추출
    모든 보호된 API에 Depends(get_current_user) 추가
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "로그인이 필요합니다. Authorization: Bearer <token> 헤더를 포함하세요.")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    return {
        "user_id": payload["user_id"],
        "username": payload["username"],
    }


# ─── 사용자별 데이터 경로 ─────────────────────────────────────────────────────
def get_user_data_dir(user_id: str) -> Path:
    """
    사용자별 독립 data 디렉토리 반환
    예: /data/users/abc123/abstracts, /data/users/abc123/chroma_db
    """
    user_dir = ROOT_DATA_DIR / "users" / user_id
    (user_dir / "abstracts").mkdir(parents=True, exist_ok=True)
    (user_dir / "graph").mkdir(parents=True, exist_ok=True)
    (user_dir / "chroma_db").mkdir(parents=True, exist_ok=True)
    return user_dir


def copy_seed_data(user_dir: Path):
    """
    신규 사용자 디렉토리에 seed 데이터(126편 논문 + 그래프 + ChromaDB) 복사
    seed가 없거나 이미 데이터가 있으면 스킵
    """
    import shutil

    # abstracts 복사
    seed_abstracts = SEED_DATA_DIR / "abstracts"
    user_abstracts = user_dir / "abstracts"
    if seed_abstracts.exists() and not any(user_abstracts.glob("*.json")):
        for f in seed_abstracts.glob("*.json"):
            shutil.copy2(f, user_abstracts / f.name)
        for f in seed_abstracts.glob("*.txt"):
            shutil.copy2(f, user_abstracts / f.name)
        print(f"  → abstracts {len(list(user_abstracts.glob('*.json')))}편 복사")

    # graph 복사
    seed_graph = SEED_DATA_DIR / "graph" / "phytochemical_graph.json"
    user_graph  = user_dir / "graph" / "phytochemical_graph.json"
    if seed_graph.exists() and not user_graph.exists():
        shutil.copy2(seed_graph, user_graph)
        print(f"  → graph 복사")

    # chroma_db 복사
    seed_chroma = SEED_DATA_DIR / "chroma_db"
    user_chroma  = user_dir / "chroma_db"
    if (seed_chroma / "chroma.sqlite3").exists() and not (user_chroma / "chroma.sqlite3").exists():
        shutil.copytree(seed_chroma, user_chroma, dirs_exist_ok=True)
        print(f"  → chroma_db 복사")


# ─── 회원가입 / 로그인 비즈니스 로직 ─────────────────────────────────────────
def register_user(username: str, password: str, email: str = "") -> dict:
    """신규 사용자 등록 + seed 데이터 초기화"""
    if len(username) < 3:
        raise HTTPException(400, "사용자명은 3자 이상이어야 합니다.")
    if len(password) < 6:
        raise HTTPException(400, "비밀번호는 6자 이상이어야 합니다.")

    conn = get_db()
    existing = conn.execute(
        "SELECT user_id FROM users WHERE username = ?", (username,)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(409, "이미 사용 중인 사용자명입니다.")

    user_id = secrets.token_hex(8)  # 16자리 랜덤 ID
    password_hash = hash_password(password)

    conn.execute(
        "INSERT INTO users (user_id, username, email, password_hash) VALUES (?, ?, ?, ?)",
        (user_id, username, email, password_hash)
    )
    conn.commit()
    conn.close()

    # 사용자 데이터 디렉토리 생성 + seed 복사
    user_dir = get_user_data_dir(user_id)
    print(f"[Auth] 신규 사용자 '{username}' (ID: {user_id}) — seed 데이터 복사 중...")
    copy_seed_data(user_dir)

    token = create_token(user_id, username)
    return {
        "token": token,
        "user_id": user_id,
        "username": username,
        "message": f"환영합니다, {username}님! 논문 {len(list((user_dir / 'abstracts').glob('*.json')))}편이 로드되었습니다."
    }


def login_user(username: str, password: str) -> dict:
    """로그인 → JWT 토큰 반환"""
    conn = get_db()
    row = conn.execute(
        "SELECT user_id, username, email, password_hash FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()

    if not row or not verify_password(password, row["password_hash"]):
        raise HTTPException(401, "사용자명 또는 비밀번호가 올바르지 않습니다.")

    # 데이터 디렉토리 존재 보장 (Volume 재마운트 등으로 사라진 경우)
    user_dir = get_user_data_dir(row["user_id"])
    if not any((user_dir / "abstracts").glob("*.json")):
        print(f"[Auth] '{username}' 데이터 없음 → seed 재복사")
        copy_seed_data(user_dir)

    token = create_token(row["user_id"], row["username"])
    return {
        "token": token,
        "user_id": row["user_id"],
        "username": row["username"],
        "message": f"로그인 성공"
    }


def get_user_info(user_id: str) -> dict:
    """사용자 정보 + 데이터 통계 반환"""
    conn = get_db()
    row = conn.execute(
        "SELECT user_id, username, email, created_at FROM users WHERE user_id = ?",
        (user_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(404, "사용자를 찾을 수 없습니다.")

    user_dir = get_user_data_dir(user_id)
    paper_count = len(list((user_dir / "abstracts").glob("*.json")))
    graph_exists = (user_dir / "graph" / "phytochemical_graph.json").exists()

    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "email": row["email"],
        "created_at": row["created_at"],
        "stats": {
            "papers": paper_count,
            "graph_built": graph_exists,
            "data_dir": str(user_dir)
        }
    }


# DB 초기화 (모듈 임포트 시 자동 실행)
init_db()
