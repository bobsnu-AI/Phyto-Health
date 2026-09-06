#!/bin/bash
# ============================================================
# Railway 시작 스크립트
# - Railway Volume(/data)이 마운트되어 있으면 → Volume 사용 (영속)
# - Volume 없으면 → 앱 내부 data/ 사용 (재배포 시 초기화)
# ============================================================

APP_DATA_DIR="$(dirname "$0")/data"
VOLUME_MOUNT="/data"

echo "🌿 PhytoGraph 시작 중..."

# Railway Volume 마운트 감지
if [ -d "$VOLUME_MOUNT" ]; then
    echo "✅ Railway Volume 감지됨: $VOLUME_MOUNT"
    export DATA_DIR="$VOLUME_MOUNT"

    # Volume에 seed abstracts가 없으면 앱 내부에서 복사
    if [ ! -d "$VOLUME_MOUNT/abstracts" ] || [ -z "$(ls -A $VOLUME_MOUNT/abstracts 2>/dev/null)" ]; then
        echo "📋 abstracts 없음 → seed 복사 중..."
        mkdir -p "$VOLUME_MOUNT/abstracts" "$VOLUME_MOUNT/graph" "$VOLUME_MOUNT/chroma_db"
        [ -d "$APP_DATA_DIR/abstracts" ]  && cp -r "$APP_DATA_DIR/abstracts/."  "$VOLUME_MOUNT/abstracts/"
        [ -d "$APP_DATA_DIR/graph" ]      && cp -r "$APP_DATA_DIR/graph/."      "$VOLUME_MOUNT/graph/"
        [ -d "$APP_DATA_DIR/chroma_db" ]  && cp -r "$APP_DATA_DIR/chroma_db/."  "$VOLUME_MOUNT/chroma_db/"
        [ -f "$APP_DATA_DIR/papers_metadata.csv" ] && cp "$APP_DATA_DIR/papers_metadata.csv" "$VOLUME_MOUNT/"
        echo "✅ seed 데이터 복사 완료 ($(ls $VOLUME_MOUNT/abstracts/*.json 2>/dev/null | wc -l)편)"
    else
        echo "✅ Volume 기존 데이터 사용: $(ls $VOLUME_MOUNT/abstracts/*.json 2>/dev/null | wc -l)편"
    fi

else
    # ── Volume 없음 → 앱 내부 data/ 사용 ──────────────────────────────
    echo "⚠️  Volume 미연결 → 앱 내부 data/ 사용"
    echo "   💡 영속 저장: Railway 대시보드 → Volumes → Mount Path: /data"
    export DATA_DIR="$APP_DATA_DIR"
fi

echo "📂 DATA_DIR=$DATA_DIR"

# ── users.db 위치: Volume 있으면 Volume, 없으면 /tmp (재시작 간 유지) ──
# Railway는 Volume 없어도 /tmp는 프로세스 재시작 간 유지됨
# 단, 재배포(새 컨테이너) 시에는 초기화됨 → Volume이 유일한 영속 해결책
if [ -z "$USERS_DB_DIR" ]; then
    if [ -d "$VOLUME_MOUNT" ]; then
        export USERS_DB_DIR="$VOLUME_MOUNT"
    else
        export USERS_DB_DIR="$DATA_DIR"
    fi
fi
echo "🗄️  USERS_DB_DIR=$USERS_DB_DIR"

exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
