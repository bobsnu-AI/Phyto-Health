#!/bin/bash
# ============================================================
# Railway 시작 스크립트
# - Railway Volume(/data)이 마운트되어 있으면 → Volume 사용
# - Volume 없으면 → 앱 내부 data/ (git seed 데이터) 사용
# ============================================================

APP_DATA_DIR="/app/data"
VOLUME_MOUNT="/data"

echo "🌿 PhytoGraph 시작 중..."

# Railway Volume 마운트 감지
if [ -d "$VOLUME_MOUNT" ] && [ "$(ls -A $VOLUME_MOUNT 2>/dev/null)" ]; then
    echo "✅ Railway Volume 감지됨: $VOLUME_MOUNT"
    # Volume이 있으면 Volume 경로를 DATA_DIR로 사용
    export DATA_DIR="$VOLUME_MOUNT"
    
    # Volume에 seed 데이터가 없으면 앱 내부에서 복사
    if [ ! -d "$VOLUME_MOUNT/abstracts" ] || [ -z "$(ls -A $VOLUME_MOUNT/abstracts 2>/dev/null)" ]; then
        echo "📋 Volume이 비어 있음 → seed 데이터 복사 중..."
        mkdir -p "$VOLUME_MOUNT/abstracts" "$VOLUME_MOUNT/graph" "$VOLUME_MOUNT/chroma_db"
        
        # abstracts 복사
        if [ -d "$APP_DATA_DIR/abstracts" ]; then
            cp -r "$APP_DATA_DIR/abstracts/." "$VOLUME_MOUNT/abstracts/"
            echo "  → abstracts: $(ls $VOLUME_MOUNT/abstracts/*.json 2>/dev/null | wc -l)편 복사"
        fi
        
        # graph 복사
        if [ -d "$APP_DATA_DIR/graph" ]; then
            cp -r "$APP_DATA_DIR/graph/." "$VOLUME_MOUNT/graph/"
            echo "  → graph: $(ls $VOLUME_MOUNT/graph/ 2>/dev/null)"
        fi
        
        # chroma_db 복사
        if [ -d "$APP_DATA_DIR/chroma_db" ]; then
            cp -r "$APP_DATA_DIR/chroma_db/." "$VOLUME_MOUNT/chroma_db/"
            echo "  → chroma_db: 복사 완료"
        fi
        
        # papers_metadata.csv 복사
        if [ -f "$APP_DATA_DIR/papers_metadata.csv" ]; then
            cp "$APP_DATA_DIR/papers_metadata.csv" "$VOLUME_MOUNT/"
            echo "  → papers_metadata.csv 복사 완료"
        fi
        
        echo "✅ seed 데이터 복사 완료"
    else
        PAPER_COUNT=$(ls "$VOLUME_MOUNT/abstracts/"*.json 2>/dev/null | wc -l)
        echo "✅ Volume에 기존 데이터 존재: 논문 ${PAPER_COUNT}편"
    fi

else
    echo "ℹ️  Railway Volume 미연결 → 앱 내부 data/ 사용 (재배포 시 초기화됨)"
    echo "   💡 영속 저장을 원하면 Railway 대시보드 → Volumes → /data 마운트 필요"
    export DATA_DIR="$APP_DATA_DIR"
    PAPER_COUNT=$(ls "$APP_DATA_DIR/abstracts/"*.json 2>/dev/null | wc -l)
    echo "   현재 seed 데이터: 논문 ${PAPER_COUNT}편"
fi

echo "📂 사용 데이터 경로: $DATA_DIR"
echo ""

# FastAPI 서버 시작
exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
