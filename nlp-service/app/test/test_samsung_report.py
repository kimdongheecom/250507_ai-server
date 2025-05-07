from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import pytest
from httpx import AsyncClient
from app.domain.service.samsung_report import SamsungReport
import os
from app.main import app

# FastAPI 앱 설정
app = FastAPI(
    title="🧪 Samsung Report API (Test 중입니다)",
    description="""
    삼성 리포트 분석 API 테스트 환경입니다.
    현재 테스트가 진행 중이며, 운영 환경이 아닙니다.
    """,
    version="1.0.0-test",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 엔드포인트
@app.get("/test")
async def read_test():
    """테스트 엔드포인트"""
    return {"message": "테스트 파일입니다 1."}

@app.get("/wordcloud")
async def generate_wordcloud():
    """워드클라우드 생성 엔드포인트
    
    삼성 보고서를 분석하여 워드클라우드를 생성하고 이미지 파일을 반환합니다.
    """
    try:
        # SamsungReport 인스턴스 생성
        sr = SamsungReport()
        
        # 전체 프로세스 실행
        sr.read_file()
        sr.extract_hangul()
        sr.change_token()
        sr.extract_noun()
        sr.read_stopword()
        sr.remove_stopword()
        sr.find_frequency()
        
        # 워드클라우드 생성
        output_path = sr.draw_wordcloud()
        
        # 파일이 존재하는지 확인
        if os.path.exists(output_path):
            return FileResponse(output_path, media_type="image/png", filename="samsung_wordcloud.png")
        else:
            return {"error": "워드클라우드 생성에 실패했습니다."}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"detail": str(e)}

# 테스트 코드
@pytest.fixture
async def async_client():
    """비동기 클라이언트 픽스처"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.mark.asyncio
async def test_read_test(async_client):
    """테스트 엔드포인트 테스트"""
    response = await async_client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "테스트 파일입니다."}

@pytest.mark.asyncio
async def test_generate_wordcloud(async_client):
    """워드클라우드 생성 테스트"""
    response = await async_client.get("/wordcloud")
    assert response.status_code == 200

# 직접 실행을 위한 코드
if __name__ == "__main__":
    import os
    import sys
    # 현재 디렉토리를 Python 경로에 추가
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    import uvicorn
    uvicorn.run(
        "app.test.test_samsung_report:app",
        host="0.0.0.0",
        port=9003,
        reload=True
    )
