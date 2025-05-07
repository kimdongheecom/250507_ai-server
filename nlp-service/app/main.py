import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from app.domain.service.samsung_report import SamsungReport

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("nlp_api")

# .env 파일 로드
load_dotenv()

# ✅ 애플리케이션 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 NLP API 서비스 시작")
    yield
    logger.info("🛑 NLP API 서비스 종료")

# ✅ FastAPI 앱 생성 
app = FastAPI(
    title="🧪 Samsung Report API",
    description="""
    삼성 리포트 분석 API 서비스입니다.
    워드클라우드 생성 및 분석 기능을 제공합니다.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def read_test():
    """테스트 엔드포인트"""
    return {"message": "테스트 파일입니다."}

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
            return FileResponse(
                output_path,
                media_type="image/png",
                filename="samsung_wordcloud.png",
                headers={"Content-Disposition": "inline; filename=samsung_wordcloud.png"}
            )
        else:
            raise HTTPException(status_code=500, detail="워드클라우드 생성에 실패했습니다.")
    except Exception as e:
        logger.error(f"워드클라우드 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9003, reload=True)
