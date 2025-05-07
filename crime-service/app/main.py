import os
import sys
import logging
from datetime import datetime
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.domain.controller.crime_controller import CrimeController
from app.api.crime_router import router as predict_router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("crime_api")

# .env 파일 로드
load_dotenv()

# ✅ 애플리케이션 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Crime API 서비스 시작")
    yield
    logger.info("🛑 Crime API 서비스 종료")

# ✅ FastAPI 앱 생성 
app = FastAPI(
    title="Crime API",
    description="Crime Prediction API Service",
    version="0.1.0",
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

# ✅ 서브 라우터 생성
crime_router = APIRouter(prefix="/crime", tags=["Crime API"])

# ✅ 서브 라우터와 엔드포인트를 연결함
app.include_router(predict_router)

def run_crime_model():
    try:
        logger.info("Crime 모델 실행 시작")
        crime_controller = CrimeController()
        
        logger.info("Crime 모델 전처리 시작")
        crime_controller.preprocess()
        
        logger.info("Crime 모델 학습 시작")
        crime_controller.learning()
        
        logger.info("Crime 모델 평가 시작")
        accuracy = crime_controller.evaluation()
        logger.info(f"Crime 모델 정확도: {accuracy:.4f}")
        
        logger.info("Crime 모델 배포 시작")
        crime_controller.deployment()
        
        logger.info("Crime 모델 실행 완료")
        return True
    except Exception as e:
        logger.error(f"Crime 모델 실행 중 오류 발생: {str(e)}")
        return False

def main():
    logger.info("Crime 서비스 시작")
    success = run_crime_model()
    
    if success:
        logger.info("Crime 서비스가 성공적으로 실행되었습니다.")
    else:
        logger.error("Crime 서비스 실행에 실패했습니다.")

if __name__ == "__main__":
    main()
