import os
import sys
import pandas as pd
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from app.domain.controller.titanic_controller import Controller
from app.api.titanic_router import router as predict_router

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("titanic_api")

# .env 파일 로드
load_dotenv()

# ✅ 애플리케이션 시작 시 실행
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Titanic API 서비스 시작")
    yield
    logger.info("🛑 Titanic API 서비스 종료")

# ✅ FastAPI 앱 생성 
app = FastAPI(
    title="Titanic API",
    description="Titanic Prediction API Service",
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
titanic_router = APIRouter(prefix="/titanic", tags=["Titanic API"])

# ✅ 서브 라우터와 엔드포인트를 연결함
app.include_router(predict_router)

def run_titanic_model():
    try:
        logger.info("Titanic 모델 실행 시작")
        titanic_controller = Controller()
        
        logger.info("Titanic 모델 전처리 시작")
        titanic_controller.preprocess("train.csv", "test.csv")
        
        logger.info("Titanic 모델 학습 시작")
        titanic_controller.learning()
        
        logger.info("Titanic 모델 평가 시작")
        accuracy = titanic_controller.evaluation()
        logger.info(f"Titanic 모델 정확도: {accuracy:.4f}")
        
        titanic_controller.submit()
        titanic_controller.tune()
        titanic_controller.tune_svm()
        titanic_controller.tune_voting()
        titanic_controller.feature_importance()
        
        logger.info("Titanic 모델 실행 완료")
        return True
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return False

def main():
    success = run_titanic_model()
    
    if success:
        logger.info("Titanic 서비스가 성공적으로 실행되었습니다.")
    else:
        logger.error("Titanic 서비스 실행에 실패했습니다.")

if __name__ == "__main__":
    main()

