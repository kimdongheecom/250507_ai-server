from fastapi import APIRouter, Request
import logging
from app.domain.controller.titanic_controller import Controller


# 로거 설정
logger = logging.getLogger("titanic_router")
logger.setLevel(logging.INFO)
router = APIRouter(tags=["Titanic API"])

# 더미 데이터
DUMMY_PASSENGERS = [
    {
        "pclass": 1,
        "sex": "female",
        "age": 25,
        "sibsp": 1,
        "parch": 0,
        "fare": 100.0,
        "embarked": "S",
        "survived": True,
        "probability": 0.95
    },
    {
        "pclass": 3,
        "sex": "male",
        "age": 30,
        "sibsp": 0,
        "parch": 0,
        "fare": 20.0,
        "embarked": "C",
        "survived": False,
        "probability": 0.15
    }
]

# GET
@router.get("/predict", summary="모든 타이타닉 승객 정보 조회")
async def get_all_passengers():
    """
    등록된 모든 타이타닉 승객의 정보를 조회합니다.
    """
    print("📋 모든 타이타닉 승객 정보 조회")
    logger.info("📋 모든 타이타닉 승객 정보 조회")
    return {"passengers": DUMMY_PASSENGERS}

# POST
@router.post("/predict")
async def predict_survival(request: Request):
    """
    승객 정보로 생존 여부를 예측합니다.
    """
    print("🔥 타이타닉 생존 예측 서비스 호출")
    logger.info("🌊 타이타닉 생존 예측 서비스 호출됨")
    return DUMMY_PASSENGERS[0]

# PUT
@router.put("/predict", summary="타이타닉 승객 정보 전체 수정")
async def update_passenger(request: Request):
    """
    타이타닉 승객 정보를 전체 수정합니다.
    """
    print("📝 타이타닉 승객 정보 전체 수정")
    logger.info("📝 타이타닉 승객 정보 전체 수정")
    return {
        "message": "승객 정보가 성공적으로 수정되었습니다.",
        "updated_data": DUMMY_PASSENGERS[0]
    }

# DELETE
@router.delete("/predict", summary="타이타닉 승객 정보 삭제")
async def delete_passenger():
    """
    타이타닉 승객 정보를 삭제합니다.
    """
    print("🗑️ 타이타닉 승객 정보 삭제")
    logger.info("🗑️ 타이타닉 승객 정보 삭제")
    return {
        "message": "승객 정보가 성공적으로 삭제되었습니다."
    }

# PATCH
@router.patch("/predict", summary="타이타닉 승객 정보 부분 수정")
async def patch_passenger(request: Request):
    """
    타이타닉 승객 정보를 부분적으로 수정합니다.
    """
    print("✏️ 타이타닉 승객 정보 부분 수정")
    logger.info("✏️ 타이타닉 승객 정보 부분 수정")
    return {
        "message": "승객 정보가 부분적으로 수정되었습니다.",
        "updated_fields": {
            "age": 27,
            "fare": 130.0,
            "probability": 0.97
        }
    }



# GET
@router.get("/csv", summary="타이타닉 모델 실행 및 CSV 생성")
def run_titanic_model():
    """
    타이타닉 모델을 실행하고 CSV 파일을 생성합니다.
    """
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
        return {"message": "csv파일이 생성되었습니다."}
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"message": "csv파일 생성 중 오류가 발생했습니다."}