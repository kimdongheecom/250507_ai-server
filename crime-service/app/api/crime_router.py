from fastapi import APIRouter, Request
import logging
from app.domain.controller.crime_controller import CrimeController
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import HTMLResponse
import os

# 로거 설정
logger = logging.getLogger("crime_router")
logger.setLevel(logging.INFO)
router = APIRouter()

# GET
@router.get("/preprocess", summary="범죄상세")
async def preprocess():
    controller = CrimeController()
    controller.preprocess('cctv_in_seoul.csv', 'crime_in_seoul.csv', 'pop_in_seoul.xls')
    return {"message": '서울시의 범죄 데이터가 전처리 되었습니다.'}

@router.get("/map", summary="범죄지도 그리기")
async def draw_crime_map():
    controller = CrimeController()
    controller.draw_crime_map()
    return {"message": '서울시의 범죄 지도가 완성되었습니다.'}

@router.get("/view-map", summary="범죄지도 보기", response_class=HTMLResponse)
async def view_crime_map():
    map_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stored-map', 'crime_map.html')
    if not os.path.exists(map_path):
        return HTMLResponse(content="지도 파일이 아직 생성되지 않았습니다. /map 엔드포인트를 먼저 호출해주세요.", status_code=404)
    
    with open(map_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)