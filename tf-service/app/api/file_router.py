import os
from fastapi import APIRouter, File,  UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import logging
import cv2

router = APIRouter()
logger = logging.getLogger("tf_main")

# 업로드 디렉토리와 출력 디렉토리를 app 내부로 고정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"파일 업로드 디렉토리: {UPLOAD_DIR}")
logger.info(f"파일 출력 디렉토리: {OUTPUT_DIR}")

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    logger.info(f"파일 업로드 시작: {file.filename}, 저장 위치: {file_location}")
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"파일 업로드 성공: {file.filename}")
        if os.path.exists(file_location):
            file_size = os.path.getsize(file_location)
            logger.info(f"파일 저장 확인: {file_location}, 크기: {file_size} bytes")
        else:
            logger.error(f"파일이 저장되지 않음: {file_location}")
        return JSONResponse(content={"filename": file.filename, "message": "파일 업로드 성공!", "path": file_location})
    except Exception as e:
        logger.error(f"파일 업로드 실패: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@router.get("/mosaic")
async def mosaic_all_uploads():
    cascade = os.path.join(BASE_DIR, 'data', 'haarcascade_frontalface_alt.xml')
    face_cascade = cv2.CascadeClassifier(cascade)
    processed_files = []
    failed_files = []

    for filename in os.listdir(UPLOAD_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(UPLOAD_DIR, filename)
            img = cv2.imread(img_path)
            if img is None:
                failed_files.append(filename)
                continue
            face = face_cascade.detectMultiScale(img, minSize=(30,30))
            if len(face) == 0:
                logger.error(f'얼굴인식 실패: {filename}')
                failed_files.append(filename)
                continue
            for (x, y, w, h) in face:
                logger.info(f'{filename} 얼굴의 좌표 = {x}, {y}, {w}, {h}')
                # 얼굴 영역 잘라내기
                face_img = img[y:y+h, x:x+w]
                # 모자이크(픽셀화) 적용
                mosaic = cv2.resize(face_img, (16, 16), interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(mosaic, (w, h), interpolation=cv2.INTER_NEAREST)
                # 원본 이미지에 다시 붙이기
                img[y:y+h, x:x+w] = mosaic
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}-face.png")
            cv2.imwrite(output_path, img)
            processed_files.append(output_path)

    return JSONResponse(content={
        "message": "모자이크 처리 완료",
        "processed_files": processed_files,
        "failed_files": failed_files
    })


