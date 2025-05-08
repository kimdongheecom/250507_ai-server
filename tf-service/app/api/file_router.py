import os
from typing import Optional
import datetime
from fastapi import APIRouter, File, Form, UploadFile, Path, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import shutil
import logging
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow import keras
import numpy as np
from app.domain.controller.mnist_controller import read_mnist_image

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

# MNIST 이미지를 칼라로 변환하는 함수
def apply_colormap(gray_image):
    """흑백 이미지를 칼라로 변환합니다."""
    # 이미지 정규화 (0-1 범위로)
    normalized = gray_image.astype(np.float32) / 255.0
    
    # 칼라맵 적용 (jet, viridis, plasma, inferno, magma, cividis 등 선택 가능)
    colored = cm.viridis(normalized)
    
    # RGBA에서 RGB로 변환 (알파 채널 제거)
    colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return colored_rgb

# MNIST 샘플 이미지 생성 엔드포인트
@router.get("/mnist-sample")
async def get_mnist_sample():
    """
    MNIST 데이터셋에서 100번째 이미지를 칼라로 변환하여 반환합니다.
    이미지는 mnist 디렉토리에 mnist_sample.png 파일로 저장되고, 레이블은 JSON으로 반환됩니다.
    
    **Returns**:
    - **label**: 이미지의 레이블 (숫자 0-9)
    - **image_path**: 저장된 이미지 파일 경로
    """
    try:
        # MNIST 데이터셋 로드
        mnist = keras.datasets.mnist
        (train_images, train_labels), (_, _) = mnist.load_data()
        
        # 100번째 이미지 선택
        mnist_idx = 100
        image = train_images[mnist_idx]
        label = int(train_labels[mnist_idx])
        
        # 이미지 파일 저장 경로 설정
        mnist_dir = os.path.join(OUTPUT_DIR, "mnist")
        os.makedirs(mnist_dir, exist_ok=True)
        image_path = os.path.join(mnist_dir, "mnist_sample_color.png")
        
        # Matplotlib을 사용하여 칼라 이미지 저장
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='viridis')  # 칼라맵 적용
        plt.axis('off')  # 축 제거
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        logger.info(f"MNIST 칼라 이미지(인덱스: {mnist_idx}, 레이블: {label})가 {image_path}에 저장되었습니다.")
        
        # 응답 반환
        return {
            "label": label,
            "image_path": image_path
        }
    
    except Exception as e:
        logger.error(f"MNIST 이미지 생성 오류: {str(e)}")
        return JSONResponse(
            content={"error": f"MNIST 이미지 생성 중 오류 발생: {str(e)}"},
            status_code=500
        )
    
@router.post("/mnist-sample")
async def post_mnist_sample(
    index: int = Form(100),
    filename: str = Form(""),  # 빈 문자열로 기본값 변경
    add_noise: bool = Form(False),
    colormap: str = Form("viridis"),  # 칼라맵 선택 옵션 추가
    file: Optional[UploadFile] = File(None)
):
    try:
        # 변수 초기화
        label = None
        
        if file:
            # 파일이 업로드된 경우
            file_location = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_location, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"업로드된 파일 저장 완료: {file_location}")
            
            # 이미지 읽기 (컬러로 읽기)
            image = cv2.imread(file_location)
            if image is None:
                raise ValueError("이미지를 읽을 수 없습니다.")
                
            # 컬러 이미지를 그대로 사용
            # OpenCV는 BGR 순서로 읽어오므로 RGB로 변환
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # 업로드된 이미지는 레이블 정보가 없음
            label = "uploaded_image"
        else:
            # MNIST 데이터셋에서 이미지 로드
            mnist = keras.datasets.mnist
            (train_images, train_labels), (_, _) = mnist.load_data()
            
            # 인덱스 유효성 검사
            if index < 0 or index >= len(train_images):
                return JSONResponse(
                    content={"error": f"유효하지 않은 인덱스: {index}, 0-{len(train_images)-1} 범위 내에서 지정해주세요."},
                    status_code=400
                )
                
            # 이미지와 레이블 가져오기
            gray_image = train_images[index].copy()
            label = int(train_labels[index])
            
            # 흑백 이미지를 칼라로 변환
            # 사용 가능한 칼라맵: viridis, plasma, inferno, magma, cividis, jet 등
            valid_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet']
            if colormap not in valid_colormaps:
                colormap = 'viridis'  # 기본값
                
            # 칼라맵 적용
            image = gray_image  # 원본 흑백 이미지는 plt.imshow로 칼라맵 적용

        # 노이즈 추가 (요청된 경우)
        has_noise = False
        if add_noise and file is None:  # 업로드된 파일에는 노이즈 적용 안 함
            # 노이즈 생성 및 적용
            noise = np.random.normal(0, 15, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            has_noise = True

        # 출력 디렉토리 설정
        mnist_dir = os.path.join(OUTPUT_DIR, "mnist")
        os.makedirs(mnist_dir, exist_ok=True)

        # 파일명 생성 - 사용자 지정 또는 현재 시간 기반
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not filename:
            # 사용자가 파일명을 지정하지 않은 경우 현재 시간 사용
            if isinstance(label, int):
                filename = f"mnist_{label}_color_{timestamp}.png"
            else:
                filename = f"custom_image_{timestamp}.png"
        else:
            # 사용자가 파일명을 지정한 경우 확장자만 확인
            filename_base, filename_ext = os.path.splitext(filename)
            if not filename_ext:
                filename = f"{filename}.png"
        
        # 노이즈와 레이블 정보를 파일명에 추가
        noise_suffix = "_noise" if has_noise else ""
        label_info = f"_label{label}" if isinstance(label, int) else ""
        color_suffix = "_color" if file is None else ""
        
        # 최종 파일명 생성 (중복 방지를 위해 항상 타임스탬프 포함)
        filename_base, filename_ext = os.path.splitext(filename)
        final_filename = f"{filename_base}{label_info}{noise_suffix}{color_suffix}_{timestamp}{filename_ext}"
        image_path = os.path.join(mnist_dir, final_filename)

        # 이미지 저장
        plt.figure(figsize=(5, 5))
        if file is None:
            # MNIST 이미지에 칼라맵 적용
            plt.imshow(image, cmap=colormap)
        else:
            # 업로드된 이미지는 이미 칼라
            plt.imshow(image)
        plt.axis('off')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        logger.info(f"이미지 저장 완료: {image_path}")

        # 응답 반환
        return {
            "label": label,
            "image_path": image_path,
            "has_noise": has_noise,
            "is_color": True,
            "colormap": colormap if file is None else "Original colors"
        }

    except Exception as e:
        logger.error(f"이미지 생성 오류: {str(e)}")
        return JSONResponse(
            content={"error": f"이미지 생성 중 오류 발생: {str(e)}"},
            status_code=500
        )

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

# 새로운 엔드포인트: MNIST 이미지 읽기 (데이터셋 또는 파일)
@router.get("/read")
async def read_mnist_image_endpoint(
    index: Optional[int] = Query(None, description="MNIST 데이터셋 인덱스 (0-59999)"),
    filepath: Optional[str] = Query(None, description="읽을 이미지 파일 경로 (UPLOADS_DIR 기준)")
):
    """
    MNIST 이미지를 데이터셋에서 읽거나 파일에서 로드합니다.
    
    **Query Parameters**:
    - **index**: MNIST 데이터셋의 인덱스 (0-59999)
    - **filepath**: 이미지 파일 경로 (UPLOADS_DIR 기준)
    
    **Note**: index 또는 filepath 중 하나만 제공해야 합니다. 둘 다 제공된 경우 filepath가 우선됩니다.
    """
    logger.info(f"MNIST 이미지 읽기 요청 시작: index={index}, filepath={filepath}")
    try:
        # 파라미터 확인
        if index is None and filepath is None:
            logger.error("필수 파라미터 누락: index 또는 filepath가 제공되지 않음")
            return JSONResponse(
                content={"error": "index 또는 filepath 중 하나는 반드시 제공해야 합니다."},
                status_code=400
            )
            
        # 파일 경로 처리
        file_path = None
        if filepath:
            # uploads 디렉토리 기준 경로 구성
            file_path = os.path.join(UPLOAD_DIR, filepath)
            logger.info(f"요청 처리 경로: filepath={filepath}, 절대경로={file_path}")
            
            # 파일 존재 여부 확인
            if not os.path.exists(file_path):
                logger.error(f"파일이 존재하지 않음: {file_path}")
                
                # 업로드 디렉토리에 있는 파일 목록 확인
                available_files = [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]
                logger.info(f"업로드 디렉토리 내 파일 목록: {available_files}")
                
                return JSONResponse(
                    content={
                        "error": f"파일을 찾을 수 없습니다: {filepath}",
                        "message": "먼저 /upload 엔드포인트를 통해 파일을 업로드하세요.",
                        "upload_dir": UPLOAD_DIR,
                        "available_files": available_files
                    },
                    status_code=404
                )
            
            logger.info(f"파일 존재 확인 완료: {file_path}")
            # 파일 크기 확인
            file_size = os.path.getsize(file_path)
            logger.info(f"파일 크기: {file_size} bytes")
        else:
            logger.info(f"MNIST 데이터셋 인덱스 사용: {index}")
            
        # 컨트롤러 호출
        logger.info(f"컨트롤러 호출 전: index={index}, filepath={file_path}")
        image, label, success, error_msg = read_mnist_image(index=index, filepath=file_path)
        logger.info(f"컨트롤러 응답: success={success}, label={label}, error_msg={error_msg}")
        
        if not success:
            logger.error(f"이미지 읽기 실패: {error_msg}")
            return JSONResponse(
                content={"error": error_msg},
                status_code=400
            )
        
        # 이미지를 PNG로 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filepath:
            # 업로드된 파일 이름 사용
            filename_base = os.path.splitext(os.path.basename(filepath))[0]
            output_filename = f"{filename_base}_processed_{timestamp}.png"
        else:
            # MNIST 인덱스 사용
            label_str = f"_label{label}" if label is not None else ""
            output_filename = f"mnist_{index}{label_str}_{timestamp}.png"
            
        # 출력 디렉토리 및 파일 경로
        mnist_dir = os.path.join(OUTPUT_DIR, "mnist")
        os.makedirs(mnist_dir, exist_ok=True)
        output_path = os.path.join(mnist_dir, output_filename)
        logger.info(f"이미지 저장 경로: {output_path}")
        
        # 이미지 저장
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        if label is not None:
            plt.title(f"Label: {label}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        logger.info(f"이미지 저장 완료: {output_path}, 형태: {image.shape}")
        
        # 응답 반환
        response_data = {
            "success": True,
            "image_path": output_path,
            "label": label,
            "image_shape": image.shape,
            "source": "dataset" if filepath is None else "file"
        }
        logger.info(f"응답 데이터 준비 완료: {response_data}")
        return response_data
        
    except Exception as e:
        logger.error(f"MNIST 이미지 읽기 오류: {str(e)}")
        return JSONResponse(
            content={"error": f"MNIST 이미지 읽기 중 오류 발생: {str(e)}"},
            status_code=500
        )
        
# 새로운 엔드포인트: 이미지 파일 업로드 후 MNIST 형식으로 읽기
@router.post("/read")
async def upload_and_read_mnist_image(
    file: UploadFile = File(...),
):
    """
    이미지 파일을 업로드하고 바로 MNIST 형식(28x28)으로 변환하여 읽습니다.
    """
    logger.info(f"파일 업로드 및 MNIST 변환 요청 시작: filename={file.filename}")
    try:
        # 파일 업로드
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        logger.info(f"파일 저장 경로: {file_location}")
        
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"파일 업로드 성공: {file.filename}, 크기: {os.path.getsize(file_location)} bytes")
        
        # 컨트롤러 호출
        logger.info(f"컨트롤러 호출 전: filepath={file_location}")
        image, label, success, error_msg = read_mnist_image(filepath=file_location)
        logger.info(f"컨트롤러 응답: success={success}, error_msg={error_msg}")
        
        if not success:
            logger.error(f"이미지 처리 실패: {error_msg}")
            return JSONResponse(
                content={"error": error_msg},
                status_code=400
            )
        
        # 이미지를 PNG로 저장
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = os.path.splitext(file.filename)[0]
        output_filename = f"{filename_base}_mnist_format_{timestamp}.png"
        logger.info(f"출력 파일명: {output_filename}")
            
        # 출력 디렉토리 및 파일 경로
        mnist_dir = os.path.join(OUTPUT_DIR, "mnist")
        os.makedirs(mnist_dir, exist_ok=True)
        output_path = os.path.join(mnist_dir, output_filename)
        logger.info(f"출력 파일 경로: {output_path}")
        
        # 이미지 저장
        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        logger.info(f"MNIST 형식 이미지 저장 완료: {output_path}, 크기: {image.shape}")
        
        # 응답 반환
        response_data = {
            "success": True,
            "original_filename": file.filename,
            "uploaded_path": file_location,
            "processed_path": output_path,
            "image_shape": image.shape
        }
        logger.info(f"응답 데이터 준비 완료: {response_data}")
        return response_data
        
    except Exception as e:
        logger.error(f"파일 업로드 및 MNIST 변환 오류: {str(e)}")
        return JSONResponse(
            content={"error": f"파일 업로드 및 MNIST 변환 중 오류 발생: {str(e)}"},
            status_code=500
        )