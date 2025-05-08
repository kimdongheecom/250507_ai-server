# app/domain/controller/mnist_controller.py
from app.domain.service.mnist_service import MnistService
import logging

mnist_service = MnistService()

# 데이터 조회
def get_mnist_data(index: int):
    return mnist_service.get_mnist_data(index)

# 이미지 조회
def get_mnist_image(index: int, save: bool = False):
    return mnist_service.plot_mnist_image(index, save)

# 이미지 Base64 조회
def get_mnist_base64(index: int):
    return mnist_service.plot_mnist_base64(index)

# 데모 실행
def run_demo(index: int):
    return mnist_service.run_demo(index)

# MNIST 이미지 읽기 (데이터셋 또는 파일)
def read_mnist_image(index=None, filepath=None):
    """
    MNIST 이미지를 데이터셋에서 읽거나 파일에서 로드합니다.
    
    Args:
        index: MNIST 데이터셋의 인덱스
        filepath: 이미지 파일 경로
        
    Returns:
        (image, label, success, error_msg) 튜플
    """
   
    logger = logging.getLogger(__name__)
    
    logger.info(f"컨트롤러: MNIST 이미지 읽기 요청 - index={index}, filepath={filepath}")
    
    # 파라미터 검증
    if index is None and filepath is None:
        logger.error("컨트롤러: 필수 파라미터 누락 - index 또는 filepath가 제공되지 않음")
        return None, None, False, "index 또는 filepath 중 하나는 반드시 제공해야 합니다."
    
    if filepath is not None:
        logger.info(f"컨트롤러: 파일 경로로 처리 - {filepath}")
    else:
        logger.info(f"컨트롤러: 인덱스로 처리 - {index}")
    
    # 서비스 호출
    logger.info(f"컨트롤러: 서비스 호출 - index={index}, filepath={filepath}")
    result = mnist_service.read_mnist_image(index=index, filepath=filepath)
    image, label, success, error_msg = result
    
    if success:
        logger.info(f"컨트롤러: 서비스 호출 성공 - label={label}, image_shape={image.shape if image is not None else None}")
    else:
        logger.error(f"컨트롤러: 서비스 호출 실패 - {error_msg}")
    
    return result