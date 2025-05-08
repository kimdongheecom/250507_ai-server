import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import base64
from io import BytesIO
import cv2
import logging

logger = logging.getLogger(__name__)

class MnistService:
    def __init__(self):
        # 절대 경로 설정
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(BASE_DIR, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # MNIST 데이터 로드
        self.mnist = keras.datasets.mnist
        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = self.mnist.load_data()

    def get_mnist_data(self, index=100):
        """지정된 인덱스의 MNIST 데이터 정보를 반환합니다."""
        return {
            'label': int(self.train_labels[index]),
            'image': self.train_images[index].tolist()  # numpy.ndarray → list 변환
        }
    
    def read_mnist_image(self, index=None, filepath=None):
        """
        MNIST 이미지를 반환합니다.
        
        Args:
            index: MNIST 데이터셋의 인덱스 (기본값: 100)
            filepath: 외부 이미지 파일 경로 (지정된 경우 index는 무시됨)
            
        Returns:
            이미지가 성공적으로 로드된 경우:
                - dataset 사용 시: (image, label, True, "")
                - 파일 사용 시: (image, None, True, "")
            실패한 경우:
                - (None, None, False, error_message)
        """
        logger.info(f"서비스: MNIST 이미지 읽기 요청 - index={index}, filepath={filepath}")
        
        # 외부 파일이 지정된 경우
        if filepath is not None:
            try:
                # 이미지 파일 존재 확인
                if not os.path.exists(filepath):
                    logger.error(f"서비스: 파일이 존재하지 않습니다: {filepath}")
                    return None, None, False, f"파일이 존재하지 않습니다: {filepath}"
                
                logger.info(f"서비스: 파일 존재 확인 완료 - {filepath}")
                
                # 이미지 파일 읽기
                image = cv2.imread(filepath)
                if image is None:
                    logger.error(f"서비스: 이미지를 읽을 수 없습니다: {filepath}")
                    return None, None, False, f"이미지를 읽을 수 없습니다: {filepath}"
                
                logger.info(f"서비스: 이미지 로드 완료 - 크기: {image.shape}")
                
                # 컬러 이미지인 경우 그레이스케일로 변환
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    logger.info(f"서비스: 컬러 이미지를 그레이스케일로 변환 - 새 크기: {image.shape}")
                
                # 이미지 크기 조정 (28x28)
                resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
                logger.info(f"서비스: 이미지 크기 조정 완료 - 원본: {image.shape}, 조정 후: {resized_image.shape}")
                
                # 픽셀값 정규화 (0-255 범위로)
                if resized_image.max() > 1.0:
                    # MNIST 형식과 일치하도록 반전시킬 수 있음 (선택 사항)
                    # resized_image = 255 - resized_image  # 배경은 검은색, 숫자는 흰색으로
                    pass
                
                logger.info(f"서비스: 외부 이미지 파일 처리 완료: {filepath}")
                return resized_image, None, True, ""
                
            except Exception as e:
                logger.error(f"서비스: 이미지 처리 중 오류 발생: {str(e)}")
                return None, None, False, f"이미지 처리 중 오류 발생: {str(e)}"
        
        # MNIST 데이터셋 사용
        else:
            if index is None:
                index = 100  # 기본값
                logger.info(f"서비스: 인덱스가 지정되지 않아 기본값 사용: {index}")
                
            try:
                if index < 0 or index >= len(self.train_images):
                    logger.error(f"서비스: 유효하지 않은 인덱스: {index}, 범위: 0-{len(self.train_images)-1}")
                    return None, None, False, f"유효하지 않은 인덱스: {index}, 범위: 0-{len(self.train_images)-1}"
                
                image = self.train_images[index]
                label = int(self.train_labels[index])
                logger.info(f"서비스: MNIST 데이터셋에서 인덱스 {index}의 이미지 로드 완료 (레이블: {label})")
                return image, label, True, ""
                
            except Exception as e:
                logger.error(f"서비스: MNIST 이미지 로딩 중 오류 발생: {str(e)}")
                return None, None, False, f"MNIST 이미지 로딩 중 오류 발생: {str(e)}"

    def print_mnist_info(self, index=100):
        """지정된 인덱스의 MNIST 데이터 정보를 출력합니다."""
        print('[label]')
        print('number label = ', self.train_labels[index])
        print('\n[image]')
        for row in self.train_images[index]:
            for col in row:
                print("%10f" % col, end="")
            print('\n')

    def plot_mnist_image(self, index=100, save: bool = False) -> str:
        """지정된 인덱스의 MNIST 이미지를 시각화합니다. 필요시 파일로 저장."""
        image = self.train_images[index]
        label = self.train_labels[index]

        plt.figure(figsize=(5, 5))
        plt.imshow(image, cmap='gray')
        plt.axis("off")
        plt.title(f"Label: {label}")

        if save:
            filename = f"mnist_{index}_label_{label}.png"
            path = os.path.join(self.output_dir, filename)
            plt.savefig(path)
            plt.close()
            return path
        else:
            plt.show()
            return "이미지 시각화 완료"

    def plot_mnist_base64(self, index=100) -> str:
        """지정된 인덱스의 MNIST 이미지를 base64 인코딩된 문자열로 반환"""
        image = self.train_images[index]
        label = self.train_labels[index]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image, cmap='gray')
        ax.axis("off")
        ax.set_title(f"Label: {label}")

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return image_base64

    def run_demo(self, index=100):
        """기존 print + plot(show) 실행"""
        self.print_mnist_info(index)
        self.plot_mnist_image(index)

# 사용 예시
if __name__ == "__main__":
    service = MnistService()
    service.run_demo(100)