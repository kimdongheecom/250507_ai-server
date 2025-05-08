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
        
        # MNIST 모델 로드 또는 생성
        self.model = self._load_or_create_model()
        logger.info("MNIST 모델 초기화 완료")

    def _load_or_create_model(self):
        """MNIST 분류 모델을 로드하거나 새로 생성합니다."""
        # 모델 저장 경로
        model_path = os.path.join(self.output_dir, 'mnist_model.h5')
        
        # 저장된 모델이 있으면 로드
        if os.path.exists(model_path):
            logger.info(f"저장된 모델 로드: {model_path}")
            return keras.models.load_model(model_path)
        
        # 없으면 새로 모델 생성 및 학습
        logger.info("새 MNIST 모델 생성 및 학습 시작")
        
        # 데이터 정규화
        train_images = self.train_images / 255.0
        test_images = self.test_images / 255.0
        
        # 간단한 CNN 모델 구성
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(28, 28, 1)),
            keras.layers.Reshape(target_shape=(28, 28, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # 모델 컴파일
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 모델 학습
        train_images = train_images.reshape((-1, 28, 28, 1))
        test_images = test_images.reshape((-1, 28, 28, 1))
        
        model.fit(
            train_images, self.train_labels,
            epochs=3,
            batch_size=128,
            validation_data=(test_images, self.test_labels),
            verbose=1
        )
        
        # 모델 평가
        test_loss, test_acc = model.evaluate(test_images, self.test_labels, verbose=2)
        logger.info(f"모델 정확도: {test_acc:.4f}")
        
        # 모델 저장
        model.save(model_path)
        logger.info(f"학습된 모델 저장: {model_path}")
        
        return model

    def predict_digit(self, image):
        """이미지에서 숫자를 예측합니다."""
        if image is None:
            logger.error("예측할 이미지가 없습니다.")
            return None, 0.0
            
        # 이미지 전처리 (정규화 및 차원 변경)
        processed_image = image.astype('float32') / 255.0
        processed_image = processed_image.reshape(1, 28, 28, 1)
        
        # 예측 수행
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_digit = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_digit])
        
        # numpy.int64를 기본 Python int로 변환 (JSON 직렬화 가능하도록)
        predicted_digit = int(predicted_digit)
        
        logger.info(f"예측 결과: 숫자 {predicted_digit}, 확률: {confidence:.4f}")
        return predicted_digit, confidence

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
                - 파일 사용 시: (image, predicted_digit, True, "")
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
                
                # 숫자 예측
                predicted_digit, confidence = self.predict_digit(resized_image)
                logger.info(f"서비스: 숫자 인식 결과 - 숫자: {predicted_digit}, 확률: {confidence:.4f}")
                
                logger.info(f"서비스: 외부 이미지 파일 처리 완료: {filepath}")
                return resized_image, predicted_digit, True, ""
                
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
                
                # 데이터셋 이미지도 예측해보기 (레이블과 비교용)
                predicted_digit, confidence = self.predict_digit(image)
                logger.info(f"서비스: 데이터셋 이미지 예측 - 레이블: {label}, 예측: {predicted_digit}, 확률: {confidence:.4f}")
                
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