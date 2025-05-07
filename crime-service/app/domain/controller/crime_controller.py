from app.domain.service.crime_prepocessor import CrimePrepocessor
from app.domain.service.map_visualizer import MapVisualizer
from app.domain.model.crime_schema import CrimeSchema
from app.domain.service.crime_prepocessor import CrimePrepocessor

class CrimeController:
    def __init__(self):
        self.dataset = CrimeSchema()
        self.preprocessor = CrimePrepocessor()
        self.visualizer = MapVisualizer()
        self.correlation_service = CrimePrepocessor()

    def preprocess(self, *args):
        """데이터 전처리를 수행합니다."""
        self.preprocessor.preprocess(*args)

    def correlation(self):
        """상관계수 분석을 수행합니다."""
        print("Controller: Calling correlation_service.load_and_analyze...")
        results = self.correlation_service.load_and_analyze()
        print("Controller: Correlation analysis completed")
        return results
    
    def get_correlation_results(self):
        """상관계수 분석 결과를 반환합니다."""
        return self.correlation()

    def draw_crime_map(self):
        """범죄 지도를 생성합니다."""
        result = self.visualizer.draw_crime_map()
        if result.get("status") == "success":
            print(f"Controller: Crime map created successfully at {result.get('file_path')}")
        else:
            print(f"Controller: Failed to create crime map - {result.get('message')}")
        return result

        