import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from app.domain.model.data_schema import DataSchema
from scipy.stats import reciprocal


class ModelingService:
    """
    머신러닝 모델 학습 및 평가를 담당하는 서비스 클래스 (팩토리 패턴 적용)
    """

    @staticmethod
    def create_k_fold(this):
        """K-Fold 객체 생성"""
        X_train = this.train
        y_train = this.label
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        return X_train, y_train, k_fold

    @staticmethod
    def create_random_variable(this):
        """모델별 정확도 저장용 딕셔너리 생성"""
        return {
            'logistic_regression': [],
            'decision_tree': [],
            'random_forest': [],
            'naive_bayes': [],
            'knn': [],
            'svm': []
        }

    @staticmethod
    def create_model(model_name):
        """팩토리 패턴으로 모델 객체 생성"""
        model_factory = {
            '랜덤포레스트': RandomForestClassifier(n_estimators=100, random_state=0),
            '나이브베이즈': GaussianNB(),
            '베이스라인(로지스틱 회귀)': LogisticRegression(max_iter=3000, class_weight='balanced'),
            '결정트리': DecisionTreeClassifier(random_state=0),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=0, class_weight='balanced')
        }

        if model_name not in model_factory:
            raise ValueError(f"알 수 없는 모델명: {model_name}")
        
        scaler_needed = (model_name == 'SVM')  # SVM은 반드시 스케일링
        return model_factory[model_name], scaler_needed

    @staticmethod
    def accuracy_by_model(this, model_name):
        """주어진 모델명으로 교차 검증 정확도 계산"""
        X_train, y_train, k_fold = ModelingService.create_k_fold(this)
        model, scaler_needed = ModelingService.create_model(model_name)

        if scaler_needed:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

        scores = cross_val_score(model, X_train, y_train, cv=k_fold, scoring='accuracy')
        print(f"[{model_name}] 각 폴드 정확도: {scores}")
        print(f"[{model_name}] 평균 정확도: {scores.mean():.4f}, 표준편차: {scores.std():.4f}")
        return scores.mean()

    # 아래 기존 방식은 그대로 두어 Controller 호환성 유지

    @staticmethod
    def accuracy_by_logistic_regression(this):
        return ModelingService.accuracy_by_model(this, '베이스라인(로지스틱 회귀)')

    @staticmethod
    def accuracy_by_dtree(this):
        return ModelingService.accuracy_by_model(this, '결정트리')

    @staticmethod
    def accuracy_by_random_forest(this):
        return ModelingService.accuracy_by_model(this, '랜덤포레스트')

    @staticmethod
    def accuracy_by_naive_bayes(this):
        return ModelingService.accuracy_by_model(this, '나이브베이즈')

    @staticmethod
    def accuracy_by_knn(this):
        return ModelingService.accuracy_by_model(this, 'KNN')

    @staticmethod
    def accuracy_by_svm(this):
        return ModelingService.accuracy_by_model(this, 'SVM')

    @staticmethod
    def predict_with_model(this, model_name):
        """최적 모델로 예측"""
        model, scaler_needed = ModelingService.create_model(model_name)

        if scaler_needed:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(this.train)
            X_test_scaled = scaler.transform(this.test)
            model.fit(X_train_scaled, this.label)
            return model.predict(X_test_scaled)
        else:
            model.fit(this.train, this.label)
            return model.predict(this.test)
        

    @staticmethod
    def predict_with_voting(this):
        """랜덤포레스트 + SVM Voting 앙상블로 예측"""
        X_train = this.train
        y_train = this.label
        X_test = this.test

        # 스케일링은 SVM만 필요하므로 파이프라인 생성
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel='rbf', 
                C=10, 
                gamma='scale', 
                probability=True,  # Soft voting을 위해 필요
                random_state=0
            ))
        ])

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=0
        )

        # VotingClassifier 생성 (Soft Voting)
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('svm', svm_pipeline)
            ],
            voting='soft'
        )

        # 학습
        voting_clf.fit(X_train, y_train)

        # 예측
        predictions = voting_clf.predict(X_test)

        return predictions


    @staticmethod
    def tune_random_forest(this):
        """랜덤포레스트 모델 하이퍼파라미터 튜닝"""
        X_train = this.train
        y_train = this.label

        # 하이퍼파라미터 그리드 정의
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [5, 10, 20, None],
            'min_samples_split': [2, 5, 10]
        }

        rf = RandomForestClassifier(random_state=0)

        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,              # 5-Fold 교차검증
            scoring='accuracy', # 정확도 기준
            n_jobs=-1,          # CPU 모두 사용
            verbose=2           # 진행상황 보여줌
        )

        # 학습
        grid_search.fit(X_train, y_train)

        print(f"🏆 최적 파라미터: {grid_search.best_params_}")
        print(f"✅ 최적 교차검증 정확도: {grid_search.best_score_:.4f}")

        # 최적 모델 반환
        return grid_search.best_estimator_
    
    @staticmethod
    def tune_svm(this):
        """SVM 모델 하이퍼파라미터 튜닝"""
        X_train = this.train
        y_train = this.label

        # 파이프라인: 스케일링 + SVM
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(probability=True, random_state=0))
        ])

        # 탐색할 하이퍼파라미터 공간
        param_distributions = {
            'svc__C': reciprocal(1e-2, 1e2),  # 0.01 ~ 100 사이 로그 분포
            'svc__gamma': reciprocal(1e-4, 1e-1)  # 0.0001 ~ 0.1 사이 로그 분포
        }

        random_search = RandomizedSearchCV(
            pipe,
            param_distributions,
            n_iter=30,       # 30회 샘플링
            cv=5,            # 5-Fold 교차검증
            scoring='accuracy',
            n_jobs=-1,
            random_state=0,
            verbose=2
        )

        # 학습
        random_search.fit(X_train, y_train)

        print(f"🏆 최적 파라미터: {random_search.best_params_}")
        print(f"✅ 최적 교차검증 정확도: {random_search.best_score_:.4f}")

        # 최적 모델 반환
        return random_search.best_estimator_
    
    