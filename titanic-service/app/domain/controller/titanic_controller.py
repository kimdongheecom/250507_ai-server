from app.domain.service.titanic_service import TitanicService
from app.domain.service.modelling_service import ModelingService
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class Controller :
     
    service = TitanicService()
    modeling_service = ModelingService()
    '''
    print(f'결정트리 활용한 검증 정확도 {None}')
    print(f'랜덤포레스트 활용한 검증 정확도 {None}')
    print(f'나이브베이즈 활용한 검증 정확도 {None}')
    print(f'KNN 활용한 검증 정확도 {None}')
    print(f'SVM 활용한 검증 정확도 {None}')
    '''

    def preprocess(self, train_fname, test_fname): #데이터 전처리
        return self.service.preprocess(train_fname, test_fname)
    
    def learning(self): #모델 학습
        """
        여러 머신러닝 모델을 학습하고 각 모델의 정확도를 계산
        
        Returns:
            dict: 모델별 정확도를 담은 사전
        """
        print("\n" + "="*50)
        print("📊 타이타닉 생존자 예측 모델 학습 시작 📊".center(50))
        print("="*50)
        this = self.service.dataset
        
        # 베이스라인 모델(로지스틱 회귀) 정확도 계산
        print("\n🔍 베이스라인 모델(로지스틱 회귀) 학습 중...")
        baseline_accuracy = self.modeling_service.accuracy_by_logistic_regression(this)
        print(f"✅ 베이스라인 모델 정확도: {baseline_accuracy:.4f}")
        
        # 각 알고리즘별 정확도 계산
        accuracy = {}
        accuracy['베이스라인(로지스틱 회귀)'] = baseline_accuracy
        
        print("\n🔍 의사결정 트리 모델 학습 중...")
        tree_acc = self.modeling_service.accuracy_by_dtree(this)
        accuracy['결정트리'] = tree_acc
        if tree_acc > baseline_accuracy:
            print(f"✅ 의사결정 트리 정확도: {tree_acc:.4f} (베이스라인 대비: {tree_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ 의사결정 트리 정확도: {tree_acc:.4f} (베이스라인 대비: {tree_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 랜덤 포레스트 모델 학습 중...")
        rf_acc = self.modeling_service.accuracy_by_random_forest(this)
        accuracy['랜덤포레스트'] = rf_acc
        if rf_acc > baseline_accuracy:
            print(f"✅ 랜덤 포레스트 정확도: {rf_acc:.4f} (베이스라인 대비: {rf_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ 랜덤 포레스트 정확도: {rf_acc:.4f} (베이스라인 대비: {rf_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 나이브 베이즈 모델 학습 중...")
        nb_acc = self.modeling_service.accuracy_by_naive_bayes(this)
        accuracy['나이브베이즈'] = nb_acc
        if nb_acc > baseline_accuracy:
            print(f"✅ 나이브 베이즈 정확도: {nb_acc:.4f} (베이스라인 대비: {nb_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ 나이브 베이즈 정확도: {nb_acc:.4f} (베이스라인 대비: {nb_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 K-최근접 이웃 모델 학습 중...")
        knn_acc = self.modeling_service.accuracy_by_knn(this)
        accuracy['KNN'] = knn_acc
        if knn_acc > baseline_accuracy:
            print(f"✅ KNN 정확도: {knn_acc:.4f} (베이스라인 대비: {knn_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ KNN 정확도: {knn_acc:.4f} (베이스라인 대비: {knn_acc - baseline_accuracy:+.4f})")
        
        print("\n🔍 서포트 벡터 머신 모델 학습 중...")
        svm_acc = self.modeling_service.accuracy_by_svm(this)
        accuracy['SVM'] = svm_acc
        if svm_acc > baseline_accuracy:
            print(f"✅ SVM 정확도: {svm_acc:.4f} (베이스라인 대비: {svm_acc - baseline_accuracy:+.4f})")
        else:
            print(f"❌ SVM 정확도: {svm_acc:.4f} (베이스라인 대비: {svm_acc - baseline_accuracy:+.4f})")
        
        # 결과 저장
        this.accuracy = accuracy
        
        return accuracy

    def evaluation(self): #모델 평가
        """
        학습된 모델들의 정확도를 비교하여 최적의 모델 선택
        
        Returns:
            tuple: 가장 높은 정확도와 해당 모델 이름
        """
        print("\n" + "="*50)
        print("📊 타이타닉 생존자 예측 모델 평가 결과 📊".center(50))
        print("="*50)
        this = self.service.dataset
        
        if not hasattr(this, 'accuracy'):
            print("❌ 모델이 학습되지 않았습니다. 먼저 learning() 메소드를 실행하세요.")
            return None, None
        
        # 최고 성능 모델 찾기
        best_model = max(this.accuracy.items(), key=lambda x: x[1])
        
        print(f"\n🏆 최고 성능 모델: {best_model[0]}, 정확도: {best_model[1]:.4f}")
        print("\n📋 모델별 정확도 비교:")
        
        # 베이스라인과 비교한 성능 향상 출력
        baseline_accuracy = this.accuracy.get('베이스라인(로지스틱 회귀)', 0)
        
        # 정렬된 결과 표시
        sorted_results = sorted(this.accuracy.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "-"*60)
        print(f"{'모델명':<20} {'정확도':<10} {'베이스라인 대비':<15} {'평가'}")
        print("-"*60)
        
        for i, (model, acc) in enumerate(sorted_results):
            if model == '베이스라인(로지스틱 회귀)':
                if i == 0:  # 베이스라인이 가장 좋은 경우
                    print(f"{model:<20} {acc:.4f}      {'---':<15} {'🏆 최고'}")
                else:
                    print(f"{model:<20} {acc:.4f}      {'---':<15} {'📊 기준'}")
            else:
                improvement = acc - baseline_accuracy
                status = ""
                if i == 0:
                    status = "🏆 최고"
                elif improvement > 0:
                    status = "✅ 개선"
                else:
                    status = "❌ 저조"
                    
                print(f"{model:<20} {acc:.4f}      {improvement:+.4f}        {status}")
        
        print("-"*60)
        
        this.best_model = best_model[0]
        this.best_accuracy = best_model[1]
        
        return best_model[1], best_model[0]
        
    def submit(self): #모델 배포
        """
        최적의 모델을 사용하여 테스트 데이터에 대한 예측 결과를 생성하고 CSV 파일로 저장
        
        Returns:
            str: 제출 파일 생성 결과 메시지
        """
        print("\n" + "="*50)
        print("📊 타이타닉 생존자 예측 결과 제출 준비 📊".center(50))
        print("="*50)
        this = self.service.dataset
        
        if not hasattr(this, 'best_model'):
            print("❌ 모델 평가가 완료되지 않았습니다. 먼저 evaluation() 메소드를 실행하세요.")
            return None
        
        print(f"\n🏆 {this.best_model} 모델로 제출 파일 생성 (정확도: {this.best_accuracy:.4f})")
        
        try:
            # 서비스 레이어에서 예측 수행
            prediction = self.modeling_service.predict_with_voting(this)
            
            # 제출용 데이터프레임 생성
            submission_df = pd.DataFrame({
                'PassengerId': this.id,
                'Survived': prediction
            })
            
            # CSV 파일로 저장
            submission_path = './update-data/submission.csv'
            submission_df.to_csv(submission_path, index=False)
            
            print(f"📋 제출 파일이 생성되었습니다: {submission_path}")
            
            return f"✅ submission.csv 생성 완료 (정확도: {this.best_accuracy:.4f})"
            
        except ValueError as e:
            print(f"❌ 예측 중 오류 발생: {str(e)}")
            return None
        
        
    def tune(self):
        """
        RandomForest 모델 하이퍼파라미터 튜닝 및 제출 파일 생성
        """
        print("\n" + "="*50)
        print("🎯 RandomForest 튜닝 및 제출 파일 생성 시작 🎯".center(50))
        print("="*50)

        this = self.service.dataset

        best_rf = self.modeling_service.tune_random_forest(this)
        best_rf.fit(this.train, this.label)
        predictions = best_rf.predict(this.test)

        submission = pd.DataFrame({
            "PassengerId": this.id,
            "Survived": predictions
        })
        submission_path = './update-data/submission_tuned.csv'
        submission.to_csv(submission_path, index=False)

        print(f"✅ 튜닝된 RandomForest 결과 저장 완료! 파일 위치: {submission_path}")

    def tune_svm(self):
        """
        SVM 모델 하이퍼파라미터 튜닝 및 제출 파일 생성
        """
        print("\n" + "="*50)
        print("🎯 SVM 튜닝 및 제출 파일 생성 시작 🎯".center(50))
        print("="*50)

        this = self.service.dataset

        best_svm = self.modeling_service.tune_svm(this)
        best_svm.fit(this.train, this.label)
        predictions = best_svm.predict(this.test)

        submission = pd.DataFrame({
            "PassengerId": this.id,
            "Survived": predictions
        })
        submission_path = './update-data/submission_svm_tuned.csv'
        submission.to_csv(submission_path, index=False)

        print(f"✅ 튜닝된 SVM 결과 저장 완료! 파일 위치: {submission_path}")


    def tune_voting(self):
        """
        튜닝된 RandomForest와 튜닝된 SVM을 이용한 Voting 앙상블 및 제출 파일 생성
        """
        print("\n" + "="*50)
        print("🎯 RandomForest + SVM Voting 앙상블 제출 파일 생성 시작 🎯".center(50))
        print("="*50)

        this = self.service.dataset

        # 튜닝된 랜덤포레스트 모델 가져오기
        best_rf = self.modeling_service.tune_random_forest(this)

        # 튜닝된 SVM 모델 가져오기
        best_svm = self.modeling_service.tune_svm(this)

        # 앙상블 모델 생성
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', best_rf),
                ('svm', best_svm)
            ],
            voting='soft'  # 소프트 보팅
        )

        # 학습
        voting_clf.fit(this.train, this.label)

        # 예측
        predictions = voting_clf.predict(this.test)

        # 제출 파일 생성
        submission = pd.DataFrame({
            "PassengerId": this.id,
            "Survived": predictions
        })

        submission_path = './update-data/submission_voting.csv'
        submission.to_csv(submission_path, index=False)

        print(f"✅ Voting 앙상블 결과 저장 완료! 파일 위치: {submission_path}")

    def feature_importance(self):
        """
        RandomForest를 사용하여 Feature Importance 출력
        """
        print("\n" + "="*50)
        print("📊 Feature Importance 출력 시작 📊".center(50))
        print("="*50)

        this = self.service.dataset

        # 튜닝된 RandomForest 모델 가져오기
        best_rf = self.modeling_service.tune_random_forest(this)
        best_rf.fit(this.train, this.label)

        # Feature Importance 추출
        importances = best_rf.feature_importances_
        feature_names = this.train.columns

        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print("\n📋 Feature Importance 분석 결과:")
        print(feature_importance_df)

        # 필요하면 나중에 반환도 가능
        return feature_importance_df
    