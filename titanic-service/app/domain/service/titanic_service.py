import pandas as pd 
import numpy as np
from app.domain.model.data_schema import DataSchema

class TitanicService:
    
    dataset = DataSchema()  

    def new_model(self, fname) -> pd.DataFrame: 
        context = './stored-data/'
        return pd.read_csv(context + fname)
    
    def preprocess(self, train_fname, test_fname) -> object: 
        print("------모델 전처리 시작------")
        train_df = self.new_model(train_fname)
        test_df = self.new_model(test_fname)
        
        self.dataset.id = test_df['PassengerId']
        y_train = train_df['Survived']
        train_df = train_df.drop('Survived', axis=1)

        # 기본 파생변수 생성
        train_df, test_df = self.add_basic_features(train_df, test_df)
        
        # 불필요한 특성 제거
        drop_features = ['SibSp', 'Parch', 'Ticket', 'Cabin']
        train_df, test_df = self.drop_feature(train_df, test_df, *drop_features)

        # 이름에서 타이틀 추출 및 처리
        train_df, test_df = self.extract_title_from_name(train_df, test_df)
        title_mapping = self.remove_duplicate_title(train_df, test_df)
        train_df, test_df = self.title_nominal(train_df, test_df, title_mapping)
        train_df, test_df = self.drop_feature(train_df, test_df, 'Name')

        # 성별 변환 및 원본 컬럼 제거
        train_df, test_df = self.gender_nominal(train_df, test_df)
        train_df, test_df = self.drop_feature(train_df, test_df, 'Sex')

        # 승선 항구 변환
        train_df, test_df = self.embarked_nominal(train_df, test_df)

        # 나이 처리
        train_df, test_df = self.age_ratio(train_df, test_df)
        # Age 컬럼 원본은 유지 (drop 안함)

        # 클래스 서수화
        train_df, test_df = self.pclass_ordinal(train_df, test_df)

        # 요금 변환 및 그룹화
        train_df, test_df = self.fare_processing(train_df, test_df)
        
        # 추가 특성 생성
        train_df, test_df = self.add_interaction_features(train_df, test_df)

        self.dataset.train = train_df
        self.dataset.test = test_df
        self.dataset.label = y_train
        
        return self.dataset

    @staticmethod
    def drop_feature(train_df, test_df, *features) -> tuple: 
        for feature in features:
            train_df = train_df.drop(feature, axis=1)
            test_df = test_df.drop(feature, axis=1)
        return train_df, test_df

    @staticmethod
    def remove_duplicate_title(train_df, test_df):
        titles = set(train_df['Title'].unique()) | set(test_df['Title'].unique())
        print("타이틀 종류:", titles)
        title_mapping = {'Mr': 1, 'Ms': 2, 'Mrs': 3, 'Master': 4, 'Royal': 5, 'Rare': 6}
        return title_mapping

    @staticmethod
    def extract_title_from_name(train_df, test_df):
        train_df['Title'] = train_df['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
        test_df['Title'] = test_df['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)
        return train_df, test_df

    @staticmethod
    def title_nominal(train_df, test_df, title_mapping):
        for df in [train_df, test_df]:
            df['Title'] = df['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            df['Title'] = df['Title'].replace(['Capt','Col','Don','Dr','Major','Rev','Jonkheer','Dona','Mme'], 'Rare')
            df['Title'] = df['Title'].replace(['Mlle'], 'Mr')
            df['Title'] = df['Title'].replace(['Miss'], 'Ms')
            df['Title'] = df['Title'].fillna('Rare')
            df['Title'] = df['Title'].map(title_mapping)
        return train_df, test_df

    @staticmethod
    def gender_nominal(train_df, test_df):
        train_df['Gender'] = train_df['Sex'].map({'male': 0, 'female': 1})
        test_df['Gender'] = test_df['Sex'].map({'male': 0, 'female': 1})
        return train_df, test_df

    @staticmethod
    def embarked_nominal(train_df, test_df):
        for df in [train_df, test_df]:
            df['Embarked'] = df['Embarked'].fillna('S')
            df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
        return train_df, test_df

    @staticmethod
    def age_ratio(train_df, test_df): 
        age_mapping = {'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4,
                       'Young Adult': 5, 'Adult': 6, 'Senior': 7}
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

        for df in [train_df, test_df]:
            df['Age'] = df['Age'].fillna(df['Age'].median())
            df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
            df['AgeGroup'] = df['AgeGroup'].map(age_mapping)
        return train_df, test_df

    @staticmethod
    def pclass_ordinal(train_df, test_df):
        return train_df, test_df

    @staticmethod
    def add_basic_features(train_df, test_df):
        for df in [train_df, test_df]:
            df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
            df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
            df['FamilySizeGroup'] = df['FamilySize'].apply(lambda x: 1 if x == 1 else (2 if x <= 4 else 3))
            df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
        return train_df, test_df

    @staticmethod
    def fare_processing(train_df, test_df):
        for df in [train_df, test_df]:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df['Fare_log'] = np.log1p(df['Fare'])  # log(1+Fare)
            df['FareGroup'] = pd.qcut(df['Fare'], 4, labels=[1, 2, 3, 4])
        return train_df, test_df

    @staticmethod
    def add_interaction_features(train_df, test_df):
        for df in [train_df, test_df]:
            df['Pclass_Fare'] = df['Pclass'] * df['Fare_log']
        return train_df, test_df


    

    

