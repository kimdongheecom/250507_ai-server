import pandas as pd
import numpy as np
import os
import traceback
import logging

logger = logging.getLogger(__name__)

def analyze_correlation(cctv_data, pop_data):
    """CCTV와 인구 데이터의 상관관계를 분석하는 함수"""
    try:
        # 데이터 확인
        if '자치구' in cctv_data.columns and '자치구' in pop_data.columns:
            merge_col = '자치구'
        elif '구별' in cctv_data.columns and '구별' in pop_data.columns:
            merge_col = '구별'
        else:
            # 컬럼명이 일치하지 않는 경우 - 첫 번째 컬럼을 기준으로 rename
            first_col_cctv = cctv_data.columns[0]
            first_col_pop = pop_data.columns[0]
            logger.info(f"컬럼명이 일치하지 않아 첫 번째 컬럼을 기준으로 병합합니다.")
            logger.info(f"CCTV 첫 번째 컬럼: {first_col_cctv}, 인구 첫 번째 컬럼: {first_col_pop}")
            
            # 첫 번째 컬럼을 '자치구'로 통일
            cctv_data = cctv_data.rename(columns={first_col_cctv: '자치구'})
            pop_data = pop_data.rename(columns={first_col_pop: '자치구'})
            merge_col = '자치구'
            
            logger.info(f"컬럼명 변경 후 CCTV 컬럼: {cctv_data.columns.tolist()}")
            logger.info(f"컬럼명 변경 후 인구 컬럼: {pop_data.columns.tolist()}")
        
        logger.info(f"데이터 병합에 사용할 컬럼: {merge_col}")
        
        # 인구 데이터 처리
        pop_data = pop_data.rename(columns={
            pop_data.columns[1]: '인구수',   
            pop_data.columns[2]: '한국인',
            pop_data.columns[3]: '외국인',
            pop_data.columns[4]: '고령자',
        })
        
        # 인덱스 26이 실제로 존재하는지 확인
        if 26 in pop_data.index:
            pop_data.drop([26], inplace=True)
        
        pop_data['외국인비율'] = pop_data['외국인'].astype(int) / pop_data['인구수'].astype(int) * 100
        pop_data['고령자비율'] = pop_data['고령자'].astype(int) / pop_data['인구수'].astype(int) * 100
        
        # 병합
        cctv_pop = pd.merge(cctv_data, pop_data, on=merge_col)
        logger.info(f"병합된 데이터 형태: {cctv_pop.shape}")
        logger.info(f"병합된 데이터 컬럼: {cctv_pop.columns.tolist()}")
        
        # 상관계수 계산
        corr1 = np.corrcoef(cctv_pop['고령자비율'], cctv_pop['소계'])
        corr2 = np.corrcoef(cctv_pop['외국인비율'], cctv_pop['소계'])
        
        logger.info(f'고령자비율과 CCTV의 상관계수: {corr1[0, 1]:.2f}')
        logger.info(f'외국인비율과 CCTV의 상관계수: {corr2[0, 1]:.2f}')
        
        return {
            '고령자비율_CCTV': corr1[0, 1],
            '외국인비율_CCTV': corr2[0, 1]
        }
    except Exception as e:
        logger.error(f"상관관계 분석 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def analyze_crime_correlation(cctv_data, crime_data, police_norm_data):
    """범죄 데이터와 CCTV의 상관관계를 분석하는 함수"""
    try:
        # 데이터 병합
        merged_data = pd.merge(cctv_data, crime_data, on='자치구')
        merged_data = pd.merge(merged_data, police_norm_data, on='자치구')
        
        # 상관계수 계산
        correlations = {}
        for col in ['살인', '강도', '강간', '절도', '폭력']:
            corr = np.corrcoef(merged_data['소계'], merged_data[col])[0, 1]
            correlations[f'CCTV_{col}'] = corr
            logger.info(f'CCTV와 {col}의 상관계수: {corr:.2f}')
        
        return correlations
    except Exception as e:
        logger.error(f"범죄 상관관계 분석 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_interpretation_text(corr, var1, var2):
    """상관계수에 대한 해석 텍스트를 반환하는 함수"""
    if abs(corr) < 0.3:
        return f"{var1}와 {var2}는 거의 상관관계가 없습니다."
    elif abs(corr) < 0.5:
        return f"{var1}와 {var2}는 약한 상관관계가 있습니다."
    elif abs(corr) < 0.7:
        return f"{var1}와 {var2}는 중간 정도의 상관관계가 있습니다."
    else:
        return f"{var1}와 {var2}는 강한 상관관계가 있습니다."

def load_and_analyze(data_dir='/app/stored-data'):
    """데이터를 로드하고 상관관계를 분석하는 함수"""
    try:
        # 데이터 로드
        cctv_data = pd.read_csv(os.path.join(data_dir, 'cctv_in_seoul.csv'))
        pop_data = pd.read_csv(os.path.join(data_dir, 'pop_in_seoul.csv'))
        crime_data = pd.read_csv(os.path.join(data_dir, 'crime_in_seoul.csv'))
        police_norm_data = pd.read_csv(os.path.join(data_dir, 'police_norm_in_seoul.csv'))
        
        # 상관관계 분석
        cctv_pop_corr = analyze_correlation(cctv_data, pop_data)
        crime_corr = analyze_crime_correlation(cctv_data, crime_data, police_norm_data)
        
        # 결과 통합
        results = {
            'cctv_pop_correlation': cctv_pop_corr,
            'crime_correlation': crime_corr,
            'interpretation': {
                '고령자비율_CCTV': get_interpretation_text(
                    cctv_pop_corr['고령자비율_CCTV'], '고령자비율', 'CCTV'
                ),
                '외국인비율_CCTV': get_interpretation_text(
                    cctv_pop_corr['외국인비율_CCTV'], '외국인비율', 'CCTV'
                )
            }
        }
        
        return results
    except Exception as e:
        logger.error(f"데이터 로드 및 분석 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise