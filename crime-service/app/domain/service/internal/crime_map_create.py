import os
import json
import pandas as pd
import numpy as np
import folium
from fastapi import HTTPException
import logging
import traceback

logger = logging.getLogger(__name__)

def create_map(data_dir=None, output_dir=None) -> str:
        """범죄 지도를 생성하고 저장된 파일 경로를 반환합니다."""
        try:
            # 상대 경로 설정
            if data_dir is None:
                data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'update_data')
            if output_dir is None:
                output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'stored-map')
                
                logger.info("범죄 지도 생성 시작...")
            police_norm, state_geo = _load_required_data(data_dir)
            folium_map = _create_folium_map(police_norm, state_geo)
            output_map_file = os.path.join(output_dir, 'crime_map.html')
            _save_map_html(folium_map, output_map_file)
            logger.info(f"범죄 지도가 성공적으로 생성되었습니다: {output_map_file}")
            return output_map_file
        except FileNotFoundError as e:
            logger.error(f"필수 파일 로드 실패: {e}")
            raise HTTPException(status_code=404, detail=f"필수 데이터 파일을 찾을 수 없습니다: {e}")
        except KeyError as e:
             logger.error(f"데이터 처리 중 필수 컬럼 부재: {e}")
             raise HTTPException(status_code=400, detail=f"데이터 처리 중 필요한 컬럼({e})이 없습니다.")
        except ValueError as e:
             logger.error(f"데이터 처리 중 값 또는 형식 오류: {e}")
             raise HTTPException(status_code=400, detail=f"데이터 처리 중 오류 발생: {e}")
        except Exception as e:
            logger.error(f"지도 생성 중 예상치 못한 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"지도 생성 중 서버 오류 발생: {type(e).__name__} - {str(e)}")

def _load_required_data(data_dir):
        """지도 생성에 필요한 데이터를 로드하고 기본적인 핸들링을 수행합니다."""
        logger.info("필수 데이터 로드 중...")

            # police_norm 데이터 로드
        police_norm_file = os.path.join(data_dir, 'police_norm_in_seoul.csv')
        if not os.path.exists(police_norm_file):
            raise FileNotFoundError(police_norm_file)
        
            try:
                police_norm = pd.read_csv(police_norm_file)
                logger.info(f"{police_norm_file} 파일 로드 완료")
                police_norm = _preprocess_police_norm(police_norm)
            except Exception as e:
                logger.error(f"{police_norm_file} 파일 처리 중 오류: {e}")
            raise ValueError(f"{police_norm_file} 파일을 처리하는 중 오류가 발생했습니다: {e}")

        # GeoJSON 데이터 로드 (stored-data 폴더에서)
        stored_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'stored-data')
        geo_json_file = os.path.join(stored_data_dir, 'geo_simple.json')
        if not os.path.exists(geo_json_file):
            raise FileNotFoundError(geo_json_file)
        
            try:
                with open(geo_json_file, 'r', encoding='utf-8') as f:
                    state_geo = json.load(f)
                logger.info(f"{geo_json_file} 파일 로드 완료")
            except json.JSONDecodeError as e:
                logger.error(f"{geo_json_file} 파일 로드 중 JSON 디코딩 오류: {e}")
                raise ValueError(f"{geo_json_file} 파일의 형식이 올바르지 않습니다.")
            except Exception as e:
                logger.error(f"{geo_json_file} 파일 처리 중 오류: {e}")
            raise ValueError(f"{geo_json_file} 파일을 처리하는 중 오류가 발생했습니다: {e}")

            return police_norm, state_geo

def _preprocess_police_norm(police_norm_df: pd.DataFrame) -> pd.DataFrame:
    """police_norm 데이터를 전처리합니다."""
    logger.info("police_norm 데이터 전처리 중...")
    
    # 필수 컬럼 존재 여부 확인
    required_cols = ['자치구', '범죄']
    for col in required_cols:
        if col not in police_norm_df.columns:
            if col == '범죄' and '범죄율' in police_norm_df.columns:
                    police_norm_df['범죄'] = police_norm_df['범죄율']
                    logger.info("컬럼명 변경 시도: '범죄율' -> '범죄'")
            else:
                logger.error(f"police_norm 데이터에 필수 컬럼 '{col}'이 없습니다.")
                raise KeyError(f"police_norm 데이터에 필수 컬럼 '{col}'이 없습니다.")

        logger.info(f"police_norm 데이터 전처리 완료. 컬럼: {police_norm_df.columns.tolist()}")
        return police_norm_df

def _create_folium_map(police_norm, state_geo):
    """Folium을 사용하여 지도를 생성합니다."""
    logger.info("Folium 지도 생성 중...")
    
    # 기본 지도 생성
    folium_map = folium.Map(location=[37.5502, 126.982], zoom_start=12, tiles='OpenStreetMap')

    # 범죄율 Choropleth 레이어 추가
    crime_choropleth = folium.Choropleth(
                geo_data=state_geo,
                data=police_norm,
                columns=['자치구', '범죄'],
                key_on='feature.id',
                fill_color='YlOrRd',
        fill_opacity=0.6,
        line_opacity=0.8,
        line_weight=2,
        legend_name='자치구별 범죄 지수',
        name='범죄율',
        reset=True
    ).add_to(folium_map)

    # 스타일 콜백 함수 정의
    style_function = lambda x: {
        'fillColor': '#00000000',
        'color': '#000000',
        'weight': 2,
        'fillOpacity': 0
    }

    # 구 경계선 레이어 추가
    folium.GeoJson(
        state_geo,
        style_function=style_function,
        name='구 경계선'
            ).add_to(folium_map)

    # 모든 자치구에 마커와 정보 추가
    for feature in state_geo['features']:
        gu_name = feature['id']
        district_data = police_norm[police_norm['자치구'].str.strip() == gu_name]
        
        if not district_data.empty:
            district_data = district_data.iloc[0]
            
            # 범죄 유형별 검거율 정보 생성
            crime_info = "<br>".join([
                f"<h4>{gu_name}</h4>",
                f"살인 검거율: {district_data['살인검거율']:.1f}%",
                f"강도 검거율: {district_data['강도검거율']:.1f}%",
                f"강간 검거율: {district_data['강간검거율']:.1f}%",
                f"절도 검거율: {district_data['절도검거율']:.1f}%",
                f"폭력 검거율: {district_data['폭력검거율']:.1f}%",
                f"<br><b>범죄 지수: {district_data['범죄']:.2f}</b>",
                f"<b>검거 지수: {district_data['검거']:.2f}</b>"
            ])
            
            # 구 중심점 좌표 계산
            coords = feature['geometry']['coordinates'][0]
            if coords:
                center_lat = sum(coord[1] for coord in coords) / len(coords)
                center_lng = sum(coord[0] for coord in coords) / len(coords)
                
                # 범죄율과 검거율 계산
                crime_value = district_data['범죄']
                arrest_value = district_data['검거']
                
                # 범죄율과 검거율 정규화 (0~1)
                crime_weight = crime_value / police_norm['범죄'].max()
                arrest_weight = arrest_value / police_norm['검거'].max()
                
                # 범죄율에 더 큰 가중치(0.7)를 주고, 검거율은 작은 가중치(0.3)로 반영
                # 범죄율이 높을수록, 검거율이 낮을수록 큰 값이 나옴
                size_ratio = (crime_weight * 0.7) * (1 + (1 - arrest_weight) * 0.3)
                size_factor = max(10, min(50, size_ratio * 70))  # 10~50 사이로 조정
                
                # 아이콘 색상 결정
                if crime_weight > 0.7:  # 범죄율이 매우 높음
                    icon_color = 'red'
                elif crime_weight > 0.4:  # 범죄율이 중간
                    icon_color = 'orange'
                else:  # 범죄율이 낮음
                    icon_color = 'green'
                
                # Circle 사용 (zoom level에 영향받지 않는 고정 크기)
                folium.Circle(
                    location=[center_lat, center_lng],
                    radius=size_factor * 10,  # 작은 크기 유지
                    popup=folium.Popup(crime_info, max_width=200),
                    tooltip=f"{gu_name}",
                    color=icon_color,
                    fill=True,
                    fill_color=icon_color,
                    fill_opacity=0.7,
                    weight=2
                ).add_to(folium_map)

    # 레이어 컨트롤 추가
        folium.LayerControl().add_to(folium_map)

        return folium_map

def _save_map_html(folium_map, output_file):
        """생성된 Folium 지도를 HTML 파일로 저장합니다."""
        try:
            logger.info(f"생성된 지도를 HTML 파일로 저장 중: {output_file}")
        # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            folium_map.save(output_file)
            logger.info("지도 저장 완료.")
        except Exception as e:
            logger.error(f"지도 저장 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
        raise IOError(f"지도를 HTML 파일로 저장하는 데 실패했습니다: {str(e)}")

# 사용 예시 (테스트용)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     creator = CrimeMapCreator()
#     try:
#         map_file_path = creator.create_map()
#         print(f"지도 생성 완료: {map_file_path}")
#     except HTTPException as e:
#         print(f"지도 생성 실패 (HTTPException): {e.status_code} - {e.detail}")
#     except Exception as e:
#         print(f"지도 생성 실패 (일반 오류): {e}")
