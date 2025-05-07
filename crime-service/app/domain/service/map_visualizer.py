import numpy as np
import pandas as pd
import os
from app.domain.model.reader_schema import ReaderSchema
from app.domain.model.google_map_schema import GoogleMapSchema
import folium
import logging
from fastapi import HTTPException
import traceback    
from app.domain.service.internal.crime_map_create import create_map

logger = logging.getLogger("crime_service")

class MapVisualizer:
    def __init__(self):
        # 현재 파일 기준으로 상대 경로 설정
        self.stored_data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'update_data')
        self.updated_data = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'update_data')
        self.stored_map = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'stored-map')
        self.reader = ReaderSchema()

    def draw_crime_map(self) -> dict:
        """범죄 지도를 생성하고 결과를 반환합니다."""
        try:
            map_file_path = create_map(self.stored_data, self.stored_map)
            return {"status": "success", "file_path": map_file_path}
        except HTTPException as e:
            logger.error(f"지도 생성 실패 (HTTPException): {e.status_code} - {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"지도 생성 중 예상치 못한 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"지도 생성 중 예상치 못한 서버 오류: {type(e).__name__}")

    def draw_crime_map2(self) -> dict:
        """범죄 지도를 생성하고 결과를 반환합니다."""
        try:
            # 데이터 로드
            self.reader.fname = 'police_norm'
            police_norm = self.reader.csv_to_dframe()
            
            self.reader.fname = 'geo_simple'
            state_geo = self.reader.json_load()
            
            self.reader.fname = 'crime_in_seoul'
            crime = self.reader.csv_to_dframe()
            
            self.reader.fname = 'police_pos'
            police_pos = self.reader.csv_to_dframe()

            # 경찰서 위치 정보 수집
            station_names = ['서울' + str(name[:-1]) + '경찰서' for name in crime['관서명']]
            station_addrs = []
            station_lats = []
            station_lngs = []
            
            gmaps = GoogleMapSchema()
            for name in station_names:
                tmp = gmaps.geocode(name, language='ko')
                station_addrs.append(tmp[0].get('formatted_address'))
                tmp_loc = tmp[0].get('geometry')
                station_lats.append(tmp_loc['location']['lat'])
                station_lngs.append(tmp_loc['location']['lng'])

            # 데이터 처리
            police_pos['lat'] = station_lats
            police_pos['lng'] = station_lngs
            col = ['살인 검거', '강도 검거', '강간 검거', '절도 검거', '폭력 검거']
            tmp = police_pos[col] / police_pos[col].max()
            police_pos['검거'] = np.sum(tmp, axis=1)

            # 지도 생성
            folium_map = folium.Map(location=[37.5502, 126.982], zoom_start=12, title='Stamen Toner')

            # Choropleth 레이어 추가
            folium.Choropleth(
                geo_data=state_geo,
                data=tuple(zip(police_norm['구별'], police_norm['범죄'])),
                columns=["State", "Crime Rate"],
                key_on="feature.id",
                fill_color="PuRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="Crime Rate (%)",
                reset=True,
            ).add_to(folium_map)

            # 경찰서 마커 추가
            for i in police_pos.index:
                folium.CircleMarker(
                    [police_pos['lat'][i], police_pos['lng'][i]],
                    radius=police_pos['검거'][i] * 10,
                    fill_color='#0a0a32'
                ).add_to(folium_map)

            # 지도 저장
            map_path = os.path.join(self.stored_map, 'crime_map.html')
            folium_map.save(map_path)

            return {"message": '서울시의 범죄 지도가 완성되었습니다.', "file_path": map_path}
        except Exception as e:
            logger.error(f"지도 생성 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"지도 생성 중 오류 발생: {str(e)}")
        
