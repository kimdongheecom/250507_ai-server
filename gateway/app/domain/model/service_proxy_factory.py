from typing import Optional
from fastapi import HTTPException
import httpx
import logging
from app.domain.model.service_type import SERVICE_URLS, ServiceType

logger = logging.getLogger("gateway_api")

class ServiceProxyFactory:
    def __init__(self, service_type: ServiceType):
        self.service_type = service_type
        self.base_url = SERVICE_URLS.get(service_type)
        if not self.base_url:
            raise ValueError(f"Service URL not found for {service_type}")
        print(f"🔍 Service URL: {self.base_url}")

    async def request(
        self,
        method: str,
        path: str,
        headers: list = None,
        json: dict = None,
        data: bytes = None,
        files: dict = None
    ) -> httpx.Response:
        """
        서비스로 요청을 전달합니다.
        
        Args:
            method: HTTP 메서드 (GET, POST, 등)
            path: 요청 경로
            headers: HTTP 헤더
            json: JSON 데이터
            data: raw 데이터
            files: 파일 데이터
        """
        if path == "titanic":
            path = "titanic/predict"
        elif path == "matzip":
            path = "matzip/predict"
        elif path == "crime":
            path = "crime/predict"
        elif path == "nlp":
            path = "nlp/wordcloud"
        url = f"{self.base_url}/{path}"
        logger.info(f"🌐 요청 전송: {method} {url}")
        print(f"🔍 Requesting URL: {url}")
        # 헤더 설정
        headers_dict = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=dict(headers) if headers else None,
                    json=json,
                    data=data,
                    files=files
                )
                logger.info(f"📥 응답 수신: {response.status_code}")
                print(f"Response status: {response.status_code}")
                print(f"Request URL: {url}")
                print(f"Request body: {data}")
                return response
            except Exception as e:
                logger.error(f"❌ 요청 실패: {str(e)}")
                print(f"Request failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))