"""요청/응답 데이터 모델 (Pydantic) — 입력 검증 자동화."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    temperature: float = Field(..., description="기온(°C)")
    humidity: float = Field(..., ge=0, le=100, description="습도(%)")
    precipitation: float = Field(..., ge=0, description="강수량(mm)")


class Recommendation(BaseModel):
    label: str
    items: List[str]
    model_info: Optional[Dict[str, Any]] = None
