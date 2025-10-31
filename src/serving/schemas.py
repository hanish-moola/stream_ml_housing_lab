"""Pydantic schemas used by the FastAPI inference service."""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal

import pandas as pd
from pydantic import BaseModel, Field


FurnishingStatus = Literal["furnished", "semi-furnished", "unfurnished"]


class HousingFeatures(BaseModel):
    area: float = Field(gt=0, description="Total area of the property in square feet")
    bedrooms: int = Field(ge=0, description="Number of bedrooms")
    bathrooms: float = Field(ge=0, description="Number of bathrooms")
    stories: int = Field(ge=0, description="Number of stories")
    parking: int = Field(ge=0, description="Number of parking spots")
    mainroad: bool = Field(description="Whether the property has access to the main road")
    guestroom: bool = Field(description="Presence of a guest room")
    basement: bool = Field(description="Whether the property has a basement")
    hotwaterheating: bool = Field(description="Availability of hot water heating")
    airconditioning: bool = Field(description="Availability of air conditioning")
    prefarea: bool = Field(description="Is the property in a preferred area")
    furnishingstatus: FurnishingStatus = Field(description="Furnishing status of the property")

    def to_model_payload(self, boolean_columns: Iterable[str]) -> Dict[str, object]:
        data = self.model_dump()
        for column in boolean_columns:
            if column in data:
                data[column] = "yes" if bool(data[column]) else "no"
        if "furnishingstatus" in data:
            data["furnishingstatus"] = data["furnishingstatus"].lower()
        return data

    def to_dataframe(self, feature_order: List[str], boolean_columns: Iterable[str]) -> pd.DataFrame:
        payload = self.to_model_payload(boolean_columns)
        ordered = {feature: payload[feature] for feature in feature_order}
        return pd.DataFrame([ordered], columns=feature_order)


class PredictionResponse(BaseModel):
    estimated_price: float = Field(description="Predicted price for the property")
    model_version: str | None = Field(default=None, description="Identifier of the model used")
