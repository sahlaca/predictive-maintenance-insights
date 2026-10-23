from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.model import FEATURES, NUMERIC_FEATURES, get_model

app = FastAPI(
    title="Predictive Maintenance Demo API",
    description="Prototype failure-category inference from machine operating measurements.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class MachineReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"] = Field(description="Product quality type")
    air_temperature: float = Field(description="Air temperature in kelvin")
    process_temperature: float = Field(description="Process temperature in kelvin")
    rotational_speed: float = Field(description="Rotational speed in rpm")
    torque: float = Field(description="Torque in newton-metres")
    tool_wear: float = Field(description="Tool wear in minutes")

    def as_model_row(self) -> dict[str, str | float]:
        return {
            "Type": self.type,
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
        }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/metadata")
def metadata() -> dict[str, object]:
    try:
        bundle = get_model()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {
        "row_count": bundle.row_count,
        "accuracy": bundle.accuracy,
        "classes": sorted(bundle.class_counts),
        "class_counts": bundle.class_counts,
        "feature_ranges": bundle.feature_ranges,
        "non_failure_ranges": bundle.non_failure_ranges,
        "feature_importances": bundle.feature_importances,
        "features": FEATURES,
    }


@app.get("/api/metrics")
def metrics() -> dict[str, object]:
    try:
        bundle = get_model()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    categories = {
        label: values
        for label, values in bundle.report.items()
        if isinstance(values, dict) and "precision" in values
    }
    return {"accuracy": bundle.accuracy, "categories": categories}


@app.post("/api/predict")
def predict(reading: MachineReading) -> dict[str, object]:
    try:
        bundle = get_model()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    row = reading.as_model_row()
    frame = pd.DataFrame([row], columns=FEATURES)
    prediction = str(bundle.pipeline.predict(frame)[0])
    probabilities = bundle.pipeline.predict_proba(frame)[0]
    classes = bundle.pipeline.named_steps["classifier"].classes_
    scores = sorted(
        [
            {"category": str(label), "score": float(score)}
            for label, score in zip(classes, probabilities)
        ],
        key=lambda result: result["score"],
        reverse=True,
    )
    outside_training_range = [
        feature
        for feature in NUMERIC_FEATURES
        if float(row[feature]) < bundle.feature_ranges[feature]["min"]
        or float(row[feature]) > bundle.feature_ranges[feature]["max"]
    ]
    return {
        "predicted_category": prediction,
        "predicted_status": "possible_failure" if prediction != "No Failure" else "no_failure_detected",
        "scores": scores,
        "score_note": "Model scores are not calibrated real-world probabilities.",
        "outside_training_range": outside_training_range,
        "range_notice": (
            "One or more values are outside the range seen in the demo dataset. "
            "The model is extrapolating, so treat this result with extra caution."
            if outside_training_range
            else None
        ),
        "prototype_notice": "Sample data only; not for safety-critical or live-equipment decisions.",
    }
