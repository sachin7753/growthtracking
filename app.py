# app.py
import re
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pandas as pd
import numpy as np
from functools import lru_cache

# ====== IMPORT YOUR EXISTING CODE FROM test.py ======
# (you can copy functions: GrowthNet, load_model, load_ref, interp_curve, est_percentile, ai_predict, get_ai_recommendations, generate_report)

# -------- CONFIG --------
HFA_BOYS_FILE = "tab_lhfa_boys_p_2_5.xlsx"
HFA_GIRLS_FILE = "tab_lhfa_girls_p_2_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
CLASS_LABELS = {0:"Underweight", 1:"Healthy", 2:"Overweight", 3:"Obese", 4:"Stunted", 5:"Normal Ht"}

# -------- AI MODEL --------
class GrowthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(CLASS_LABELS))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def load_model(path: str) -> GrowthNet:
    model = GrowthNet()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# -------- FastAPI App --------
app = FastAPI(title="Growth Advisor API")

# Load model once
growth_model = load_model(MODEL_PATH)

# -------- Request/Response Schemas --------
class GrowthRequest(BaseModel):
    age_months: int
    height_cm: float
    weight_kg: float
    sex: str

class GrowthResponse(BaseModel):
    bmi: float
    wfh_p: float
    hfa_p: float
    ai_status: str
    confidence: float
    who_msgs: List[str]
    recommendations: List[str]

# -------- API Endpoint --------
@app.post("/predict", response_model=GrowthResponse)
def predict_growth(data: GrowthRequest):
    report = generate_report(data.age_months, data.height_cm, data.weight_kg, data.sex, growth_model)
    return GrowthResponse(
        bmi=report["bmi"],
        wfh_p=report["wfh_p"],
        hfa_p=report["hfa_p"],
        ai_status=report["ai_status"],
        confidence=report["confidence"],
        who_msgs=report["who_msgs"],
        recommendations=report["recommendations"]
    )
