import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from functools import lru_cache

# -------- CONFIG --------
HFA_BOYS_FILE = "tab_lhfa_boys_p_2_5.xlsx"
HFA_GIRLS_FILE = "tab_lhfa_girls_p_2_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
DAYS_PER_MONTH = 30.4375
CLASS_LABELS = {0:"Underweight", 1:"Healthy", 2:"Overweight", 3:"Obese", 4:"Stunted", 5:"Normal Ht"}

# -------- AI MODEL DEFINITION --------
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
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# -------- WHO UTILITIES --------
@lru_cache(maxsize=None)
def load_ref(path: str, primary_col_regex: str) -> tuple[pd.DataFrame, list[str]]:
    print(f"Reading file: {path}...")
    df = pd.read_excel(path)
    primary_col = next((c for c in df.columns if re.search(primary_col_regex, str(c), re.I)), None)
    if not primary_col:
        raise ValueError(f"No column matching '{primary_col_regex}' found in {path}")
    pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
    df = df[[primary_col] + pcols].copy()
    df.columns = ["primary"] + pcols
    return df, pcols

def interp_curve(ref_df: pd.DataFrame, pcols: list[str], val: float) -> dict[float, float]:
    values = ref_df.iloc[:, 0].values.astype(float)
    if val <= values.min():
        row = ref_df.iloc[0]
    elif val >= values.max():
        row = ref_df.iloc[-1]
    else:
        idx = np.searchsorted(values, val, side="right")
        v0, v1 = values[idx-1], values[idx]
        frac = (val - v0) / (v1 - v0)
        row0, row1 = ref_df.iloc[idx-1], ref_df.iloc[idx]
        return {float(re.findall(r"\d+",c)[0]): row0[c]+frac*(row1[c]-row0[c]) for c in pcols}
    return {float(re.findall(r"\d+",c)[0]): float(row[c]) for c in pcols}

def est_percentile(value: float, curve: dict[float, float]) -> float:
    pts = sorted(curve.items(), key=lambda item: item[1])
    values = [v for p,v in pts]
    percs = [p for p,v in pts]
    if value <= values[0]: return percs[0]
    if value >= values[-1]: return percs[-1]
    j = np.searchsorted(values, value, side="right")
    v0,v1,p0,p1 = values[j-1],values[j],percs[j-1],percs[j]
    return p0 + (value - v0) / (v1 - v0) * (p1 - p0)

# -------- PREDICTION & REPORTING --------
def ai_predict(model: GrowthNet, age_m: int, ht: float, wt: float, sex: str, wfh_p: float, hfa_p: float) -> tuple[str, float]:
    x = torch.tensor([[age_m, ht, wt, 1 if sex == "M" else 0]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx_tensor = torch.max(probabilities, dim=1)
        pred_idx = pred_idx_tensor.item()
        confidence_score = confidence.item()

    status = CLASS_LABELS.get(pred_idx, "Unknown")
    bmi = wt / ((ht / 100) ** 2)

    if wfh_p < 3: status = "Underweight"
    elif wfh_p > 97: status = "Obese" if bmi >= 30 else "Overweight"
    elif bmi >= 30: status = "Obese"
    elif bmi >= 25: status = "Overweight"
    elif hfa_p < 3 and status in ["Healthy", "Normal Height"]: status = "Stunted"
    elif status == "Underweight" and wfh_p >= 5 and hfa_p < 5: status = "Stunted"
    return status, confidence_score

def get_ai_recommendations(status: str, age_m: int, wfh_p: float, hfa_p: float, bmi: float) -> list[str]:
    recs = [f"🤖 <b>AI Recommendation Engine Analysis</b>"]

    if status in ["Obese", "Overweight"]:
        recs.append(f"⚠️ <b>Status: {status}</b> (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f})")
        if bmi >= 35:
            recs.append("- <b>Immediate pediatric consultation is critical</b> due to severe obesity.")
        else:
            recs.append("- A pediatric consultation is strongly recommended to create a management plan.")
        recs.append("- Avoid sugary drinks, juices, and processed snacks. Focus on whole foods.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- At least 60 minutes/day of active play (running, cycling, sports).")
        recs.append("- Limit screen time to <1 hour/day.")
        recs.append("- Encourage family-based activities: walking, dancing, playground games.")

        if hfa_p < 5:
            recs.append("- ⚠️ Child is both overweight and stunted → focus on balanced nutrition + safe physical activity.")

    elif status == "Underweight":
        recs.append(f"⚠️ <b>Status: Underweight</b> (Weight-for-Height: P{wfh_p:.1f})")
        if wfh_p < 1:
            recs.append("- <b>Severe Wasting:</b> Medical evaluation is urgently needed.")
        else:
            recs.append("- Increase intake of healthy, energy-dense foods like avocado, nuts, and dairy.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- Allow free play but avoid overexertion.")
        recs.append("- Light activities: walking, gentle play, building stamina.")
        recs.append("- Ensure adequate rest and recovery.")

    elif status == "Stunted":
        recs.append(f"⚠️ <b>Status: Stunted</b> (Height-for-Age: P{hfa_p:.1f})")
        recs.append("- Provide a diet rich in iron, zinc, vitamin A, and protein.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- Moderate daily play (30–45 minutes).")
        recs.append("- Encourage outdoor play for sunlight (Vitamin D).")
        recs.append("- Avoid excessive screen time.")

    else:
        recs.append(f"✅ <b>Status: Healthy Growth Track</b>")
        recs.append("- Balanced meals and regular pediatric check-ups.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- 60 minutes/day of varied play (running, cycling, ball games).")
        recs.append("- Try different activities: swimming, dancing, climbing, etc.")
        recs.append("- Maintain good sleep hygiene (10–12 hrs/night).")

    return recs

def generate_report(age_m: int, ht: float, wt: float, sex: str, model: GrowthNet) -> dict:
    hfa_ref, hfa_pcols = load_ref(HFA_BOYS_FILE if sex == "M" else HFA_GIRLS_FILE, r'month')
    hfa_curve = interp_curve(hfa_ref, hfa_pcols, age_m)
    hfa_p = est_percentile(ht, hfa_curve)

    wfh_ref, wfh_pcols = load_ref(WFH_BOYS_FILE if sex == "M" else WFH_GIRLS_FILE, r'height|length')
    wfh_curve = interp_curve(wfh_ref, wfh_pcols, ht)
    wfh_p = est_percentile(wt, wfh_curve)

    ai_status, confidence = ai_predict(model, age_m, ht, wt, sex, wfh_p, hfa_p)
    bmi = wt / ((ht / 100) ** 2)

    who_msgs = []
    if wfh_p < 3:
        who_msgs.append("⚠️ <font color='red'>Weight-for-height below 3rd percentile (Wasting risk).</font>")
    elif wfh_p > 97:
        who_msgs.append("⚠️ <font color='red'>Weight-for-height above 97th percentile (Overweight risk).</font>")
    else:
        who_msgs.append("✅ <font color='green'>Weight-for-height is in a healthy range.</font>")

    if hfa_p < 3:
        who_msgs.append("⚠️ <font color='red'>Height-for-age below 3rd percentile (Stunting risk).</font>")
    else:
        who_msgs.append("✅ <font color='green'>Height-for-age is in a healthy range.</font>")

    recommendations = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    return {
        "wfh_p": wfh_p, "hfa_p": hfa_p, "bmi": bmi,
        "who_msgs": who_msgs, "recommendations": recommendations,
        "ai_status": ai_status, "confidence": confidence
    }

# -------- INPUT HANDLERS, PDF EXPORT, MAIN --------
def get_age() -> int:
    while True:
        age_input = input("Enter age (e.g., '3y 6m' or '42m'): ").strip().lower()
        try:
            years, months = 0, 0
            if "y" in age_input:
                parts = re.split(r'y', age_input.replace("years","y").replace("year","y"))
                years = int(parts[0].strip()) if parts[0].strip() else 0
                if "m" in parts[1]: months = int(re.sub(r'[^0-9]', '', parts[1]))
            elif "m" in age_input:
                months = int(re.sub(r'[^0-9]', '', age_input))
            else:
                months = int(age_input)
            total_months = years*12 + months
            if 24 <= total_months <= 60: return total_months
            else: print("❌ Age must be between 2 and 5 years (24–60 months).")
        except: print("❌ Invalid age format. Try again.")

def get_float(prompt: str, min_val: float, max_val: float) -> float:
    while True:
        try:
            val = float(input(prompt))
            if min_val < val < max_val: return val
            print(f"❌ Value must be between {min_val} and {max_val}.")
        except ValueError: print("❌ Invalid number. Try again.")

def get_sex() -> str:
    while True:
        s = input("Enter sex (M/F): ").strip().upper()
        if s in ["M","F"]: return s
        print("❌ Invalid input. Enter 'M' or 'F'.")

def save_pdf(report_lines: list[str], filename="growth_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MyTitle", fontSize=16, leading=20, spaceAfter=16, alignment=1, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name="MySection", fontSize=12, leading=14, spaceAfter=8, textColor=colors.darkgreen))
    flow = [Paragraph("🧒 <b>Child Growth & Health Report</b>", styles["MyTitle"])]
    for line in report_lines:
        style = styles["MySection"] if line.startswith("<b>") else styles["Normal"]
        flow.append(Paragraph(line, style)); flow.append(Spacer(1,6))
    doc.build(flow); print(f"📄 Report saved successfully as {filename}")

if __name__ == "__main__":
    try:
        growth_model = load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Model file '{MODEL_PATH}' not found. Run train.py first."); exit()

    print("=== Upgraded Hybrid Growth Advisor (Ages 2–5 years) ===")
    age_months = get_age()
    height_cm = get_float("Enter height (cm): ", 75, 125)
    weight_kg = get_float("Enter weight (kg): ", 5, 30)
    sex = get_sex()

    report = generate_report(age_months, height_cm, weight_kg, sex, growth_model)

    print("\n" + "="*50)
    print("        🧒 CHILD GROWTH REPORT")
    print("="*50 + "\n")
    print(f"📊 MEASUREMENTS\n- Age: {age_months//12}y {age_months%12}m\n- Height: {height_cm:.1f} cm\n- Weight: {weight_kg:.1f} kg\n- BMI: {report['bmi']:.1f}\n")
    print("📈 WHO ASSESSMENT")
    [print("-", re.sub(r"<.*?>","",msg)) for msg in report['who_msgs']]
    print("\n" + "-"*50)
    print(f"🤖 AI Recommendation Engine Analysis (Final Status: {report['ai_status']})")
    [print("-", re.sub(r"<.*?>","",tip)) for tip in report['recommendations'][1:]]
    print(f"(Model confidence: {report['confidence']:.1%})")
    print("-"*50)

    if input("\nDo you want to save this report as a PDF? (Y/N): ").strip().upper() == "Y":
        pdf_lines = [
            f"<b>Age:</b> {age_months//12}y {age_months%12}m | <b>Height:</b> {height_cm:.1f} cm | <b>Weight:</b> {weight_kg:.1f} kg | <b>BMI:</b> {report['bmi']:.1f}",
            Spacer(1,12), "<b>WHO Percentile Analysis:</b>"
        ] + [re.sub(r"<.*?>","",m) for m in report['who_msgs']] + [Spacer(1,12)] + report['recommendations'] + [f"<i>(Model confidence: {report['confidence']:.1%})</i>"]
        save_pdf(pdf_lines)
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from functools import lru_cache

# -------- CONFIG --------
HFA_BOYS_FILE = "tab_lhfa_boys_p_2_5.xlsx"
HFA_GIRLS_FILE = "tab_lhfa_girls_p_2_5.xlsx"
WFH_BOYS_FILE = "tab_wfh_boys_p_0_5.xlsx"
WFH_GIRLS_FILE = "tab_wfh_girls_p_0_5.xlsx"
MODEL_PATH = "growth_model.pth"
DAYS_PER_MONTH = 30.4375
CLASS_LABELS = {0:"Underweight", 1:"Healthy", 2:"Overweight", 3:"Obese", 4:"Stunted", 5:"Normal Ht"}

# -------- AI MODEL DEFINITION --------
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
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# -------- WHO UTILITIES --------
@lru_cache(maxsize=None)
def load_ref(path: str, primary_col_regex: str) -> tuple[pd.DataFrame, list[str]]:
    print(f"Reading file: {path}...")
    df = pd.read_excel(path)
    primary_col = next((c for c in df.columns if re.search(primary_col_regex, str(c), re.I)), None)
    if not primary_col:
        raise ValueError(f"No column matching '{primary_col_regex}' found in {path}")
    pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
    df = df[[primary_col] + pcols].copy()
    df.columns = ["primary"] + pcols
    return df, pcols

def interp_curve(ref_df: pd.DataFrame, pcols: list[str], val: float) -> dict[float, float]:
    values = ref_df.iloc[:, 0].values.astype(float)
    if val <= values.min():
        row = ref_df.iloc[0]
    elif val >= values.max():
        row = ref_df.iloc[-1]
    else:
        idx = np.searchsorted(values, val, side="right")
        v0, v1 = values[idx-1], values[idx]
        frac = (val - v0) / (v1 - v0)
        row0, row1 = ref_df.iloc[idx-1], ref_df.iloc[idx]
        return {float(re.findall(r"\d+",c)[0]): row0[c]+frac*(row1[c]-row0[c]) for c in pcols}
    return {float(re.findall(r"\d+",c)[0]): float(row[c]) for c in pcols}

def est_percentile(value: float, curve: dict[float, float]) -> float:
    pts = sorted(curve.items(), key=lambda item: item[1])
    values = [v for p,v in pts]
    percs = [p for p,v in pts]
    if value <= values[0]: return percs[0]
    if value >= values[-1]: return percs[-1]
    j = np.searchsorted(values, value, side="right")
    v0,v1,p0,p1 = values[j-1],values[j],percs[j-1],percs[j]
    return p0 + (value - v0) / (v1 - v0) * (p1 - p0)

# -------- PREDICTION & REPORTING --------
def ai_predict(model: GrowthNet, age_m: int, ht: float, wt: float, sex: str, wfh_p: float, hfa_p: float) -> tuple[str, float]:
    x = torch.tensor([[age_m, ht, wt, 1 if sex == "M" else 0]], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx_tensor = torch.max(probabilities, dim=1)
        pred_idx = pred_idx_tensor.item()
        confidence_score = confidence.item()

    status = CLASS_LABELS.get(pred_idx, "Unknown")
    bmi = wt / ((ht / 100) ** 2)

    if wfh_p < 3: status = "Underweight"
    elif wfh_p > 97: status = "Obese" if bmi >= 30 else "Overweight"
    elif bmi >= 30: status = "Obese"
    elif bmi >= 25: status = "Overweight"
    elif hfa_p < 3 and status in ["Healthy", "Normal Height"]: status = "Stunted"
    elif status == "Underweight" and wfh_p >= 5 and hfa_p < 5: status = "Stunted"
    return status, confidence_score

def get_ai_recommendations(status: str, age_m: int, wfh_p: float, hfa_p: float, bmi: float) -> list[str]:
    recs = [f"🤖 <b>AI Recommendation Engine Analysis</b>"]

    if status in ["Obese", "Overweight"]:
        recs.append(f"⚠️ <b>Status: {status}</b> (BMI: {bmi:.1f} | Wt-for-Ht: P{wfh_p:.1f})")
        if bmi >= 35:
            recs.append("- <b>Immediate pediatric consultation is critical</b> due to severe obesity.")
        else:
            recs.append("- A pediatric consultation is strongly recommended to create a management plan.")
        recs.append("- Avoid sugary drinks, juices, and processed snacks. Focus on whole foods.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- At least 60 minutes/day of active play (running, cycling, sports).")
        recs.append("- Limit screen time to <1 hour/day.")
        recs.append("- Encourage family-based activities: walking, dancing, playground games.")

        if hfa_p < 5:
            recs.append("- ⚠️ Child is both overweight and stunted → focus on balanced nutrition + safe physical activity.")

    elif status == "Underweight":
        recs.append(f"⚠️ <b>Status: Underweight</b> (Weight-for-Height: P{wfh_p:.1f})")
        if wfh_p < 1:
            recs.append("- <b>Severe Wasting:</b> Medical evaluation is urgently needed.")
        else:
            recs.append("- Increase intake of healthy, energy-dense foods like avocado, nuts, and dairy.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- Allow free play but avoid overexertion.")
        recs.append("- Light activities: walking, gentle play, building stamina.")
        recs.append("- Ensure adequate rest and recovery.")

    elif status == "Stunted":
        recs.append(f"⚠️ <b>Status: Stunted</b> (Height-for-Age: P{hfa_p:.1f})")
        recs.append("- Provide a diet rich in iron, zinc, vitamin A, and protein.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- Moderate daily play (30–45 minutes).")
        recs.append("- Encourage outdoor play for sunlight (Vitamin D).")
        recs.append("- Avoid excessive screen time.")

    else:
        recs.append(f"✅ <b>Status: Healthy Growth Track</b>")
        recs.append("- Balanced meals and regular pediatric check-ups.")

        # 💪 Physical Activity
        recs.append("\n💪 <b>Physical Activity</b>")
        recs.append("- 60 minutes/day of varied play (running, cycling, ball games).")
        recs.append("- Try different activities: swimming, dancing, climbing, etc.")
        recs.append("- Maintain good sleep hygiene (10–12 hrs/night).")

    return recs

def generate_report(age_m: int, ht: float, wt: float, sex: str, model: GrowthNet) -> dict:
    hfa_ref, hfa_pcols = load_ref(HFA_BOYS_FILE if sex == "M" else HFA_GIRLS_FILE, r'month')
    hfa_curve = interp_curve(hfa_ref, hfa_pcols, age_m)
    hfa_p = est_percentile(ht, hfa_curve)

    wfh_ref, wfh_pcols = load_ref(WFH_BOYS_FILE if sex == "M" else WFH_GIRLS_FILE, r'height|length')
    wfh_curve = interp_curve(wfh_ref, wfh_pcols, ht)
    wfh_p = est_percentile(wt, wfh_curve)

    ai_status, confidence = ai_predict(model, age_m, ht, wt, sex, wfh_p, hfa_p)
    bmi = wt / ((ht / 100) ** 2)

    who_msgs = []
    if wfh_p < 3:
        who_msgs.append("⚠️ <font color='red'>Weight-for-height below 3rd percentile (Wasting risk).</font>")
    elif wfh_p > 97:
        who_msgs.append("⚠️ <font color='red'>Weight-for-height above 97th percentile (Overweight risk).</font>")
    else:
        who_msgs.append("✅ <font color='green'>Weight-for-height is in a healthy range.</font>")

    if hfa_p < 3:
        who_msgs.append("⚠️ <font color='red'>Height-for-age below 3rd percentile (Stunting risk).</font>")
    else:
        who_msgs.append("✅ <font color='green'>Height-for-age is in a healthy range.</font>")

    recommendations = get_ai_recommendations(ai_status, age_m, wfh_p, hfa_p, bmi)
    return {
        "wfh_p": wfh_p, "hfa_p": hfa_p, "bmi": bmi,
        "who_msgs": who_msgs, "recommendations": recommendations,
        "ai_status": ai_status, "confidence": confidence
    }

# -------- INPUT HANDLERS, PDF EXPORT, MAIN --------
def get_age() -> int:
    while True:
        age_input = input("Enter age (e.g., '3y 6m' or '42m'): ").strip().lower()
        try:
            years, months = 0, 0
            if "y" in age_input:
                parts = re.split(r'y', age_input.replace("years","y").replace("year","y"))
                years = int(parts[0].strip()) if parts[0].strip() else 0
                if "m" in parts[1]: months = int(re.sub(r'[^0-9]', '', parts[1]))
            elif "m" in age_input:
                months = int(re.sub(r'[^0-9]', '', age_input))
            else:
                months = int(age_input)
            total_months = years*12 + months
            if 24 <= total_months <= 60: return total_months
            else: print("❌ Age must be between 2 and 5 years (24–60 months).")
        except: print("❌ Invalid age format. Try again.")

def get_float(prompt: str, min_val: float, max_val: float) -> float:
    while True:
        try:
            val = float(input(prompt))
            if min_val < val < max_val: return val
            print(f"❌ Value must be between {min_val} and {max_val}.")
        except ValueError: print("❌ Invalid number. Try again.")

def get_sex() -> str:
    while True:
        s = input("Enter sex (M/F): ").strip().upper()
        if s in ["M","F"]: return s
        print("❌ Invalid input. Enter 'M' or 'F'.")

def save_pdf(report_lines: list[str], filename="growth_report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MyTitle", fontSize=16, leading=20, spaceAfter=16, alignment=1, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name="MySection", fontSize=12, leading=14, spaceAfter=8, textColor=colors.darkgreen))
    flow = [Paragraph("🧒 <b>Child Growth & Health Report</b>", styles["MyTitle"])]
    for line in report_lines:
        style = styles["MySection"] if line.startswith("<b>") else styles["Normal"]
        flow.append(Paragraph(line, style)); flow.append(Spacer(1,6))
    doc.build(flow); print(f"📄 Report saved successfully as {filename}")

if __name__ == "__main__":
    try:
        growth_model = load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Model file '{MODEL_PATH}' not found. Run train.py first."); exit()

    print("=== Upgraded Hybrid Growth Advisor (Ages 2–5 years) ===")
    age_months = get_age()
    height_cm = get_float("Enter height (cm): ", 75, 125)
    weight_kg = get_float("Enter weight (kg): ", 5, 30)
    sex = get_sex()

    report = generate_report(age_months, height_cm, weight_kg, sex, growth_model)

    print("\n" + "="*50)
    print("        🧒 CHILD GROWTH REPORT")
    print("="*50 + "\n")
    print(f"📊 MEASUREMENTS\n- Age: {age_months//12}y {age_months%12}m\n- Height: {height_cm:.1f} cm\n- Weight: {weight_kg:.1f} kg\n- BMI: {report['bmi']:.1f}\n")
    print("📈 WHO ASSESSMENT")
    [print("-", re.sub(r"<.*?>","",msg)) for msg in report['who_msgs']]
    print("\n" + "-"*50)
    print(f"🤖 AI Recommendation Engine Analysis (Final Status: {report['ai_status']})")
    [print("-", re.sub(r"<.*?>","",tip)) for tip in report['recommendations'][1:]]
    print(f"(Model confidence: {report['confidence']:.1%})")
    print("-"*50)

    if input("\nDo you want to save this report as a PDF? (Y/N): ").strip().upper() == "Y":
        pdf_lines = [
            f"<b>Age:</b> {age_months//12}y {age_months%12}m | <b>Height:</b> {height_cm:.1f} cm | <b>Weight:</b> {weight_kg:.1f} kg | <b>BMI:</b> {report['bmi']:.1f}",
            Spacer(1,12), "<b>WHO Percentile Analysis:</b>"
        ] + [re.sub(r"<.*?>","",m) for m in report['who_msgs']] + [Spacer(1,12)] + report['recommendations'] + [f"<i>(Model confidence: {report['confidence']:.1%})</i>"]
        save_pdf(pdf_lines)
