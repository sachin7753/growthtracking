import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -------- CONFIG --------
WFA_BOYS_FILE = "tab_wfa_boys_p_0_5.xlsx"
WFA_GIRLS_FILE = "tab_wfa_girls_p_0_5.xlsx"
# --- UPDATED FILENAMES (NOTE: These are for ages 2-5 years ONLY) ---
HFA_BOYS_FILE = "tab_lhfa_boys_p_2_5.xlsx"
HFA_GIRLS_FILE = "tab_lhfa_girls_p_2_5.xlsx"
# -------------------------------------------------------------------
BFA_FILE = "bmi.csv.xlsx"
MODEL_SAVE_PATH = "growth_model.pth"

# Training parameters
LEARNING_RATE = 0.005
EPOCHS = 500
PATIENCE = 20

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

# -------- DATA UTILITIES --------
def load_who_reference(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # This logic handles both "Month" and "Age in days" columns
    age_col = next((c for c in df.columns if re.search(r'age|day|month', str(c), re.I)), None)
    if not age_col: raise ValueError(f"No Age/Month/Day column found in {path}")

    pcols = [c for c in df.columns if re.match(r"P\d+", str(c))]
    df = df[[age_col] + pcols].copy()
    
    # Standardize the age column
    if 'day' in age_col.lower():
        df.columns = ["age_days"] + pcols
        df["age_months"] = df["age_days"] / 30.4375
    else: # Assumes it's 'Month'
        df.columns = ["age_months"] + pcols
        df["age_days"] = df["age_months"] * 30.4375
        
    return df

# -------- DATASET BUILDER --------
def build_dataset() -> pd.DataFrame:
    print("Building enhanced dataset with jitter...")
    wfa_boys = load_who_reference(WFA_BOYS_FILE)
    wfa_girls = load_who_reference(WFA_GIRLS_FILE)
    hfa_boys = load_who_reference(HFA_BOYS_FILE)
    hfa_girls = load_who_reference(HFA_GIRLS_FILE)
    bfa_df = pd.read_excel(BFA_FILE)
    
    dataset = []
    # --- Synthetic data generation ---
    # Merge WFA and HFA dataframes on the nearest month to ensure alignment
    wfa_boys['month_key'] = wfa_boys['age_months'].round()
    hfa_boys['month_key'] = hfa_boys['age_months'].round()
    merged_boys = pd.merge(wfa_boys, hfa_boys, on='month_key', suffixes=('_wfa', '_hfa'))

    wfa_girls['month_key'] = wfa_girls['age_months'].round()
    hfa_girls['month_key'] = hfa_girls['age_months'].round()
    merged_girls = pd.merge(wfa_girls, hfa_girls, on='month_key', suffixes=('_wfa', '_hfa'))

    for sex, merged_df in [("M", merged_boys), ("F", merged_girls)]:
        p_cols_wfa = [c for c in merged_df.columns if c.startswith('P') and c.endswith('_wfa')]
        
        for i, row in merged_df.iterrows():
            age = row["age_months_wfa"]
            for col_wfa in p_cols_wfa:
                p_val = col_wfa.split('_')[0]
                col_hfa = f"{p_val}_hfa"
                if col_hfa not in merged_df.columns: continue

                perc = float(re.findall(r"\d+", p_val)[0])
                wt = row[col_wfa]
                ht = row[col_hfa]
                
                wt_jitter = wt * np.random.normal(1, 0.02)
                ht_jitter = ht * np.random.normal(1, 0.02)
                
                if perc < 3: wfa_lbl = 0
                elif perc < 85: wfa_lbl = 1
                elif perc < 97: wfa_lbl = 2
                else: wfa_lbl = 3
                
                hfa_lbl = 4 if perc < 3 else 5
                
                final_label = wfa_lbl if wfa_lbl != 1 else hfa_lbl
                dataset.append([age, ht_jitter, wt_jitter, 1 if sex == "M" else 0, final_label])

    # --- BMI dataset samples ---
    for _, row in bfa_df.iterrows():
        bmi_class = str(row["BmiClass"]).lower()
        if "under" in bmi_class: lbl = 0
        elif "over" in bmi_class: lbl = 2
        elif "obese" in bmi_class: lbl = 3
        else: lbl = 1
        dataset.append([row["Age"], row["Height in centimeter"], row["Weight"], 1, lbl])

    print("Dataset built.")
    return pd.DataFrame(dataset, columns=["age", "height", "weight", "sex", "label"])

# -------- MAIN TRAINING SCRIPT --------
if __name__ == "__main__":
    data = build_dataset()
    X = torch.tensor(data.drop("label", axis=1).values, dtype=torch.float32)
    y = torch.tensor(data["label"].values, dtype=torch.long)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Data split: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples.")

    model = GrowthNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\nStarting model training with early stopping...")
    for epoch in range(EPOCHS):
        model.train(); optimizer.zero_grad()
        outputs = model(X_train); loss = criterion(outputs, y_train)
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_outputs = model(X_val); val_loss = criterion(val_outputs, y_val)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss; epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else: epochs_no_improve += 1
        if epochs_no_improve == PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}."); break
            
    print("\n--- Final Model Evaluation on Unseen Test Data ---")
    best_model = GrowthNet(); best_model.load_state_dict(torch.load(MODEL_SAVE_PATH)); best_model.eval()
    with torch.no_grad():
        y_pred_tensor = best_model(X_test); y_pred = torch.argmax(y_pred_tensor, dim=1).numpy(); y_test_np = y_test.numpy()

    print("\nClassification Report:"); print(classification_report(y_test_np, y_pred, labels=list(CLASS_LABELS.keys()), target_names=list(CLASS_LABELS.values()), zero_division=0))
    print("Confusion Matrix:"); cm = confusion_matrix(y_test_np, y_pred, labels=list(CLASS_LABELS.keys()))
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS.values(), yticklabels=CLASS_LABELS.values())
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('Confusion Matrix'); plt.show()
    print(f"\n✅ Best model saved to {MODEL_SAVE_PATH}")