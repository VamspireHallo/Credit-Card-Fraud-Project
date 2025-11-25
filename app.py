import gradio as gr
import numpy as np
import torch
import joblib
import pandas as pd

# =====================================
# 1. Define DL model
# =====================================
class FraudMLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

# =====================================
# 2. Load models
# =====================================
xgb_model = joblib.load("models/xgb_creditcard.pkl")
rf_model  = joblib.load("models/rf_creditcard.pkl")

def try_load(path, fallback):
    try:
        return joblib.load(path)
    except:
        return fallback

xgb_threshold = try_load("models/xgb_threshold.pkl", 0.80)
rf_threshold  = try_load("models/rf_threshold.pkl", 0.80)
mlp_threshold = try_load("models/mlp_threshold.pkl", 0.80)

# DL model
num_features = 29
mlp_model = FraudMLP(num_features)
mlp_model.load_state_dict(torch.load("models/mlp_fraud_model.pth", map_location="cpu"))
mlp_model.eval()

# Scalers
ml_scaler  = joblib.load("models/ml_scaler_creditcard.joblib")
mlp_scaler = joblib.load("models/mlp_scaler_creditcard.joblib")

# =====================================
# 3. Preprocessing shared function
# =====================================
def preprocess_input(transaction, model_choice):
    transaction["Amount_log"] = np.log1p(transaction["Amount"])
    feat_order = [f"V{i}" for i in range(1, 29)] + ["Amount_log"]

    arr = np.array([transaction[f] for f in feat_order]).reshape(1, -1)

    # ML and DL use separate scalers
    if model_choice in ["XGBoost", "RandomForest"]:
        return ml_scaler.transform(arr)
    else:
        return mlp_scaler.transform(arr)

# =====================================
# 4. Unified prediction logic
# =====================================
def run_prediction(feature_dict, model_choice):
    X_scaled = preprocess_input(feature_dict, model_choice)

    if model_choice == "XGBoost":
        prob = xgb_model.predict_proba(X_scaled)[:, 1][0]
        thr = xgb_threshold

    elif model_choice == "RandomForest":
        prob = rf_model.predict_proba(X_scaled)[:, 1][0]
        thr = rf_threshold

    else:  # MLP
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        prob = mlp_model(X_tensor).item()
        thr = mlp_threshold

    label = "Fraud" if prob >= thr else "Legit"
    return f"Prediction: {label} (Probability: {prob:.4f})"

# =====================================
# 5-A. Manual Input Prediction
# =====================================
def predict_manual(*inputs):
    feature_dict = {f"V{i}": inputs[i - 1] for i in range(1, 29)}
    feature_dict["Amount"] = inputs[28]
    model_choice = inputs[29]
    return run_prediction(feature_dict, model_choice)

# =====================================
# 5-B. CSV Upload Prediction
# =====================================
def predict_csv(file, model_choice):
    if file is None:
        return "Upload a CSV with columns V1..V28 and Amount."

    try:
        df = pd.read_csv(file.name)
    except:
        return "Error: Could not read the CSV file."

    required = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    for col in required:
        if col not in df.columns:
            return f"Missing column: {col}"

    # Use ONLY the first row
    row = df.iloc[0].to_dict()

    return run_prediction(row, model_choice)

# =====================================
# 5-C. CSV Text Input Prediction
# =====================================
def predict_csv_text(csv_line, model_choice):
    """
    csv_line: str, a single CSV row like:
    2,-0.4259,0.9605,...,3.67,0
    """
    try:
        values = csv_line.strip().split(',')
        if len(values) != 31:  # 1 Time + 28 V + 1 Amount + 1 Class
            return "Error: Expected 31 columns in the CSV row."

        # Map V1..V28
        feature_dict = {f"V{i}": float(values[i]) for i in range(1,29)}
        # Amount
        feature_dict["Amount"] = float(values[29])
        # Ignore Time (values[0]) and Class (values[30])

        return run_prediction(feature_dict, model_choice)
    except Exception as e:
        return f"Error parsing input: {e}"

# =====================================
# 6. Gradio Interface (Tabs)
# =====================================
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

with gr.Blocks() as iface:
    gr.Markdown("# Credit Card Fraud Detection")

    with gr.Tabs():
        # ------------------------- CSV Upload ---------------------------
        with gr.Tab("Upload CSV File"):
            gr.Markdown("Upload a CSV containing **V1..V28 and Amount**. Only the **first row input** is used.")

            csv_file = gr.File(label="Upload CSV")
            model_select_2 = gr.Dropdown(["XGBoost", "RandomForest", "MLP"], label="Model")

            csv_output = gr.Textbox(label="Prediction")
            gr.Button("Predict From CSV").click(
                predict_csv,
                inputs=[csv_file, model_select_2],
                outputs=csv_output
            )

        # ------------------------- CSV Text Input -------------------------
        with gr.Tab("CSV Text Input"):
            gr.Markdown("Paste a single CSV row (Time,V1..V28,Amount,Class). (Time) and (Class) will be ignored and filtered out.")
            csv_text_input = gr.Textbox(lines=2, placeholder="Paste CSV row here")
            model_select_3 = gr.Dropdown(["XGBoost", "RandomForest", "MLP"], label="Model")
            text_output = gr.Textbox(label="Prediction")
            gr.Button("Predict From CSV Text").click(
                predict_csv_text,
                inputs=[csv_text_input, model_select_3],
                outputs=text_output
            )


        # ------------------------- Manual Input -------------------------
        with gr.Tab("Manual Input"):
            manual_inputs = [gr.Number(label=f) for f in feature_names]
            model_select_1 = gr.Dropdown(["XGBoost", "RandomForest", "MLP"], label="Model")
            manual_inputs.append(model_select_1)

            manual_output = gr.Textbox(label="Prediction")
            gr.Button("Predict").click(
                predict_manual,
                inputs=manual_inputs,
                outputs=manual_output
            )



# =====================================
# 7. Launch
# =====================================
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
