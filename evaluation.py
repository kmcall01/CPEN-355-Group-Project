import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
import random
import kagglehub

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW = 64
HORIZON = 5
NUM_STOCKS = 5
STEP = 5

# =========================
# FEATURES (MUST MATCH TRAINING)
# =========================
def compute_features(df):
    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df["ret"] = df["Close"].pct_change()
    df["logret"] = np.log(df["Close"]).diff()
    df["mom5"] = df["Close"].pct_change(5)
    df["vol"] = df["ret"].rolling(20).std()
    df["ma_ratio"] = df["Close"] / df["Close"].rolling(20).mean()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["vol_change"] = df["Volume"].pct_change()

    df = df.dropna()
    return df


def normalize(X):
    X = np.nan_to_num(X)
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1
    return (X - mu) / std


# =========================
# MODELS
# =========================
class AlphaModel(nn.Module):
    def __init__(self, n_features=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW * n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.net(x).squeeze(-1)


class LogisticAlpha(nn.Module):
    def __init__(self, n_features=7):
        super().__init__()
        self.linear = nn.Linear(WINDOW * n_features, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.linear(x).squeeze(-1)


# =========================
# MODEL LOADER
# =========================
def load_model(path, n_features):
    state = torch.load(path, map_location=DEVICE)

    if "net.0.weight" in state:
        print("Loading Neural Network model")
        model = AlphaModel(n_features)
    else:
        print("Loading Logistic Regression model")
        model = LogisticAlpha(n_features)

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# =========================
# RUN PREDICTIONS (FIXED)
# =========================
def run_predictions(df, model, use_logistic=False):

   
    feats = df[[
        "ret",
        "logret",
        "mom5",
        "vol",
        "ma_ratio",
        "hl_range",
        "vol_change"
    ]].values

    prices = df["Close"].values

    preds, actuals, times = [], [], []

    for i in range(0, len(df) - WINDOW - HORIZON, STEP):

        X = feats[i:i+WINDOW]

        if not np.isfinite(X).all():
            continue

        X = normalize(X)
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(X_t).cpu().item()

        # logistic model probability
        if use_logistic:
            pred = 1 / (1 + np.exp(-out))
        else:
            pred = out

        p0 = prices[i + WINDOW]
        p1 = prices[i + WINDOW + HORIZON]

        actual = (p1 - p0) / p0

        preds.append(pred)
        actuals.append(actual)
        times.append(i + WINDOW + HORIZON)

    return np.array(preds), np.array(actuals), np.array(times)


# =========================
# PLOT
# =========================
def plot_stock(df, preds, actuals, times, title):
    close = df["Close"].values

    plt.figure(figsize=(14, 5))
    plt.plot(close, linewidth=1)

    for p, a, t in zip(preds, actuals, times):
        if t >= len(close):
            continue

        correct = np.sign(p - 0.5 if p <= 1 else p) == np.sign(a)

        color = "green" if correct else "red"
        marker = "o" if correct else "x"

        plt.scatter(t, close[t], color=color, s=25, marker=marker)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():

    print("Loading dataset...")

    path = kagglehub.dataset_download(
        "borismarjanovic/price-volume-data-for-all-stocks-etfs"
    )

    files = glob(os.path.join(path, "Stocks", "*.txt"))
    files = random.sample(files, NUM_STOCKS)

    print("Loading models...")

    nn_model = load_model("alpha_model.pth", n_features=7)
    log_model = load_model("logistic_alpha.pth", n_features=7)

    print("\nEvaluating models...\n")

    for f in files:

        try:
            df = pd.read_csv(f)
            df = compute_features(df)
        except:
            continue

        if len(df) < WINDOW + 100:
            continue

        # =====================
        # NN evaluation
        # =====================
        preds, actuals, times = run_predictions(df, nn_model, use_logistic=False)

        if len(preds) > 0:
            acc = (np.sign(preds) == np.sign(actuals)).mean()
            print(f"[NN] {os.path.basename(f)} | directional accuracy: {acc:.2%}")
            plot_stock(df, preds, actuals, times, f"NN - {os.path.basename(f)}")

        # =====================
        # Logistic evaluation
        # =====================
        preds, actuals, times = run_predictions(df, log_model, use_logistic=True)

        if len(preds) > 0:
            acc = (np.sign(preds - 0.5) == np.sign(actuals)).mean()
            print(f"[LOG] {os.path.basename(f)} | directional accuracy: {acc:.2%}")
            plot_stock(df, preds, actuals, times, f"LOG - {os.path.basename(f)}")


if __name__ == "__main__":
    main()
