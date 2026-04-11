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
STEP = 5  # skip to reduce clutter

# =========================
# FEATURES (must match training)
# =========================
def compute_features(df):
    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()

    df["ret"] = df["Close"].pct_change()
    df["logret"] = np.log(df["Close"]).diff()
    df["mom"] = df["Close"].diff(5)
    df["vol"] = df["ret"].rolling(20).std()

    df = df.dropna()
    return df


def normalize(X):
    X = np.nan_to_num(X)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1
    return (X - mu) / sd


# =========================
# MODEL
# =========================
class AlphaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.net(x)


# =========================
# RUN SLIDING PREDICTIONS
# =========================
def run_predictions(df, model):
    feats = df[["ret", "logret", "mom", "vol"]].values
    prices = df["Close"].values

    preds = []
    actuals = []
    times = []

    for i in range(0, len(df) - WINDOW - HORIZON, STEP):

        X = feats[i:i+WINDOW]
        if not np.isfinite(X).all():
            continue

        X = normalize(X)
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(X_t).cpu().item()

        p0 = prices[i + WINDOW]
        p1 = prices[i + WINDOW + HORIZON]

        actual = (p1 - p0) / p0

        preds.append(pred)
        actuals.append(actual)
        times.append(i + WINDOW + HORIZON)

    return np.array(preds), np.array(actuals), np.array(times), prices


# =========================
# PLOT SINGLE STOCK
# =========================
def plot_stock(df, preds, actuals, times, title):
    close = df["Close"].values

    plt.figure(figsize=(14, 5))
    plt.plot(close, color="black", linewidth=1, label="Close Price")

    for p, a, t in zip(preds, actuals, times):

        if t >= len(close):
            continue

        correct = np.sign(p) == np.sign(a)

        color = "green" if correct else "red"
        marker = "o" if correct else "x"

        plt.scatter(t, close[t],
                    color=color,
                    s=25,
                    marker=marker)

    plt.title(f"{title} | Green=Correct Direction, Red=Wrong")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# MAIN
# =========================
def main():

    print("Loading dataset...")

    path = kagglehub.dataset_download(
        "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    )

    files = glob(os.path.join(path, "Stocks", "*.txt"))
    files = random.sample(files, NUM_STOCKS)

    print("Loading model...")

    model = AlphaModel().to(DEVICE)
    model.load_state_dict(torch.load("alpha_model.pth", map_location=DEVICE))
    model.eval()

    print("Evaluating stocks...\n")

    for f in files:

        try:
            df = pd.read_csv(f)
            df = compute_features(df)
        except:
            continue

        if len(df) < WINDOW + 100:
            continue

        preds, actuals, times, prices = run_predictions(df, model)

        if len(preds) == 0:
            continue

        acc = (np.sign(preds) == np.sign(actuals)).mean()
        print(f"{os.path.basename(f)} | directional accuracy: {acc:.2%}")

        plot_stock(df, preds, actuals, times, os.path.basename(f))


if __name__ == "__main__":
    main()