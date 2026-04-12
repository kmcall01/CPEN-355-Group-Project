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
NUM_STOCKS = 10  

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# FEATURES
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
class LSTMAlpha(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.2 if layers > 1 else 0.0
        )

        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),   
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class LogisticAlpha(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features * WINDOW, 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.linear(x).squeeze(-1)


# =========================
# LOAD MODEL
# =========================
def load_model(path, n_features):
    state = torch.load(path, map_location=DEVICE)

    if "lstm.weight_ih_l0" in state:
        print("Loading LSTM model")
        model = LSTMAlpha(n_features)
    else:
        print("Loading Logistic model")
        model = LogisticAlpha(n_features)

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# =========================
# PREDICTIONS
# =========================
def run_predictions(df, model, is_logistic=False):

    feats = df[[
        "ret", "logret", "mom5", "vol",
        "ma_ratio", "hl_range", "vol_change"
    ]].values

    prices = df["Close"].values

    preds, actuals, times = [], [], []

    for i in range(len(df) - WINDOW - HORIZON):

        X = feats[i:i+WINDOW]

        if not np.isfinite(X).all():
            continue

        X = normalize(X)
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(X_t).cpu().item()

        if is_logistic:
            pred = 1 / (1 + np.exp(-out))
            pred = pred - 0.5   # center it
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
# PLOT + SAVE
# =========================
def plot_and_save(df, preds, actuals, times, title, filename):

    close = df["Close"].values

    plt.figure(figsize=(14, 5))
    plt.plot(close, linewidth=1)

    correct_count = 0

    for p, a, t in zip(preds, actuals, times):

        if t >= len(close):
            continue

        correct = np.sign(p) == np.sign(a)
        if correct:
            correct_count += 1

        color = "green" if correct else "red"
        marker = "o" if correct else "x"

        plt.scatter(t, close[t], color=color, s=20, marker=marker)

    acc = correct_count / len(preds) if len(preds) > 0 else 0

    plt.title(f"{title} | Accuracy: {acc:.2%}")
    plt.xlabel("Time")
    plt.ylabel("Price")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close()


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

    print("Loading models...")

    lstm_model = load_model("lstm_alpha.pth", 7)
    log_model = load_model("logistic_alpha.pth", 7)

    print("\nEvaluating...\n")

    for f in files:

        name = os.path.basename(f)

        try:
            df = pd.read_csv(f)
            df = compute_features(df)
        except:
            continue

        if len(df) < WINDOW + 50:
            continue

        # =====================
        # LSTM
        # =====================
        preds, actuals, times = run_predictions(df, lstm_model, False)

        if len(preds) > 0:
            plot_and_save(
                df, preds, actuals, times,
                f"LSTM - {name}",
                f"LSTM_{name}.png"
            )

        # =====================
        # LOGISTIC
        # =====================
        preds, actuals, times = run_predictions(df, log_model, True)

        if len(preds) > 0:
            plot_and_save(
                df, preds, actuals, times,
                f"LOG - {name}",
                f"LOG_{name}.png"
            )


if __name__ == "__main__":
    main()
