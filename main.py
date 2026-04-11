import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from glob import glob
import kagglehub
import random

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW = 64
TOP_K = 20
FEE = 0.0005
EPOCHS = 50
SAMPLES = 500
LR = 1e-3


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
# MODEL (returns prediction)
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
# BUILD GLOBAL CROSS-SECTIONAL DATASET
# =========================
def load_all_data(files):
    X_all = []
    y_all = []
    stock_ids = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df = compute_features(df)
        except:
            continue

        if len(df) < WINDOW + 10:
            continue

        feats = df[["ret", "logret", "mom", "vol"]].values
        prices = df["Close"].values

        for i in range(len(df) - WINDOW - 5):

            X = feats[i:i+WINDOW]
            if not np.isfinite(X).all():
                continue

            X = normalize(X)

            y = (prices[i+WINDOW+5] - prices[i+WINDOW]) / prices[i+WINDOW]

            if not np.isfinite(y):
                continue

            X_all.append(X)
            y_all.append(y)
            stock_ids.append(f)

    return np.array(X_all), np.array(y_all), np.array(stock_ids)


# =========================
# CROSS-SECTIONAL BACKTEST
# =========================
def backtest_cross_section(model, X, y, stock_ids):
    model.eval()

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten()

    # build dataframe
    df = pd.DataFrame({
        "pred": preds,
        "ret": y,
        "stock": stock_ids
    })

    equity = 1.0
    curve = []

    # simulate daily cross-section trading
    for i in range(0, len(df), 1000):

        batch = df.iloc[i:i+1000]
        if len(batch) < 50:
            continue

        # rank stocks
        batch = batch.sort_values("pred")

        longs = batch.tail(TOP_K)
        shorts = batch.head(TOP_K)

        long_ret = longs["ret"].mean()
        short_ret = shorts["ret"].mean()

        pnl = long_ret - short_ret

        # transaction cost
        pnl -= FEE * 2 * TOP_K

        equity *= (1 + pnl)
        curve.append(equity)

    return np.array(curve)


# =========================
# SHARPE RATIO
# =========================
def sharpe(curve):
    rets = np.diff(curve) / curve[:-1]
    if rets.std() == 0:
        return 0
    return np.sqrt(252) * rets.mean() / rets.std()


# =========================
# TRAIN
# =========================
def train(model, X, y, epochs=3):
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(DEVICE)

    for e in range(epochs):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"epoch {e} loss {loss.item():.6f}")


# =========================
# MAIN PIPELINE
# =========================
def main():

    path = kagglehub.dataset_download(
        "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    )

    files = glob(os.path.join(path, "Stocks", "*.txt"))
    files = random.sample(files, SAMPLES)  # controlled research set

    print("stocks:", len(files))

    print("building dataset...")
    X, y, stock_ids = load_all_data(files)

    print("total samples:", len(X))

    # train/val split (time-aware approximate)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    ids_val = stock_ids[split:]

    model = AlphaModel().to(DEVICE)

    print("training...")
    train(model, X_train, y_train, epochs=EPOCHS)

    print("backtesting...")
    curve = backtest_cross_section(model, X_val, y_val, ids_val)

    s = sharpe(curve)
    print("\nFINAL SHARPE:", s)

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), "alpha_model.pth")
    print("Saved model → alpha_model.pth")


if __name__ == "__main__":
    main()
