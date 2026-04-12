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
EPOCHS = 100
SAMPLES = 700
LR = 1e-4

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


# =========================
# BUILD DATASET 
# =========================
def build_dataset(files):
    data_by_date = {}

    for f in files:
        try:
            df = pd.read_csv(f)
            df = compute_features(df)
        except:
            continue

        if len(df) < WINDOW + 10:
            continue

        feats = df[[
            "ret", "logret", "mom5", "vol",
            "ma_ratio", "hl_range", "vol_change"
        ]].values

        prices = df["Close"].values
        dates = pd.to_datetime(df["Date"]).values

        for i in range(len(df) - WINDOW - 5):
            X = feats[i:i+WINDOW]
            y = (prices[i+WINDOW+5] - prices[i+WINDOW]) / prices[i+WINDOW]
            date = dates[i+WINDOW]

            if not np.isfinite(X).all() or not np.isfinite(y):
                continue

            if date not in data_by_date:
                data_by_date[date] = {"X": [], "y": []}

            data_by_date[date]["X"].append(X)
            data_by_date[date]["y"].append(y)

    
    for date in data_by_date:
        data_by_date[date]["X"] = np.stack(data_by_date[date]["X"])
        data_by_date[date]["y"] = np.array(data_by_date[date]["y"])

    return data_by_date


# =========================
# NORMALIZATION
# =========================
def compute_normalization(data_by_date):
    all_X = []

    for date in data_by_date:
        all_X.append(data_by_date[date]["X"])

    all_X = np.concatenate(all_X, axis=0)

    mu = all_X.mean(axis=0)
    std = all_X.std(axis=0)
    std[std < 1e-6] = 1

    return mu, std


def apply_normalization(data_by_date, mu, std):
    for date in data_by_date:
        data_by_date[date]["X"] = (data_by_date[date]["X"] - mu) / std


# =========================
# MODEL
# =========================
class AlphaModel(nn.Module):
    def __init__(self, n_features):
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


# =========================
# RANKING LOSS
# =========================
def ranking_loss(pred, target):
    pred = pred - pred.mean()
    target = target - target.mean()

    cov = (pred * target).mean()
    pred_std = pred.std() + 1e-6
    target_std = target.std() + 1e-6

    corr = cov / (pred_std * target_std)
    return -corr


# =========================
# TRAIN
# =========================
def train(model, train_dates, data_by_date):
    opt = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        random.shuffle(train_dates)

        for date in train_dates:
            batch = data_by_date[date]

            if len(batch["X"]) < 50:
                continue

            X = torch.from_numpy(batch["X"]).float().to(DEVICE)
            y = torch.from_numpy(batch["y"]).float().to(DEVICE)

            pred = model(X)
            loss = ranking_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"epoch {epoch} loss {np.mean(losses):.6f}")


# =========================
# BACKTEST
# =========================
def backtest(model, val_dates, data_by_date):
    model.eval()

    equity = 1.0
    curve = []

    with torch.no_grad():
        for date in sorted(val_dates):
            batch = data_by_date[date]

            if len(batch["X"]) < 50:
                continue

            X = torch.from_numpy(batch["X"]).float().to(DEVICE)
            y = batch["y"]

            pred = model(X).cpu().numpy()

            df = pd.DataFrame({
                "pred": pred,
                "ret": y
            })

            df = df.sort_values("pred")

            longs = df.tail(TOP_K)
            shorts = df.head(TOP_K)

            pnl = longs["ret"].mean() - shorts["ret"].mean()

            pnl -= FEE  # realistic cost

            equity *= (1 + pnl)
            curve.append(equity)

    return np.array(curve)


# =========================
# SHARPE
# =========================
def sharpe(curve):
    rets = np.diff(curve) / curve[:-1]
    if len(rets) < 2 or rets.std() == 0:
        return 0
    return np.sqrt(252) * rets.mean() / rets.std()


# =========================
# MAIN
# =========================
def main():

    path = kagglehub.dataset_download(
        "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    )

    files = glob(os.path.join(path, "Stocks", "*.txt"))
    files = random.sample(files, SAMPLES)

    print("stocks:", len(files))

    print("building dataset...")
    data_by_date = build_dataset(files)

    dates = sorted(data_by_date.keys())
    print("total dates:", len(dates))

    split = int(0.8 * len(dates))
    train_dates = dates[:split]
    val_dates = dates[split:]

    print("computing normalization...")
    train_data = {d: data_by_date[d] for d in train_dates}
    mu, std = compute_normalization(train_data)
    apply_normalization(data_by_date, mu, std)

    n_features = data_by_date[dates[0]]["X"].shape[2]

    model = AlphaModel(n_features).to(DEVICE)

    print("training...")
    train(model, train_dates, data_by_date)

    print("backtesting...")
    curve = backtest(model, val_dates, data_by_date)

    s = sharpe(curve)
    print("\nFINAL SHARPE:", s)

    torch.save(model.state_dict(), "alpha_model.pth")
    print("Saved -> alpha_model.pth")


if __name__ == "__main__":
    main()
