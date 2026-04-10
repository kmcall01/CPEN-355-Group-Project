import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

from optimizer import run_optimizer

LR = 0.01
N_ITERS = 500
TRAIN_STOCKS = 30


# ─────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────
def add_indicators(df):

    df = df.copy()

    df['Return'] = df['Close'].pct_change()

    df['MA_5'] = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()

    df['EMA_10'] = df['Close'].ewm(span=10).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()

    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()

    df['MACD'] = ema12 - ema26

    return df


# ─────────────────────────────────────────────
# Load stock
# ─────────────────────────────────────────────
def load_random_stock(path, file):

    df = pd.read_csv(os.path.join(path, "Stocks", file))

    if len(df) < 50:
        raise ValueError("Stock too short")

    df = add_indicators(df).dropna()

    if len(df) < 50:
        raise ValueError("Indicators removed too much data")

    features = [
        'Open','High','Low','Close','Volume',
        'Return','MA_5','MA_10','EMA_10','RSI','MACD'
    ]

    X = df[features].values
    close = df['Close'].values

    # alpha target
    fwd_return = np.roll(close, -5) / close - 1

    momentum = df['MA_5'] - df['MA_10']
    mean_reversion = close - df['EMA_10']

    alpha = (
        fwd_return
        + 0.1 * momentum.values
        - 0.1 * mean_reversion.values
    )

    alpha = alpha[:-5]
    X = X[:-5]
    close = close[:-5]

    if len(X) < 50:
        raise ValueError("Not enough usable samples")

    # normalize features
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X = (X - X.mean(axis=0)) / X_std

    # normalize alpha
    alpha_std = alpha.std()
    if alpha_std == 0:
        alpha_std = 1
    alpha = (alpha - alpha.mean()) / alpha_std

    return X, alpha, close


# ─────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────
def get_alpha_problem(X, alpha):

    N = len(X)
    REG = 0.01

    def f(w):

        pred = X @ w
        position = np.tanh(pred)

        pnl = np.mean(position * alpha)
        penalty = REG * np.mean(position**2)

        return -(pnl - penalty)

    def grad_f(w):

        pred = X @ w
        position = np.tanh(pred)

        dpos = 1 - position**2

        pnl_grad = (X.T @ (dpos * alpha)) / N
        penalty_grad = REG * (X.T @ (dpos * position)) / N

        return -(pnl_grad - penalty_grad)

    return f, grad_f


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def visualize_trading(close, pred_signal, alpha, stock_name):

    if len(close) < 2:
        return

    position = np.tanh(pred_signal)

    pred_dir = np.sign(position)

    true_dir = np.sign(
        np.diff(close, append=close[-1])
    )

    correct = pred_dir == true_dir

    pnl = np.cumsum(position * alpha)

    fig = plt.figure(figsize=(14,8))

    # price panel
    ax1 = plt.subplot(3,1,1)

    ax1.plot(close, color="black")

    for i in range(len(close)-1):

        if position[i] > 0:
            ax1.axvspan(i, i+1, color="green", alpha=0.05)
        else:
            ax1.axvspan(i, i+1, color="red", alpha=0.05)

    ax1.set_title(f"{stock_name} Price with Long/Short Zones")

    # accuracy panel
    ax2 = plt.subplot(3,1,2)

    for i in range(len(close)-1):

        color = "green" if correct[i] else "red"

        ax2.plot(
            [i,i+1],
            [close[i],close[i+1]],
            color=color,
            linewidth=2
        )

    ax2.set_title("Prediction Accuracy")

    # pnl panel
    ax3 = plt.subplot(3,1,3)

    ax3.plot(pnl, label="Strategy PnL")

    ax3.set_title("Cumulative Strategy PnL")
    ax3.legend()

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Build training dataset
# ─────────────────────────────────────────────
def build_training_dataset(path, files, n_stocks):

    chosen = np.random.choice(files, n_stocks, replace=False)

    Xs = []
    alphas = []

    for f in chosen:

        try:

            X, alpha, _ = load_random_stock(path, f)

            Xs.append(X)
            alphas.append(alpha)

        except:
            continue

    if len(Xs) == 0:
        raise RuntimeError("No valid training stocks found")

    X_all = np.vstack(Xs)
    alpha_all = np.concatenate(alphas)

    return X_all, alpha_all


# ─────────────────────────────────────────────
# Evaluate random stocks
# ─────────────────────────────────────────────
def evaluate_on_random_stocks(path, files, w, n=5):

    chosen = np.random.choice(files, n, replace=False)

    corrs = []
    pnls = []
    used_names = []

    for f in chosen:

        try:

            X, alpha, close = load_random_stock(path, f)

            pred = X @ w
            position = np.tanh(pred)

            corr = np.corrcoef(position, alpha)[0,1]
            pnl = np.mean(position * alpha)

            corrs.append(corr)
            pnls.append(pnl)
            used_names.append(f)

            visualize_trading(close, pred, alpha, f)

            print(f"\nStock: {f}")
            print(f"Correlation: {corr:.4f}")
            print(f"Alpha Capture: {pnl:.6f}")

        except:
            continue

    if len(corrs) == 0:
        print("No valid evaluation stocks")
        return

    # summary chart
    plt.figure(figsize=(10,4))

    x = np.arange(len(corrs))

    plt.bar(x-0.2, corrs, width=0.4, label="Correlation")
    plt.bar(x+0.2, pnls, width=0.4, label="Alpha Capture")

    plt.xticks(x, used_names, rotation=45)

    plt.legend()
    plt.title("Model Performance")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():

    path = kagglehub.dataset_download(
        "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    )

    files = os.listdir(os.path.join(path,"Stocks"))

    X_train, alpha_train = build_training_dataset(
        path,
        files,
        TRAIN_STOCKS
    )

    f, grad_f = get_alpha_problem(X_train, alpha_train)

    w0 = np.zeros(X_train.shape[1])

    results = run_optimizer(
        f,
        grad_f,
        w0,
        learning_rates=[LR],
        n_iters=N_ITERS,
        method="adam"
    )

    ws, fs = results[LR]
    w_final = ws[-1]

    print("Training complete")

    evaluate_on_random_stocks(
        path,
        files,
        w_final,
        n=5
    )


if __name__ == "__main__":
    main()

