# ---------------------------------------------------------------------
#  Single-file demo: Alpha Vantage → Bayesian HMM → POMDP allocator
#  (c) 2025  —  free-tier safe (≤5 calls / 24 h thanks to on-disk cache)
# ---------------------------------------------------------------------
import os, time, pickle, datetime as dt, functools, requests
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# =====================================================================
# 0.  Small disk-cache helper
# =====================================================================
_CACHE_DIR = Path.home() / ".av_cache"
_CACHE_DIR.mkdir(exist_ok=True)

def _cache_paths(name: str):
    return _CACHE_DIR / f"{name}.pkl", _CACHE_DIR / f"{name}.stamp"

def _cache_load(name: str, max_age_h: int = 24):
    pkl, stamp = _cache_paths(name)
    if not (pkl.exists() and stamp.exists()):
        return None
    age = dt.datetime.utcnow() - dt.datetime.fromisoformat(stamp.read_text())
    if age.total_seconds() > max_age_h * 3600:
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)

def _cache_save(name: str, obj):
    pkl, stamp = _cache_paths(name)
    with open(pkl, "wb") as f:
        pickle.dump(obj, f)
    stamp.write_text(dt.datetime.utcnow().isoformat())


# =====================================================================
# 1.  Alpha Vantage fetcher  (FREE endpoints only, with cache)
# =====================================================================
class AlphaVantageFetcher:
    BASE = "https://www.alphavantage.co/query"
    ECON_FUNCS = {
        "FED_RATE":     ("FEDERAL_FUNDS_RATE", "monthly"),
        "CPI":          ("CPI",                "monthly"),
        "UNEMPLOYMENT": ("UNEMPLOYMENT",      "monthly"),
        "GDP":          ("REAL_GDP",          "quarterly"),
    }

    def __init__(self, api_key: str, max_retries: int = 5):
        if not api_key:
            raise ValueError("Alpha Vantage API key required")
        self.key, self.retries = api_key, max_retries

    # ---------- internal REST helper ----------------------------------
    def _call_api(self, **params) -> dict:
        params["apikey"] = self.key
        tries, backoff = 0, 12
        while True:
            js = requests.get(self.BASE, params=params, timeout=30).json()
            if any(k in js for k in ("Note", "Information")):
                tries += 1
                if tries > self.retries:
                    raise RuntimeError(js)
                time.sleep(backoff * tries)
                continue
            if "Error Message" in js:
                raise RuntimeError(js["Error Message"])
            return js

    # ---------- daily prices (free) -----------------------------------
    def get_equity_daily(self, symbol: str, full: bool = True) -> pd.DataFrame:
        key = f"price_{symbol}_{'full' if full else 'compact'}"
        cached = _cache_load(key)
        if cached is not None:
            return cached

        js = self._call_api(function="TIME_SERIES_DAILY",
                            symbol=symbol,
                            outputsize="full" if full else "compact")
        if "Time Series (Daily)" not in js:
            raise RuntimeError(js)
        df = (pd.DataFrame(js["Time Series (Daily)"]).T
                .apply(pd.to_numeric, errors="coerce"))
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={"4. close": "close", "5. volume": "volume"})
        df["returns"] = np.log(df["close"]).diff()
        _cache_save(key, df)
        return df

    # ---------- macro indicators (free) -------------------------------
    def get_indicator(self, tag: str) -> pd.DataFrame:
        key = f"econ_{tag}"
        cached = _cache_load(key)
        if cached is not None:
            return cached

        func, interval = self.ECON_FUNCS[tag]
        js = self._call_api(function=func, interval=interval)
        rows = [(pd.to_datetime(d["date"]),
                 pd.to_numeric(str(d["value"]).replace("%", ""), errors="coerce"))
                for d in js["data"]]
        df = pd.DataFrame(rows, columns=["date", tag]).set_index("date")
        _cache_save(key, df)
        return df

    # ---------- combined data frame -----------------------------------
    def build_dataset(self, symbol="SPY") -> pd.DataFrame:
        df = self.get_equity_daily(symbol)
        for tag in self.ECON_FUNCS:
            df = df.join(self.get_indicator(tag), how="left")
        df = df.ffill().bfill()
        return df.dropna()


# =====================================================================
# 2.  Two-state Bayesian HMM (with scaling factors)
# =====================================================================
class BayesianHMM:
    def __init__(self, n_states=2):
        self.K = n_states
        self.A = np.eye(self.K) * 0.9 + np.ones((self.K, self.K)) * 0.1 / (self.K - 1)
        self.means = self.covs = None

    def _init_params(self, X):
        self.means = np.array([X.mean(axis=0), X.mean(axis=0) * 1.1])
        self.covs  = np.array([np.cov(X.T)] * self.K)

    # ---------- scaled forward–backward --------------------------------
    def _forward_backward(self, X):
        T = len(X)
        alpha, beta = np.zeros((T, self.K)), np.zeros((T, self.K))
        c = np.zeros(T)

        # forward
        alpha[0] = [multivariate_normal.pdf(X[0], self.means[k], self.covs[k])
                    for k in range(self.K)]
        c[0] = 1.0 / max(alpha[0].sum(), 1e-300)
        alpha[0] *= c[0]

        for t in range(1, T):
            for k in range(self.K):
                alpha[t, k] = multivariate_normal.pdf(X[t], self.means[k], self.covs[k]) * \
                               np.dot(alpha[t-1], self.A[:, k])
            c[t] = 1.0 / max(alpha[t].sum(), 1e-300)
            alpha[t] *= c[t]

        # backward
        beta[-1] = c[-1]
        for t in range(T - 2, -1, -1):
            for k in range(self.K):
                beta[t, k] = np.dot(
                    self.A[k, :],
                    [multivariate_normal.pdf(X[t+1], self.means[j], self.covs[j]) *
                     beta[t+1, j] for j in range(self.K)]
                )
            beta[t] *= c[t]
        return alpha, beta, c

    # ---------- Baum–Welch EM -----------------------------------------
    def fit(self, df: pd.DataFrame, n_iter: int = 50):
        X = df.values
        self._init_params(X)

        for _ in range(n_iter):
            a, b, _ = self._forward_backward(X)
            gamma = a * b
            gamma /= gamma.sum(axis=1, keepdims=True)

            # M-step: means & covs
            for k in range(self.K):
                w = gamma[:, k][:, None]
                self.means[k] = (w * X).sum(axis=0) / w.sum()
                diff = X - self.means[k]
                self.covs[k] = (w * diff).T @ diff / w.sum()

            # M-step: transitions
            xi = np.zeros((self.K, self.K))
            for t in range(len(X) - 1):
                denom = sum(a[t, i] * self.A[i, j] *
                            multivariate_normal.pdf(X[t+1], self.means[j], self.covs[j]) *
                            b[t+1, j]
                            for i in range(self.K) for j in range(self.K))
                for i in range(self.K):
                    for j in range(self.K):
                        xi[i, j] += a[t, i] * self.A[i, j] * \
                                    multivariate_normal.pdf(X[t+1], self.means[j], self.covs[j]) * \
                                    b[t+1, j] / denom
            self.A = xi / xi.sum(axis=1, keepdims=True)

    # ---------- emission likelihood -----------------------------------
    def emission_prob(self, x):
        return np.array([multivariate_normal.pdf(x, m, c) for m, c in zip(self.means, self.covs)])


# =====================================================================
# 3.  POMDP allocator
# =====================================================================
class AssetAllocationPOMDP:
    def __init__(self, data: pd.DataFrame):
        self.data = data.reset_index(drop=True)
        self.t, self.wealth = 0, 1_000_000.0
        self.wealth_hist = [self.wealth]

        self.hmm = BayesianHMM(); self.hmm.fit(data[["returns","FED_RATE","CPI","GDP"]])
        self.actions = [np.array([.7,.2,.1]), np.array([.4,.4,.2]), np.array([.2,.3,.5])]

    def _obs(self):
        return self.data.loc[self.t, ["returns","FED_RATE","CPI","GDP"]].values
    
    def belief(self):
        obs_prob = self.hmm.emission_prob(self._obs())
        # Assuming no prior info, use uniform prior or use forward algorithm to get filtered beliefs
        prior = np.array([0.5, 0.5])  # or the last time step's filtered belief if available
        posterior = obs_prob * prior
        return posterior / posterior.sum()

    def belief(self):
        probs = self.hmm.emission_prob(self._obs())
        return probs / probs.sum()  # Normalize to sum to 1
    
    def step(self, a_idx: int):
        r_eq_log = self._obs()[0]                 
        r_eq = np.expm1(r_eq_log)             
        r_bond, r_cash = 0.02, -0.01             

        old_wealth = self.wealth
        self.wealth *= 1 + self.actions[a_idx] @ np.array([r_eq, r_bond, r_cash])
        self.t = min(self.t + 1, len(self.data) - 1)
        self.wealth_hist.append(self.wealth)

        return np.log(self.wealth / old_wealth)


# =====================================================================
# 4.  Main
# =====================================================================
if __name__ == "__main__":
    fetcher = AlphaVantageFetcher(api_key="API_KEY") # Please create your own API key through Alpha Vantage
    df = fetcher.build_dataset("SPY")                           

    pomdp = AssetAllocationPOMDP(df)

    n_steps, rng = 252, np.random.default_rng(0)
    rewards, beliefs = [], []
    actions_chosen = []

    for _ in range(n_steps):
        a = rng.integers(0, 3)  # Random action selection (can be improved to policy)
        actions_chosen.append(a)
        rewards.append(pomdp.step(a))
        beliefs.append(pomdp.belief())

    # Convert actions to allocations weights
    allocations = np.array([pomdp.actions[a] for a in actions_chosen])

    x_wealth = np.arange(n_steps + 1)
    x_returns = np.arange(n_steps)

    plt.figure(figsize=(12, 10))

    init_cap = 1_000_000.0
    bh_wealth = init_cap * np.exp(np.cumsum(df["returns"].values[: n_steps + 1]))

    # 1) Portfolio wealth plot
    plt.subplot(4, 1, 1)
    plt.plot(x_wealth, pomdp.wealth_hist, label="HMM–POMDP", color="#0F2540")
    plt.plot(x_wealth, bh_wealth,        label="Buy-&-Hold SPY", color="#F17C67", alpha=0.75)
    plt.title("Portfolio wealth")
    plt.ylabel("USD")
    plt.legend()

    # 2) Returns comparison plot
    plt.subplot(4, 1, 2)
    plt.plot(x_returns, df["returns"].values[:n_steps], label="Market", color="#F17C67")
    plt.plot(x_returns, rewards, label="Strategy", color="#0F2540")
    plt.legend()
    plt.title("Returns")
    plt.ylabel("Return")

    # 3) Portfolio Allocation Over Time
    plt.subplot(4, 1, 3)
    plt.stackplot(
        x_returns,
        allocations[:, 0],  # Equities
        allocations[:, 1],  # Bonds
        allocations[:, 2],  # Cash
        labels=["Equities", "Bonds", "Cash"],
        colors=["#4A475C", "#CEAEB9", "#E9CEC3"]
    )
    plt.title("Portfolio Allocation Over Time")
    plt.ylabel("Allocation weight")
    plt.legend(loc="upper left")
    plt.ylim(0, 1)

    # 4) Rolling Volatility of Strategy Returns
    window = 20  # days rolling window
    returns_series = np.array(rewards)
    rolling_volatility = (
        pd.Series(returns_series).rolling(window=window).std()
        .fillna(method='bfill')
        .values
    )

    plt.subplot(4, 1, 4)
    plt.plot(x_returns, rolling_volatility, color="#0F2540")
    plt.title(f"{window}-Day Rolling Volatility of Strategy Returns")
    plt.xlabel("Trading days")
    plt.ylabel("Volatility (std dev)")

    plt.tight_layout()
    plt.show()