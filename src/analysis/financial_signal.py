# src/analysis/financial_signal.py
"""
Cross-Sectional Return Dispersion via TDA: H₀ Gap as a Regime Signal

Applies the same persistent homology methodology from the income inequality
analysis to financial markets.  Each month, the cross-sectional distribution
of 49 industry portfolio returns is analysed for its largest adjacent gap
(H₀ feature lifespan).  A large gap indicates industry bifurcation — a
structural break in the return distribution that scalar dispersion misses.

Data:    Ken French 49 Industry Portfolios (free, 1926–present)
Signal:  Monthly H₀ gap vs naive cross-sectional std dev
Backtest: Volatility-timing strategy using gap as regime indicator
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import requests
import zipfile
import io

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent / "regression"
PLOT_DIR = BASE / "results" / "plots"

INDUSTRY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/49_Industry_Portfolios_CSV.zip"
)
FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/F-F_Research_Data_Factors_CSV.zip"
)
INDUSTRY_CACHE = DATA_DIR / "49_Industry_Portfolios.CSV"
FACTORS_CACHE = DATA_DIR / "F-F_Research_Data_Factors.CSV"

START_YEAR = 1963          # CRSP quality threshold
MIN_LOOKBACK = 60          # months before backtest begins

CRISES = [
    ("Black Monday", "1987-09", "1987-12"),
    ("LTCM", "1998-07", "1998-10"),
    ("Dot-Com", "2000-03", "2002-10"),
    ("GFC", "2007-12", "2009-06"),
    ("COVID", "2020-02", "2020-04"),
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _download_zip_csv(url: str, cache_path: Path) -> None:
    """Download a Ken French zip, extract the CSV, and cache locally."""
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[financial] Downloading {cache_path.name} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        with open(cache_path, "wb") as f:
            f.write(zf.read(names[0]))
    print(f"[financial] Cached → {cache_path}")


def _parse_french_monthly(csv_path: Path) -> pd.DataFrame:
    """Parse the first (monthly) data block from a Ken French CSV.

    Returns DataFrame indexed by month-end Timestamps with float columns.
    Values are in percentage points (2.37 means 2.37 %).
    """
    lines = csv_path.read_text(encoding="latin-1").splitlines()

    # Locate the first row whose first token is a 6-digit date (YYYYMM).
    data_start = header_idx = None
    for i, line in enumerate(lines):
        tok = line.split(",")[0].strip()
        if tok.isdigit() and len(tok) == 6:
            data_start = i
            for j in range(i - 1, -1, -1):
                if lines[j].strip():
                    header_idx = j
                    break
            break
    if data_start is None:
        raise ValueError(f"No monthly data block found in {csv_path}")

    # Read until first non-date row.
    data_end = len(lines)
    for i in range(data_start, len(lines)):
        tok = lines[i].split(",")[0].strip()
        if not tok or not tok.isdigit() or len(tok) != 6:
            data_end = i
            break

    headers = [h.strip() for h in lines[header_idx].split(",")]
    rows = [[p.strip() for p in lines[i].split(",")] for i in range(data_start, data_end)]

    df = pd.DataFrame(rows)
    if len(headers) >= df.shape[1]:
        df.columns = headers[: df.shape[1]]
    date_col = df.columns[0]
    df.index = pd.to_datetime(df[date_col].str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.drop(columns=[date_col])
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([-99.99, -999.0], np.nan)
    return df


def load_financial_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (industry_returns, ff_factors) aligned and trimmed."""
    _download_zip_csv(INDUSTRY_URL, INDUSTRY_CACHE)
    _download_zip_csv(FACTORS_URL, FACTORS_CACHE)

    ind = _parse_french_monthly(INDUSTRY_CACHE)
    ff = _parse_french_monthly(FACTORS_CACHE)

    common = ind.index.intersection(ff.index)
    ind, ff = ind.loc[common], ff.loc[common]
    mask = ind.index >= pd.Timestamp(f"{START_YEAR}-01-01")
    return ind.loc[mask], ff.loc[mask]


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def compute_signals(ind: pd.DataFrame) -> pd.DataFrame:
    """Monthly H₀ gap and naive dispersion across industry returns."""
    records = []
    for date, row in ind.iterrows():
        rets = row.dropna().values
        if len(rets) < 10:
            continue
        sorted_rets = np.sort(rets)
        diffs = np.diff(sorted_rets)
        gap_idx = int(np.argmax(diffs))
        records.append({
            "date": date,
            "h0_gap": diffs[gap_idx],
            "dispersion": float(np.std(rets, ddof=1)),
            "n_industries": len(rets),
            "gap_pctile": gap_idx / (len(rets) - 1) * 100,
        })
    return pd.DataFrame(records).set_index("date")


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def _backtest_timing(signal: pd.Series, mkt_excess: pd.Series,
                     rf: pd.Series) -> pd.DataFrame:
    """Binary timing: invest in market when signal < expanding median,
    otherwise park in risk-free.  No look-ahead bias."""
    common = signal.index.intersection(mkt_excess.index).intersection(rf.index)
    signal, mkt, rf_s = signal.loc[common], mkt_excess.loc[common], rf.loc[common]
    mkt_total = mkt + rf_s

    rows = []
    for i in range(MIN_LOOKBACK, len(signal) - 1):
        thresh = signal.iloc[: i + 1].median()
        is_equity = signal.iloc[i] < thresh
        rows.append({
            "date": signal.index[i + 1],
            "strategy_return": mkt_total.iloc[i + 1] if is_equity else rf_s.iloc[i + 1],
            "market_return": mkt_total.iloc[i + 1],
            "rf_return": rf_s.iloc[i + 1],
            "position": "equity" if is_equity else "risk-free",
        })
    return pd.DataFrame(rows).set_index("date")


def _metrics(returns: pd.Series, rf: pd.Series) -> dict:
    """Annualised performance metrics (returns in pct points)."""
    r = returns / 100
    rf_d = rf / 100
    n = len(r)
    cagr = (1 + r).prod() ** (12 / n) - 1 if n else 0
    vol = r.std() * np.sqrt(12)
    excess = r - rf_d
    sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0
    cum = (1 + r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return {
        "CAGR": f"{cagr:.1%}", "Vol": f"{vol:.1%}",
        "Sharpe": f"{sharpe:.2f}", "Max DD": f"{max_dd:.1%}",
        "Hit Rate": f"{(r > 0).mean():.1%}", "Months": str(n),
    }


def _partial_corr(x, y, z):
    """Partial Pearson r of x, y controlling for z."""
    rx = stats.linregress(z, x)
    ry = stats.linregress(z, y)
    return stats.pearsonr(
        x - (rx.slope * z + rx.intercept),
        y - (ry.slope * z + ry.intercept),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_financial_signal() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────
    ind, ff = load_financial_data()
    print(f"[financial] {len(ind)} months, {ind.shape[1]} industries, "
          f"{ind.index[0]:%Y-%m} to {ind.index[-1]:%Y-%m}")

    signals = compute_signals(ind)
    mkt, rf = ff["Mkt-RF"], ff["RF"]

    # ── predictive analysis ───────────────────────────────────────────────
    next_abs_mkt = mkt.shift(-1).abs()
    common = signals.index.intersection(next_abs_mkt.dropna().index)
    gap_s = signals.loc[common, "h0_gap"]
    disp_s = signals.loc[common, "dispersion"]
    vol_t = next_abs_mkt.loc[common]
    next_ret = mkt.shift(-1).reindex(common)

    r_gv, p_gv = stats.pearsonr(gap_s, vol_t)
    r_dv, p_dv = stats.pearsonr(disp_s, vol_t)
    r_part, p_part = _partial_corr(gap_s.values, vol_t.values, disp_s.values)
    r_gd, _ = stats.pearsonr(gap_s, disp_s)

    # ── backtest ──────────────────────────────────────────────────────────
    bt_gap = _backtest_timing(signals["h0_gap"], mkt, rf)
    bt_disp = _backtest_timing(signals["dispersion"], mkt, rf)
    common_bt = bt_gap.index.intersection(bt_disp.index)
    bt_gap, bt_disp = bt_gap.loc[common_bt], bt_disp.loc[common_bt]
    rf_bt = rf.reindex(common_bt).fillna(0)

    m_bh = _metrics(bt_gap["market_return"], rf_bt)
    m_gap = _metrics(bt_gap["strategy_return"], rf_bt)
    m_disp = _metrics(bt_disp["strategy_return"], rf_bt)

    # ── quintile analysis ─────────────────────────────────────────────────
    labels = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    qcut = pd.qcut(gap_s, 5, labels=labels)
    q_mean = next_ret.groupby(qcut).mean()
    q_std = next_ret.groupby(qcut).std()

    # ── report ────────────────────────────────────────────────────────────
    lines = [
        "=" * 70,
        "FINANCIAL SIGNAL: Cross-Sectional H₀ Gap as Regime Indicator",
        "=" * 70,
        f"Data: Ken French 49 Industry Portfolios, "
        f"{ind.index[0]:%Y-%m} – {ind.index[-1]:%Y-%m}",
        f"Signal months: {len(signals)}  |  Backtest months: {len(bt_gap)}",
        "",
        "--- Signal Properties ---",
        f"  Corr(gap, dispersion) = {r_gd:.4f}",
        "",
        "--- Predictive Power: Signal → Next-Month |Market Return| ---",
        f"  {'Signal':<25} {'r':>8} {'p':>12}",
        f"  {'-'*25} {'-'*8} {'-'*12}",
        f"  {'H₀ Gap':<25} {r_gv:>8.4f} {p_gv:>12.4e}",
        f"  {'Cross-Sect. Std Dev':<25} {r_dv:>8.4f} {p_dv:>12.4e}",
        f"  {'Gap | Dispersion':<25} {r_part:>8.4f} {p_part:>12.4e}",
        "",
        "--- Quintile Analysis: Next-Month Mkt Return by Gap Quintile ---",
        f"  {'Quintile':<12} {'Mean (%)':>10} {'Std (%)':>10}",
        f"  {'-'*12} {'-'*10} {'-'*10}",
    ]
    for q in labels:
        lines.append(f"  {q:<12} {q_mean[q]:>10.2f} {q_std[q]:>10.2f}")

    lines += [
        "",
        "--- Backtest: Vol-Timing (expanding median, no look-ahead) ---",
        f"  {'Metric':<12} {'Buy&Hold':>10} {'TDA Gap':>10} {'Dispersion':>10}",
        f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}",
    ]
    for k in ["CAGR", "Vol", "Sharpe", "Max DD", "Hit Rate", "Months"]:
        lines.append(f"  {k:<12} {m_bh[k]:>10} {m_gap[k]:>10} {m_disp[k]:>10}")

    lines += [
        "",
        "Interpretation:",
        "  The H₀ gap captures the largest structural break in the cross-",
        "  section of industry returns each month — where the distribution",
        "  'tears' between winner and loser industries.  Like the income",
        "  inequality analysis (gap vs Gini), this detects distributional",
        "  structure that a scalar summary (std dev) can miss.",
    ]
    report = "\n".join(lines)
    (OUT_DIR / "financial_signal.txt").write_text(report)
    print(report)

    # ── plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.28)

    # (a) signal time-series
    ax1 = fig.add_subplot(gs[0, :])
    gap_sm = signals["h0_gap"].rolling(12, min_periods=6).mean()
    ax1.plot(signals.index, signals["h0_gap"], color="steelblue",
             lw=0.4, alpha=0.35)
    ax1.plot(gap_sm.index, gap_sm, color="steelblue", lw=2,
             label="H₀ Gap (12-mo avg)")
    ax1b = ax1.twinx()
    disp_sm = signals["dispersion"].rolling(12, min_periods=6).mean()
    ax1b.plot(disp_sm.index, disp_sm, color="darkorange", lw=2,
              label="Dispersion (12-mo avg)")
    ax1b.set_ylabel("Dispersion (%)", color="darkorange")
    for name, s, e in CRISES:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if s_ts >= signals.index[0]:
            ax1.axvspan(s_ts, e_ts, alpha=0.12, color="red")
            mid = s_ts + (e_ts - s_ts) / 2
            ax1.text(mid, ax1.get_ylim()[1] * 0.92 if ax1.get_ylim()[1] else 1,
                     name, ha="center", fontsize=7, color="red")
    ax1.set(ylabel="H₀ Gap (pct pts)",
            title="Cross-Sectional H₀ Gap & Dispersion with Crisis Periods")
    ax1.legend(loc="upper left", fontsize=9)
    ax1b.legend(loc="upper right", fontsize=9)

    # (b) equity curves
    ax2 = fig.add_subplot(gs[1, :])
    cum_bh = (1 + bt_gap["market_return"] / 100).cumprod()
    cum_g = (1 + bt_gap["strategy_return"] / 100).cumprod()
    cum_d = (1 + bt_disp["strategy_return"] / 100).cumprod()
    ax2.plot(cum_bh.index, cum_bh, "gray", lw=1.3,
             label=f"Buy & Hold  Sharpe {m_bh['Sharpe']}")
    ax2.plot(cum_g.index, cum_g, "steelblue", lw=2,
             label=f"TDA Gap Timing  Sharpe {m_gap['Sharpe']}")
    ax2.plot(cum_d.index, cum_d, "darkorange", lw=1.5, ls="--",
             label=f"Dispersion Timing  Sharpe {m_disp['Sharpe']}")
    for name, s, e in CRISES:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if s_ts >= bt_gap.index[0]:
            ax2.axvspan(s_ts, e_ts, alpha=0.08, color="red")
    ax2.set(ylabel="Growth of $1 (log scale)", title="Backtest Equity Curves",
            yscale="log")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # (c) quintile bar chart
    ax3 = fig.add_subplot(gs[2, 0])
    colours = ["#2ecc71", "#82e0aa", "#f9e79f", "#f5b041", "#e74c3c"]
    bars = ax3.bar(range(5), q_mean.values, color=colours,
                   edgecolor="k", alpha=0.85)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.axhline(0, color="k", lw=0.5)
    for b, v in zip(bars, q_mean.values):
        ax3.text(b.get_x() + b.get_width() / 2, v + 0.03,
                 f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax3.set(xlabel="H₀ Gap Quintile",
            ylabel="Mean Next-Month Mkt Return (%)",
            title="Return by Gap Regime")

    # (d) gap vs future vol scatter
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.scatter(gap_s, vol_t, c="steelblue", alpha=0.12, s=8,
                edgecolor="none")
    z = np.polyfit(gap_s, vol_t, 1)
    xl = np.linspace(gap_s.min(), gap_s.quantile(0.99), 100)
    ax4.plot(xl, np.polyval(z, xl), "r-", lw=2)
    ax4.set(xlabel="H₀ Gap (pct pts)",
            ylabel="|Next-Month Mkt Return| (%)",
            title=f"Gap → Volatility  r={r_gv:.3f}  partial r={r_part:.3f}")

    fig.suptitle("Financial TDA: Industry Return Gap as Regime Signal",
                 fontsize=15, y=1.01)
    fig.savefig(PLOT_DIR / "financial_signal.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ── save signal CSV for dashboard ─────────────────────────────────────
    out_csv = BASE / "results" / "timeseries" / "financial_signal.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(out_csv)

    print(f"[financial] Saved plot   → {PLOT_DIR / 'financial_signal.png'}")
    print(f"[financial] Saved report → {OUT_DIR / 'financial_signal.txt'}")
    print(f"[financial] Saved CSV    → {out_csv}")


if __name__ == "__main__":
    run_financial_signal()
