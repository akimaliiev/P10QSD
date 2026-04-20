# P10QSD — Do Words Move Markets?
### Predicting Abnormal Stock Returns from Quarter-over-Quarter Changes in SEC 10-Q Filings

> **Core question:** Do changes in the language of SEC 10-Q Risk Factors predict whether a stock will outperform or underperform the market in the following week?

**Short answer: Yes — and finance-specific text features beat price-based technical indicators.**

---

## Why This Project Exists

Stock prediction models overwhelmingly rely on price and volume data — moving averages, RSI, momentum, volatility. These features capture what the market *has already priced in*. They are blind to the qualitative information companies disclose through regulatory filings.

Every publicly traded company must file a 10-Q quarterly report with the SEC. Item 1A of this filing — the Risk Factors section — is legally required to be updated whenever the company's risk landscape changes materially. This creates a unique signal: when a company rewrites its risk disclosures, it is telling investors something has changed, often before that change appears in the financial numbers.

Prior work (Cohen, Malloy & Nguyen, 2020, *Journal of Finance*) showed that investors largely ignore these changes — leading to delayed and predictable price reactions. We build a machine learning pipeline to systematically exploit this inattention across 186 S&P 500 companies over 10 years.

---

## What Makes This Different from Prior Work

### 1. Filing-aligned evaluation (not daily)

The most common mistake in finance NLP is forward-filling quarterly text features into daily price rows. A company files once per quarter — so forward-filling creates 90 consecutive rows with **identical feature vectors**. Those are not 90 independent observations. They inflate apparent accuracy from ~52% to ~61% while adding zero genuine signal.

We use one row per filing. Every observation is a genuine quarterly event.

### 2. Abnormal return target (not raw return)

Predicting whether a stock goes up or down conflates two different things: (1) market-wide movements and (2) company-specific information. If the S&P 500 rises 3% in a week, nearly every stock rises regardless of what its 10-Q says.

We predict **abnormal return = stock return − SPY return** over 5 trading days starting from t+1 after the filing date. This isolates the company-specific signal. The t+1 start skips the filing day itself, which is dominated by algorithmic traders reacting to headline EPS numbers within milliseconds — not reading the risk factors section.

### 3. Loughran-McDonald finance sentiment (not VADER)

VADER was designed for Twitter and product reviews. It scores "liability", "risk", and "adverse" as neutral because in everyday language those words often are. In SEC filings they are strongly negative signals.

The Loughran-McDonald (LM) word lists were built specifically from 10-K filings. LM marks "liability", "impairment", "litigation", "adverse", "forfeit" as negative — correctly. This distinction matters: our ablation shows LM sentiment alone achieves F1=0.525 while VADER-based models struggle to beat the majority baseline.

### 4. Sector contagion signal (novel contribution)

When companies in a sector simultaneously rewrite their risk factors, the information spills across supply chains before the market fully prices it in. We compute the average cosine similarity of all sector peers who filed within ±45 days as a feature for each filing. No prior work has used peer filing changes to predict individual company abnormal returns.

---

## Results

### Main Results (186 companies, 1,082 test samples, 2023–2024)

| Model | Accuracy | Macro F1 | p-value | Significant? |
|---|---|---|---|---|
| Majority-class baseline | 46.3% | — | — | — |
| **Logistic Regression** | **51.7%** | **0.515** | **0.003** | ✅ Yes |
| LightGBM | 50.3% | 0.503 | 0.029 | ✅ Yes |
| Ensemble (soft voting) | 48.7% | 0.487 | 0.128 | No |
| XGBoost | 48.7% | 0.486 | 0.143 | No |
| Random Forest | 48.2% | 0.481 | 0.173 | No |

Two independent models are statistically significant. The baseline is 46.3% (not 50%) because abnormal returns in 2023–2024 were skewed by the S&P 500 mega-cap rally — most stocks underperformed the index in absolute terms. Our +5.4pp lift over baseline with p=0.003 is genuine.

### Ablation Study — The Core Finding

| Feature Group | Accuracy | Macro F1 | Features |
|---|---|---|---|
| **LM sentiment only** | **52.8%** | **0.525** | 6 |
| Text only (no price) | 51.5% | 0.514 | 17 |
| All features combined | 51.2% | 0.512 | 24 |
| Price only | 46.9% | 0.449 | 7 |

**LM sentiment features alone outperform all price-based technical indicators by +0.076 F1.** More striking: adding price features back actually slightly reduces performance. For predicting company-specific abnormal returns, the text signal is purer without noise from momentum and volatility.

### Walk-Forward Rolling Cross-Validation (3-year window, 7 folds)

| Test Year | Accuracy | Macro F1 |
|---|---|---|
| 2018 | 49.5% | 0.494 |
| 2019 | 48.1% | 0.481 |
| 2020 | 50.4% | 0.487 |
| 2021 | 49.1% | 0.489 |
| 2022 | 48.0% | 0.477 |
| 2023 | 48.2% | 0.473 |
| 2024 | 50.3% | 0.496 |
| **Mean** | **49.1%** | **0.485** |
| **Std** | **±1.0%** | **±0.009** |

Remarkably stable across all 7 market regimes including COVID crash (2020), Fed tightening (2022), and recovery (2023–2024). Standard deviation of ±1.0% is the tightest we observed at any scale.

### LR Coefficients — What the Model Learned

| Feature | Direction | Coefficient | Interpretation |
|---|---|---|---|
| `lm_litigious` | → Down | −0.031 | Legal risk language predicts underperformance |
| `lm_positive` | → Down | −0.030 | Optimistic language predicts underperformance |
| `price_return_20d` | → Down | −0.028 | Mean reversion: recent winners underperform |
| `text_length_norm` | → Down | −0.024 | Longer risk sections = more disclosed risks |
| `lm_uncertainty` | → Down | −0.019 | Hedging language predicts underperformance |
| `lm_negative` | → Down | −0.017 | Negative language predicts underperformance |
| `price_volatility_20d` | → Up | +0.015 | High volatility stocks outperform after filings |
| `cosine_sim_prev` | → Down | −0.008 | No text change = ignoring real risks |

Every LM text feature predicts **down**. Companies using more optimistic, litigious, or uncertain language in their Risk Factors section underperform the market in the following week. This is the "cheap talk" effect: positive language in mandatory risk disclosures is not reassuring — it is predictive of trouble.

---

## Dataset

| Property | Value |
|---|---|
| Companies | 186 S&P 500 constituents |
| Sectors | Technology, Finance, Healthcare, Consumer, Energy, Industrial |
| Filing period | April 2015 — December 2024 |
| Total observations | 4,840 (1 per 10-Q filing) |
| Training set | 3,758 filings (pre-2023) |
| Test set | 1,082 filings (2023–2024) |
| Target balance | 49.8% up / 50.2% down (near perfect) |
| Cosine sim median | 0.918 (most quarters barely change) |

The median cosine similarity of 0.918 is itself informative: most companies copy-paste their risk factors quarter to quarter. The ~8% of filings with cosine similarity below 0.80 represent genuine rewrites — and these are where the predictive signal is strongest.

---

## Features

### Text features (from Item 1A Risk Factors)

| Feature | Description |
|---|---|
| `cosine_sim_prev` | TF-IDF cosine similarity between current and previous quarter's text |
| `risk_drift_4q` | Rolling 4-quarter trend — is this company rewriting more or less over time? |
| `filing_surprise` | Company-specific z-score — how unusual is this quarter's change? |
| `sector_contagion` | Average peer cosine similarity within ±45 days (novel) |
| `lm_negative` | Loughran-McDonald negative word ratio |
| `lm_positive` | LM positive word ratio |
| `lm_uncertainty` | LM uncertainty/hedging word ratio |
| `lm_litigious` | LM legal/regulatory risk word ratio |
| `lm_constraining` | LM constraining word ratio |
| `lm_net_sentiment` | (positive − negative) / (positive + negative + 1) |
| `lm_*_delta` | Quarter-over-quarter change in each LM score |
| `lm_neg_x_cosine` | Interaction: negative sentiment × text change magnitude |
| `text_price_divergence` | Interaction: positive text tone vs falling price |

### Price features (at filing date, no lookahead)

| Feature | Description |
|---|---|
| `price_return_1d/5d/20d` | Momentum over 1, 5, 20 trading days before filing |
| `price_volatility_20d` | Annualized 20-day realized volatility |
| `price_ma_ratio_5/20` | Price relative to 5-day and 20-day moving average |
| `price_rsi` | 14-day Relative Strength Index |

---

## Setup

```bash
git clone https://github.com/Sarthak-Malla/P10QSD.git
cd P10QSD
python3 -m venv venv
source venv/bin/activate
venv/bin/pip install -r requirements.txt
venv/bin/pip install xgboost lightgbm torch seaborn
venv/bin/python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Running the Pipeline

```bash
# Step 1: Download 10-Q filings from SEC EDGAR (checkpointed, safe to interrupt)
venv/bin/python -m src.dataloader.sec_loader

# Step 2: Build filing-aligned dataset (all features + abnormal returns)
venv/bin/python -m src.dataloader.filing_dataset

# Step 3: Exploratory data analysis (7 plots saved to outputs/eda/)
venv/bin/python -m src.analysis.eda

# Step 4: Train and evaluate all models (walk-forward CV, grid search, ablation)
venv/bin/python -m src.models.baseline

# Step 5: Sparse model + sector analysis + portfolio evaluation
venv/bin/python -m src.models.final_analysis

# Optional: LSTM temporal model over per-company sequences
venv/bin/python -m src.models.temporal_model

# Scale to full S&P 500 overnight (checkpointed)
caffeinate -i bash scale_to_sp500.sh
```

---

## Project Structure

```
P10QSD/
├── conf/
│   └── config.yaml              # Hydra config (tickers, dates, model params)
├── src/
│   ├── analysis/
│   │   └── eda.py               # EDA: 7 plots across time, sector, regime
│   ├── dataloader/
│   │   ├── loader.py            # yfinance price downloader
│   │   ├── sec_loader.py        # SEC EDGAR 10-Q downloader (checkpointed)
│   │   └── filing_dataset.py    # Filing-aligned dataset builder
│   ├── features/
│   │   └── lm_features.py       # Loughran-McDonald finance sentiment
│   └── models/
│       ├── baseline.py          # Walk-forward CV, grid search, ensemble, ablation
│       ├── temporal_model.py    # LSTM over filing sequences
│       └── final_analysis.py    # Sparse model, sector models, portfolio eval
├── sp500_50.txt                 # 50-company subset for quick experiments
├── sp500_tickers.txt            # Full 503 S&P 500 tickers
└── scale_to_sp500.sh            # One-command overnight full-scale run
```

---

## Key Methodological Choices (and Why They Matter)

**Why not forward-fill?** Forward-filling 1 quarterly filing into 90 daily rows means the model sees 90 identical feature vectors as 90 "independent" training examples. This inflates accuracy from ~52% to ~61% with zero added signal. Our filing-aligned approach is the only honest evaluation.

**Why abnormal returns?** Raw return prediction conflates market-wide movements with company-specific filing signals. A rising market lifts all stocks regardless of their 10-Q disclosures. Subtracting SPY return isolates what the filing should actually predict.

**Why t+1?** Filing day (t=0) prices are dominated by algorithmic traders reacting to headline EPS numbers in milliseconds. The text-driven informed reaction from analysts who read the filing happens from t+1 onward.

**Why LM over VADER?** VADER was trained on social media sentiment. It marks "liability", "risk", "adverse" as neutral. The Loughran-McDonald word lists were constructed from 10-K/10-Q filings and correctly classify financial domain terminology. The difference in F1 between LM-only (0.525) and VADER-based models demonstrates this empirically.

**Why rolling window CV?** An expanding window CV keeps 2015–2016 filing patterns in every training set. Post-COVID, pre-AI disclosure norms are different from 2015 boilerplate. A 3-year rolling window trains on the most relevant recent context and is more realistic for deployment.

---

## Literature Foundation

- Cohen, Malloy & Nguyen (2020). *Lazy Prices.* Journal of Finance — investors don't read filings, changes predict returns
- Loughran & McDonald (2011). *When is a Liability not a Liability?* Journal of Finance — finance-specific sentiment wordlists
- Li (2010). *The Information Content of Forward-Looking Statements.* Journal of Accounting Research
- Yang et al. (2020). *FinBERT.* arXiv:2006.08097 — BERT fine-tuned on financial text
- Kogan et al. (2009). *Predicting Risk from Financial Reports.* NAACL-HLT

---

## Authors

**Sultan Akimaliyev** — sultan.akimaliyev@mbzuai.ac.ae
**Sarthak Malla** — sarthak.malla@mbzuai.ac.ae

Mohamed bin Zayed University of Artificial Intelligence
