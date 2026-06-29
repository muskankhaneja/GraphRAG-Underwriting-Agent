# GraphRAG Underwriting Agent

A **Dual-Engine Agentic Credit Underwriting and Audit System** that combines a traditional XGBoost predictive model with a GraphRAG pipeline (Neo4j + LLM) orchestrated by a LangGraph multi-agent framework to produce audit-ready Credit Underwriting Memorandums.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                   │
│   Data Auditor Agent → Graph Traversal Agent → Synthesis    │
│              Agent                 ↑ self-correction loop   │
└────────────────┬────────────────────────────────────────────┘
                 │
     ┌───────────┴────────────┐
     │                        │
┌────▼──────────┐    ┌────────▼──────────┐
│  Quantitative │    │  Knowledge Graph  │
│    Engine     │    │   (GraphRAG)      │
│   XGBoost     │    │  Neo4j + Vectors  │
│  P(default)   │    │  SEC 10-K filings │
└───────────────┘    └───────────────────┘
```

| Engine | Technology | Output |
|---|---|---|
| Quantitative | XGBoost binary classifier | $P_d$ — probability of speculative grade |
| Qualitative | Neo4j + LLM vector search | Narrative risk context from 10-K filings |
| Orchestration | LangGraph state machine | Audit-ready Credit Underwriting Memorandum |

---

## Development Phases

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Complete | XGBoost credit risk model on S&P 500 financial ratios |
| **Phase 2** | 🔜 Next | Neo4j graph DB + SEC 10-K vector ingestion |
| **Phase 3** | ⏳ Pending | LangChain `@tool` wrappers for ML inference and graph queries |
| **Phase 4** | ⏳ Pending | LangGraph multi-agent orchestration + self-correction loop |
| **Phase 5** | ⏳ Pending | Ragas/TruLens evaluation and RegTech guardrails |

---

## Phase 1 — Quantitative Foundation

### Dataset
**Kaggle: Corporate Credit Rating with Financial Ratios** (`kirtandelwadia/corporate-credit-rating-with-financial-ratios`)
- 7,804 S&P 500 corporate records
- 16 financial ratio features (liquidity, leverage, profitability, efficiency)
- Real company `Ticker` identifiers → Phase 2 entity linking to SEC EDGAR

### Binary Target

| Class | Rating Labels | Meaning |
|---|---|---|
| `0` — Investment Grade | `AAA` → `BBB-` | Low/Moderate Risk |
| `1` — Speculative Grade | `BB+` → `D` | High Risk |

### Model Performance

| Metric | Value |
|---|---|
| CV PR-AUC (5-fold) | **0.9110 ± 0.0124** |
| CV ROC-AUC | **0.9465** |
| Hold-out PR-AUC | **0.9319** |
| Hold-out ROC-AUC | **0.9637** |
| Hold-out F1 | **0.8771** |
| Optimal Threshold | **0.435** |

---

## Project Structure

```
GraphRAG-Underwriting-Agent/
├── src/
│   └── phase1_quantitative/
│       ├── data_engineering.py     # Kaggle ingest, rating mapping, scaling, split
│       ├── train_credit_model.py   # XGBoost CV training + artifact export
│       └── inference.py            # CreditRiskPredictor inference class
├── tests/
│   └── test_phase1.py              # 39 pytest unit tests
├── data/
│   ├── raw/                        # .gitignored — auto-downloaded by pipeline
│   └── processed/                  # .gitignored — parquet splits
├── models/                         # .gitignored — exported artifacts
│   ├── xgb_credit_model.json
│   ├── scaler.pkl
│   ├── metrics.json
│   └── feature_importance.csv
├── .github/workflows/ci.yml        # GitHub Actions: pytest on push
└── requirements.txt
```

---

## Quickstart

### Prerequisites
- Python 3.9+
- Kaggle API credentials at `~/.kaggle/kaggle.json`

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
# Downloads dataset automatically, engineers features, trains model, exports artifacts
python -m src.phase1_quantitative.train_credit_model
```

### Run Tests
```bash
pytest tests/test_phase1.py -v
```

### Inference
```python
from src.phase1_quantitative.inference import CreditRiskPredictor

predictor = CreditRiskPredictor()
result = predictor.predict({
    "Current Ratio": 1.8,
    "Debt/Equity Ratio": 0.6,
    "Net Profit Margin": 0.12,
    "ROA - Return On Assets": 0.08,
    "Asset Turnover": 0.9,
    "Gross Margin": 0.35,
    "Operating Margin": 0.15,
    "EBIT Margin": 0.14,
    "EBITDA Margin": 0.20,
    "Pre-Tax Profit Margin": 0.13,
    "ROE - Return On Equity": 0.18,
    "Return On Tangible Equity": 0.20,
    "ROI - Return On Investment": 0.10,
    "Long-term Debt / Capital": 0.30,
    "Operating Cash Flow Per Share": 2.50,
    "Free Cash Flow Per Share": 1.80,
})
# {
#   "probability_of_speculative": 0.083,
#   "predicted_class": 0,
#   "risk_tier": "Investment Grade",
#   "confidence": "High",
#   "model_version": "xgb_credit_model_v1"
# }
```

---

## CI/CD

GitHub Actions runs `pytest tests/ -v` on every push to `phase1-quantitative` and `main`.

---

## License

MIT
