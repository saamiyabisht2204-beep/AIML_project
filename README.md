# AI-Based Student Stress Level Prediction Using Machine Learning

An end-to-end Machine Learning project to analyze student lifestyle and academic factors (e.g., study hours, sleep duration, screen time, exercise frequency, attendance, and academic workload) to predict and classify student stress into **Low**, **Moderate**, or **High** categories.

---

## 📁 Repository Structure

```text
├── data/
│   ├── raw/             # Unprocessed, original dataset dumps
│   ├── processed/       # Cleaned, transformed, and train/test splits
│   └── external/        # External reference datasets or schemas
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── docs/                # Project specifications, reports, and architecture
│   └── project_specification.md
├── src/                 # Production-ready, modular ML codebase
│   ├── data/            # Data ingestion, loading, and cleaning pipelines
│   ├── features/        # Feature extraction, scaling, and engineering
│   ├── models/          # Model training, evaluation, tuning, and serialization
│   └── utils/           # Configuration loaders, logging, and metrics helpers
├── configs/             # YAML configurations for experiments & hyperparameters
│   └── config.yaml
├── tests/               # Unit and integration test suites
├── requirements.txt     # Python project dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.9+
- **Machine Learning & Data:** `scikit-learn`, `pandas`, `numpy`, `scipy`
- **Visualization:** `matplotlib`, `seaborn`
- **Web App / UI:** `streamlit`
- **Testing & Tooling:** `pytest`, `pyyaml`

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # macOS / Linux
# .\venv\Scripts\activate  # Windows

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pipeline Execution
- **Exploratory Data Analysis:** Run notebooks in `notebooks/`
- **Training & Evaluation:** Use `src/models/` modules with parameters governed by `configs/config.yaml`
- **Interactive UI:** Run `streamlit run app.py` (when UI is implemented)

### Interactive UI on stresmlit : https://aimlproject-x6khjlpwdfld3dbcftjqha.streamlit.app/
