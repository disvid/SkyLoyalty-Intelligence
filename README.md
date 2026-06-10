# SkyLoyalty Intelligence

An AI-powered customer retention and churn prediction platform that enables businesses to identify at-risk customers, understand churn drivers, segment customers, and generate data-driven retention strategies.

---

## Features

### Customer Segmentation
- Behavioral customer grouping
- Identification of loyal, regular, and at-risk customers
- Segment-wise analytics

### Churn Prediction
- Machine learning-based churn risk scoring
- Probability estimation for each customer
- Early identification of customers likely to leave

### Customer 360 Analysis
- Unified customer profile generation
- Demographic, transactional, and engagement insights
- Comprehensive customer understanding

### Explainable AI
- SHAP-based model interpretation
- Feature importance analysis
- Transparent prediction explanations

### Retention Intelligence
- Personalized retention recommendations
- Action prioritization based on churn risk
- Business-focused intervention strategies

### Interactive Dashboard
- Real-time analytics and visualizations
- KPI monitoring
- Downloadable reports and insights

---

## Implementation Workflow

```text
Raw Customer Data
        │
        ▼
Data Loading & Validation
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Customer Segmentation
        │
        ▼
Churn Prediction Model
        │
        ▼
Explainability Analysis
        │
        ▼
Retention Recommendation Engine
        │
        ▼
Customer 360 Generation
        │
        ▼
Interactive Dashboard & Reports
```

---

## Tech Stack

### Frontend
- Streamlit

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-Learn
- XGBoost
- Imbalanced-Learn

### Explainability
- SHAP

### Visualization
- Plotly
- Matplotlib
- Seaborn

### Utilities
- Joblib
- OpenPyXL

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/disvid/skyloyalty-intelligence.git
cd skyloyalty-intelligence
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Train Model
```bash
python run_pipeline.py
```
### 4. Run Application

```bash
streamlit run app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## Output Artifacts

The application automatically generates:

- Trained churn prediction models
- Customer segmentation results
- Customer 360 profiles
- SHAP explainability reports
- Retention recommendations
- Business intelligence visualizations

All generated artifacts are stored inside the `outputs/` directory.

---

## Use Cases

- Customer Retention Analytics
- Churn Risk Identification
- Customer Lifetime Value Optimization
- Loyalty Program Analysis
- Customer Segmentation
- Data-Driven Retention Campaigns

