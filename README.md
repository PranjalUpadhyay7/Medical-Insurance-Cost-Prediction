<div align="center">
  <h1>🩺 Medical Insurance Cost Prediction</h1>
  <p><strong>An end-to-end Machine Learning pipeline utilizing 9 regression architectures to accurately forecast health insurance premiums based on demographic and lifestyle factors.</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-%23179c3e.svg?style=flat)](https://xgboost.readthedocs.io/)
  [![Jupyter Notebook](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
</div>

---

## 🎯 Executive Summary for Reviewers
This project demonstrates proficiency in **Exploratory Data Analysis (EDA), Feature Engineering, and Applied Predictive Modeling**. By systematically preprocessing data (handling categorical variables, scaling) and evaluating multiple regression models—from standard Linear Regression to advanced ensembles like **XGBoost and Gradient Boosting**—this repository serves as a blueprint for solving continuous variable prediction problems.

<details>
<summary><b>💡 Core Takeaways (Click to Expand)</b></summary>

- **Robust Preprocessing**: Includes One-Hot Encoding and explicit mitigation of the Dummy Variable Trap.
- **Statistical Rigor**: Feature selection validated through Pearson Correlation Coefficients & P-values.
- **Comprehensive Evaluation**: Models are judged across 4 key metrics: `MAE`, `MSE`, `R²`, and `MPE` (Mean Percentage Error).
- **Ablation Studies**: Tests the isolated impacts of feature scaling (StandardScaler) and feature selection (All vs. Top 3 features).
</details>

---

## 🏗️ System Architecture & Data Flow

The following interactive flowchart maps out the end-to-end machine learning lifecycle implemented in the notebook.

```mermaid
graph TD
    classDef dataset fill:#1A5F7A,stroke:#ffffff,stroke-width:2px,color:#ffffff,rx:10px,ry:10px
    classDef process fill:#22A39F,stroke:#ffffff,stroke-width:2px,color:#ffffff,rx:10px,ry:10px
    classDef model fill:#F3E99F,stroke:#333333,stroke-width:2px,color:#333333,rx:10px,ry:10px
    classDef eval fill:#C82C36,stroke:#ffffff,stroke-width:2px,color:#ffffff,rx:10px,ry:10px

    A[(insurance.csv)]:::dataset --> B[Data Preprocessing]:::process
    
    subgraph Feature Engineering
        B --> C[Categorical Encoding <br/> One-Hot]:::process
        C --> D[Correlation Matrix & Heatmap]:::process
        D --> E{Feature Selection & Scaling}:::process
    end
    
    E -->|1. All Features| F1(Train-Test Split):::process
    E -->|2. Top Features age, bmi, smoker| F1
    E -->|3. Standard Scaling| F1
    E -->|4. Unscaled Data| F1
    
    F1 --> G[Model Training Arena]:::model
    
    subgraph Regression Architectures
        G --> H1[Linear Models: <br/>Linear, Ridge, Lasso, Polynomial]:::model
        G --> H2[Tree-Based: <br/>Decision Tree, Random Forest]:::model
        G --> H3[Boosting Ensembles: <br/>XGBoost, Gradient, AdaBoost]:::model
    end
    
    H1 & H2 & H3 --> I[Performance Evaluation]:::eval
    
    subgraph Metrics & Selection
        I --> J1(MAE & MSE):::eval
        I --> J2(R² Score):::eval
        I --> J3(Mean Percentage Error):::eval
    end
    
    J1 & J2 & J3 --> K{Best Model Identification <br/> & GridSearchCV}:::process
```

---

## 📊 Dataset Overview

The dataset (`insurance (1).csv`) dictates the individual medical costs billed by health insurance.

| Feature | DataType | Description |
| :--- | :--- | :--- |
| **`age`** | `Numeric` | Age of primary beneficiary. |
| **`sex`** | `Categorical` | Insurance contractor gender (female / male). |
| **`bmi`** | `Numeric` | Body mass index (ideal range: 18.5 - 24.9). |
| **`children`**| `Numeric` | Number of kids/dependents covered by insurance. |
| **`smoker`** | `Categorical` | Smoking status (yes / no). |
| **`region`** | `Categorical` | Beneficiary's residential area in the US (NE, SE, SW, NW). |
| **`charges`** | `Numeric` | **Target Variable:** Medical costs billed. |

---

## 🚀 Quick Start & Installation

To run this project locally and explore the predictive models:

**1. Clone the environment and navigate to the directory**
Ensure you are in the `Medical-Insurance-Cost-Prediction` workspace.

**2. Install Dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tabulate scipy
```

**3. Launch the Pipeline**
Run the Jupyter Notebook to execute the data flow.
```bash
jupyter notebook InsurancePricePrediction_code.ipynb
```
*(Pro-tip: Inside the notebook, hit `Ctrl + F9` or `Cell -> Run All` to execute the full pipeline and generate the comparative Actual vs. Predicted scatter plots).*

---

## 🔍 Key Findings (Spoiler Alert)

- **Smoking is heavily correlated** with higher insurance charges, acting as the primary pivot node in tree-based architectures.
- Complex ensemble models (like **Gradient Boosting** and **XGBoost**) tend to outperform basic linear regression, particularly when navigating the non-linear relationship between BMI, smoking status, and charges.
- **Standard Scaling** produces varying impacts depending on the architecture; linear methods (Ridge, Lasso) stabilize, whereas tree-based ensembles handle unscaled data natively without performance degradation.
