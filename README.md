# Serie A Decision Support System (DSS) & Machine Learning Engine

[![R Version](https://img.shields.io/badge/R-%3E%3D%204.2.0-blue.svg)](https://www.r-project.org/)
[![Framework](https://img.shields.io/badge/Framework-R%20Shiny-green.svg)](https://shiny.posit.co/)
[![Machine Learning](https://img.shields.io/badge/ML-27%20Sections-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Exam%20Ready%20%2F%20Active-brightgreen.svg)]()

A comprehensive **Statistical Learning** framework and interactive **R Shiny** application built as a Decision Support System (DSS) for predictive analytics, Expected Goals ($xG$) modeling, and tactical simulation in Serie A football matches.

This project bridges complex backend statistical modeling (a 27-section Machine Learning pipeline) with an intuitive operational frontend designed to translate raw data and model predictions into actionable managerial insights.

# Dashboard Preview

| Tab 1: Match Predictor | Tab 2: Scenario Simulator |
| :---: | :---: |
| ![Match Predictor](app/www/tab1_preview.jpg) | ![Scenario Simulator](app/www/tab2_preview.jpg) |

# Frontend Architecture (R Shiny DSS)

The application translates complex predictive models into dynamic, real-time UI components:

# Tab 1 — Match Predictor
* **Multi-Model Consensus Matrix:** Estimates $1X2$ match outcome probabilities and Expected Goals ($xG$) by running multiple models in parallel (**Poisson GLM**, **Multinomial Logit**, **Random Forest**, and **GAM**).
* **Bivariate Poisson Score Matrix:** Generates an exact score probability heat map based on joint goal expectation distributions for Home and Away teams.
* **Risk & Expected Metrics:** Dynamically calculates **Over 2.5 Goals** probability and combined expected goals driven by user-adjusted shot inputs (Shots on Target / Total Shots).

# Tab 2 — Scenario Simulator ("What-If" Elasticity)
* **Dynamic Tactical Simulation:** Evaluates tactical shifts and marginal returns by adjusting team performance deltas ($\Delta$ Shots on Target).
* **GAM Non-Linear Spline Response Curves:** Evaluates diminishing marginal returns using *Natural Cubic Splines* to model how incremental shots affect win probability, holding all other match conditions constant.
* **Baseline vs. What-If Comparison:** Instantaneous comparative visual analysis between baseline pre-match expectations and simulated tactical adjustments.

# Modules in Development (Post-Exam Scope)
* **Tab 3 — League Analytics:** Evaluates overall league efficiency and compares actual points against Expected Points ($xPTS$).
* **Tab 4 — Team Analysis:** Profiles individual club performance metrics across offensive and defensive dimensions.
* **Tab 5 — Model Analytics:** Scientific model validation suite featuring confusion matrices, ROC/AUC curves, residual diagnostics, and cross-validated $MSE$ benchmarks.

# Backend Engine (27-Section Machine Learning Pipeline)

The core analytical pipeline residing in `serie_a_project` covers an extensive statistical learning curriculum:

1. **Preprocessing & EDA:** Data imputation, feature scaling, interaction terms, and correlation matrices.
2. **Parametric & Linear Models:** Multiple Linear Regression, Ridge Regression, LASSO, Best Subset Selection.
3. **Count Models & Classification:** 
   * **Poisson GLM** for modeling Expected Goals ($xG$).
   * **Multinomial Logit**, LDA, QDA, and Naïve Bayes for $1X2$ outcome classification.
4. **Non-Linear Methods & Splines:** GAM (Generalized Additive Models) with *Natural Cubic Splines*, Smoothing Splines, LOESS, and KNN.
5. **Tree-Based Methods & Ensembles:** Decision Trees with Cost-Complexity Pruning, Random Forest, Gradient Boosting (GBM), and BART.
6. **Support Vector Machines & Neural Networks:** Linear and RBF-kernel SVMs, Feedforward Neural Networks, and Time-Series/Momentum modeling.
7. **Diagnostics & Validation:** K-Fold Cross-Validation, Bootstrapping, Permutation Testing, Outlier/Leverage analysis, and multi-class ROC/AUC curves.

# Repository Structure

serie-a-dss/
├── analysis/
│   ├── 01_serie_a_ml_engine.R        # Complete 27-section Statistical Learning pipeline
│   └── outputs/                      # Serialized models (.rds) and coefficients
│
├── app/
│   ├── global.R                      # Package loading, helper functions, model imports
│   ├── ui.R                          # Shiny user interface layout (Tab 1 & Tab 2)
│   ├── server.R                      # Reactive logic, simulation engines, Plotly renders
│   └── www/                          # Custom CSS, assets, and preview screenshots
│
├── data/
│   ├── raw/                          # Raw Serie A historical match data
│   └── processed/                    # Cleaned and feature-engineered datasets
│
├── README.md                         # Project documentation
└── SerieA_DSS.Rproj                  # RStudio project file

# Installation & Local Execution
Ensure you have R (version $\ge 4.2.0$) and RStudio installed. Install required packages by running:
install.packages(c(
  "shiny", "shinydashboard", "tidyverse", "plotly", 
  "caret", "glmnet", "randomForest", "gbm", 
  "mgcv", "e1071", "nnet", "pROC"
))

# Clone the repository and launch the Shiny app via RStudio or the R console:
# Clone repository
# git clone [https://github.com/JBale451/SerieA-DSS.git](https://github.com/JBale451/SerieA-DSS.git)

# Launch the interactive DSS
shiny::runApp("app/")

# Author & Academic Context
Author: Jarim Natan Balesini (@JBale451)
Context: Exam Project / Thesis Defense in Statistical Learning & Data Science
Focus: Bridging theoretical machine learning algorithms with interactive Decision Support Systems (DSS).
