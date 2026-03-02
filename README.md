# RBI Claims Risk & Reserve Optimization Engine

![Databricks](https://img.shields.io/badge/Databricks-Serverless-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-MLOps-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)

## Executive Summary
This project implements an end-to-end Machine Learning pipeline to predict the **Ultimate Incurred Cost** of general liability and worker's compensation claims for a multi-brand restaurant corporation (simulating Restaurant Brands International: Tim Hortons, Burger King, Popeyes).

By leveraging Gradient Boosted Trees (GBT) and Generalized Linear Models (GLM) within a Databricks Unity Catalog environment, this solution enables:
* **More Accurate Reserving:** Reducing "trapped capital" by predicting claim severity early.
* **Risk Engineering:** Identifying high-risk operational patterns (e.g., aging stores with specific fryer equipment in Florida).
* **Fairer Pricing:** Segmenting risk by brand and region rather than a flat-rate actuarial approach.

## The Tech Stack
* **Core Logic:** Python, PySpark (Spark MLlib)
* **Platform:** Databricks Serverless (Unity Catalog)
* **Ops & Governance:** MLflow (Experiment Tracking & Model Registry)
* **Data Storage:** Managed Delta Tables

## Key Results & Business Impact
A model "Bake-Off" was conducted to compare the actuarial standard (Generalized Linear Regression with a Gamma family) against a modern challenger (Gradient Boosted Trees). The **Gradient Boosted Tree** won by successfully capturing non-linear interactions between `Store_Age` and `Claim_Type`.

### Decile Analysis (Lift Chart)
To validate the model's economic value, predictions were ranked into deciles to prove the model can effectively separate routine claims from catastrophic "shock losses":
* **Decile 1 (Safest):** Avg Actual Cost **~$3,819**
* **Decile 10 (Most Risky):** Avg Actual Cost **~$35,709**

*Result: The steep 9.3x slope confirms the model can effectively flag high-exposure claims immediately upon First Notice of Loss (FNOL) for proactive intervention.*

## Solution Architecture

This repository is modularized into a 4-stage enterprise pipeline:

### `01_data_simulation.py`
Architected a synthetic data generator to create 50,000 realistic records reflecting specific operational risks, bypassing public DBFS and writing directly to secure **Unity Catalog Managed Delta Tables**.

### `02_feature_engineering.py`
Developed a PySpark ML Pipeline utilizing `StringIndexer`, `OneHotEncoder`, and `VectorAssembler` to transform categorical data (Region, Brand, Claim Type) and extracted temporal features (Seasonality, Weekend Flags) into machine-readable vectors. 

### `03_modeling_and_evaluation.py`
Conducted the champion/challenger algorithm evaluation, calculating Root Mean Squared Error (RMSE) to penalize shock-claim misses, and Mean Absolute Error (MAE) for business explainability. Extracted decile distributions for executive lift-chart reporting.

### `04_mlflow_registry.py`
Simulated a production-grade MLOps lifecycle by tracking hyperparameters and metrics via Databricks MLflow. Governed the winning model by inferring an explicit Unity Catalog signature schema and deploying the artifact to the MLflow Model Registry via secure UC Volumes.

## How to Run This Project
1. Clone this repository to your local machine using VS Code.
2. Link the repository to your Databricks Workspace via Git integration.
3. Ensure you are running a Databricks cluster with Unity Catalog enabled.
4. Run the scripts sequentially (`01` through `04`).