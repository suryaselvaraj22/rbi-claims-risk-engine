# 04_mlflow_registry.py
# Objective: Train the winning GBT model and log it to the MLflow Model Registry
# to simulate a production-ready MLOps deployment.

import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("RBI_MLOps_Registry").getOrCreate()
print("Starting MLOps & MLflow Registry Phase...")

# 1. Load the Engineered Data from Script 2
input_table = "default.rbi_engineered_claims"
df_engineered = spark.table(input_table)
train_data, test_data = df_engineered.randomSplit([0.8, 0.2], seed=42)

# 2. Set up the MLflow Experiment Run
# Everything inside this 'with' block gets recorded securely by Databricks
run_name = "GBT_Production_Candidate"
print(f"Initiating MLflow run: {run_name}")

with mlflow.start_run(run_name=run_name) as run:
    # 3. Define and Train the Winning Model
    max_iter = 20
    max_depth = 5

    gbt = GBTRegressor(
        featuresCol="features", labelCol="total_claim_cost",
        maxIter=max_iter, maxDepth=max_depth, seed=42
    )

    print("Training GBT model for registry...")
    gbt_model = gbt.fit(train_data)
    predictions = gbt_model.transform(test_data)

    # 4. Calculate Final Metrics
    evaluator_rmse = RegressionEvaluator(
        labelCol="total_claim_cost", predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol="total_claim_cost", predictionCol="prediction", metricName="mae"
    )

    final_rmse = evaluator_rmse.evaluate(predictions)
    final_mae = evaluator_mae.evaluate(predictions)

    # 5. Log Parameters (The "Ingredients")
    mlflow.log_param("model_type", "GBTRegressor")
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("train_rows", train_data.count())

   # 6. Log Metrics (The "Results")
    mlflow.log_metric("rmse", final_rmse)
    mlflow.log_metric("mae", final_mae)

    # 7. Log and Register the Model (The "Artifact")
    # In Serverless/Unity Catalog, MLflow needs a secure Volume to temporarily write the model
    volume_path = "/Volumes/workspace/default/mlflow_tmp"
    print(f"Ensuring temporary UC Volume exists at {volume_path}...")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS workspace.default.mlflow_tmp") 

    model_name = "RBI_Claims_Predictor"
    print(f"Logging model to MLflow Registry as: {model_name}...")

    mlflow.spark.log_model(
        spark_model=gbt_model, 
        artifact_path="model", 
        registered_model_name=model_name,
        dfs_tmpdir=volume_path
    )

print("\n" + "=" * 50)
print("✅ SUCCESS: Model successfully registered in MLflow!")
print("=" * 50)
print("You can view the tracked experiment by clicking 'Experiments' on the left sidebar.")