# 03_modeling_and_evaluation.py
# Objective: Train a traditional GLM and a modern GBT model to predict claim severity,
# evaluate their performance, and generate Decile data for the Lift Chart.

from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, ntile, avg
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("RBI_Modeling").getOrCreate()
print("Starting Modeling and Evaluation Phase...")

# 1. Load the Engineered Data from Script 2
input_table = "default.rbi_engineered_claims"
df_engineered = spark.table(input_table)

# 2. Train-Test Split (80% for training, 20% for testing)
# We set a seed so the random split is identical every time we run it
train_data, test_data = df_engineered.randomSplit([0.8, 0.2], seed=42)
print(f"Training data count: {train_data.count()}, Testing data count: {test_data.count()}")

# 3. Model 1: Generalized Linear Model (GLM) - Gamma Distribution
# The Actuarial standard for right-skewed insurance costs (cannot be negative)
print("\nTraining GLM (Gamma) Model...")
glm = GeneralizedLinearRegression(
    family="gamma", link="log", featuresCol="features", labelCol="total_claim_cost",
    maxIter=10, regParam=0.3
)

glm_model = glm.fit(train_data)
glm_predictions = glm_model.transform(test_data)

# 4. Model 2: Gradient Boosted Trees (GBT)
# The modern challenger capable of finding complex, non-linear risk interactions
print("\nTraining Gradient Boosted Trees (GBT) Model...")
gbt = GBTRegressor(
    featuresCol="features", labelCol="total_claim_cost",
    maxIter=20, maxDepth=5, seed=42
)
gbt_model = gbt.fit(train_data)
gbt_predictions = gbt_model.transform(test_data)

# 5. Evaluation: RMSE and MAE
# RMSE punishes big misses (shock claims), MAE gives the average dollar amount we are off by
evaluator_rmse = RegressionEvaluator(
    labelCol="total_claim_cost", predictionCol="prediction", metricName="rmse"
)
evaluator_mae = RegressionEvaluator(
    labelCol="total_claim_cost", predictionCol="prediction", metricName="mae"
)

glm_rmse = evaluator_rmse.evaluate(glm_predictions)
glm_mae = evaluator_mae.evaluate(glm_predictions)
gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
gbt_mae = evaluator_mae.evaluate(gbt_predictions)

print("\n" + "=" * 40)
print("🏆 MODEL PERFORMANCE (BAKE-OFF) 🏆")
print("=" * 40)
print(f"GLM (Actuarial Standard) - RMSE: ${glm_rmse:,.2f}, MAE: ${glm_mae:,.2f}")
print(f"GBT (Modern Challenger) - RMSE: ${gbt_rmse:,.2f}, MAE: ${gbt_mae:,.2f}")

# Determine the winner dynamically based on the lowest RMSE error
winning_predictions = gbt_predictions if gbt_rmse < glm_rmse else glm_predictions
winner_name = "GBT" if gbt_rmse < glm_rmse else "GLM"
print(f"\nWinner: {winner_name} Model!")

# 6. Decile Analysis (Data for the Lift Chart)
# We rank the test predictions into 10 buckets (1 = Low Risk, 10 = High Risk)
print("\nCalculating Lift Chart Deciles...")
windowSpec = Window.orderBy("prediction")
predictions_with_deciles = winning_predictions.withColumn("decile", ntile(10).over(windowSpec))

# Aggregate the average actual cost vs predicted cost per decile bucket
lift_data = predictions_with_deciles.groupBy("decile").agg(
    avg("prediction").alias("avg_predicted_cost"),
    avg("total_claim_cost").alias("avg_actual_cost")
).orderBy("decile")

# Save the lift chart data to a Delta Table so it can be easily visualized in Databricks
lift_table_name = "default.rbi_lift_chart_data"
lift_data.write.format("delta").mode("overwrite").saveAsTable(lift_table_name)

print("Modeling complete! Lift chart data safely exported to the Catalog.")
display(lift_data)