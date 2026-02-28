# 02_feature_engineering.py
# Objective: Transform raw simulated claims data into machine-readable numerical vectors
# using PySpark ML pipelines.

from pyspark.sql import SparkSession
from pyspark.sql.functions import month, dayofweek, when, col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("RBI_Feature_Engineering").getOrCreate()
print("Starting Feature Engineering...")

# 1. Load the Managed Delta Table from Script 1
input_table = "default.rbi_simulated_claims"
df_raw = spark.table(input_table)

# 2. Extract Temporal (Time-based) Features
# Insurance claims are highly seasonal. Weekends and winter months matter.
# dayofweek: 1 = Sunday, 7 = Saturday
df_enhanced = df_raw.withColumn("incident_month", month(col("incident_date")))\
                    .withColumn("day_of_week", dayofweek(col("incident_date")))\
                    .withColumn("is_weekend", when(col("day_of_week").isin(1, 7), 1).otherwise(0))\
                    .withColumn("is_winter", when(col("incident_month").isin(12, 1, 2, 3), 1).otherwise(0))

# 3. Define the MLlib Pipeline Stages
# Stage A: StringIndexer assigns a unique number to each category (e.g., Tim Hortons = 0, Burger King = 1)
brand_indexer = StringIndexer(inputCol="brand", outputCol="brand_idx", handleInvalid="keep")
region_indexer = StringIndexer(inputCol="region", outputCol="region_idx", handleInvalid="keep")
claim_type_indexer = StringIndexer(inputCol="claim_type", outputCol="claim_type_idx", handleInvalid="keep")

# Stage B: OneHotEncoder converts those numbers into binary vectors (e.g., [1, 0, 0])
encoder = OneHotEncoder(
    inputCols=["brand_idx", "region_idx", "claim_type_idx"],
    outputCols=["brand_vec", "region_vec", "claim_type_vec"]
)

# Stage C: VectorAssembler combines ALL features into a single array column called "features"
assembler = VectorAssembler(
    inputCols=["brand_vec", "region_vec", "claim_type_vec", "store_age_years", "is_weekend", "is_winter"],
    outputCol="features"
)

# 4. Build and Run the Pipeline
print("Building and fitting the feature engineering pipeline...")
pipeline = Pipeline(stages=[
    brand_indexer, region_indexer, claim_type_indexer, encoder, assembler
])

# Fit the pipeline to the data (learn the categories) and transform it (apply the changes)
pipeline_model = pipeline.fit(df_enhanced)
df_engineered = pipeline_model.transform(df_enhanced)

# 5. Save the engineered data for the Modeling script
output_table = "default.rbi_engineered_claims"
print(f"Saving engineered features to Delta Table: {output_table}...")

# We select only the columns we need for modeling to keep the final table clean and efficient
modeling_dataset = df_engineered.select(
"claim_id", "total_claim_cost", "is_litigated", "features"
)

# Save to the Catalog
modeling_dataset.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("Feature Engineering complete!")
display(modeling_dataset.limit(5))
