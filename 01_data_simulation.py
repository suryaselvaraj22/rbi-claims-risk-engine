# 01_data_simulation.py
# Objective: Generate a synthetic dataset for Restaurant Brands International (RBI) claims
# to simulate an enterprise risk environment without exposing proprietary data.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, rand, when, lit, month
from pyspark.sql.types import IntegerType

# 1. Initialize Spark Session 
# (Databricks provides 'spark' automatically, but this makes the code robust/portable)
spark = SparkSession.builder.appName("RBI_Risk_Simulation").getOrCreate()

print("Starting data simulation for RBI Claims...")

# Define how large our dataset should be
num_rows = 50000

# 2. Base DataFrame Creation
# Assigning Brands, Regions, Store Age, and Incident Dates with specific probability distributions
df_raw = spark.range(0, num_rows).select(
    col(id).alias("claim_id"),
    expr("CASE WHEN rand() < 0.4 THEN 'Tim Hortons' " +
         "WHEN rand() < 0.7 THEN 'Burger King' " +
         "ELSE 'Popeyes' END").alias("brand"),
    expr("CASE WHEN rand() < 0.5 THEN 'Ontario' " +
         "WHEN rand() < 0.8 THEN 'Florida' " +
         "ELSE 'Alberta' END").alias("region"),
    (rand() * 20).cast(IntegerType()).alias("store_age_years"),
    expr("date_add(current_date(), -cast(rand() * 1800 as int))").alias("incident_date")
)

# 3. Add Business Logic for Claim Types and Base Costs
# This simulates the "Risk Engineering" aspect where certain brands/regions have specific hazards
df_claims = df.raw.withColumn(
    "claim_type",
    expr("CASE " +
         "WHEN brand = 'Tim Hortons' AND rand() < 0.3 THEN 'Hot Liquid Burn' " +
         "WHEN brand IN ('Burger King', 'Popeyes') AND rand() < 0.4 THEN 'Fryer/Grease Burn' " +
         "WHEN region IN ('Ontario', 'Alberta') AND month(incident_date) IN (12, 1, 2, 3) AND rand() < 0.5 THEN 'Slip and Fall (Ice)' " +
         "ELSE 'General Liability/Other' END"
    )
).withColumn(
    "initial_reserve",
    expr("CASE " +
         "WHEN claim_type LIKE '%Fryer%' THEN 5000 + (rand() * 45000) " + # Fryer burns are expensive
         "WHEN claim_type LIKE '%Ice%' THEN 2000 + (rand() * 20000) " + # Ice slips are medium cost
         "ELSE 500 + (rand() * 5000) END"                               # General/Coffee burns are cheaper
    )
)

# 4. Finalize Target Variables with Noise
# Real-world data is messy, so we add variance to our costs and flag litigated claims
df_final = df_claims.withColumn(
    "total_claim_cost",
    col("intial_reserve") * (lit(1.0) + (rand() - 0.2)) # Add a variance between -20% and +80%
).withColumn(
    "is_litigated",
    expr("total_claim_cost > 20000").cast("int") # 1 if litigated (cost > 20k), 0 otherwise
)

# 5. Save to Delta format
# We write this to a temporary DBFS path so the next script in our pipeline can read it
output_path = "dbfs:/tmp/rbi_portfolio/simulated_claims"
print(f"Saving {num_rows} simulated records to {output_path}...")

# Mode 'overwrite' ensures we can run this script multiple times safely
df_final.write.format("delta").mode("overwrite").save(output_path)

print("Data simulation complete. Ready for Feature Engineering.")

# Display the first 10 rows to verify (this will output a nice table in the Databricks UI)
display(df_final.limit(10))