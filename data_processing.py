from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pymongo
import pandas as pd
from pymongo import UpdateOne
from datetime import datetime

# Initialize Spark session (no MongoDB connector needed)
spark = SparkSession.builder \
    .appName("Sentiment Analysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Data Processing Pipeline

# Load data from MongoDB using pymongo
print("=== Loading Data from MongoDB using pymongo ===")
uri = "mongodb+srv://saad:mongoPass@dataprojectid2221.2uplah7.mongodb.net/?retryWrites=true&w=majority&appName=DataProjectID2221"
client = pymongo.MongoClient(uri)
db = client["Our_Database"]
collection = db["collection_1"]


# Check MongoDB collection count first
mongo_count = collection.count_documents({})
print(f"MongoDB collection has {mongo_count} documents")

# Get data as pandas DataFrame first (exclude _id field)
data = list(collection.find({}, {"_id": 0}))
pandas_df = pd.DataFrame(data)

# Convert to Spark DataFrame
df = spark.createDataFrame(pandas_df)

print(f"Total records loaded into Spark: {df.count()}")


print("\nData Schema:")
df.printSchema()

# Partition data across workers for parallel processing
print("\n=== Partitioning Data ===")
partitioned_df = df.repartition(8)
print(f"Number of partitions: {partitioned_df.rdd.getNumPartitions()}")

# Initial data validation and cleaning
print("\n=== Data Validation and Cleaning ===")

# Check for null values
print("Null values per column:")
for column in df.columns:
    null_count = df.filter(col(column).isNull()).count()
    print(f"{column}: {null_count}")

# Remove rows with null text or sentiment
cleaned_df = df.filter(
    col("text").isNotNull() &
    col("sentiment").isNotNull() &
    (col("text") != "")
)

print(f"\nRecords after cleaning: {cleaned_df.count()}")

# Check sentiment distribution
print("\nSentiment distribution:")
cleaned_df.groupBy("sentiment").count().show()

# Sample of cleaned data
print("\nSample of cleaned data:")
cleaned_df.select("text", "sentiment", "Time of Tweet", "Age of User").show(8, truncate=False)

# Cache the cleaned data for future operations
cleaned_df.cache()

print("\n=== Storing Processed Data ===")

# Create a new collection for processed data
processed_collection = db["processed_sentiment_data"]

# Create indexes for optimized queries
print("Creating indexes for optimized queries...")
try:
    # Index for time-based queries
    processed_collection.create_index([("Time of Tweet", 1)])
    # Index for sentiment-based queries
    processed_collection.create_index([("sentiment", 1)])
    # Compound index for time and sentiment queries
    processed_collection.create_index([("Time of Tweet", 1), ("sentiment", 1)])
    # Unique index on textID to prevent duplicates
    processed_collection.create_index([("textID", 1)], unique=True)
    print("Indexes created successfully")
except Exception as e:
    print(f"Index creation completed (some may already exist): {e}")

# Convert Spark DataFrame to list of dictionaries for MongoDB storage
processed_data = cleaned_df.collect()
processed_records = []

for row in processed_data:
    record = {
        "textID": row["textID"],
        "text": row["text"],
        "sentiment": row["sentiment"],
        "Time of Tweet": row["Time of Tweet"],
        "Age of User": row["Age of User"],
        "processed_timestamp": datetime.now(),
       
    }
    processed_records.append(record)

# Bulk upsert operation to prevent duplicates
print(f"Upserting {len(processed_records)} records...")
upsert_count = 0
update_count = 0
error_count = 0



# Create bulk operations for upsert
bulk_operations = []
for record in processed_records:
    operation = UpdateOne(
        {"textID": record["textID"]},  # Filter by textID
        {"$set": record},              # Update with new data
        upsert=True                    # Insert if doesn't exist
    )
    bulk_operations.append(operation)

# Execute bulk operations in batches
batch_size = 1000
for i in range(0, len(bulk_operations), batch_size):
    batch = bulk_operations[i:i + batch_size]
    try:
        result = processed_collection.bulk_write(batch)
        upsert_count += result.upserted_count
        update_count += result.modified_count
    except Exception as e:
        error_count += len(batch)
        print(f"Error in batch {i//batch_size + 1}: {e}")

print(f"Storage completed:")
print(f"  - New records inserted: {upsert_count}")
print(f"  - Existing records updated: {update_count}")
print(f"  - Errors: {error_count}")

# Verify storage
final_count = processed_collection.count_documents({})
print(f"Total records in processed collection: {final_count}")

print("\n" + "="*60)
print("=== DATA PROCESSING COMPLETE ===")
print("="*60)
print(f"Processed {final_count} records and stored in 'processed_sentiment_data' collection")
print("Run 'sentiment_analysis.py' to analyze the processed data")

# Close MongoDB connection
client.close()

spark.stop()