import pymongo
import csv

uri = "mongodb+srv://saad:mongoPass@dataprojectid2221.2uplah7.mongodb.net/?retryWrites=true&w=majority&appName=DataProjectID2221"

# Create a new client and connect to the server
client = pymongo.MongoClient(uri, server_api=pymongo.server_api.ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

print("Databases:", client.list_database_names())

myDb = client["Our_Database"]
myCollection = myDb["collection_1"]

myCollection.drop()

# Open and insert CSV
with open("test.csv", newline="", encoding="iso-8859-1") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    if rows:
        myCollection.insert_many(rows)
        print(f"Inserted {len(rows)} rows into MongoDB!")
    else:
        print("No rows found in CSV file.") 