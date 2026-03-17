from pymongo import MongoClient
import pandas as pd

client = MongoClient("mongodb://localhost:27017")

db = client["quantum-sim"]

col = db["training_data"]
data = list(col.find({}, {"_id":0}))

df = pd.DataFrame(data)

df.to_csv("mongo.csv", index=False)

print("Exported → mongo.csv")