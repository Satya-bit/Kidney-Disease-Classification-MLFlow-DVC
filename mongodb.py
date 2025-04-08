import pymongo
from pymongo import MongoClient
import json

client = MongoClient("mongodb://localhost:27017")
mydb = client["Kidney_tumor"]

collection = mydb.Personal_details

# record={
#     "name":"Rahul",
#     "age":21
# }

# collection.insert_one(record)