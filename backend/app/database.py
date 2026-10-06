from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI is missing from .env")


print("Testing MongoDB connection...")


client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=10000
)


try:
    result = client.admin.command("ping")
    print("MongoDB connection successful:", result)
except Exception as e:
    print("MongoDB connection failed:")
    print(e)
    raise


# =========================
# DATABASE
# =========================

db = client["ai_video_repurposer"]


# =========================
# JOB COLLECTION
# =========================

jobs_collection = db["jobs"]