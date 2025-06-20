# filepath: scripts/check_milvus.py
from database.milvus_handler import MilvusHandler

milvus_handler = MilvusHandler()
collection = milvus_handler.collection
print("Total entities:", collection.num_entities)
