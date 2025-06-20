from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


class MilvusHandler:
    def __init__(
        self, host="localhost", port="19530", collection_name="face_embeddings", dim=128
    ):
        self.collection_name = collection_name
        self.dim = dim
        connections.connect(host=host, port=port)
        self.collection = self.create_collection()

    def get_milvus_collection(self, host, port):
        connections.connect(host=host, port=port)

    def create_collection(self):
        fields = [
            FieldSchema(name="student_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]
        schema = CollectionSchema(fields, description="Face embeddings")

        # Ganti pengecekan koleksi
        if not utility.has_collection(self.collection_name):
            collection = Collection(name=self.collection_name, schema=schema)
            print(f"Collection '{self.collection_name}' created.")
        else:
            collection = Collection(name=self.collection_name)
            print(f"Collection '{self.collection_name}' already exists.")

        # Index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        return collection
