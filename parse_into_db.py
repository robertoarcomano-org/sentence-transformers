import os
os.environ["HF_HUB_OFFLINE"] = "1"

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParseIntoDB:

    def init_db(self, delete_collection):
        self.db = QdrantClient(url="http://localhost:6333")
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # 384 dimensioni, leggero
        self.COLLECTION = "lapiazza"
        
        if not self.db.collection_exists(self.COLLECTION):
            self.db.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
        else:
            if delete_collection:
                self.db.delete_collection(collection_name=self.COLLECTION)            
            self.db.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def __init__(self, chunk_size=800, chunk_overlap=100, vector_size=384, delete_collection=False):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_size = vector_size
        self.init_db(delete_collection=delete_collection)

    def add(self, document):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""] 
        )
        documents = splitter.split_text(document)
        # print(len(documents))
        # for i, d in enumerate(documents):
        #     print(i,len(d),d)
        #     print()

        embeddings = self.model.encode(documents)
        points = [
            PointStruct(id=i, vector=emb.tolist(), payload={"text": doc})
            for i, (emb, doc) in enumerate(zip(embeddings, documents))
        ]
        self.db.upsert(collection_name=self.COLLECTION, points=points)
        print("Inseriti", len(points), "documenti")

    def search(self, query):
        query_vector = self.model.encode(query).tolist()
        results = self.db.query_points(
            collection_name=self.COLLECTION,
            query=query_vector,
            limit=3,
            with_payload=True
        ).points

        print(f"\nRisultati per: '{query}'\n")
        for r in results:
            print(f"score={r.score:.4f} | id={r.id} | text={r.payload['text']}\n")

if __name__ == "__main__":
    p = ParseIntoDB(chunk_size=800, chunk_overlap=50, vector_size=384, delete_collection=True)
    with open("article1.txt", "r") as f:
        text = f.read()
    p.add(text)
    p.search("Parlami di un autore di quadri")


# // List all collections
# GET collections

# // Get collection info
# GET collections/test_docs

# // List points in a collection, using filter
# POST /collections/test_docs/points/scroll
# {
#   "limit": 10,
#   "with_payload": true,
#   "with_vector": false
# }
