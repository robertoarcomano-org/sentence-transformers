import { pipeline } from "@huggingface/transformers";
import { QdrantClient } from "@qdrant/js-client-rest";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import fs from "fs";

class ParseIntoDB {
  constructor({ chunkSize = 800, chunkOverlap = 100, vectorSize = 384 } = {}) {
    this.chunkSize = chunkSize;
    this.chunkOverlap = chunkOverlap;
    this.vectorSize = vectorSize;
    this.COLLECTION = "lapiazza";
  }

  async init(deleteCollection = false) {
    this.db = new QdrantClient({ url: "http://localhost:6333" });
    this.model = await pipeline("feature-extraction", "Xenova/paraphrase-multilingual-MiniLM-L12-v2");

    const { exists } = await this.db.collectionExists(this.COLLECTION);
    if (exists && deleteCollection) {
      await this.db.deleteCollection(this.COLLECTION);
    }
    if (!exists || deleteCollection) {
      await this.db.createCollection(this.COLLECTION, {
        vectors: { size: this.vectorSize, distance: "Cosine" },
      });
    }
  }

  async encode(texts) {
    const output = await this.model(texts, { pooling: "mean", normalize: true });
    return output.tolist();
  }

  async add(document) {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
      separators: ["\n\n", "\n", ". ", " ", ""],
    });
    const documents = await splitter.splitText(document);

    console.log(documents.length);

    const embeddings = await this.encode(documents);
    const points = documents.map((doc, i) => ({
      id: i,
      vector: embeddings[i],
      payload: { text: doc },
    }));

    await this.db.upsert(this.COLLECTION, { points });
    console.log("Inseriti", points.length, "documenti");
  }

  async search(query) {
    const [queryVector] = await this.encode([query]);
    const results = await this.db.query(this.COLLECTION, {
      query: queryVector,
      limit: 3,
      with_payload: true,
    });

    console.log(`\nRisultati per: '${query}'\n`);
    for (const r of results.points) {
      console.log(`score=${r.score.toFixed(4)} | id=${r.id} | text=${r.payload.text}\n`);
    }
  }
}

async function main() {
  const p = new ParseIntoDB({ chunkSize: 800, chunkOverlap: 50, vectorSize: 384 });
  await p.init(true);
  const text = fs.readFileSync("../article1.txt", "utf-8");
  await p.add(text);
  await p.search("Parlami di un autore di quadri");
}

main();
