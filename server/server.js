const { OllamaEmbeddings } = require("@langchain/ollama");
const { MongoClient } = require("mongodb");
const { Binary } = require("bson");

const MONGO_URI =
  "mongodb+srv://sd6200:JQ7GhhWLZxgNyNAe@cluster0.aagcme2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";
const DB_NAME = "embedded_health";
const COLLECTION_NAME = "embedded_health_collection";

const OLLAMA_MODEL = "qwen3-embedding";
const OLLAMA_BASE_URL = "http://100.64.0.1:11434";
const NEW_EMBEDDING_FIELD = "description_embedding_qwen3";
const BATCH_SIZE = 50;

const embeddings = new OllamaEmbeddings({
  model: OLLAMA_MODEL,
  baseUrl: OLLAMA_BASE_URL,
});
const client = new MongoClient(MONGO_URI);

async function main() {
  try {
    await client.connect();
    console.log("Connected to MongoDB");

    const db = client.db(DB_NAME);
    const collection = db.collection(COLLECTION_NAME);
    const query = {
      description: { $exists: true, $ne: "" },
      //[NEW_EMBEDDING_FIELD]: { $exists: false },
    };
    const cursor = collection.find(query);
    let operations = [];
    let processedCount = 0;

    console.log(
      "Starting to process documents into the correct BSON Binary format..."
    );

    for await (const doc of collection.find(query)) {
      const vector = await embeddings.embedQuery(doc.description); // use description
      const vectorBinary = Binary.fromFloat32Array(new Float32Array(vector));

      operations.push({
        updateOne: {
          filter: { _id: doc._id },
          update: { $set: { [NEW_EMBEDDING_FIELD]: vectorBinary } },
        },
      });

      if (operations.length >= BATCH_SIZE) {
        await collection.bulkWrite(operations);
        processedCount += operations.length;
        console.log(`... processed and updated ${processedCount} documents`);
        operations = [];
      }
    }

    if (operations.length > 0) {
      await collection.bulkWrite(operations);
      processedCount += operations.length;
      console.log(`... processed and updated ${processedCount} documents`);
    }

    console.log("Batch update complete.");
  } catch (error) {
    console.error("An error occurred:", error);
  } finally {
    await client.close();
    console.log("Disconnected from MongoDB.");
  }
}

main();
