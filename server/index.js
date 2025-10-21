const express = require("express");
const util = require("util");
require("dotenv").config();

const { Binary } = require("bson");
const { MongoClient } = require("mongodb");
const { OllamaEmbeddings } = require("@langchain/ollama");

const OLLAMA_BASE_URL = "http://100.64.0.1:11434";
const OLLAMA_MODEL = "qwen3-embedding";

const MONGO_URI =
  "mongodb+srv://sd6200:JQ7GhhWLZxgNyNAe@cluster0.aagcme2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";
const DB_NAME = "embedded_health"; // your database name
const COLLECTION_NAME = "embedded_health_collection";
const EMBEDDING_FIELD = "description_embedding_qwen3";
const INDEX_NAME = "vector_index";

const embeddingsModel = new OllamaEmbeddings({
  model: OLLAMA_MODEL,
  baseUrl: OLLAMA_BASE_URL,
});

const mongoClient = new MongoClient(MONGO_URI);

const app = express();
const port = 4000;

app.use(express.json());
app.use(express.static("public"));

app.get("/symptoms", async function (req, res) {
  let searchQuery = req.query.searchQuery;

  try {
    await mongoClient.connect();
    console.log("Connected to MongoDB");
    const db = mongoClient.db(DB_NAME);
    const collection = db.collection(COLLECTION_NAME);

    console.log(`Generating embedding for: "${searchQuery}"`);
    const queryVector = await embeddingsModel.embedQuery(searchQuery);
    const vectorBinary = Binary.fromFloat32Array(new Float32Array(queryVector));

    const pipeline = [
      {
        $vectorSearch: {
          index: INDEX_NAME,
          path: EMBEDDING_FIELD,
          queryVector: vectorBinary,
          numCandidates: 100,
          limit: 20,
        },
      },
      {
        $project: {
          _id: 0,
          disease: 1,
          description: 1,
          score: { $meta: "vectorSearchScore" },
        },
      },
    ];

    console.log("Searching for similar symptoms...");
    const results = await collection.aggregate(pipeline).toArray();
    res.json(results);
  } catch (error) {
    console.error("An error occurred:", error);
    res.sendStatus(500);
  } finally {
    await mongoClient.close();
    console.log("Disconnected from MongoDB.");
  }
});

app.listen(port, function () {
  console.log(`Server running on port ${port}`);
});
