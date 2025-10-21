// CommonJS version
const fs = require("fs");
const { MongoClient } = require("mongodb");

const MONGO_URI =
  "mongodb+srv://sd6200:JQ7GhhWLZxgNyNAe@cluster0.aagcme2.mongodb.net/?retryWrites=true&w=majority";
const DB_NAME = "embedded_health"; // your database name
const COLLECTION_NAME = "embedded_health_collection"; // your collection name
const JSON_FILE = "./data.json"; // path to your JSON file

async function uploadData() {
  const client = new MongoClient(MONGO_URI);

  try {
    await client.connect();
    console.log("Connected to MongoDB");

    const db = client.db(DB_NAME);
    const collection = db.collection(COLLECTION_NAME);

    // Read JSON file
    const jsonData = JSON.parse(fs.readFileSync(JSON_FILE, "utf8"));

    // Insert all documents into the collection
    const result = await collection.insertMany(jsonData);
    console.log(`✅ Inserted ${result.insertedCount} documents`);
  } catch (err) {
    console.error(err);
  } finally {
    await client.close();
    console.log("Disconnected from MongoDB");
  }
}

uploadData();
