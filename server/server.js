const express = require("express");
const util = require("util");

const { Binary } = require("bson");
const { MongoClient } = require("mongodb");
const { ChatOllama, OllamaEmbeddings } = require("@langchain/ollama");
const { HumanMessage, SystemMessage } = require("@langchain/core/messages");
const { StateGraph, START, END } = require("@langchain/langgraph");
const { StructuredTool } = require("@langchain/core/tools");
const { z } = require("zod");

const app = express();
const port = 4000;

app.use(express.json());
app.use(express.static("public"));

const MONGO_URI =
  "mongodb+srv://sd6200:JQ7GhhWLZxgNyNAe@cluster0.aagcme2.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";
const DB_NAME = "embedded_health";
const COLLECTION_NAME = "embedded_health_collection";

const OLLAMA_BASE_URL = "http://100.64.0.1:11434";
const EMBEDDING_FIELD = "description_embedding_qwen3";
const INDEX_NAME = "vector_index";

const LLM_MODEL = "gpt-oss:20b";
const EMBEDDING_MODEL = "qwen3-embedding";

const llm = new ChatOllama({
  baseUrl: OLLAMA_BASE_URL,
  model: LLM_MODEL,
});

const embeddings = new OllamaEmbeddings({
  baseUrl: OLLAMA_BASE_URL,
  model: EMBEDDING_MODEL,
});

const mongoClient = new MongoClient(MONGO_URI);

class QuerySymptomsDatabaseTool extends StructuredTool {
  name = "QuerySymptomsDatabase";
  description =
    "Query symptoms database to find relevant diseases based on a symptom description.";

  schema = z.object({
    numResults: z.number().describe("The number of search results to return."),
    query: z
      .string()
      .describe(
        "The search query: perhaps a disease description or general disease idea."
      ),
  });

  async _call({ numResults, query }) {
    try {
      await mongoClient.connect();
      console.log("Connected to MongoDB");

      const db = mongoClient.db(DB_NAME);
      const collection = db.collection(COLLECTION_NAME);

      // generate an embedding for the search query
      console.log(`Generating embedding for: "${query}"`);
      const queryVector = await embeddings.embedQuery(query);
      const vectorBinary = Binary.fromFloat32Array(
        new Float32Array(queryVector)
      );

      // define the vector search pipeline
      const pipeline = [
        {
          $vectorSearch: {
            index: INDEX_NAME,
            path: EMBEDDING_FIELD,
            queryVector: vectorBinary,
            numCandidates: 100, // number of candidates to consider
            limit: numResults, // number of top results to return
          },
        },
        {
          // fields to return
          $project: {
            _id: 0,
            disease: 1,
            description: 1,
            score: { $meta: "vectorSearchScore" },
          },
        },
      ];

      // execute the query
      console.log("Searching for similar symtoms...");
      const results = await collection.aggregate(pipeline).toArray();

      return results;
    } catch (error) {
      console.error("An error occurred:", error);
      return [];
    } finally {
      await mongoClient.close();
      console.log("Disconnected from MongoDB");
    }
  }
}

class SummarizeDiseaseSuggestionsTool extends StructuredTool {
  name = "SummarizeDiseaseSuggestions";
  description =
    "Convey directly to the end-user a summary of suggested disease ideas.";

  schema = z.object({
    text: z
      .string()
      .describe(
        "A brief description of the disease suggestions. 3-5 suggestions is appropriate. Markdown preferred. Include with each result a match value as a percentage, e.g. 92%."
      ),
  });
}

const querySymptomsDatabaseTool = new QuerySymptomsDatabaseTool();
const summarizeDiseaseSuggestionsTool = new SummarizeDiseaseSuggestionsTool();

const graphStateData = {
  userInput: "",
  summary: "",
  toolCalls: [],
};

// NODE: query symptoms database
async function querySymptomsNode(state) {
  console.log("Query SYMPTOMS STATE:", state);

  // INVOKE THE TOOL!
  const toolResult = await querySymptomsDatabaseTool.invoke({
    query: state.userInput,
    numResults: 5, // or whatever number you want
  });
  console.log("TOOL RESULT:", toolResult);

  const message = new HumanMessage({
    content: [
      {
        type: "text",
        text: `Here are the results from querying relevant disease suggestions based on user input. Evaluate and report back based on the tools available to you. ${JSON.stringify(
          toolResult
        )}`,
      },
    ],
  });

  const llmWithTool = llm.bind({ tools: [summarizeDiseaseSuggestionsTool] });
  const response = await llmWithTool.invoke([message]);
  console.log("RAW LLM RESPONSE:", response);

  return {
    toolCalls: response.tool_calls,
  };
}

// step 1: define a graph
const workflow = new StateGraph({ channels: graphStateData });

workflow.addNode("querySymptomsNode", querySymptomsNode);

// step 3: define edges
workflow.addEdge(START, "querySymptomsNode");

workflow.addEdge("querySymptomsNode", END);

// step 4: compile workflow/graph
const graph = workflow.compile();

// EXPRESS API CODE GOES HERE

app.post("/agenttest", async function (req, res) {
  const result = await graph.invoke({
    userInput: req.body.diseasesLike,
  });

  console.log("GRAPH RESULT:", result);

  res.json(result);
});

app.listen(port, function () {
  console.log(`Server running on port ${port}`);
});
