const express = require("express");
const util = require("util");

const { Binary } = require("bson");
const { MongoClient } = require("mongodb");
const { ChatOllama, OllamaEmbeddings } = require("@langchain/ollama");
const { HumanMessage, SystemMessage } = require("@langchain/core/messages");
const { StateGraph, START, END } = require("@langchain/langgraph");
const { StructuredTool } = require("@langchain/core/tools");
const { z } = require("zod");
const axios = require("axios");

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

class GenerateTreatmentPlanTool extends StructuredTool {
  name = "GenerateTreatmentPlan";
  description =
    "Generate potential treatment plans, home remedies, and recommended lifestyle changes for a given disease.";

  schema = z.object({
    disease: z.string().describe("The diagnosed or suspected disease name."),
    severity: z
      .string()
      .optional()
      .describe(
        "Severity or stage of the disease if known (mild, moderate, severe)."
      ),
  });

  async _call({ disease, severity }) {
    // you can later connect this to a treatments database or just use the LLM for now
    const prompt = `Provide a clear, evidence-based treatment plan for ${disease}${
      severity ? " (severity: " + severity + ")" : ""
    }. Include:
    - Common medications (if applicable)
    - Home care or lifestyle tips
    - When to seek professional help`;

    const response = await llm.invoke([new HumanMessage(prompt)]);
    return response.content;
  }
}

class AssessSeriousnessTool extends StructuredTool {
  name = "AssessSeriousness";
  description =
    "Assess the seriousness of a disease or symptoms as mild, moderate, or severe.";

  schema = z.object({
    disease: z.string().describe("The identified or suspected disease."),
    symptoms: z.string().describe("Description of the user’s symptoms."),
  });

  async _call({ disease, symptoms }) {
    const prompt = `Based on the following:
    - Disease: ${disease}
    - Symptoms: ${symptoms}

    Assess the seriousness level (mild, moderate, or severe) and give a 1-2 sentence justification. 
    Respond in JSON format like this:
    { "seriousness": "moderate", "reason": "Symptoms indicate persistent fever and fatigue." }`;

    const response = await llm.invoke([new HumanMessage(prompt)]);
    return response.content;
  }
}

class FindNearbyCareTool extends StructuredTool {
  name = "FindNearbyCare";
  description =
    "Suggest nearby clinics, urgent care centers, or hospitals based on ZIP code.";

  schema = z.object({
    zipCode: z.string().describe("The ZIP code for the user's location."),
    urgency: z.string().describe("Urgency level: mild, moderate, or severe."),
  });

  async _call({ zipCode, urgency }) {
    const axios = require("axios");

    // Nominatim search URL with US country code
    const url = `https://nominatim.openstreetmap.org/search`;
    const params = {
      q: `hospital near ${zipCode}`,
      format: "json",
      countrycodes: "us",
      limit: 5,
    };

    try {
      const response = await axios.get(url, {
        params,
        headers: {
          "User-Agent": "MyAppName/1.0 (myemail@example.com)",
        },
      });

      const hospitals = response.data.map((h) => ({
        name: h.display_name,
        lat: h.lat,
        lon: h.lon,
      }));

      return {
        urgency,
        recommendedOptions: hospitals,
      };
    } catch (err) {
      console.error("Error fetching hospitals:", err.message);
      return {
        urgency,
        recommendedOptions: [],
      };
    }
  }
}

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
    "Summarize suggested diseases and identify the most likely disease to store for further analysis.";

  schema = z.object({
    summary: z
      .string()
      .describe(
        "A readable summary for the user listing the top 3–5 suspected diseases, with confidence percentages."
      ),
    suspectedDisease: z
      .string()
      .describe(
        "The single most likely disease (best match) to use for subsequent steps in reasoning or treatment planning."
      ),
  });

  async _call({ summary, suspectedDisease }) {
    // You could store or log this if needed
    console.log("Top suspected disease:", suspectedDisease);
    console.log("Summary for user:", summary);
    return { summary, suspectedDisease };
  }
}

const querySymptomsDatabaseTool = new QuerySymptomsDatabaseTool();
const summarizeDiseaseSuggestionsTool = new SummarizeDiseaseSuggestionsTool();
const generateTreatmentPlanTool = new GenerateTreatmentPlanTool();
const assessSeriousnessTool = new AssessSeriousnessTool();
const findNearbyCareTool = new FindNearbyCareTool();

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

  const toolCall = response.tool_calls?.[0]?.args || {};
  const { summary, suspectedDisease } = toolCall;

  return {
    toolCalls: response.tool_calls,
    summary,
    suspectedDisease,
  };
}

async function assessSeriousnessNode(state) {
  console.log("Assess SERIOUSNESS STATE:", state);

  // Use the suspected disease identified earlier
  const disease = state.suspectedDisease || "Unknown condition";
  const symptoms = state.userInput || "";

  // Actually invoke the AssessSeriousnessTool
  const seriousnessRaw = await assessSeriousnessTool.invoke({
    disease,
    symptoms,
  });

  console.log("Assess Seriousness Result:", seriousnessRaw);

  // Try to safely parse the JSON (since model responses are strings)
  let seriousnessResult;
  try {
    seriousnessResult = JSON.parse(seriousnessRaw);
  } catch {
    seriousnessResult = { seriousness: "mild", reason: "Default fallback" };
  }

  return {
    ...state,
    seriousnessResult,
  };
}

async function decideNextStepNode(state) {
  const { seriousnessResult } = state;
  const seriousness = seriousnessResult?.seriousness?.toLowerCase?.() || "mild";

  console.log("Deciding next step based on seriousness:", seriousness);

  if (seriousness === "mild" || seriousness === "moderate") {
    return { next: "generateTreatmentPlanNode" };
  } else {
    return { next: "findNearbyCareNode" };
  }
}

async function generateTreatmentPlanNode(state) {
  console.log("Generating treatment plan for:", state);

  const likelyDisease = state.toolCalls?.[0]?.disease || "Unknown condition";

  const treatmentPlan = await generateTreatmentPlanTool.invoke({
    disease: likelyDisease,
    severity: state.seriousnessResult?.seriousness || "mild",
  });

  console.log("Treatment plan:", treatmentPlan);

  return {
    ...state,
    treatmentPlan,
  };
}

async function findNearbyCareNode(state) {
  console.log("Finding nearby care for:", state);

  // you’ll need to pass the ZIP from the API request into the graph invocation (see below)
  const zipCode = state.zipCode || "00000";

  const careRecommendations = await findNearbyCareTool.invoke({
    zipCode,
    urgency: state.seriousnessResult?.seriousness || "moderate",
  });

  console.log("Nearby care recommendations:", careRecommendations);

  return {
    ...state,
    careRecommendations,
  };
}

// step 1: define a graph
const workflow = new StateGraph({ channels: graphStateData });

workflow.addNode("querySymptomsNode", querySymptomsNode);
// workflow.addNode("assessSeriousnessNode", assessSeriousnessNode);
// workflow.addNode("decideNextStepNode", decideNextStepNode);
// workflow.addNode("generateTreatmentPlanNode", generateTreatmentPlanNode);
workflow.addNode("findNearbyCareNode", findNearbyCareNode);

// step 3: define edges
workflow.addEdge(START, "querySymptomsNode");

workflow.addEdge("querySymptomsNode", "findNearbyCareNode");
workflow.addEdge("findNearbyCareNode", END);
// workflow.addConditionalEdges("assessSeriousnessNode", (state) => {
//   const seriousness =
//     state.seriousnessResult?.seriousness?.toLowerCase?.() || "mild";

//   if (seriousness === "severe") return "findNearbyCareNode";
//   return "generateTreatmentPlanNode";
// });
// workflow.addEdge("generateTreatmentPlanNode", END);
// workflow.addEdge("findNearbyCareNode", END);
// step 4: compile workflow/graph
const graph = workflow.compile();

// EXPRESS API CODE GOES HERE
app.post("/agenttest", async function (req, res) {
  const result = await graph.invoke({
    userInput: req.body.diseasesLike,
    zipCode: req.body.zipCode, // 👈 add this line
  });

  console.log("GRAPH RESULT:", result);
  res.json(result);
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
