import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

function App() {
  const [zipCode, setZipCode] = useState("");
  const [symptoms, setSymptoms] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://localhost:4000/symptoms", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ diseasesLike: symptoms, zipCode }),
      });
      if (!response.ok) throw new Error("API request failed");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      setError("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>AI Health Assistant</h1>
        <p className="disclaimer">
          ⚠️ This AI is for informational purposes only. Not a substitute for
          professional medical advice.
        </p>

        <form onSubmit={handleSubmit} className="form-container">
          <input
            type="text"
            placeholder="ZIP Code (e.g., 84770)"
            value={zipCode}
            onChange={(e) => setZipCode(e.target.value)}
            required
            className="input-field"
          />
          <textarea
            placeholder="Describe your symptoms..."
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            required
            rows={4}
            className="textarea-field"
          />
          <button type="submit" disabled={loading} className="submit-btn">
            {loading ? "Checking..." : "Submit"}
          </button>
        </form>

        {error && <p className="error">{error}</p>}

        {result && (
          <div className="result-container">
            <h2>Results</h2>

            {result.summary && (
              <div className="result-section">
                <h3>Most Likely Disease</h3>
                <p>{result.summary}</p>
              </div>
            )}

            {result.seriousnessResult && (
              <div className="result-section">
                <h3>Seriousness Assessment</h3>
                <p>
                  <strong>Seriousness:</strong>{" "}
                  {result.seriousnessResult.seriousness}
                  <br />
                  <strong>Reason:</strong> {result.seriousnessResult.reason}
                </p>
              </div>
            )}

            {result.treatmentPlan && (
              <div className="result-section">
                <h3>Treatment Plan</h3>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    table: ({ node, ...props }) => (
                      <div style={{ overflowX: "auto" }}>
                        <table className="markdown-table" {...props} />
                      </div>
                    ),
                  }}
                >
                  {result.treatmentPlan}
                </ReactMarkdown>
              </div>
            )}
            {result.careRecommendations?.recommendedOptions?.length > 0 && (
              <div className="result-section">
                <h3>Nearby Care Facilities</h3>
                <p>
                  Based on the seriousness of your symptoms, we recommend the
                  following treatment centers:
                </p>
                <ul>
                  {result.careRecommendations.recommendedOptions.map((h, i) => (
                    <li key={i}>
                      <strong>{h.name}</strong> ({h.lat}, {h.lon})
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
