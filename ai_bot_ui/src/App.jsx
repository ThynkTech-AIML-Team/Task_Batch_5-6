import { useEffect, useRef, useState } from "react";

export default function App() {
  const [domain, setDomain] = useState("");
  const [candidateName, setCandidateName] = useState("");
  const [sessionId, setSessionId] = useState("");

  const [questions, setQuestions] = useState([]);
  const [qIndex, setQIndex] = useState(0);
  const [question, setQuestion] = useState("");

  const [answerText, setAnswerText] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastAnalyzedQuestion, setLastAnalyzedQuestion] = useState("");

  const [finalResults, setFinalResults] = useState(null);
  const [error, setError] = useState("");

  const endpoints = {
    start: "/start-interview",
    ask: "/ask-question",
    results: "/results",
  };

  // ---------- Mic (Speech to Text) ----------
  const recognitionRef = useRef(null);
  const [isListening, setIsListening] = useState(false);

  useEffect(() => {
    if (!("webkitSpeechRecognition" in window)) return;

    const rec = new window.webkitSpeechRecognition();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-IN";

    rec.onresult = (e) => {
      let text = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        text += e.results[i][0].transcript;
      }
      setAnswerText(text);
    };

    rec.onend = () => setIsListening(false);
    rec.onerror = () => setIsListening(false);

    recognitionRef.current = rec;
  }, []);

  function startMic() {
    try {
      recognitionRef.current?.start();
      setIsListening(true);
    } catch {}
  }

  function stopMic() {
    recognitionRef.current?.stop();
    setIsListening(false);
  }

  // ---------- Styles (Dark Theme) ----------
  const page = {
    minHeight: "100vh",
    width: "100vw",
    margin: 0,
    padding: "18px",
    backgroundColor: "#000",
    color: "#fff",
    fontFamily: "Arial",
    boxSizing: "border-box",
  };

  const title = { margin: "4px 0 14px 0", fontSize: 26 };

  const card = {
    border: "1px solid #333",
    borderRadius: 12,
    padding: 14,
    background: "#111",
    color: "#fff",
    boxShadow: "0 0 0 1px rgba(255,255,255,0.02)",
  };

  const subCard = {
    border: "1px solid #2a2a2a",
    borderRadius: 10,
    padding: 12,
    background: "#0b0b0b",
    color: "#fff",
  };

  const label = { fontSize: 13, opacity: 0.9 };

  const input = {
    padding: "10px 10px",
    marginRight: 10,
    background: "#000",
    color: "#fff",
    border: "1px solid #555",
    borderRadius: 8,
    outline: "none",
    width: 220,
  };

  const textarea = {
    width: "100%",
    height: 120,
    padding: 12,
    background: "#000",
    color: "#fff",
    border: "1px solid #555",
    borderRadius: 10,
    outline: "none",
    resize: "vertical",
    boxSizing: "border-box",
  };

  const btn = (disabled = false, variant = "primary") => {
    const base = {
      padding: "10px 14px",
      borderRadius: 10,
      border: "1px solid transparent",
      cursor: disabled ? "not-allowed" : "pointer",
      fontWeight: 600,
      transition: "transform 0.06s ease, opacity 0.2s ease",
      opacity: disabled ? 0.6 : 1,
      userSelect: "none",
    };

    if (variant === "primary") {
      return {
        ...base,
        background: disabled ? "#2b2b2b" : "#1f6feb",
        color: disabled ? "#aaa" : "#fff",
      };
    }

    if (variant === "ghost") {
      return {
        ...base,
        background: disabled ? "#111" : "#111",
        color: disabled ? "#aaa" : "#fff",
        border: "1px solid #444",
      };
    }

    return base;
  };

  const hr = { border: "none", borderTop: "1px solid #222", margin: "14px 0" };

  const grid2 = {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 14,
  };

  const tableWrap = { overflowX: "auto" };

  const table = { width: "100%", borderCollapse: "collapse", minWidth: 700 };

  const th = {
    border: "1px solid #333",
    padding: 10,
    background: "#0f0f0f",
    textAlign: "left",
    fontSize: 13,
  };

  const td = { border: "1px solid #333", padding: 10, fontSize: 13 };

  const muted = { opacity: 0.8, fontSize: 13 };

  // ---------- API Calls ----------
  async function startInterview() {
    setError("");
    setAnalysis(null);
    setIsAnalyzing(false);
    setLastAnalyzedQuestion("");
    setFinalResults(null);

    if (!candidateName.trim()) {
      alert("Enter candidate name first");
      return;
    }

    try {
      const res = await fetch(`/api${endpoints.start}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain, candidate_name: candidateName }),
      });

      const data = await res.json();

      if (!res.ok || data?.error) {
        setError(data?.error || "Failed to start interview");
        return;
      }

      const qs = Array.isArray(data.questions) ? data.questions : [];
      setSessionId(data.candidate_id || "");
      setQuestions(qs);
      setQIndex(0);
      setQuestion(qs[0] || "");
      setAnswerText("");
    } catch (e) {
      setError(String(e));
    }
  }

  // ✅ FAST UI: move to next question immediately, analysis comes async
  async function submitAnswer() {
    setError("");

    if (!sessionId) return alert("Start interview first");
    if (!question) return alert("No question loaded");
    if (!answerText.trim()) return alert("Speak or type your answer first");

    const currentQuestion = questions[qIndex];
    const currentAnswer = answerText;

    // Move next immediately (fast)
    const nextIndex = qIndex + 1;
    const nextQ =
      nextIndex < questions.length
        ? questions[nextIndex]
        : "Interview completed. Click Show Results.";

    setQIndex(nextIndex);
    setQuestion(nextQ);
    setAnswerText("");

    // Show analyzing status
    setIsAnalyzing(true);
    setLastAnalyzedQuestion(currentQuestion);

    try {
      const res = await fetch(`/api${endpoints.ask}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          candidate_id: sessionId,
          question: currentQuestion,
          answer_text: currentAnswer,
        }),
      });

      const data = await res.json();

      if (!res.ok || data?.error) {
        setError(data?.error || "Ask-question failed");
        return;
      }

      const a = data?.analysis || data;

      setAnalysis({
        sentiment: a?.sentiment ?? "-",
        confidence_score: a?.confidence_score ?? "-",
        sentiment_confidence: a?.sentiment_confidence ?? "-",
      });
    } catch (e) {
      setError(String(e));
    } finally {
      setIsAnalyzing(false);
    }
  }

  async function showResults() {
    setError("");
    if (!sessionId) return alert("Start interview first");

    try {
      const res = await fetch(`/api${endpoints.results}/${sessionId}`);
      const data = await res.json();
      if (!res.ok || data?.error) {
        setError(data?.error || "Failed to fetch results");
        return;
      }
      setFinalResults(data);
    } catch (e) {
      setError(String(e));
    }
  }

  // ---------- Summary from DB records ----------
  const records = finalResults?.database_records;

  function computeSummaryTuples(rows) {
    if (!Array.isArray(rows) || rows.length === 0) return null;

    const sentiments = rows.map((r) => r?.[6]).filter(Boolean);
    const confs = rows.map((r) => Number(r?.[7])).filter((n) => !Number.isNaN(n));

    const counts = sentiments.reduce((acc, s) => {
      acc[s] = (acc[s] || 0) + 1;
      return acc;
    }, {});

    const avgConf = confs.length
      ? Math.round((confs.reduce((a, b) => a + b, 0) / confs.length) * 100) / 100
      : "-";

    const topSentiment =
      Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "-";

    return { counts, avgConf, topSentiment };
  }

  const summary = computeSummaryTuples(records);

  const progressText =
    questions.length > 0 && !question.includes("Interview completed")
      ? `Question ${Math.min(qIndex + 1, questions.length)} / ${questions.length}`
      : questions.length > 0
      ? `Completed ${questions.length} / ${questions.length}`
      : "-";

  return (
    <div style={page}>
      <div style={{ maxWidth: 1400, margin: "0 auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 10 }}>
          <h2 style={title}>AI Interview Bot</h2>
          <div style={{ textAlign: "right" }}>
            <div style={muted}><b>Session:</b> {sessionId || "-"}</div>
            <div style={muted}><b>Progress:</b> {progressText}</div>
          </div>
        </div>

        {/* Top controls */}
        <div style={card}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center" }}>
            <div>
              <div style={label}>Candidate Name</div>
              <input
                value={candidateName}
                onChange={(e) => setCandidateName(e.target.value)}
                style={input}
                placeholder="e.g. Rahul"
              />
            </div>

            <div>
              <div style={label}>Domain</div>
              <input
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                style={input}
                placeholder="e.g. AI/ML"
              />
            </div>

            <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
              <button
                onClick={startInterview}
                style={btn(false, "primary")}
                onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.99)")}
                onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                Start Interview
              </button>

              <button
                onClick={showResults}
                style={btn(!sessionId, "ghost")}
                disabled={!sessionId}
                onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.99)")}
                onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                Show Results
              </button>
            </div>
          </div>

          {error && (
            <div style={{ marginTop: 12, padding: 10, borderRadius: 10, background: "#2a0f0f", border: "1px solid #5a1a1a" }}>
              <b style={{ color: "#ff6b6b" }}>Error:</b> {error}
            </div>
          )}
        </div>

        <div style={{ height: 14 }} />

        {/* Two panels */}
        <div style={grid2}>
          {/* Left: Bot */}
          <div style={card}>
            <h3 style={{ marginTop: 0 }}>Interview Bot</h3>
            <div style={label}>Current Question</div>
            <div style={{ ...subCard, marginTop: 10, minHeight: 90, display: "flex", alignItems: "center" }}>
              <div style={{ fontSize: 16, lineHeight: 1.35 }}>
                {question || "Click Start Interview to begin."}
              </div>
            </div>

            <button
              onClick={submitAnswer}
              disabled={!sessionId || !question || question.includes("Interview completed")}
              style={btn(!sessionId || !question || question.includes("Interview completed"), "primary")}
              onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.99)")}
              onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            >
              Submit Answer → Next Question
            </button>

            <div style={{ marginTop: 10, ...muted }}>
              Tip: If mic is slow, you can type the answer in the box too.
            </div>
          </div>

          {/* Right: User */}
          <div style={card}>
            <h3 style={{ marginTop: 0 }}>You (Mic Answer)</h3>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <button
                onClick={startMic}
                disabled={isListening}
                style={btn(isListening, "primary")}
                onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.99)")}
                onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                {isListening ? "Listening..." : "Start Mic"}
              </button>

              <button
                onClick={stopMic}
                style={btn(false, "ghost")}
                onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.99)")}
                onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                Stop
              </button>
            </div>

            <div style={{ marginTop: 12 }}>
              <div style={label}>Your Answer</div>
              <textarea
                value={answerText}
                onChange={(e) => setAnswerText(e.target.value)}
                style={textarea}
                placeholder="Your speech will appear here (or you can type)."
              />
            </div>
          </div>
        </div>

        <div style={{ height: 14 }} />

        {/* Analysis */}
        <div style={card}>
          <h3 style={{ marginTop: 0 }}>Analysis</h3>

          {isAnalyzing ? (
            <div style={{ ...subCard }}>
              <b>Analyzing:</b> {lastAnalyzedQuestion} ...
            </div>
          ) : analysis ? (
            <div style={{ ...subCard }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 10 }}>
                <div>
                  <div style={label}>Sentiment / Emotion</div>
                  <div style={{ fontSize: 18, fontWeight: 700, marginTop: 6 }}>{analysis.sentiment}</div>
                </div>
                <div>
                  <div style={label}>Confidence Score</div>
                  <div style={{ fontSize: 18, fontWeight: 700, marginTop: 6 }}>{analysis.confidence_score}</div>
                </div>
                <div>
                  <div style={label}>Sentiment Confidence</div>
                  <div style={{ fontSize: 18, fontWeight: 700, marginTop: 6 }}>
                    {typeof analysis.sentiment_confidence === "number"
                      ? analysis.sentiment_confidence.toFixed(3)
                      : analysis.sentiment_confidence}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div style={muted}>-</div>
          )}
        </div>

        <div style={{ height: 14 }} />

        {/* Final results */}
        <div style={card}>
          <h3 style={{ marginTop: 0 }}>Final Results</h3>

          {!finalResults ? (
            <div style={muted}>-</div>
          ) : (
            <>
              {summary && (
                <div style={{ ...subCard, marginBottom: 12 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 10 }}>
                    <div>
                      <div style={label}>Overall Sentiment (most frequent)</div>
                      <div style={{ fontSize: 18, fontWeight: 700, marginTop: 6 }}>{summary.topSentiment}</div>
                    </div>
                    <div>
                      <div style={label}>Average Confidence Score</div>
                      <div style={{ fontSize: 18, fontWeight: 700, marginTop: 6 }}>{summary.avgConf}</div>
                    </div>
                    <div>
                      <div style={label}>Sentiment Counts</div>
                      <div style={{ marginTop: 6, fontFamily: "monospace", fontSize: 12, opacity: 0.9 }}>
                        {JSON.stringify(summary.counts)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div style={{ ...muted, marginBottom: 8 }}>
                Per-question results (answer text hidden):
              </div>

              <div style={tableWrap}>
                <table style={table}>
                  <thead>
                    <tr>
                      <th style={th}>Question</th>
                      <th style={th}>Sentiment</th>
                      <th style={th}>Sentiment Confidence</th>
                      <th style={th}>Confidence Score</th>
                      <th style={th}>Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Array.isArray(records) && records.length > 0 ? (
                      records.map((r) => (
                        <tr key={r[0]}>
                          <td style={td}>{r[3]}</td>
                          <td style={td}>{r[6]}</td>
                          <td style={td}>
                            {Number(r[5]).toFixed ? Number(r[5]).toFixed(3) : r[5]}
                          </td>
                          <td style={td}>{r[7]}</td>
                          <td style={td}>{r[8]}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td style={td} colSpan={5}>
                          No records found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>

              <hr style={hr} />

              <div style={muted}>
                Note: Emotion labels come from the model (joy, anger, sadness, etc.).
              </div>
            </>
          )}
        </div>

        <div style={{ height: 18 }} />
      </div>
    </div>
  );
}