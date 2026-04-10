"use client";

import { useEffect, useState } from "react";

type PredictionItem = {
  label: string;
  probability: number;
};

type PredictResponse = {
  filename: string | null;
  top_class: string;
  top_probability: number;
  predictions: PredictionItem[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const CONFIDENCE_THRESHOLD = 0.5;
const FRIENDLY_LABELS: Record<string, string> = {
  airplane: "Airplane",
  automobile: "Car",
  bird: "Bird",
  cat: "Cat",
  deer: "Deer",
  dog: "Dog",
  frog: "Frog",
  horse: "Horse",
  ship: "Ship",
  truck: "Truck",
};

function formatProbability(value: number) {
  return `${(value * 100).toFixed(value >= 0.1 ? 1 : 2)}%`;
}

function friendlyLabel(label: string) {
  return FRIENDLY_LABELS[label] ?? label.replaceAll("_", " ");
}

function confidenceSummary(value: number) {
  if (value >= 0.95) return "Very sure";
  if (value >= 0.8) return "Sure";
  if (value >= CONFIDENCE_THRESHOLD) return "Fairly sure";
  return "Not fully sure";
}

async function fetchPrediction(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: string }
      | null;
    throw new Error(payload?.detail ?? "Prediction request failed.");
  }

  return (await response.json()) as PredictResponse;
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const showAlternatives = Boolean(result && result.top_probability < CONFIDENCE_THRESHOLD);
  const alternativePredictions =
    result && result.top_probability < CONFIDENCE_THRESHOLD ? result.predictions.slice(1, 4) : [];

  async function runPrediction(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await fetchPrediction(file);
      setResult(data);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  function handleFile(file: File | null) {
    if (!file) {
      setSelectedFile(null);
      setResult(null);
      setError(null);
      return;
    }

    setSelectedFile(file);
    void runPrediction(file);
  }

  function clearSelection() {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setLoading(false);
  }

  return (
    <main className="page">
      <div className="ambient ambientOne" />
      <div className="ambient ambientTwo" />
      <div className="ambient ambientThree" />

      <div className="container">
        <section className="hero">
          <div className="eyebrow">Photo checker</div>
          <h1 className="headline">
            What is in this photo?
          </h1>
          <p className="lede">
            Upload a clear picture and get a simple answer in seconds. If the image is fuzzy or
            busy, we’ll show a few extra ideas instead of guessing too hard.
          </p>

          <div className="chips">
            <span className="chip">One clear answer</span>
            <span className="chip">More ideas when unsure</span>
            <span className="chip">Looks great on mobile</span>
          </div>
        </section>

        <section className="grid">
          <article className="panel">
            <div
              className={`dropzone ${isDragging ? "dragging" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(event) => {
                event.preventDefault();
                setIsDragging(false);
                const droppedFile = event.dataTransfer.files?.[0] ?? null;
                handleFile(droppedFile);
              }}
            >
              <div className="previewFrame">
                {previewUrl ? (
                  <img src={previewUrl} alt="Selected preview" className="previewImage" />
                ) : (
                  <div className="placeholder">
                    <div className="placeholderMark">Upload</div>
                    <p>Drag an image here or choose one from your phone or computer.</p>
                  </div>
                )}
              </div>

              <div className="actions">
                <label className="fileButton">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(event) => handleFile(event.target.files?.[0] ?? null)}
                  />
                  Choose photo
                </label>

                <button
                  type="button"
                  className="primaryButton"
                  onClick={() => {
                    if (selectedFile) {
                      void runPrediction(selectedFile);
                    }
                  }}
                  disabled={!selectedFile || loading}
                >
                  {loading ? "Checking photo..." : "Check photo"}
                </button>

                {selectedFile ? (
                  <button type="button" className="secondaryButton" onClick={clearSelection}>
                    Clear
                  </button>
                ) : null}
              </div>

              <div className="helperCard">
                <div className="helperCardTitle">Best results</div>
                <p className="helperCardText">
                  Use a clear photo of one main subject. Bright, simple images work best.
                </p>
              </div>
            </div>
          </article>

          <article className="panel">
            <div className="resultHeader">
              <div>
                <div className="sectionLabel">Result</div>
                <h2 className="sectionTitle">Best guess</h2>
              </div>
              <div
                className={`statusPill ${
                  loading ? "busy" : error ? "error" : result ? "ready" : ""
                }`}
              >
                {loading ? "Checking" : error ? "Error" : result ? "Done" : "Waiting"}
              </div>
            </div>

            {error ? (
              <div className="errorBox">{error}</div>
            ) : loading ? (
              <div className="emptyState">
                <div className="emptyMark">Checking</div>
                <p>We are looking at your photo now.</p>
              </div>
            ) : result ? (
              <>
                <div className="highlight">
                  <div className="highlightLabel">Best guess</div>
                  <div className="highlightValue">{friendlyLabel(result.top_class)}</div>
                  <div className="highlightScore">
                    {confidenceSummary(result.top_probability)} ·{" "}
                    {formatProbability(result.top_probability)} sure
                  </div>
                  <div className="confidenceTrack" aria-hidden="true">
                    <div
                      className="confidenceFill"
                      style={{ width: `${Math.max(result.top_probability * 100, 4)}%` }}
                    />
                  </div>
                  <p className="resultNote">
                    {showAlternatives
                      ? "The model is not fully sure, so we are showing a few other ideas below."
                      : "The model feels confident, so the extra ideas stay hidden."}
                  </p>
                </div>

                {showAlternatives ? (
                  <div className="predictionList">
                    <div className="suggestionsHeader">Other ideas</div>
                    {alternativePredictions.map((item) => (
                      <div key={item.label} className="predictionRow">
                        <div className="predictionMeta">
                          <span className="predictionLabel">{friendlyLabel(item.label)}</span>
                          <span className="predictionValue">{formatProbability(item.probability)}</span>
                        </div>
                        <div className="barTrack" aria-hidden="true">
                          <div
                            className="barFill"
                            style={{ width: `${Math.max(item.probability * 100, 2)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="successBox">
                    <div className="successTitle">Looks clear enough</div>
                    <p className="successText">
                      Try another photo anytime if you want to compare results.
                    </p>
                  </div>
                )}
              </>
            ) : (
              <div className="emptyState">
                <div className="emptyMark">Ready</div>
                <p>Choose a photo to see the model's best guess.</p>
              </div>
            )}
          </article>
        </section>

        <footer className="footerRow">
          <span>Best results come from clear, well-lit photos with one main subject.</span>
          <span>{selectedFile ? "Photo loaded" : "Ready when you are"}</span>
        </footer>
      </div>
    </main>
  );
}
