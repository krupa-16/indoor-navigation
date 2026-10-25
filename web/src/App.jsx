
import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [detections, setDetections] = useState([]);
  const [localization, setLocalization] = useState(null);
  const [loading, setLoading] = useState(false);
  const [localizing, setLocalizing] = useState(false);
  const [error, setError] = useState("");
  async function detectObjects() {
    if (!file) {
      setError("Please select an indoor image first.");
      return;
    }
    setLoading(true);
    setError("");
    setDetections([]);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("http://localhost:8000/api/detect", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Detection failed.");
      setDetections(data.detections || []);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  }
  async function findLocation() {
    if (!file) {
      setError("Please select an indoor image first.");
      return;
    }
    setLocalizing(true);
    setError("");
    setLocalization(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("http://localhost:8000/api/localize", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Localization failed.");
      setLocalization(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLocalizing(false);
    }
  }
  function handleImage(event) {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setDetections([]);
    setLocalization(null);
    setError("");
  }
  return (
    <main className="container">
      <h1>Indoor Navigation</h1>
      <p>Computer Vision-Based Indoor Navigation</p>
      <section className="panel">
        <h2>Upload an Indoor Image</h2>
        <input type="file" accept="image/jpeg,image/png,image/webp" onChange={handleImage} />
        {preview && <img src={preview} alt="Uploaded indoor scene" className="preview" />}
        <div className="actions">
          <button onClick={detectObjects} disabled={!file || loading || localizing}>
            {loading ? "Detecting..." : "Detect Objects"}
          </button>
          <button onClick={findLocation} disabled={!file || loading || localizing}>
            {localizing ? "Finding Location..." : "Find Location"}
          </button>
        </div>
        <p className="hint">Upload a photo, then choose object detection or landmark localization.</p>
      </section>
      {error && <p className="error">{error}</p>}
      <section className="panel">
        <h2>Object Detection Results</h2>
        {detections.length === 0 && !loading && <p>No detection results yet.</p>}
        {detections.map((item, index) => (
          <div className="result" key={index}>
            <strong>{item.label}</strong>
            <span>{(item.confidence * 100).toFixed(1)}% confidence</span>
          </div>
        ))}
        {loading && <p>YOLO is analyzing the image...</p>}
      </section>
      <section className="panel">
        <h2>Landmark Localization Results</h2>
        {!localization && !localizing && <p>Upload an image and click Find Location.</p>}
        {localizing && <p>ORB is comparing visual features with your reference landmarks...</p>}
        {localization && (
          <>
            <h3>{localization.localized ? `Detected Landmark: ${localization.best_match}` : "Landmark Not Confirmed"}</h3>
            <p>{localization.message}</p>
            <div className="result">
              <strong>Feature matches</strong>
              <span>{localization.good_matches}</span>
            </div>
            <div className="result">
              <strong>Geometrically verified matches</strong>
              <span>{localization.verified_matches}</span>
            </div>
            <div className="result">
              <strong>Inlier ratio</strong>
              <span>{(localization.inlier_ratio * 100).toFixed(1)}%</span>
            </div>
            <h3>Top Reference Matches</h3>
            {(localization.matches || []).map((item, index) => (
              <div className="match-card" key={`${item.landmark}-${index}`}>
                <h4>{index + 1}. {item.landmark}</h4>
                <p>Feature matches: {item.good_matches}</p>
                <p>Verified matches: {item.verified_matches}</p>
                <p>Inlier ratio: {(item.inlier_ratio * 100).toFixed(1)}%</p>
                {item.match_image && (
                  <img
                    src={item.match_image}
                    alt={`Visual feature matches for ${item.landmark}`}
                    className="match-preview"
                  />
                )}
              </div>
            ))}
          </>
        )}
      </section>
    </main>
  );
}
export default App;
