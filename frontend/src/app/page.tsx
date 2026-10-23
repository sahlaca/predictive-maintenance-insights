"use client";

import { type FormEvent, useEffect, useState } from "react";
import Image from "next/image";

type Metadata = {
  row_count: number;
  accuracy: number;
  classes: string[];
  class_counts: Record<string, number>;
  feature_ranges: Record<string, { min: number; max: number; median: number }>;
  non_failure_ranges: Record<string, { lower: number; upper: number }>;
  feature_importances: Array<{ feature: string; importance: number }>;
};

type Reading = {
  type: "L" | "M" | "H";
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
};

type Prediction = {
  predicted_category: string;
  predicted_status: string;
  scores: Array<{ category: string; score: number }>;
  score_note: string;
  outside_training_range: string[];
  range_notice: string | null;
  prototype_notice: string;
};

const featureFields: Array<{
  key: keyof Omit<Reading, "type">;
  label: string;
  unit: string;
  feature: string;
  step: number;
}> = [
  { key: "air_temperature", label: "Air temperature", unit: "K", feature: "Air temperature [K]", step: 0.1 },
  { key: "process_temperature", label: "Process temperature", unit: "K", feature: "Process temperature [K]", step: 0.1 },
  { key: "rotational_speed", label: "Rotational speed", unit: "rpm", feature: "Rotational speed [rpm]", step: 1 },
  { key: "torque", label: "Torque", unit: "Nm", feature: "Torque [Nm]", step: 0.1 },
  { key: "tool_wear", label: "Tool wear", unit: "min", feature: "Tool wear [min]", step: 1 },
];

const initialReading: Reading = {
  type: "M",
  air_temperature: 300,
  process_temperature: 310,
  rotational_speed: 1500,
  torque: 40,
  tool_wear: 100,
};

export default function Home() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [reading, setReading] = useState<Reading>(initialReading);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/metadata")
      .then(async (response) => {
        if (!response.ok) throw new Error(await response.text());
        return response.json() as Promise<Metadata>;
      })
      .then((nextMetadata) => {
        setMetadata(nextMetadata);
        setReading((current) => ({
          ...current,
          ...Object.fromEntries(
            featureFields.map(({ key, feature }) => [key, nextMetadata.feature_ranges[feature].median]),
          ),
        }));
      })
      .catch((reason: unknown) => {
        setError(reason instanceof Error ? reason.message : "Could not connect to the API.");
      });
  }, []);

  async function submitPrediction(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reading),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail ?? "Prediction request failed.");
      setPrediction(result as Prediction);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not connect to the API.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Aroha Predictive Maintenance Insights home"><span className="brand-logo"><Image src="/aroha_logo.jpeg" alt="Aroha" width={140} height={140} priority /></span><span className="brand-divider">/</span> PREDICTIVE MAINTENANCE INSIGHTS</a>
        <span className="prototype-tag"><i /> INTERACTIVE PROTOTYPE</span>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow">INDUSTRIAL ANALYTICS · MODEL EXPLORER</p>
          <h1>Keep your operations<br /><em>one step ahead.</em></h1>
          <p className="hero-description">Explore how machine operating conditions relate to failure categories. Adjust a reading and see what the prototype predicts.</p>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit orbit-three" />
          <div className="core"><span>PM</span></div>
          <span className="art-node node-a">T°</span><span className="art-node node-b">RPM</span><span className="art-node node-c">Nm</span>
          <div className="art-caption">SENSOR SIGNALS <b>→</b> INSIGHT</div>
        </div>
      </section>

      <section className="stat-grid" aria-label="Model overview">
        <article className="stat-card"><span className="stat-label">TOP MODEL FACTOR</span><strong className="top-factor">{metadata ? metadata.feature_importances[0]?.feature.replace("numeric__", "") ?? "—" : "—"}</strong><small>Most influential in this demo model</small></article>
        <article className="stat-card"><span className="stat-label">TRAINING RECORDS</span><strong>{metadata ? metadata.row_count.toLocaleString() : "—"}</strong><small>After prototype data cleanup</small></article>
        <article className="stat-card"><span className="stat-label">FAILURE CATEGORIES</span><strong>{metadata ? metadata.classes.length : "—"}</strong><small>Including the no-failure class</small></article>
        <article className="stat-card"><span className="stat-label">MODEL STATUS</span><strong className="status-live"><i /> READY</strong><small>Demo model loaded by API</small></article>
      </section>

      <section className="workspace-grid">
        <article className="panel prediction-panel">
          <div className="section-heading"><div><p className="eyebrow">LIVE SANDBOX</p><h2>Test a machine reading</h2></div><span className="step-chip">01 / INPUT</span></div>
          <p className="muted">Enter sensor values to classify an example operating point. The shown ranges describe the demo data; you can enter values outside them.</p>
          <form onSubmit={submitPrediction}>
            <label className="field-label" htmlFor="product-type">Product quality type</label>
            <div className="type-options" role="group" aria-label="Product quality type">
              {(["L", "M", "H"] as const).map((type) => (
                <button className={`type-option ${reading.type === type ? "selected" : ""}`} key={type} onClick={() => setReading({ ...reading, type })} type="button">
                  <b>{type}</b><span>{type === "L" ? "Low" : type === "M" ? "Medium" : "High"} quality</span>
                </button>
              ))}
            </div>
            <div className="form-fields">
              {featureFields.map(({ key, label, unit, feature, step }) => {
                const range = metadata?.feature_ranges[feature];
                return (
                  <label className="input-wrap" key={key}>
                    <span>{label}<small>{unit}</small></span>
                    <input
                      aria-label={`${label} in ${unit}`}
                      disabled={!metadata}
                      onChange={(event) => setReading({ ...reading, [key]: Number(event.target.value) })}
                      required
                      step={step}
                      type="number"
                      value={reading[key]}
                    />
                    {range && <small className="input-range">Demo data: {range.min}–{range.max} {unit}</small>}
                  </label>
                );
              })}
            </div>
            <button className="primary-button" disabled={!metadata || loading} type="submit">
              {loading ? "Analyzing reading…" : <>Analyze machine reading <span>↗</span></>}
            </button>
          </form>
          {error && <div className="error-message" role="alert">{error}</div>}
        </article>

        <article className="panel result-panel">
          <div className="section-heading"><div><p className="eyebrow">MODEL OUTPUT</p><h2>Prediction insight</h2></div><span className="step-chip">02 / RESULT</span></div>
          {!prediction ? (
            <div className="empty-result"><div className="empty-icon">⌁</div><h3>Ready when you are</h3><p>Submit a machine reading to see its predicted category and model scores.</p><div className="empty-line" /></div>
          ) : (
            <div className="result-content">
              <div className={`result-banner ${prediction.predicted_status === "possible_failure" ? "alert" : "safe"}`}>
                <span className="result-symbol">{prediction.predicted_status === "possible_failure" ? "!" : "✓"}</span>
                <div><small>PREDICTED CATEGORY</small><h3>{prediction.predicted_category}</h3><p>{prediction.predicted_status === "possible_failure" ? "Possible failure indicated" : "No failure detected by model"}</p></div>
              </div>
              <div className="score-heading"><b>Category scores</b><span>MODEL OUTPUT</span></div>
              {prediction.scores.map(({ category, score }) => (
                <div className="score-row" key={category}><div className="score-label"><span>{category}</span><b>{(score * 100).toFixed(1)}%</b></div><div className="score-track"><i style={{ width: `${score * 100}%` }} /></div></div>
              ))}
              <p className="score-note">ⓘ {prediction.score_note}</p>
              {prediction.range_notice && <div className="warning-note range-warning"><span>!</span><p>{prediction.range_notice}</p></div>}
              <div className="reading-summary"><b>Analyzed reading</b><span>{reading.type} type · {reading.rotational_speed} rpm · {reading.tool_wear} min tool wear</span></div>
            </div>
          )}
        </article>
      </section>

      <section className="operator-section">
        <div className="operator-heading"><div><p className="eyebrow">PRACTICAL TAKEAWAYS</p><h2>What the operating data suggests</h2><p>Plain-language patterns from the project analysis to support a maintenance conversation.</p></div><span className="step-chip">FOR DISCUSSION</span></div>
        <div className="takeaway-grid">
          <article className="takeaway-card"><span className="takeaway-number">01</span><div><h3>Watch load and speed together</h3><p>Higher torque paired with lower RPM appears with strain-related failures. Low torque with high RPM also appears in some failures; review operating context rather than treating one reading alone as a cause.</p></div><span className="takeaway-tags">TORQUE <b>×</b> RPM</span></article>
          <article className="takeaway-card"><span className="takeaway-number">02</span><div><h3>Consider tool wear under load</h3><p>Tool wear by itself is not a dependable warning in this analysis. Higher wear combined with higher torque is a pattern worth checking during inspection.</p></div><span className="takeaway-tags">TOOL WEAR <b>×</b> LOAD</span></article>
          <article className="takeaway-card"><span className="takeaway-number">03</span><div><h3>Use combined signals</h3><p>Temperature, speed, torque, and tool wear interact. Avoid interpreting any single threshold as a guaranteed safe/unsafe boundary.</p></div><span className="takeaway-tags">MULTI-SIGNAL REVIEW</span></article>
        </div>

        <div className="operator-columns">
          <article className="panel insight-panel">
            <div className="section-heading"><div><p className="eyebrow">MODEL-IDENTIFIED DRIVERS</p><h2>Factors affecting failures</h2></div></div>
            <p className="muted">Relative influence in this demo model—not a causal effect or a failure probability.</p>
            {metadata ? metadata.feature_importances.slice(0, 5).map(({ feature, importance }, index) => {
              const label = feature.replace("numeric__", "").replace("categorical__", "Product type ");
              return <div className="driver-row" key={feature}><div className="driver-label"><span><i>{String(index + 1).padStart(2, "0")}</i>{label}</span><b>{(importance * 100).toFixed(1)}%</b></div><div className="score-track"><i style={{ width: `${importance * 100}%` }} /></div></div>;
            }) : <p className="muted">Loading model factors…</p>}
          </article>

          <article className="panel ranges-panel">
            <div className="section-heading"><div><p className="eyebrow">REFERENCE ONLY</p><h2>Typical no-failure ranges</h2></div></div>
            <p className="muted">Middle 90% of measurements in the dataset’s no-failure records.</p>
            <div className="range-list">
              {metadata && featureFields.map(({ feature, label, unit }) => {
                const range = metadata.non_failure_ranges[feature];
                return <div className="range-item" key={feature}><span>{label}<small>{unit}</small></span><b>{range.lower.toFixed(1)} <i>to</i> {range.upper.toFixed(1)}</b></div>;
              })}
            </div>
            <div className="warning-note"><span>!</span><p>These describe historical examples, not manufacturer-approved safety limits. Confirm limits with equipment specifications and site procedures.</p></div>
          </article>
        </div>
      </section>

      <footer><span>AROHA INTELLIGENCE · PREDICTIVE MAINTENANCE POC</span><span>DEMO DATA ONLY <b>•</b> NOT FOR SAFETY-CRITICAL USE</span></footer>
    </main>
  );
}
