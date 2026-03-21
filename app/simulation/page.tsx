"use client";

import { useMemo, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const TARGETS = [
  "NR-AR",
  "NR-AR-LBD",
  "NR-AhR",
  "NR-Aromatase",
  "NR-ER",
  "NR-ER-LBD",
  "NR-PPAR-gamma",
  "SR-ARE",
  "SR-ATAD5",
  "SR-HSE",
  "SR-MMP",
  "SR-p53",
];

// ⭐ NEW TYPES
interface FinalSummary {
  final_toxicity_score: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  highest_risk_target: string;
  highest_risk_value: number;
}

interface SimulationResponse {
  status: string;
  smiles: string;
  vqe_energy: number;
  exact_energy: number;
  delta_energy: number;
  toxicity_probabilities: Record<string, number>;
  confidence_score: number;
  final_summary?: FinalSummary;
  cached: boolean;
}

export default function SimulationPage() {
  const [smiles, setSmiles] = useState("CC(=O)Oc1ccccc1C(=O)O");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);

  const sortedProbabilities = useMemo(() => {
    if (!result) return [];

    return TARGETS.map((target) => ({
      target,
      probability: result.toxicity_probabilities[target] ?? 0,
    })).sort((a, b) => b.probability - a.probability);
  }, [result]);

  const finalSummary = useMemo<FinalSummary | null>(() => {
    if (!result) return null;

    if (result.final_summary) {
      return result.final_summary;
    }

    const entries = Object.entries(result.toxicity_probabilities || {});
    if (entries.length === 0) return null;

    const avg = entries.reduce((acc, [, p]) => acc + p, 0) / entries.length;
    const [maxTarget, maxProb] = entries.reduce(
      (best, cur) => (cur[1] > best[1] ? cur : best),
      entries[0]
    );

    let risk: "LOW" | "MEDIUM" | "HIGH" = "LOW";
    if (avg >= 0.6) risk = "HIGH";
    else if (avg >= 0.3) risk = "MEDIUM";

    return {
      final_toxicity_score: Number((avg * 100).toFixed(2)),
      risk_level: risk,
      highest_risk_target: maxTarget,
      highest_risk_value: Number((maxProb * 100).toFixed(2)),
    };
  }, [result]);

  // 🎨 COLORS
  const getRiskColor = (level: string) => {
    if (level === "LOW") return "text-green-500";
    if (level === "MEDIUM") return "text-yellow-500";
    return "text-red-500";
  };

  const getBarColor = (level: string) => {
    if (level === "LOW") return "from-green-500 to-emerald-400";
    if (level === "MEDIUM") return "from-yellow-500 to-orange-400";
    return "from-red-500 to-pink-500";
  };

  const handleSimulate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_URL}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      });

      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.detail || "Simulation request failed");
      }

      setResult(payload as SimulationResponse);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Server connection failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-6 md:p-24">
      <div className="z-10 max-w-6xl w-full">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-amber-500 to-red-500 bg-clip-text text-transparent">
          Quantum Toxicity Lab
        </h1>

        <p className="text-lg mb-10 max-w-3xl text-foreground/80 border-l-4 border-orange-500/50 pl-4">
          Simulate VQE-assisted molecular toxicity with a hybrid quantum-classical
          multi-label predictor using Tox21 targets.
        </p>

        {/* INPUT */}
        <div className="rounded-lg p-6 mb-8 bg-background border border-orange-500/30 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4 text-orange-500">🧪 SMILES Input</h2>

          <input
            className="w-full p-3 rounded-lg border border-orange-500/30"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
          />

          <button
            onClick={handleSimulate}
            disabled={loading}
            className="w-full mt-4 py-3 rounded-xl bg-orange-600 hover:bg-orange-500"
          >
            {loading ? "Simulating..." : "Run Simulation"}
          </button>
        </div>

        {/* ERROR */}
        {error && <div className="text-red-500 mb-4">{error}</div>}

        {/* RESULT */}
        {result && (
          <div className="space-y-6">

            {/* 🔥 FINAL TOXICITY (MAIN HIGHLIGHT) */}
            {finalSummary && (
              <div className="rounded-2xl p-8 border border-orange-500/40 shadow-lg bg-gradient-to-br from-black/40 to-orange-900/20">

                <h2 className="text-3xl font-bold mb-6 text-orange-400">
                  🔥 Final Toxicity Score
                </h2>

                <div className="text-center mb-6">
                  <div className="text-6xl font-bold text-white">
                    {finalSummary.final_toxicity_score}%
                  </div>

                  <div className={`mt-2 text-2xl font-semibold ${getRiskColor(finalSummary.risk_level)}`}>
                    {finalSummary.risk_level} RISK
                  </div>
                </div>

                {/* Gradient bar */}
                <div className="w-full h-5 bg-gray-800 rounded-full overflow-hidden mb-6">
                  <div
                    className="h-full transition-all duration-700"
                    style={{
                      width: `${finalSummary.final_toxicity_score}%`,
                      background: "linear-gradient(to right, green, yellow, red)"
                    }}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    Highest Risk: {finalSummary.highest_risk_target}
                  </div>
                  <div>
                    Value: {finalSummary.highest_risk_value}%
                  </div>
                </div>
              </div>
            )}
            {/* EXISTING RESULTS */}
            <div className="rounded-lg p-6 bg-background border border-orange-500/30 shadow-sm">
              <h2 className="text-xl font-semibold mb-4">📊 Energy Results</h2>

              <p>Exact Energy: {result.exact_energy.toFixed(6)}</p>
              <p>VQE Energy: {result.vqe_energy.toFixed(6)}</p>
              <p>Gap: {result.delta_energy.toFixed(6)}</p>
              <p>Confidence: {(result.confidence_score * 100).toFixed(1)}%</p>
            </div>

            {/* TOXICITY */}
            <div className="rounded-lg p-6 bg-background border border-orange-500/30 shadow-sm">
              <h2 className="text-xl font-semibold mb-4">🧬 Toxicity Targets</h2>

              {sortedProbabilities.map(({ target, probability }) => (
                <div key={target} className="mb-3">
                  <div className="flex justify-between">
                    <span>{target}</span>
                    <span>{(probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded">
                    <div
                      className="h-2 bg-orange-500 rounded"
                      style={{ width: `${probability * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

          </div>
        )}
      </div>
    </main>
  );
}