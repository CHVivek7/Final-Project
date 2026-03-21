"use client";

import { useMemo, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const TARGETS = [
  "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
  "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
  "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
];

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
    if (result.final_summary) return result.final_summary;

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

  const getRiskColor = (level: string) => {
    if (level === "LOW") return "text-green-500";
    if (level === "MEDIUM") return "text-yellow-500";
    return "text-red-500";
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
      if (!res.ok) throw new Error(payload.detail || "Simulation request failed");
      setResult(payload as SimulationResponse);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Server connection failed");
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
        <div className="rounded-lg p-6 mb-8 bg-background border border-orange-500 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4 text-orange-500">🧪 SMILES Input</h2>
          <input
            className="w-full p-3 rounded-lg border border-orange-500/50 bg-background"
            value={smiles}
            onChange={(e) => setSmiles(e.target.value)}
          />
          <button
            onClick={handleSimulate}
            disabled={loading}
            className="w-full mt-4 py-3 rounded-xl bg-orange-600 hover:bg-orange-500 text-white font-medium"
          >
            {loading ? "Simulating..." : "Run Simulation"}
          </button>
        </div>

        {/* ERROR */}
        {error && <div className="text-red-500 mb-4">{error}</div>}

        {/* RESULTS */}
        {result && (
          <div className="space-y-6">

            {/* FINAL TOXICITY + ENERGY — side by side, half width each */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* 🔥 FINAL TOXICITY */}
              {finalSummary && (
                <div className="rounded-2xl p-6 border border-orange-500/40 bg-[#0a0a0a] transition-all duration-300 hover:shadow-[0_0_25px_rgba(251,146,60,0.15)] flex flex-col h-full">
                  <h2 className="text-xl font-bold mb-6 text-orange-400 flex items-center">
                    <span className="mr-2">🔥</span> Final Toxicity Score
                  </h2>

                  <div className="text-center mb-8">
                    <div className="text-6xl font-black text-white tracking-tighter">
                      {finalSummary.final_toxicity_score}%
                    </div>
                    <div className={`mt-2 text-sm font-bold ${getRiskColor(finalSummary.risk_level)} border-current bg-white/5`}>
                      {finalSummary.risk_level} RISK
                    </div>
                  </div>

                  {/* THE FIX: Forced Height and Gradient */}
                  <div 
                    className="relative w-full rounded-full overflow-hidden bg-black border border-white/10"
                    style={{ height: '24px' }} // Forced thick height
                  >
                    <div
                      className="h-full transition-all duration-1000 ease-out"
                      style={{
                        width: `${finalSummary.final_toxicity_score}%`,
                        background: `linear-gradient(90deg, 
                          #22C55E 0%, 
                          #4ADE80 20%, 
                          #A3E635 40%, 
                          #FDE047 60%, 
                          #FB923C 80%, 
                          #EF4444 100%)`,
                        backgroundSize: '100% 100%',
                        boxShadow: 'inset 0 0 10px rgba(0,0,0,0.3)'
                      }}
                    />
                  </div>

                  {/* FOOTER METADATA */}
                  <div className="grid grid-cols-2 gap-4 mt-auto pt-8">
                    <div className="flex flex-col">
                      <span className="text-gray-500 uppercase text-[10px] font-black tracking-widest mb-1">Highest Target</span>
                      <span className="text-white font-bold truncate">{finalSummary.highest_risk_target}</span>
                    </div>
                    <div className="flex flex-col items-end">
                      <span className="text-gray-500 uppercase text-[10px] font-black tracking-widest mb-1">Target Value</span>
                      <span className="text-amber-500 font-mono font-black text-lg">{finalSummary.highest_risk_value}%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* 📊 ENERGY RESULTS */}
              <div className="rounded-2xl p-6 border border-orange-500 bg-background">
                <h2 className="text-xl font-bold mb-4 text-orange-400">
                  📊 Energy Results
                </h2>

                {[
                  { label: "Exact Energy", value: `${result.exact_energy.toFixed(6)} Ha`, color: "text-blue-400" },
                  { label: "VQE Energy",   value: `${result.vqe_energy.toFixed(6)} Ha`,   color: "text-green-400" },
                  { label: "Gap",          value: `${result.delta_energy.toFixed(6)} Ha`,  color: "text-amber-400" },
                  { label: "Confidence",   value: `${(result.confidence_score * 100).toFixed(1)}%`, color: "text-purple-400" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex justify-between items-center py-2 border-b border-orange-500/20 last:border-none">
                    <span className="text-sm text-gray-400">{label}</span>
                    <span className={`text-sm font-mono font-medium ${color}`}>{value}</span>
                  </div>
                ))}

                {/* Confidence bar */}
                <div className="mt-4">
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-purple-500 transition-all duration-700"
                      style={{ width: `${(result.confidence_score * 100).toFixed(1)}%` }}
                    />
                  </div>
                </div>
              </div>

            </div>

            {/* 🧬 TOXICITY TARGETS */}
            {/* 🧬 TOXICITY TARGETS — half width */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="rounded-lg p-6 bg-background border border-orange-500 shadow-sm">
                <h2 className="text-xl font-semibold mb-4">🧬 Toxicity Targets</h2>

                {sortedProbabilities.map(({ target, probability }) => (
                  <div key={target} className="mb-3">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-300">{target}</span>
                      <span className="text-gray-300">{(probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-2 rounded-full bg-orange-500 transition-all duration-500"
                        style={{ width: `${probability * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}