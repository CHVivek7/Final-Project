"use client";

import { useMemo, useState, useEffect } from "react"; // Added useEffect

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
  molecule_info?: {
    common_name: string;
    iupac_name: string;
    formula: string;
  };
}

export default function SimulationPage() {
  const [smiles, setSmiles] = useState("CC(=O)Oc1ccccc1C(=O)O");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);
  
  // 🔥 FIX: Added mounted state to prevent hydration mismatch
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  const sortedProbabilities = useMemo(() => {
    if (!result) return [];
    return TARGETS.map((target) => ({
      target,
      probability: result.toxicity_probabilities[target] ?? 0,
    })).sort((a, b) => b.probability - a.probability);
  }, [result]);

  const finalSummary = useMemo<FinalSummary | null>(() => {
    if (!result) return null;
    // Always prioritize the summary sent by the backend
    if (result.final_summary) return result.final_summary;
    return null;
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

  // 🔥 FIX: Return null or a loader until the component is mounted on the client
  if (!mounted) return null;

  return (
    <main className="flex min-h-screen flex-col items-center p-6 md:p-24">
      {/* Rest of your UI remains EXACTLY the same */}
      <div className="z-10 max-w-6xl w-full">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 bg-gradient-to-r from-amber-500 to-red-500 bg-clip-text text-transparent">
          Quantum Toxicity Lab
        </h1>

        <p className="text-lg mb-10 max-w-3xl text-foreground/80 border-l-4 border-orange-500/50 pl-4">
          Simulate VQE-assisted molecular toxicity with a hybrid quantum-classical
          multi-label predictor using Tox21 targets.
        </p>

        <div className="rounded-lg p-6 mb-8 bg-background border border-orange-500 shadow-sm">
          <h2 className="text-2xl font-semibold mb-4 text-orange-500">🧪 SMILES Input</h2>
          <input
            className="w-full p-3 rounded-lg border border-orange-500/50 bg-background text-white"
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

        {error && <div className="text-red-500 mb-4">{error}</div>}

        {result && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {finalSummary && (
                <div className="rounded-2xl p-6 border border-orange-500/40 bg-[#0a0a0a] transition-all duration-300 hover:shadow-[0_0_25px_rgba(251,146,60,0.15)] flex flex-col h-full">
                  <h2 className="text-xl font-bold mb-6 text-orange-400 flex items-center">
                    <span className="mr-2">🔥</span> Final Toxicity Score
                  </h2>

                  <div className="text-center mb-8">
                    <div className="text-6xl font-black text-white tracking-tighter">
                      {/* Displays the backend-calculated adjusted score (e.g., 32.45%) */}
                      {(result?.final_summary?.final_toxicity_score ?? 0).toFixed(2)}%
                    </div>
                    <div className={`mt-2 text-sm font-bold ${getRiskColor(finalSummary.risk_level)} border-current bg-white/5`}>
                      {finalSummary.risk_level} RISK
                    </div>
                  </div>

                  <div 
                    className="relative w-full rounded-full overflow-hidden bg-black border border-white/10"
                    style={{ height: '24px' }}
                  >
                    <div
                      className="h-full transition-all duration-1000 ease-out"
                      style={{
                        width: `${finalSummary.final_toxicity_score}%`,
                        background: `linear-gradient(90deg, #22C55E 0%, #4ADE80 20%, #A3E635 40%, #FDE047 60%, #FB923C 80%, #EF4444 100%)`,
                        backgroundSize: '100% 100%',
                        boxShadow: 'inset 0 0 10px rgba(0,0,0,0.3)'
                      }}
                    />
                  </div>

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

              <div className="rounded-2xl p-6 border border-orange-500 bg-background">
                <h2 className="text-xl font-bold mb-4 text-orange-400">
                  📊 Energy Results
                </h2>

                {[
                  { label: "Exact Energy", value: `${result?.exact_energy?.toFixed(6) ?? "0.000000"} Ha`, color: "text-blue-400" },
                  { label: "VQE Energy",   value: `${result?.vqe_energy?.toFixed(6) ?? "0.000000"} Ha`,   color: "text-green-400" },
                  { label: "Gap",          value: `${result?.delta_energy?.toFixed(6) ?? "0.000000"} Ha`,  color: "text-amber-400" },
                  { label: "Confidence",   value: `${(result?.confidence_score * 100).toFixed(1) ?? "0.0"}%`, color: "text-purple-400" },
                ].map(({ label, value, color }) => (
                  <div key={label} className="flex justify-between items-center py-2 border-b border-orange-500/20 last:border-none">
                    <span className="text-sm text-gray-400">{label}</span>
                    <span className={`text-sm font-mono font-medium ${color}`}>{value}</span>
                  </div>
                ))}

                <div className="mt-4">
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-purple-500 transition-all duration-700"
                      style={{ width: `${(result?.confidence_score * 100).toFixed(1) ?? "0.0"}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="rounded-lg p-6 bg-background border border-orange-500 shadow-sm">
                <h2 className="text-xl font-semibold mb-4 text-white">🧬 Toxicity Targets</h2>
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
              <div className="rounded-2xl p-6 bg-[#0a0a0a] transition-all duration-300 flex flex-col items-center justify-center relative"
                style={{ border: '2px solid #f97316', minHeight: '300px' }}>
                <h2 className="text-xl font-bold mb-6 text-orange-500 w-full flex items-center">
                  <span className="mr-2">🔬</span> Molecular Structure
                </h2>
                <div className="bg-white/90 p-4 rounded-xl shadow-inner w-full flex items-center justify-center">
                  <img 
                    src={`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(smiles)}/PNG`}
                    alt="Molecular Structure"
                    className="max-w-full h-auto"
                  />
                </div>
                <div className="mt-4 w-full">
                  <div className="flex justify-between items-center py-2 border-b border-orange-500/20">
                    <span className="text-sm text-gray-400">Common Name</span>
                    <span className="text-sm font-mono text-green-400">
                      {result?.molecule_info?.common_name || "Unknown"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-orange-500/20">
                    <span className="text-sm text-gray-400">IUPAC</span>
                    <span className="text-sm font-mono text-blue-400 text-right max-w-[200px] truncate">
                      {result?.molecule_info?.iupac_name || "Unknown"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-sm text-gray-400">Formula</span>
                    <span className="text-sm font-mono text-purple-400">
                      {result?.molecule_info?.formula || "-"}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}