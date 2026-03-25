"use client";

import { useMemo, useState, useEffect } from "react";

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

interface ExpertAnalysis {
  chemical_name: string;
  reason: string;
  health_issues: string[];
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
    weight: number;
  };
  expert_analysis?: ExpertAnalysis;
}

export default function SimulationPage() {
  const [smiles, setSmiles] = useState("Nc1ccccc1");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SimulationResponse | null>(null);

  // Panel visibility states
  const [showDetails, setShowDetails] = useState(false);
  const [showToxicityTargets, setShowToxicityTargets] = useState(false);

  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  // Reset panel states whenever a new simulation runs
  const handleSimulate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setShowDetails(false);
    setShowToxicityTargets(false);

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
    return null;
  }, [result]);

  const expertAnalysis = useMemo<ExpertAnalysis | null>(() => {
    if (!result?.expert_analysis) return null;
    return result.expert_analysis;
  }, [result]);

  const getRiskColor = (level: string) => {
    if (level === "LOW") return "text-green-500";
    if (level === "MEDIUM") return "text-yellow-500";
    return "text-red-500";
  };

  const getRiskBadgeStyle = (level: string) => {
    if (level === "LOW") return "bg-green-500/10 text-green-400 border border-green-500/30";
    if (level === "MEDIUM") return "bg-yellow-500/10 text-yellow-400 border border-yellow-500/30";
    return "bg-red-500/10 text-red-400 border border-red-500/30";
  };

  if (!mounted) return null;

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

        {/* ── SMILES Input ── */}
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

            {/* ── Row 1: Toxicity Score + Energy Results ── */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* Toxicity Score Card */}
              {finalSummary && (
                <div className="rounded-2xl p-6 border border-orange-500/40 bg-[#0a0a0a] transition-all duration-300 hover:shadow-[0_0_25px_rgba(251,146,60,0.15)] flex flex-col h-full">
                  <h2 className="text-xl font-bold mb-6 text-orange-400 flex items-center">
                    <span className="mr-2">🔥</span> Final Toxicity Score
                  </h2>

                  <div className="text-center mb-8">
                    <div className="text-6xl font-black text-white tracking-tighter">
                      {(result?.final_summary?.final_toxicity_score ?? 0).toFixed(2)}%
                    </div>
                    <div
                      className={`mt-2 text-sm font-bold ${getRiskColor(finalSummary.risk_level)} bg-white/5`}
                    >
                      {finalSummary.risk_level} RISK
                    </div>
                  </div>

                  <div
                    className="relative w-full rounded-full overflow-hidden bg-black border border-white/10"
                    style={{ height: "24px" }}
                  >
                    <div
                      className="h-full transition-all duration-1000 ease-out"
                      style={{
                        width: `${finalSummary.final_toxicity_score}%`,
                        background:
                          "linear-gradient(90deg, #22C55E 0%, #4ADE80 20%, #A3E635 40%, #FDE047 60%, #FB923C 80%, #EF4444 100%)",
                        backgroundSize: "100% 100%",
                        boxShadow: "inset 0 0 10px rgba(0,0,0,0.3)",
                      }}
                    />
                  </div>

                  {/* View Details Button — replaces highest target/value */}
                  <div className="mt-auto pt-8">
                    <button
                      onClick={() => {
                        setShowDetails((prev) => !prev);
                        if (showDetails) setShowToxicityTargets(false);
                      }}
                      className="w-full py-2.5 rounded-xl border border-orange-500/60 bg-orange-500/10 hover:bg-orange-500/20 text-orange-400 font-medium text-sm transition-all duration-200 flex items-center justify-center gap-2"
                    >
                      <span>{showDetails ? "Hide Details" : "View Details"}</span>
                      <span className="text-xs">{showDetails ? "▲" : "▼"}</span>
                    </button>
                  </div>
                </div>
              )}

              {/* Energy Results Card */}
              <div className="rounded-2xl p-6 border border-orange-500 bg-background">
                <h2 className="text-xl font-bold mb-4 text-orange-400">
                  📊 Energy Results
                </h2>

                {[
                  {
                    label: "Exact Energy",
                    value: `${result?.exact_energy?.toFixed(6) ?? "0.000000"} Ha`,
                    color: "text-blue-400",
                  },
                  
                  {
                    label: "VQE Energy",
                    value: `${result?.vqe_energy?.toFixed(6) ?? "0.000000"} Ha`,
                    color: "text-green-400",
                  },
                  {
                    label: "Gap",
                    value: `${result?.delta_energy?.toFixed(6) ?? "0.000000"} Ha`,
                    color: "text-amber-400",
                  },
                  {
                    label: "Confidence",
                    value: `${((result?.confidence_score ?? 0) * 100).toFixed(1)}%`,
                    color: "text-purple-400",
                  },
                ].map(({ label, value, color }) => (
                  <div
                    key={label}
                    className="flex justify-between items-center py-2 border-b border-orange-500/20 last:border-none"
                  >
                    <span className="text-sm text-gray-400">{label}</span>
                    <span className={`text-sm font-mono font-medium ${color}`}>{value}</span>
                  </div>
                ))}

                <div className="mt-4">
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-purple-500 transition-all duration-700"
                      style={{
                        width: `${((result?.confidence_score ?? 0) * 100).toFixed(1)}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* ── Expert Analysis Section (Refined UI) ── */}
            {showDetails && expertAnalysis && (
  <div className="rounded-2xl p-6 border border-orange-500/50 bg-[#0a0a0a] animate-fadeIn">

    {/* Header */}
    <div className="flex items-center justify-between mb-5">
      <h2 className="text-[16px] font-bold text-orange-400 flex items-center gap-2">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fb923c" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        Expert Analysis
      </h2>
      {finalSummary && (
        <span className={`text-[11px] font-bold tracking-widest px-4 py-1.5 rounded-full border ${getRiskBadgeStyle(finalSummary.risk_level)}`}>
          {finalSummary.risk_level} RISK
        </span>
      )}
    </div>

    {/* ── Table-like rows with perfectly aligned columns ── */}
    <div className="w-full" style={{ display: "table", borderCollapse: "collapse" }}>

      {/* Drug Name Row */}
      <div style={{ display: "table-row" }}>
        <div style={{ display: "table-cell", width: "110px", minWidth: "110px", verticalAlign: "top", paddingTop: "14px", paddingBottom: "14px", paddingRight: "16px", borderTop: "1px solid rgba(255,255,255,0.06)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <span className="text-[9px] font-black uppercase tracking-[0.18em] text-gray-500 leading-tight block">
            Drug<br/>Name
          </span>
        </div>
        <div style={{ display: "table-cell", verticalAlign: "top", paddingTop: "14px", paddingBottom: "14px", borderTop: "1px solid rgba(255,255,255,0.06)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <p className="text-[18px] font-bold text-white leading-tight mb-1">
            {expertAnalysis.chemical_name}
          </p>
          <span className="font-mono text-[11px] bg-white/5 border border-white/10 rounded px-2 py-0.5 text-gray-400">
            {smiles}
          </span>
        </div>
      </div>

      {/* Risk Reason Row */}
      <div style={{ display: "table-row" }}>
        <div style={{ display: "table-cell", width: "110px", minWidth: "110px", verticalAlign: "top", paddingTop: "14px", paddingBottom: "14px", paddingRight: "16px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <span className="text-[9px] font-black uppercase tracking-[0.18em] text-gray-500 leading-tight block">
            Risk<br/>Reason
          </span>
        </div>
        <div style={{ display: "table-cell", verticalAlign: "top", paddingTop: "25px", paddingBottom: "25px", marginLeft: "30px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <p className="text-[13px] text-gray-300 leading-[1.75]">
            {expertAnalysis.reason}
          </p>
        </div>
      </div>

      {/* Health Issues Row */}
      <div style={{ display: "table-row" }}>
        <div style={{ display: "table-cell", width: "110px", minWidth: "110px", verticalAlign: "top", paddingTop: "14px", paddingBottom: "14px", paddingRight: "16px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <span className="text-[9px] font-black uppercase tracking-[0.18em] text-gray-500 leading-tight block">
            Health<br/>Issues
          </span>
        </div>
        <div style={{ display: "table-cell", verticalAlign: "top", paddingTop: "14px", paddingBottom: "14px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <ul className="flex flex-col gap-2">
            {expertAnalysis.health_issues.map((issue, idx) => (
              <li key={idx} className="flex items-center gap-3">
                <span className="text-gray-500 font-bold text-[16px] leading-none select-none">|</span>
                <span className="text-[13px] text-gray-200 leading-relaxed">
                  {issue}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

    </div>

    {/* Show Toxicity Targets Button */}
    <div className="mt-5 pt-4 border-t border-white/5 hover:border-white/10 transition-colors duration-200">
      <button
        onClick={() => setShowToxicityTargets((prev) => !prev)}
        className="w-full py-3 rounded-xl border border-white/10 bg-transparent hover:bg-white/5 text-white font-medium text-[14px] transition-all duration-200 flex items-center justify-center gap-2"
      >
        {showToxicityTargets ? "Hide toxicity targets" : "Show toxicity targets"}
        <svg
          width="16" height="16" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          className={`transition-transform duration-200 ${showToxicityTargets ? "rotate-180" : ""}`}
        >
          <polyline points="6 9 12 15 18 9"/>
        </svg>
      </button>
    </div>

  </div>
)}
            {/* ── Row 2: Toxicity Targets (conditional) + Molecular Structure ── */}
            <div
              className={`grid gap-6 ${
                showToxicityTargets ? "grid-cols-1 md:grid-cols-2" : "grid-cols-1"
              }`}
            >
              {/* Toxicity Targets — only shown when toggled */}
              {showToxicityTargets && (
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
              )}

              {/* Molecular Structure — always visible */}
              <div
                className="rounded-2xl p-6 bg-[#0a0a0a] transition-all duration-300 flex flex-col items-center justify-center relative"
                style={{ border: "2px solid #f97316", minHeight: "300px" }}
              >
                <h2 className="text-xl font-bold mb-6 text-orange-500 w-full flex items-center">
                  <span className="mr-2">🔬</span> Molecular Structure
                </h2>
                <div className="bg-white/90 p-4 rounded-xl shadow-inner w-full flex items-center justify-center">
                  <img
                    src={`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/${encodeURIComponent(
                      smiles
                    )}/PNG`}
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
                  <div className="flex justify-between items-center py-2 border-b border-orange-500/20">
                    <span className="text-sm text-gray-400">Formula</span>
                    <span className="text-sm font-mono text-purple-400">
                      {result?.molecule_info?.formula || "-"}
                    </span>
                  </div>
                  <div className="flex justify-between items-center py-2">
                    <span className="text-sm text-gray-400">Weight</span>
                    <span className="text-sm font-mono text-rose-400">
                      {result?.molecule_info?.weight ? `${result.molecule_info.weight.toFixed(2)} g/mol` : "-"}
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