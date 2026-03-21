'use client';

import { useEffect, useState } from 'react';
import axios from 'axios';

type RiskLevel = 'HIGH' | 'MEDIUM' | 'LOW';

interface ExampleMolecule {
  name: string;
  smiles: string;
}

interface ModelInfo {
  model_type: string;
  feature_names: string[];
  num_trained_molecules: number;
}

interface ToxicityResult {
  risk_level: RiskLevel;
  probability: number;
  smiles: string;
  quantum_features: Record<string, number>;
}

export default function ToxicityPage() {
  const [smiles, setSmiles] = useState('');
  const [result, setResult] = useState<ToxicityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [examples, setExamples] = useState<ExampleMolecule[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);

  // Load examples on mount
  useEffect(() => {
    axios
      .get('/api/toxicity/example_molecules')
      .then((res) => setExamples((res.data.examples ?? []) as ExampleMolecule[]))
      .catch(() => setExamples([]));

    axios
      .get('/api/toxicity/model_info')
      .then((res) => setModelInfo(res.data as ModelInfo))
      .catch(() => setModelInfo(null));
  }, []);

  const predictToxicity = async () => {
    if (!smiles) return;
    
    setLoading(true);
    try {
      const response = await axios.post('/api/predict_toxicity', { smiles });
      setResult(response.data as ToxicityResult);
    } catch (error) {
      console.error('Prediction failed:', error);
      alert('Prediction failed. Check SMILES string.');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (level: RiskLevel) => {
    switch(level) {
      case 'HIGH': return 'text-red-600 bg-red-100';
      case 'MEDIUM': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-green-600 bg-green-100';
    }
  };

  return (
    <div className="container mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">🧪 Quantum Drug Toxicity Predictor</h1>
      
      <div className="grid md:grid-cols-2 gap-6">
        {/* Left Column - Input */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Enter Molecule</h2>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              SMILES String
            </label>
            <input
              type="text"
              value={smiles}
              onChange={(e) => setSmiles(e.target.value)}
              placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O"
              className="w-full p-2 border rounded font-mono"
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Or select example:
            </label>
            <select
              onChange={(e) => setSmiles(e.target.value)}
              className="w-full p-2 border rounded"
              value=""
            >
              <option value="">Choose...</option>
              {examples.map((ex, i) => (
                <option key={i} value={ex.smiles}>
                  {ex.name}
                </option>
              ))}
            </select>
          </div>
          
          <button
            onClick={predictToxicity}
            disabled={loading || !smiles}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Running Quantum Simulation...' : 'Predict Toxicity'}
          </button>
          
          {modelInfo && (
            <div className="mt-4 p-3 bg-gray-50 rounded text-sm">
              <p><strong>Model:</strong> {modelInfo.model_type}</p>
              <p><strong>Quantum Features:</strong> {modelInfo.feature_names.join(', ')}</p>
              <p><strong>Trained on:</strong> {modelInfo.num_trained_molecules} molecules</p>
            </div>
          )}
        </div>
        
        {/* Right Column - Results */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">Results</h2>
          
          {result ? (
            <div>
              <div className={`p-4 rounded-lg mb-4 ${getRiskColor(result.risk_level)}`}>
                <p className="text-lg font-bold">
                  Risk Level: {result.risk_level}
                </p>
                <p className="text-2xl font-bold">
                  {result.probability.toFixed(1)}% Toxic
                </p>
              </div>
              
              <div className="mb-4">
                <p className="font-medium">SMILES:</p>
                <p className="font-mono text-sm bg-gray-100 p-2 rounded">
                  {result.smiles}
                </p>
              </div>
              
              <div>
                <p className="font-medium mb-2">Quantum Features from VQE:</p>
                <table className="w-full text-sm">
                  <tbody>
                    {Object.entries(result.quantum_features).map(([key, value]) => (
                      <tr key={key} className="border-b">
                        <td className="py-1 capitalize">{key.replace('_', ' ')}:</td>
                        <td className="py-1 font-mono text-right">{Number(value).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              Enter a SMILES string and click predict
            </div>
          )}
        </div>
      </div>
    </div>
  );
}