/**
 * TestPanel.jsx — Live test input with dynamic QNN & MLP predictions
 * Automatically updates feature labels, default presets, and class names
 * based on the active dataset (Formula 1, Iris, Wine, Breast Cancer).
 */
import React, { useState, useCallback, useEffect } from 'react';

export default function TestPanel({ testInput, mode, datasetMeta }) {
  const meta = datasetMeta || {
    featureNames: ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'],
    classNames: { '-1': 'Class A (-1)', '1': 'Class B (+1)' },
    defaults: { 'Default': [1, 1, 1, 1] }
  };

  const getInitialInputs = useCallback(() => {
    if (meta.defaults && Object.values(meta.defaults).length > 0) {
      return [...Object.values(meta.defaults)[0]];
    }
    return [0, 0, 0, 0];
  }, [meta]);

  const [inputs, setInputs] = useState(getInitialInputs);
  const [result, setResult] = useState(null);

  // Sync inputs when dataset changes
  useEffect(() => {
    setInputs(getInitialInputs());
    setResult(null);
  }, [meta.id, getInitialInputs]);

  const handleTest = useCallback(() => {
    const res = testInput(inputs);
    setResult(res);
  }, [inputs, testInput]);

  const labelNeg = meta.classNames?.['-1'] || 'Class A (−1)';
  const labelPos = meta.classNames?.['1'] || 'Class B (+1)';

  return (
    <div>
      <div className="test-inputs">
        {inputs.map((v, i) => (
          <div key={i} className="test-input-row">
            <div className="test-input-label" title={meta.featureNames[i]}>
              {meta.featureNames[i] || `F${i+1}`}
            </div>
            <input
              type="number"
              className="test-input-field"
              value={v}
              step={0.01}
              onChange={e => {
                const next = [...inputs];
                next[i] = parseFloat(e.target.value) || 0;
                setInputs(next);
              }}
            />
          </div>
        ))}
      </div>

      {meta.defaults && (
        <div style={{ display: 'flex', gap: 4, marginTop: 10, flexWrap: 'wrap' }}>
          {Object.entries(meta.defaults).map(([name, vals]) => (
            <button
              key={name}
              className="btn btn-secondary"
              style={{ flex: 1, minWidth: '45%', padding: '5px 8px', fontSize: 10 }}
              onClick={() => setInputs([...vals])}
            >
              {name}
            </button>
          ))}
        </div>
      )}

      <button className="btn btn-primary" style={{ marginTop: 8 }} onClick={handleTest}>
        ⚡ Run Inference
      </button>

      {result && (
        <div>
          {(mode === 'quantum' || mode === 'both') && (
            <div className={`prediction-result ${result.quantum.prediction === 1 ? 'virginica' : 'versicolor'}`}
              style={{ marginTop: 10 }}>
              <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>
                ⚛ QNN → {result.quantum.prediction === 1 ? labelPos : labelNeg}
              </div>
              <div style={{ fontSize: 10, fontFamily: 'monospace', opacity: 0.8 }}>
                Score: {result.quantum.score?.toFixed(4)} &nbsp;|&nbsp; ⟨Z₀⟩: {result.quantum.analyticalExpectation?.toFixed(4)}
              </div>
            </div>
          )}
          {(mode === 'classical' || mode === 'both') && (
            <div className={`prediction-result ${result.classical.prediction === 1 ? 'virginica' : 'versicolor'}`}
              style={{ marginTop: 8 }}>
              <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>
                🔢 MLP → {result.classical.prediction === 1 ? labelPos : labelNeg}
              </div>
              <div style={{ fontSize: 10, fontFamily: 'monospace', opacity: 0.8 }}>
                Score: {result.classical.score?.toFixed(4)}
              </div>
            </div>
          )}

          {/* Confidence bars */}
          {result.quantum && (
            <div style={{ marginTop: 8 }}>
              <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 4 }}>
                Quantum confidence (|score|)
              </div>
              <div className="bar-track">
                <div className="bar-fill quantum"
                  style={{ width: `${Math.min(100, Math.abs(result.quantum.score ?? 0) * 50 + 50)}%` }} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
