/**
 * WeightHeatmap.jsx — MLP weight heatmap with pruning overlay
 */
import React from 'react';

function weightToColor(w, threshold, pruneEnabled) {
  if (pruneEnabled && Math.abs(w) < threshold) return 'rgba(255,255,255,0.05)';
  const v = Math.min(1, Math.abs(w) / 2);
  if (w > 0) return `rgba(0,${Math.floor(80 + 175 * v)},${Math.floor(200 + 55 * v)},${0.3 + v * 0.7})`;
  return `rgba(${Math.floor(150 + 105 * v)},${Math.floor(50 * (1 - v))},${Math.floor(50 * (1 - v))},${0.3 + v * 0.7})`;
}

export default function WeightHeatmap({ model, pruneThreshold }) {
  if (!model) return null;
  const pruneEnabled = pruneThreshold > 0;

  return (
    <div>
      {model.weights.map((W, li) => (
        <div key={li} style={{ marginBottom: 10 }}>
          <div style={{ fontSize: 9, color: 'var(--text-muted)', marginBottom: 4, letterSpacing: 1 }}>
            Layer {li + 1}: {W[0].length}→{W.length}
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {W.map((row, ri) => (
              <div key={ri} style={{ display: 'flex', gap: 1 }}>
                {row.map((w, ci) => (
                  <div
                    key={ci}
                    style={{
                      flex: 1,
                      height: 10,
                      borderRadius: 1,
                      background: weightToColor(w, pruneThreshold, pruneEnabled),
                      transition: 'background 0.3s',
                      cursor: 'pointer',
                    }}
                    title={`W[${ri},${ci}] = ${w.toFixed(4)}`}
                  />
                ))}
              </div>
            ))}
          </div>
          {/* Bias row */}
          <div style={{ display: 'flex', gap: 1, marginTop: 2, opacity: 0.7 }}>
            {model.biases[li].map((b, i) => (
              <div
                key={i}
                style={{
                  flex: 1, height: 5, borderRadius: 1,
                  background: weightToColor(b, pruneThreshold, pruneEnabled),
                }}
                title={`b[${i}] = ${b.toFixed(4)}`}
              />
            ))}
          </div>
        </div>
      ))}
      <div style={{ fontSize: 9, color: 'var(--text-muted)', marginTop: 6 }}>
        🔵 Positive weights &nbsp;|&nbsp; 🔴 Negative weights
        {pruneEnabled && <span style={{ color: 'var(--accent-purple)', marginLeft: 6 }}>⚠ Pruned: faded</span>}
      </div>
    </div>
  );
}
