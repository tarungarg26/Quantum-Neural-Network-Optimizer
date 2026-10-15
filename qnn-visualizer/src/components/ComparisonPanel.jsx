/**
 * ComparisonPanel.jsx — Side-by-side QNN vs MLP metric comparison
 */
import React from 'react';

function Winner({ q, c }) {
  if (!q || !c) return <span className="comparison-winner tie">NO DATA</span>;
  if (Math.abs(q - c) < 0.005) return <span className="comparison-winner tie">TIE</span>;
  return q > c
    ? <span className="comparison-winner quantum">QNN ⚛</span>
    : <span className="comparison-winner classical">MLP 🔢</span>;
}

export default function ComparisonPanel({ qnnMetrics, mlpMetrics, mode }) {
  const qAcc = qnnMetrics.valAcc.at(-1);
  const cAcc = mlpMetrics.valAcc.at(-1);
  const qLoss = qnnMetrics.valLoss.at(-1);
  const cLoss = mlpMetrics.valLoss.at(-1);

  const showQ = mode === 'quantum' || mode === 'both';
  const showC = mode === 'classical' || mode === 'both';

  const fmt = (v) => v !== undefined ? (v * 100).toFixed(1) + '%' : '—';
  const fmtL = (v) => v !== undefined ? v.toFixed(4) : '—';

  return (
    <div>
      {/* Metric cards */}
      <div className="metric-cards" style={{ marginBottom: 12 }}>
        {showQ && (
          <div className="metric-card quantum">
            <div className="metric-card-label">QNN Acc</div>
            <div className="metric-card-value">{fmt(qAcc)}</div>
            <div className="metric-card-sub">Val accuracy</div>
          </div>
        )}
        {showC && (
          <div className="metric-card classical">
            <div className="metric-card-label">MLP Acc</div>
            <div className="metric-card-value">{fmt(cAcc)}</div>
            <div className="metric-card-sub">Val accuracy</div>
          </div>
        )}
        {showQ && (
          <div className="metric-card quantum">
            <div className="metric-card-label">QNN Loss</div>
            <div className="metric-card-value" style={{ fontSize: 16 }}>{fmtL(qLoss)}</div>
            <div className="metric-card-sub">MSE (val)</div>
          </div>
        )}
        {showC && (
          <div className="metric-card classical">
            <div className="metric-card-label">MLP Loss</div>
            <div className="metric-card-value" style={{ fontSize: 16 }}>{fmtL(cLoss)}</div>
            <div className="metric-card-sub">MSE (val)</div>
          </div>
        )}
      </div>

      {/* Accuracy comparison bar */}
      {showQ && showC && qAcc !== undefined && cAcc !== undefined && (
        <div className="comparison-bar">
          <div className="comparison-header">
            <span className="comparison-label">Accuracy</span>
            <Winner q={qAcc} c={cAcc} />
          </div>
          <div className="bar-track">
            <div className="bar-fill quantum" style={{ width: `${qAcc * 100}%` }} />
          </div>
          <div className="bar-labels">
            <span>QNN {fmt(qAcc)}</span>
            <span>{fmt(cAcc)} MLP</span>
          </div>
          <div className="bar-track" style={{ marginTop: 4 }}>
            <div className="bar-fill classical" style={{ width: `${cAcc * 100}%` }} />
          </div>
        </div>
      )}

      {/* Loss comparison */}
      {showQ && showC && qLoss !== undefined && cLoss !== undefined && (
        <div className="comparison-bar">
          <div className="comparison-header">
            <span className="comparison-label">Val MSE Loss</span>
            <Winner q={1 - qLoss} c={1 - cLoss} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            <div>
              <div style={{ fontSize: 9, color: 'var(--quantum-blue)', marginBottom: 2 }}>QNN</div>
              <div className="bar-track">
                <div className="bar-fill quantum" style={{ width: `${Math.min(100, qLoss * 100)}%` }} />
              </div>
              <div style={{ fontSize: 9, fontFamily: 'monospace', color: 'var(--text-muted)', marginTop: 2 }}>{fmtL(qLoss)}</div>
            </div>
            <div>
              <div style={{ fontSize: 9, color: 'var(--classical-orange)', marginBottom: 2 }}>MLP</div>
              <div className="bar-track">
                <div className="bar-fill classical" style={{ width: `${Math.min(100, cLoss * 100)}%` }} />
              </div>
              <div style={{ fontSize: 9, fontFamily: 'monospace', color: 'var(--text-muted)', marginTop: 2 }}>{fmtL(cLoss)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Which is better summary */}
      {showQ && showC && (
        <div style={{
          marginTop: 8, padding: '10px 12px', borderRadius: 10,
          background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)',
          fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.8,
        }}>
          <strong style={{ color: 'var(--text-primary)' }}>Note:</strong> Iris binary is a near-linearly-separable task.
          The MLP typically converges faster due to dense layer architecture.
          The QNN uses fewer circuit evaluations but incurs higher shot-noise variance at finite shots.
          <span style={{ display: 'block', marginTop: 4, color: 'var(--text-muted)', fontSize: 9 }}>
            No quantum advantage claims — simulation only.
          </span>
        </div>
      )}
    </div>
  );
}
