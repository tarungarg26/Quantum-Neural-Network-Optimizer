import React, { useState, useRef, useEffect } from 'react';
import { useTrainingEngine } from './hooks/useTrainingEngine';
import NeuralNetCanvas from './components/NeuralNetCanvas';
import MetricsPanel from './components/MetricsPanel';
import CircuitViz from './components/CircuitViz';
import TestPanel from './components/TestPanel';
import WeightHeatmap from './components/WeightHeatmap';
import BlochSpheres from './components/BlochSpheres';
import ComparisonPanel from './components/ComparisonPanel';

const SHOT_OPTIONS = [null, 32, 128, 512, 1024];

export default function App() {
  const engine = useTrainingEngine();
  const [activeTab, setActiveTab] = useState('network');
  const [showScanlines, setShowScanlines] = useState(true);

  const {
    epoch, isTraining, startTraining, stopTraining, reset,
    mode, setMode,
    shots, setShots,
    lr, setLr,
    pruneThreshold, setPruneThreshold,
    pruneEnabled, setPruneEnabled,
    qnnParams, mlpModel,
    qnnMetrics, mlpMetrics,
    liveCircuit,
    testInput,
    dataset,
    datasetKey, setDatasetKey,
    datasetMeta, allDatasets,
    pruneStats,
  } = engine;

  // Window controls (Electron or web fallback)
  const winCtrl = window.electronAPI ?? {
    minimize: () => {}, maximize: () => {}, close: () => window.close(),
  };

  return (
    <>
      {showScanlines && <div className="scanline-overlay" />}

      {/* ── Title Bar ─────────────────────────────────────────────────── */}
      <div className="titlebar">
        <div className="titlebar-logo">
          <div className="titlebar-logo-icon">⚛</div>
          <div>
            <div className="titlebar-title">QNNVIZ</div>
            <div className="titlebar-subtitle">Quantum Neural Network Visualizer</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className={`status-pill ${isTraining ? 'training' : 'idle'}`}>
            <div className="dot" />
            {isTraining ? `TRAINING — EPOCH ${epoch}` : 'IDLE'}
          </div>
          {(mode === 'quantum' || mode === 'both') && qnnMetrics.valAcc.length > 0 && (
            <div className="metric-badge quantum">
              <div className="dot" />
              QNN {(qnnMetrics.valAcc.at(-1) * 100).toFixed(1)}%
            </div>
          )}
          {(mode === 'classical' || mode === 'both') && mlpMetrics.valAcc.length > 0 && (
            <div className="metric-badge classical">
              <div className="dot" />
              MLP {(mlpMetrics.valAcc.at(-1) * 100).toFixed(1)}%
            </div>
          )}
        </div>

        <div className="titlebar-controls">
          <button className="titlebar-btn minimize" onClick={winCtrl.minimize} />
          <button className="titlebar-btn maximize" onClick={winCtrl.maximize} />
          <button className="titlebar-btn close" onClick={winCtrl.close} />
        </div>
      </div>

      {/* ── Main Layout ───────────────────────────────────────────────── */}
      <div className="app-layout">

        {/* ══ LEFT SIDEBAR ══════════════════════════════════════════════ */}
        <aside className="sidebar">

          {/* Dataset Selector */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Benchmark Dataset</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {allDatasets.map(d => {
                const isActive = datasetKey === d.id;
                return (
                  <button
                    key={d.id}
                    className={`btn ${isActive ? 'btn-primary' : 'btn-secondary'}`}
                    style={{
                      textAlign: 'left',
                      padding: '8px 10px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      fontSize: 11,
                      border: isActive ? '1px solid var(--accent-quantum)' : '1px solid rgba(255,255,255,0.08)',
                      background: isActive ? 'rgba(0, 212, 255, 0.15)' : 'rgba(255,255,255,0.03)',
                      transition: 'all 0.15s ease',
                    }}
                    onClick={() => setDatasetKey(d.id)}
                  >
                    <span style={{ fontSize: 16 }}>{d.icon}</span>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 600, color: isActive ? '#fff' : 'var(--text-muted)' }}>
                        {d.name}
                      </div>
                      <div style={{ fontSize: 9, opacity: 0.7, color: 'var(--text-muted)' }}>
                        {d.tag}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
            {datasetMeta && (
              <div style={{ marginTop: 8, fontSize: 10, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                {datasetMeta.description}
              </div>
            )}
          </div>

          {/* Mode Toggle */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Mode</div>
            <div className="mode-toggle">
              {['quantum', 'classical', 'both'].map(m => (
                <button
                  key={m}
                  className={`mode-btn ${m} ${mode === m ? 'active' : ''}`}
                  onClick={() => setMode(m)}
                >
                  {m === 'quantum' ? '⚛ Quantum' : m === 'classical' ? '🔢 Classic' : '⚡ Both'}
                </button>
              ))}
            </div>
          </div>

          {/* Training Controls */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Training</div>
            <div className="control-group">

              <div className="epoch-display">
                <div className="epoch-count">{epoch}</div>
                <div className="epoch-label">steps</div>
              </div>

              {!isTraining ? (
                <button className="btn btn-primary" onClick={startTraining}>
                  ▶&nbsp; Start Training
                </button>
              ) : (
                <button className="btn btn-danger" onClick={stopTraining}>
                  ⏹&nbsp; Stop
                </button>
              )}
              <button className="btn btn-secondary" onClick={() => reset(0)}>
                ↺&nbsp; Reset
              </button>
            </div>
          </div>

          {/* Hyperparameters */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Hyperparameters</div>
            <div className="control-group">
              <div className="control-row">
                <div className="control-label">
                  <span>Learning Rate</span>
                  <span className="control-value">{lr.toExponential(1)}</span>
                </div>
                <input
                  type="range" className="control-slider quantum"
                  min={-4} max={-1} step={0.1}
                  value={Math.log10(lr)}
                  onChange={e => setLr(10 ** +e.target.value)}
                  style={{ '--pct': `${((Math.log10(lr) + 4) / 3) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Shot Budget (Quantum) */}
          {(mode === 'quantum' || mode === 'both') && (
            <div className="sidebar-section">
              <div className="sidebar-section-title">Shot Budget</div>
              <div className="shot-selector">
                {SHOT_OPTIONS.map(s => (
                  <button
                    key={s ?? 'inf'}
                    className={`shot-btn ${shots === s ? 'active' : ''}`}
                    onClick={() => setShots(s)}
                  >
                    {s === null ? '∞' : s}
                  </button>
                ))}
              </div>
              <div style={{ marginTop: 8, fontSize: 9, color: 'var(--text-muted)' }}>
                {shots === null
                  ? 'Analytic (exact statevector)'
                  : `${shots} shots/circuit — adds shot noise σ²=(1−f²)/${shots}`}
              </div>
            </div>
          )}

          {/* Pruning */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Pruning</div>
            <div className="control-group">
              <div className="toggle-row">
                <span className="toggle-label">Enable Pruning</span>
                <label className="toggle">
                  <input type="checkbox" checked={pruneEnabled} onChange={e => setPruneEnabled(e.target.checked)} />
                  <span className="toggle-slider" />
                </label>
              </div>
              {pruneEnabled && (
                <>
                  <div className="control-row">
                    <div className="control-label">
                      <span>Threshold</span>
                      <span className="control-value">{pruneThreshold.toFixed(3)}</span>
                    </div>
                    <input
                      type="range" className="control-slider"
                      min={0.001} max={0.3} step={0.001}
                      value={pruneThreshold}
                      onChange={e => setPruneThreshold(+e.target.value)}
                    />
                  </div>
                  {mode !== 'quantum' && (
                    <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                      {(pruneStats.ratio * 100).toFixed(1)}% weights pruned
                      ({pruneStats.prunedCount}/{pruneStats.totalCount})
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Scanline toggle */}
          <div className="sidebar-section">
            <div className="sidebar-section-title">Display</div>
            <div className="toggle-row">
              <span className="toggle-label">Scanlines</span>
              <label className="toggle">
                <input type="checkbox" checked={showScanlines} onChange={e => setShowScanlines(e.target.checked)} />
                <span className="toggle-slider" />
              </label>
            </div>
          </div>

        </aside>

        {/* ══ MAIN CANVAS ═══════════════════════════════════════════════ */}
        <main className="main-canvas">
          <div className="canvas-header">
            <div className="canvas-tabs">
              {[
                { id: 'network', label: '🧠 Network' },
                { id: 'circuit', label: '⚛ Circuit' },
                { id: 'metrics', label: '📈 Metrics' },
                { id: 'landscape', label: '🗺 Landscape' },
              ].map(t => (
                <button
                  key={t.id}
                  className={`canvas-tab ${activeTab === t.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(t.id)}
                >
                  {t.label}
                </button>
              ))}
            </div>
            <div style={{ flex: 1 }} />
            <div className="info-pill">
              {datasetMeta?.icon} {datasetMeta?.name} &nbsp;·&nbsp; {dataset.train.length} train · {dataset.val.length} val · {dataset.test.length} test
            </div>
          </div>

          <div className="canvas-content">
            {activeTab === 'network' && (
              <NeuralNetCanvas
                mode={mode}
                qnnParams={qnnParams}
                mlpModel={mlpModel}
                liveCircuit={liveCircuit}
                pruneEnabled={pruneEnabled}
                pruneThreshold={pruneThreshold}
                isTraining={isTraining}
                epoch={epoch}
              />
            )}
            {activeTab === 'circuit' && (
              <CircuitViz
                qnnParams={qnnParams}
                liveCircuit={liveCircuit}
                shots={shots}
                mode={mode}
              />
            )}
            {activeTab === 'metrics' && (
              <MetricsPanel
                qnnMetrics={qnnMetrics}
                mlpMetrics={mlpMetrics}
                mode={mode}
                epoch={epoch}
              />
            )}
            {activeTab === 'landscape' && (
              <LossLandscape qnnParams={qnnParams} dataset={dataset} />
            )}
          </div>
        </main>

        {/* ══ RIGHT PANEL ═══════════════════════════════════════════════ */}
        <aside className="right-panel">

          {/* Comparison */}
          <div className="panel-section">
            <div className="panel-title">Live Comparison</div>
            <ComparisonPanel
              qnnMetrics={qnnMetrics}
              mlpMetrics={mlpMetrics}
              mode={mode}
            />
          </div>

          {/* Bloch Spheres */}
          {(mode === 'quantum' || mode === 'both') && liveCircuit && (
            <div className="panel-section">
              <div className="panel-title">Qubit States (Bloch)</div>
              <BlochSpheres blochVectors={liveCircuit.blochVectors} />
            </div>
          )}

          {/* State Probabilities */}
          {(mode === 'quantum' || mode === 'both') && liveCircuit && (
            <div className="panel-section">
              <div className="panel-title">Computational Basis Probs</div>
              <div className="prob-bars" style={{ maxHeight: 120, overflowY: 'auto' }}>
                {liveCircuit.probabilities.slice(0, 8).map((p, i) => (
                  <div key={i} className="prob-bar-row">
                    <div className="prob-bar-label">|{i.toString(2).padStart(4, '0')}⟩</div>
                    <div className="prob-bar-track">
                      <div className="prob-bar-fill" style={{ width: `${p * 100}%` }} />
                    </div>
                    <div className="prob-bar-value">{(p * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Weight Heatmap */}
          {(mode === 'classical' || mode === 'both') && (
            <div className="panel-section">
              <div className="panel-title">MLP Weights Heatmap</div>
              <WeightHeatmap
                model={mlpModel}
                pruneThreshold={pruneEnabled ? pruneThreshold : 0}
              />
            </div>
          )}

          {/* QNN Parameters */}
          {(mode === 'quantum' || mode === 'both') && (
            <div className="panel-section">
              <div className="panel-title">QNN θ Parameters</div>
              <div className="weight-grid">
                {qnnParams.theta.flat(2).map((angle, i) => {
                  const norm = (angle % (2 * Math.PI)) / (2 * Math.PI);
                  const h = norm * 240;
                  return (
                    <div
                      key={i}
                      className="weight-cell"
                      style={{ background: `hsl(${h},80%,55%)` }}
                      data-tooltip={`θ[${Math.floor(i/8)},${Math.floor((i%8)/2)},${i%2}] = ${angle.toFixed(3)}`}
                    />
                  );
                })}
                <div
                  className="weight-cell"
                  style={{ background: `hsl(${(Math.abs(qnnParams.bias) * 120).toFixed(0)},70%,55%)`, width: 30 }}
                  data-tooltip={`bias = ${qnnParams.bias.toFixed(4)}`}
                />
              </div>
            </div>
          )}

          {/* Test Input */}
          <div className="panel-section">
            <div className="panel-title">Test Custom Input</div>
            <TestPanel testInput={testInput} mode={mode} datasetMeta={datasetMeta} />
          </div>

        </aside>
      </div>
    </>
  );
}

// ─── Loss Landscape (simple 2D slice) ─────────────────────────────────────────
function LossLandscape({ qnnParams: _qnnParams, dataset: _dataset }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.offsetWidth;
    const H = canvas.height = canvas.offsetHeight;

    // Draw gradient placeholder
    const grd = ctx.createRadialGradient(W/2, H/2, 20, W/2, H/2, Math.min(W,H)/2);
    grd.addColorStop(0, 'rgba(0,212,255,0.5)');
    grd.addColorStop(0.4, 'rgba(155,89,255,0.3)');
    grd.addColorStop(1, 'rgba(7,13,26,0)');
    ctx.fillStyle = '#070d1a';
    ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = 'rgba(0,212,255,0.5)';
    ctx.font = '14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Loss Landscape — 2D Slice along θ₀, θ₁', W/2, H/2 - 10);
    ctx.fillStyle = 'rgba(138,163,200,0.5)';
    ctx.font = '11px Inter';
    ctx.fillText('Train more steps to see the trajectory', W/2, H/2 + 16);
  }, []);

  return (
    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
