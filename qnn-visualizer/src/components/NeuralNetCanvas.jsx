/**
 * NeuralNetCanvas.jsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Renders both QNN and Classical MLP networks side by side using HTML5 Canvas.
 * Shows:
 *  - Node activations (color-coded by value)
 *  - Edge weights (thickness + color by magnitude)
 *  - Quantum superposition rings on QNN nodes
 *  - Pruned weights (ghosted)
 *  - Live animation on training
 *  - Hover tooltips with weight/bias values
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';

const COLORS = {
  quantum: '#00d4ff',
  classical: '#ff6b35',
  bg: '#070d1a',
  node_bg: '#0f1830',
  positive: '#00d4ff',
  negative: '#ff4757',
  zero: '#4a5a7a',
  glow_q: 'rgba(0,212,255,0.3)',
  glow_c: 'rgba(255,107,53,0.3)',
};

function lerpColor(a, b, t) {
  const ah = parseInt(a.slice(1), 16);
  const bh = parseInt(b.slice(1), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const rr = Math.round(ar + (br - ar) * t);
  const rg = Math.round(ag + (bg - ag) * t);
  const rb = Math.round(ab + (bb - ab) * t);
  return `#${rr.toString(16).padStart(2,'0')}${rg.toString(16).padStart(2,'0')}${rb.toString(16).padStart(2,'0')}`;
}

function valueToColor(v) {
  // v in [-1,1]: negative=red, zero=gray, positive=cyan
  if (v > 0) return lerpColor('#4a5a7a', '#00d4ff', Math.min(1, v));
  return lerpColor('#4a5a7a', '#ff4757', Math.min(1, -v));
}

export default function NeuralNetCanvas({
  mode, qnnParams, mlpModel, liveCircuit,
  pruneEnabled, pruneThreshold, isTraining, epoch,
}) {
  const canvasRef = useRef(null);
  const frameRef = useRef(0);
  const [tooltip, setTooltip] = useState(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.02)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    const t = (frameRef.current / 60);
    frameRef.current++;

    const showQuantum = mode === 'quantum' || mode === 'both';
    const showClassical = mode === 'classical' || mode === 'both';

    if (mode === 'both') {
      drawQNN(ctx, qnnParams, liveCircuit, W * 0.05, H * 0.08, W * 0.42, H * 0.84, t, isTraining, pruneEnabled, pruneThreshold);
      drawMLP(ctx, mlpModel, W * 0.55, H * 0.08, W * 0.42, H * 0.84, t, isTraining, pruneEnabled, pruneThreshold);
      // Labels
      ctx.font = 'bold 11px Orbitron, monospace';
      ctx.fillStyle = 'rgba(0,212,255,0.7)';
      ctx.textAlign = 'center';
      ctx.fillText('QUANTUM CIRCUIT (QNN)', W * 0.26, H * 0.96);
      ctx.fillStyle = 'rgba(255,107,53,0.7)';
      ctx.fillText('CLASSICAL MLP', W * 0.76, H * 0.96);
    } else if (showQuantum) {
      drawQNN(ctx, qnnParams, liveCircuit, W * 0.05, H * 0.06, W * 0.9, H * 0.88, t, isTraining, pruneEnabled, pruneThreshold);
    } else if (showClassical) {
      drawMLP(ctx, mlpModel, W * 0.05, H * 0.06, W * 0.9, H * 0.88, t, isTraining, pruneEnabled, pruneThreshold);
    }
  }, [mode, qnnParams, mlpModel, liveCircuit, isTraining, pruneEnabled, pruneThreshold]);

  useEffect(() => {
    let animId;
    const loop = () => {
      draw();
      if (isTraining) {
        animId = requestAnimationFrame(loop);
      }
    };
    loop();
    return () => { if (animId) cancelAnimationFrame(animId); };
  }, [draw, isTraining]);

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          setTooltip({ x: e.clientX, y: e.clientY, info: `(${x.toFixed(0)}, ${y.toFixed(0)})` });
        }}
        onMouseLeave={() => setTooltip(null)}
      />
      {tooltip && (
        <div
          className="node-tooltip"
          style={{ left: tooltip.x + 12, top: tooltip.y - 30 }}
        >
          <div className="node-tooltip-title">NETWORK INFO</div>
          <div className="node-tooltip-row">
            <span>Epoch</span>
            <span className="node-tooltip-val mono">{epoch}</span>
          </div>
          {liveCircuit && (
            <div className="node-tooltip-row">
              <span>⟨Z₀⟩</span>
              <span className="node-tooltip-val mono">{liveCircuit.analyticalExpectation?.toFixed(4) ?? '—'}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Draw QNN ─────────────────────────────────────────────────────────────────

function drawQNN(ctx, qnnParams, liveCircuit, x0, y0, W, H, t, isTraining, pruneEnabled, pruneThreshold) {
  const layers = [
    { label: 'Input', nodes: 4, type: 'input' },
    { label: 'Encoding\nRY(x̃)', nodes: 4, type: 'encoding' },
    { label: 'Layer 1\nRY+RZ', nodes: 4, type: 'vqc' },
    { label: 'Layer 2\nRY+RZ', nodes: 4, type: 'vqc' },
    { label: 'Readout\n⟨Z₀⟩', nodes: 1, type: 'readout' },
  ];

  const xPositions = layers.map((_, i) => x0 + (i / (layers.length - 1)) * W);
  const nodePositions = [];

  // Get quantum activations
  const qActiv = liveCircuit?.quantumActivations;
  const bloch = liveCircuit?.blochVectors;

  // Draw connections first
  for (let li = 0; li < layers.length - 1; li++) {
    const srcCount = layers[li].nodes;
    const dstCount = layers[li + 1].nodes;
    for (let si = 0; si < srcCount; si++) {
      for (let di = 0; di < dstCount; di++) {
        // Get weight from qnn params if available
        let weight = 0.5;
        if (li === 1 && qnnParams?.theta) weight = qnnParams.theta[0]?.[si]?.[di % 2] ?? 0.5;
        if (li === 2 && qnnParams?.theta) weight = qnnParams.theta[1]?.[si]?.[di % 2] ?? 0.5;

        const sx = xPositions[li];
        const sy = y0 + ((si + 1) / (srcCount + 1)) * H;
        const dx = xPositions[li + 1];
        const dy = y0 + ((di + 1) / (dstCount + 1)) * H;

        const isPruned = pruneEnabled && Math.abs(weight) < pruneThreshold;
        const alpha = isPruned ? 0.05 : Math.min(1, Math.abs(weight) / Math.PI);

        // Animated signal pulse
        let pulse = 0;
        if (isTraining) {
          pulse = Math.sin(t * 3 - li * 1.5 + si * 0.5) * 0.5 + 0.5;
        }

        const grad = ctx.createLinearGradient(sx, sy, dx, dy);
        const col = valueToColor(weight / Math.PI);
        grad.addColorStop(0, `rgba(0,212,255,${alpha * 0.3})`);
        grad.addColorStop(0.5, `${col}${Math.floor(alpha * 255 * (0.3 + pulse * 0.4)).toString(16).padStart(2,'0')}`);
        grad.addColorStop(1, `rgba(0,212,255,${alpha * 0.3})`);

        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(dx, dy);
        ctx.strokeStyle = grad;
        ctx.lineWidth = isPruned ? 0.3 : Math.min(3, Math.abs(weight) / 0.5 + 0.5);
        ctx.stroke();

        // Signal pulse dot
        if (isTraining && !isPruned && alpha > 0.2) {
          const px = sx + (dx - sx) * ((t * 0.3 + li * 0.3 + si * 0.1) % 1);
          const py = sy + (dy - sy) * ((t * 0.3 + li * 0.3 + si * 0.1) % 1);
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fillStyle = COLORS.quantum;
          ctx.fill();
        }
      }
    }
  }

  // Draw nodes
  for (let li = 0; li < layers.length; li++) {
    const layer = layers[li];
    const x = xPositions[li];
    nodePositions.push([]);

    for (let ni = 0; ni < layer.nodes; ni++) {
      const y = y0 + ((ni + 1) / (layer.nodes + 1)) * H;
      nodePositions[li].push({ x, y });
      const r = layer.type === 'readout' ? 20 : 16;

      // Get activation value
      let activation = 0;
      if (layer.type === 'vqc' && qActiv) {
        activation = qActiv[li - 2]?.[ni] ?? 0;
      } else if (layer.type === 'encoding' && bloch) {
        activation = bloch[ni]?.z ?? 0;
      } else if (layer.type === 'readout' && liveCircuit) {
        activation = liveCircuit.analyticalExpectation ?? 0;
      }

      // Glow
      if (!pruneEnabled || Math.abs(activation) > pruneThreshold) {
        const grd = ctx.createRadialGradient(x, y, 0, x, y, r * 2.5);
        const col = valueToColor(activation);
        grd.addColorStop(0, col + '40');
        grd.addColorStop(1, 'transparent');
        ctx.fillStyle = grd;
        ctx.fillRect(x - r * 2.5, y - r * 2.5, r * 5, r * 5);
      }

      // Quantum superposition rings
      if (layer.type === 'vqc' || layer.type === 'encoding') {
        for (let ring = 1; ring <= 3; ring++) {
          const ringR = r + ring * 6 + (isTraining ? Math.sin(t * 2 + ring) * 2 : 0);
          ctx.beginPath();
          ctx.arc(x, y, ringR, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(0,212,255,${0.08 / ring})`;
          ctx.lineWidth = 0.5;
          ctx.setLineDash([3, 6]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // Node body
      const nodeGrd = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
      nodeGrd.addColorStop(0, '#1a2845');
      nodeGrd.addColorStop(1, '#0b1225');
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = nodeGrd;
      ctx.fill();

      // Border
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.strokeStyle = valueToColor(activation);
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Inner fill by activation
      const fillR = r * Math.abs(activation);
      if (fillR > 0) {
        ctx.beginPath();
        ctx.arc(x, y, fillR, 0, Math.PI * 2);
        ctx.fillStyle = `${valueToColor(activation)}30`;
        ctx.fill();
      }

      // Label
      ctx.font = `bold ${layer.type === 'readout' ? 10 : 9}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(232,240,254,0.9)';
      if (layer.type === 'input') ctx.fillText(`x${ni}`, x, y);
      else if (layer.type === 'readout') ctx.fillText('⟨Z₀⟩', x, y);
      else if (layer.type === 'encoding') ctx.fillText(`q${ni}`, x, y);
      else ctx.fillText(activation.toFixed(2), x, y);
    }

    // Layer label
    ctx.font = '8px Inter, sans-serif';
    ctx.fillStyle = 'rgba(138,163,200,0.7)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const label = layer.label.replace('\n', ' ');
    ctx.fillText(label, x, y0 - 14);
  }
}

// ─── Draw MLP ─────────────────────────────────────────────────────────────────

function drawMLP(ctx, mlpModel, x0, y0, W, H, t, isTraining, pruneEnabled, pruneThreshold) {
  if (!mlpModel) return;
  const arch = mlpModel.architecture; // [4, 8, 4, 1]
  const xPositions = arch.map((_, i) => x0 + (i / (arch.length - 1)) * W);
  const labels = ['Input', 'Hidden 1\n(ReLU)', 'Hidden 2\n(ReLU)', 'Output\n(tanh)'];

  // Connections
  for (let li = 0; li < arch.length - 1; li++) {
    const srcN = arch[li];
    const dstN = arch[li + 1];
    const W_mat = mlpModel.weights[li];

    for (let di = 0; di < dstN; di++) {
      for (let si = 0; si < srcN; si++) {
        const w = W_mat[di][si];
        const isPruned = pruneEnabled && Math.abs(w) < pruneThreshold;
        const sx = xPositions[li];
        const sy = y0 + ((si + 1) / (srcN + 1)) * H;
        const dx = xPositions[li + 1];
        const dy = y0 + ((di + 1) / (dstN + 1)) * H;

        const alpha = isPruned ? 0.03 : Math.min(0.7, Math.abs(w) * 0.8 + 0.1);
        const col = w > 0 ? '#ff6b35' : '#9b59ff';

        let pulse = 0;
        if (isTraining && !isPruned) {
          pulse = Math.sin(t * 3 - li * 1.5 + si * 0.3) * 0.5 + 0.5;
        }

        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(dx, dy);
        ctx.strokeStyle = `${col}${Math.floor(alpha * 255 * (isPruned ? 1 : (0.5 + pulse * 0.5))).toString(16).padStart(2, '0')}`;
        ctx.lineWidth = isPruned ? 0.2 : Math.min(2.5, Math.abs(w) * 1.5 + 0.3);
        ctx.stroke();
      }
    }
  }

  // Nodes
  for (let li = 0; li < arch.length; li++) {
    const n = arch[li];
    const x = xPositions[li];
    for (let ni = 0; ni < n; ni++) {
      const y = y0 + ((ni + 1) / (n + 1)) * H;
      const r = li === arch.length - 1 ? 20 : 14;

      const isOutput = li === arch.length - 1;

      // Glow
      const grd = ctx.createRadialGradient(x, y, 0, x, y, r * 2);
      grd.addColorStop(0, isOutput ? 'rgba(255,107,53,0.3)' : 'rgba(255,107,53,0.1)');
      grd.addColorStop(1, 'transparent');
      ctx.fillStyle = grd;
      ctx.fillRect(x - r * 2, y - r * 2, r * 4, r * 4);

      // Body
      const bg = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
      bg.addColorStop(0, '#1f1510');
      bg.addColorStop(1, '#0d0a07');
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = bg;
      ctx.fill();
      ctx.strokeStyle = isTraining
        ? `rgba(255,${Math.floor(107 + Math.sin(t * 4 + ni) * 40)},53,0.9)`
        : 'rgba(255,107,53,0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Label
      ctx.font = `${li === 0 ? 9 : 8}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = 'rgba(232,240,254,0.85)';
      if (li === 0) ctx.fillText(`x${ni}`, x, y);
      else if (isOutput) ctx.fillText('out', x, y);
      else ctx.fillText(`h${ni}`, x, y);
    }

    // Layer label
    ctx.font = '8px Inter, sans-serif';
    ctx.fillStyle = 'rgba(138,163,200,0.6)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(labels[li].replace('\n', ' '), x, y0 - 14);
  }
}
