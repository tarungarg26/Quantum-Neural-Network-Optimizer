/**
 * CircuitViz.jsx
 * ─────────────────────────────────────────────────────────────────────────────
 * Visualizes the 4-qubit QNN circuit with:
 *  - Gate boxes (RY, RZ, CNOT) on each wire
 *  - Live parameter values in gate boxes
 *  - Probability bars per qubit state
 *  - Shot noise overlay when shots != null
 *  - Toggle between analytic/finite-shot view
 */

import React, { useRef, useEffect } from 'react';

const GATE_W = 44;
const GATE_H = 28;
const WIRE_H = 64;
const MARGIN_LEFT = 80;
const MARGIN_TOP = 60;
const WIRE_COLOR = '#1a3050';
const QUBIT_COLORS = ['#00d4ff', '#9b59ff', '#ff6b35', '#00ff88'];

export default function CircuitViz({ qnnParams, liveCircuit, shots, mode }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = '#040810';
    ctx.fillRect(0, 0, W, H);

    if (!qnnParams || mode === 'classical') {
      ctx.fillStyle = 'rgba(138,163,200,0.4)';
      ctx.font = '14px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Switch to Quantum or Both mode to see circuit', W / 2, H / 2);
      return;
    }

    const n = 4;
    const layers = ['Enc', 'L1-RY', 'L1-RZ', 'CNOT', 'L2-RY', 'L2-RZ', 'CNOT', '⟨Z₀⟩'];
    const totalCols = layers.length;
    const colW = Math.min(80, (W - MARGIN_LEFT - 40) / totalCols);

    // Draw wires
    for (let q = 0; q < n; q++) {
      const y = MARGIN_TOP + q * WIRE_H;
      ctx.beginPath();
      ctx.moveTo(MARGIN_LEFT - 20, y);
      ctx.lineTo(MARGIN_LEFT + totalCols * colW + 20, y);
      ctx.strokeStyle = WIRE_COLOR;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Qubit label
      ctx.fillStyle = QUBIT_COLORS[q];
      ctx.font = 'bold 12px JetBrains Mono, monospace';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(`|q${q}⟩`, MARGIN_LEFT - 28, y);

      // Bloch Z value indicator
      if (liveCircuit?.blochVectors) {
        const bz = liveCircuit.blochVectors[q]?.z ?? 0;
        const col = bz > 0 ? '#00d4ff' : '#ff4757';
        ctx.fillStyle = col + 'aa';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'left';
        ctx.fillText(`z=${bz.toFixed(2)}`, MARGIN_LEFT + totalCols * colW + 28, y);
      }
    }

    // Column positions
    const colX = (i) => MARGIN_LEFT + i * colW + colW / 2;

    // Draw column labels
    layers.forEach((label, i) => {
      ctx.font = '8px Inter, sans-serif';
      ctx.fillStyle = 'rgba(138,163,200,0.6)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(label, colX(i), MARGIN_TOP - 10);
    });

    // Draw gates
    const drawGate = (col, qubit, label, value, color) => {
      const x = colX(col) - GATE_W / 2;
      const y = MARGIN_TOP + qubit * WIRE_H - GATE_H / 2;

      // Gate body
      const grd = ctx.createLinearGradient(x, y, x, y + GATE_H);
      grd.addColorStop(0, color + '40');
      grd.addColorStop(1, color + '20');
      ctx.fillStyle = grd;
      ctx.strokeStyle = color + 'cc';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(x, y, GATE_W, GATE_H, 4);
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = color;
      ctx.font = 'bold 10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, colX(col), MARGIN_TOP + qubit * WIRE_H - 5);

      // Value
      if (value !== null) {
        ctx.fillStyle = 'rgba(232,240,254,0.7)';
        ctx.font = '7px JetBrains Mono, monospace';
        ctx.fillText(value.toFixed(2), colX(col), MARGIN_TOP + qubit * WIRE_H + 7);
      }
    };

    const drawCNOT = (col, ctrl, tgt) => {
      const cx = colX(col);
      const cy = MARGIN_TOP + ctrl * WIRE_H;
      const ty = MARGIN_TOP + tgt * WIRE_H;

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx, ty);
      ctx.strokeStyle = 'rgba(0,212,255,0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Control dot
      ctx.beginPath();
      ctx.arc(cx, cy, 5, 0, Math.PI * 2);
      ctx.fillStyle = '#00d4ff';
      ctx.fill();

      // Target circle ⊕
      ctx.beginPath();
      ctx.arc(cx, ty, 10, 0, Math.PI * 2);
      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(cx - 10, ty);
      ctx.lineTo(cx + 10, ty);
      ctx.moveTo(cx, ty - 10);
      ctx.lineTo(cx, ty + 10);
      ctx.strokeStyle = '#00d4ff';
      ctx.stroke();
    };

    // Encode column (col 0)
    for (let q = 0; q < n; q++) {
      const x_val = liveCircuit?.x_scaled?.[q] ?? 0;
      drawGate(0, q, 'RY', x_val, '#9b59ff');
    }

    // Layer 1 (cols 1-3)
    for (let q = 0; q < n; q++) {
      const ry1 = qnnParams.theta[0]?.[q]?.[0] ?? 0;
      const rz1 = qnnParams.theta[0]?.[q]?.[1] ?? 0;
      drawGate(1, q, 'RY', ry1, '#00d4ff');
      drawGate(2, q, 'RZ', rz1, '#00f5ff');
    }
    // CNOT ring layer 1 (col 3)
    for (let q = 0; q < n; q++) {
      drawCNOT(3, q, (q + 1) % n);
    }

    // Layer 2 (cols 4-6)
    for (let q = 0; q < n; q++) {
      const ry2 = qnnParams.theta[1]?.[q]?.[0] ?? 0;
      const rz2 = qnnParams.theta[1]?.[q]?.[1] ?? 0;
      drawGate(4, q, 'RY', ry2, '#00d4ff');
      drawGate(5, q, 'RZ', rz2, '#00f5ff');
    }
    // CNOT ring layer 2 (col 6)
    for (let q = 0; q < n; q++) {
      drawCNOT(6, q, (q + 1) % n);
    }

    // Readout (col 7)
    const readout_y = MARGIN_TOP;
    const x_meas = colX(7);
    ctx.beginPath();
    ctx.roundRect(x_meas - GATE_W / 2, readout_y - GATE_H / 2, GATE_W, GATE_H, 4);
    ctx.fillStyle = 'rgba(0,255,136,0.15)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,255,136,0.8)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = '#00ff88';
    ctx.font = 'bold 10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('M', x_meas, readout_y);

    // Expectation value display
    if (liveCircuit) {
      const f = liveCircuit.analyticalExpectation ?? 0;
      const shotF = liveCircuit.expectation ?? 0;

      ctx.fillStyle = 'rgba(0,212,255,0.8)';
      ctx.font = '11px JetBrains Mono, monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`⟨Z₀⟩ = ${f.toFixed(4)}`, MARGIN_LEFT, H - 80);

      if (shots !== null) {
        ctx.fillStyle = 'rgba(255,107,53,0.7)';
        ctx.fillText(`⟨Z₀⟩̂ = ${shotF.toFixed(4)} (${shots} shots)`, MARGIN_LEFT, H - 62);
        ctx.fillStyle = 'rgba(155,89,255,0.6)';
        ctx.fillText(`noise σ = ${Math.sqrt(Math.max(0, 1 - f ** 2) / shots).toFixed(4)}`, MARGIN_LEFT, H - 44);
      }
    }

    // Shot noise overlay
    if (shots !== null && liveCircuit) {
      const noiseAmt = liveCircuit.shotNoise ?? 0;
      ctx.fillStyle = 'rgba(255,107,53,0.05)';
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = 'rgba(255,107,53,0.4)';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(`Shot noise: ${noiseAmt > 0 ? '+' : ''}${noiseAmt.toFixed(4)}`, W - 12, H - 12);
    }

  }, [qnnParams, liveCircuit, shots, mode]);

  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
    </div>
  );
}
