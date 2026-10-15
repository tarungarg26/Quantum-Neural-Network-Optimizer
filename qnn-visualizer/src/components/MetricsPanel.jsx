/**
 * MetricsPanel.jsx — Live training loss & accuracy charts
 */

import React, { useRef, useEffect } from 'react';

function drawChart(ctx, W, H, series, colors, labels, yLabel, xLabel) {
  ctx.clearRect(0, 0, W, H);
  const pad = { top: 24, right: 16, bottom: 32, left: 44 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  // Background grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * cH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cW, y); ctx.stroke();
  }
  for (let i = 0; i <= 5; i++) {
    const x = pad.left + (i / 5) * cW;
    ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + cH); ctx.stroke();
  }

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + cH);
  ctx.moveTo(pad.left, pad.top + cH); ctx.lineTo(pad.left + cW, pad.top + cH);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = 'rgba(138,163,200,0.6)';
  ctx.font = '8px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, pad.left + cW / 2, H - 4);
  ctx.save();
  ctx.translate(10, pad.top + cH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();

  // Find range
  const allVals = series.flat();
  if (allVals.length === 0) return;
  const yMin = Math.min(...allVals) * 0.95;
  const yMax = Math.max(...allVals) * 1.05 || 1;

  // Y axis ticks
  ctx.fillStyle = 'rgba(138,163,200,0.5)';
  ctx.font = '7px JetBrains Mono, monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = yMax - (i / 4) * (yMax - yMin);
    const y = pad.top + (i / 4) * cH;
    ctx.fillText(val.toFixed(3), pad.left - 4, y + 3);
  }

  // Draw each series
  series.forEach((data, si) => {
    if (data.length < 2) return;
    const color = colors[si];
    const n = data.length;

    // Gradient fill
    const grd = ctx.createLinearGradient(0, pad.top, 0, pad.top + cH);
    grd.addColorStop(0, color + '30');
    grd.addColorStop(1, color + '00');
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.left + (i / (n - 1)) * cW;
      const y = pad.top + cH - ((v - yMin) / (yMax - yMin)) * cH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.left + cW, pad.top + cH);
    ctx.lineTo(pad.left, pad.top + cH);
    ctx.closePath();
    ctx.fillStyle = grd;
    ctx.fill();

    // Line
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad.left + (i / (n - 1)) * cW;
      const y = pad.top + cH - ((v - yMin) / (yMax - yMin)) * cH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Last point dot
    const lastV = data.at(-1);
    const lx = pad.left + cW;
    const ly = pad.top + cH - ((lastV - yMin) / (yMax - yMin)) * cH;
    ctx.beginPath();
    ctx.arc(lx, ly, 3, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // Legend
    ctx.fillStyle = color;
    ctx.fillRect(pad.left + si * 80, 6, 10, 3);
    ctx.fillStyle = 'rgba(232,240,254,0.7)';
    ctx.font = '8px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${labels[si]}: ${lastV.toFixed(4)}`, pad.left + si * 80 + 14, 12);
  });
}

export default function MetricsPanel({ qnnMetrics, mlpMetrics, mode, epoch }) {
  const lossRef = useRef(null);
  const accRef = useRef(null);

  useEffect(() => {
    [lossRef, accRef].forEach((ref, chartIdx) => {
      const canvas = ref.current;
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

      const series = [];
      const colors = [];
      const labels = [];

      if (mode === 'quantum' || mode === 'both') {
        if (chartIdx === 0) {
          series.push(qnnMetrics.trainLoss, qnnMetrics.valLoss);
          colors.push('#00d4ff', '#9b59ff');
          labels.push('QNN Train', 'QNN Val');
        } else {
          series.push(qnnMetrics.trainAcc, qnnMetrics.valAcc);
          colors.push('#00d4ff', '#9b59ff');
          labels.push('QNN Train', 'QNN Val');
        }
      }
      if (mode === 'classical' || mode === 'both') {
        if (chartIdx === 0) {
          series.push(mlpMetrics.trainLoss, mlpMetrics.valLoss);
          colors.push('#ff6b35', '#ffa726');
          labels.push('MLP Train', 'MLP Val');
        } else {
          series.push(mlpMetrics.trainAcc, mlpMetrics.valAcc);
          colors.push('#ff6b35', '#ffa726');
          labels.push('MLP Train', 'MLP Val');
        }
      }

      drawChart(ctx, W, H, series, colors, labels,
        chartIdx === 0 ? 'MSE Loss' : 'Accuracy',
        'Step (×5)'
      );
    });
  }, [qnnMetrics, mlpMetrics, mode, epoch]);

  return (
    <div style={{
      position: 'absolute', inset: 0, display: 'grid',
      gridTemplateRows: '1fr 1fr', gap: 1,
      background: '#040810', padding: 20,
    }}>
      <div style={{ position: 'relative' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, fontSize: 10, color: 'rgba(138,163,200,0.8)', fontWeight: 600, letterSpacing: '1px', zIndex: 1, padding: '4px 8px', background: 'rgba(4,8,16,0.8)', borderRadius: 6 }}>
          TRAINING LOSS (MSE)
        </div>
        <canvas ref={lossRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>
      <div style={{ position: 'relative' }}>
        <div style={{ position: 'absolute', top: 0, left: 0, fontSize: 10, color: 'rgba(138,163,200,0.8)', fontWeight: 600, letterSpacing: '1px', zIndex: 1, padding: '4px 8px', background: 'rgba(4,8,16,0.8)', borderRadius: 6 }}>
          VALIDATION ACCURACY
        </div>
        <canvas ref={accRef} style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>
    </div>
  );
}
