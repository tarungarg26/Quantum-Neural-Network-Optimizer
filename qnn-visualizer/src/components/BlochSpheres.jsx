/**
 * BlochSpheres.jsx — Renders 4 mini Bloch spheres for each qubit
 */
import React, { useRef, useEffect } from 'react';

const COLORS = ['#00d4ff', '#9b59ff', '#ff6b35', '#00ff88'];

function drawBlochSphere(ctx, cx, cy, r, bv, color) {
  // Sphere outline
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.1)';
  ctx.lineWidth = 1;
  ctx.stroke();

  // Equator
  ctx.beginPath();
  ctx.ellipse(cx, cy, r, r * 0.2, 0, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.stroke();

  // Axes
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 0.5;
  ctx.beginPath(); ctx.moveTo(cx, cy - r); ctx.lineTo(cx, cy + r); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx - r, cy); ctx.lineTo(cx + r, cy); ctx.stroke();

  // Bloch vector arrow
  const { x, z } = bv;
  const px = cx + x * r * 0.85;
  const py = cy - z * r * 0.85;

  const grd = ctx.createLinearGradient(cx, cy, px, py);
  grd.addColorStop(0, color + '60');
  grd.addColorStop(1, color);

  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(px, py);
  ctx.strokeStyle = grd;
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Arrow head
  const angle = Math.atan2(py - cy, px - cx);
  ctx.beginPath();
  ctx.moveTo(px, py);
  ctx.lineTo(px - 5 * Math.cos(angle - 0.4), py - 5 * Math.sin(angle - 0.4));
  ctx.lineTo(px - 5 * Math.cos(angle + 0.4), py - 5 * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();

  // State dot
  ctx.beginPath();
  ctx.arc(px, py, 2.5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.shadowBlur = 8;
  ctx.shadowColor = color;
  ctx.fill();
  ctx.shadowBlur = 0;

  // |0⟩ and |1⟩ labels
  ctx.fillStyle = 'rgba(138,163,200,0.5)';
  ctx.font = '7px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.fillText('|0⟩', cx, cy - r - 3);
  ctx.fillText('|1⟩', cx, cy + r + 8);
}

export default function BlochSpheres({ blochVectors }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !blochVectors) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, W, H);

    const r = Math.min(W / 8, H / 2.5 - 12);
    blochVectors.forEach((bv, i) => {
      const cx = (i + 0.5) * (W / 4);
      const cy = H / 2;

      // Background circle
      const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grd.addColorStop(0, COLORS[i] + '08');
      grd.addColorStop(1, 'transparent');
      ctx.fillStyle = grd;
      ctx.fillRect(cx - r, cy - r, r * 2, r * 2);

      drawBlochSphere(ctx, cx, cy, r, bv, COLORS[i]);

      // Qubit label
      ctx.fillStyle = COLORS[i] + 'cc';
      ctx.font = 'bold 8px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`q${i}`, cx, H - 4);
    });
  }, [blochVectors]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: 100, display: 'block' }}
    />
  );
}
