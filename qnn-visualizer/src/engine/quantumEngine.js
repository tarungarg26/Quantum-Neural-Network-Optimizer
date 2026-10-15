/**
 * quantumEngine.js
 * ─────────────────────────────────────────────────────────────────────────────
 * Simulates a Quantum Neural Network (QNN) and a Classical Neural Network (CNN)
 * side-by-side on the Iris binary classification task.
 *
 * Quantum model:
 *   - Angle encoding: RY(π·x̃ᵢ) per qubit
 *   - 2 variational layers of [RY,RZ] + CNOT ring
 *   - Readout: ⟨Z₀⟩ + classical bias
 *   - Parameter-shift rule gradient (simulated)
 *   - Shot noise: Var(f̂) = (1−f²)/N
 *
 * Classical model:
 *   - 4 → 8 → 4 → 1 MLP with tanh activations
 *   - Adam optimizer
 *
 * Both trained on same data splits for fair comparison.
 */

// ─── Utility ───────────────────────────────────────────────────────────────

export const sigmoid = (x) => 1 / (1 + Math.exp(-x));
export const tanh = Math.tanh;
export const relu = (x) => Math.max(0, x);

function seededRng(seed) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function randn(rng) {
  // Box-Muller
  const u = rng() || 1e-10;
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ─── Iris Data (versicolor=-1, virginica=+1, 100 samples) ────────────────────

export function generateIrisData() {
  // Simplified iris-like separable data for simulation
  const rng = seededRng(42);
  const data = [];
  for (let i = 0; i < 50; i++) {
    // versicolor cluster
    data.push({
      x: [5.5 + randn(rng) * 0.4, 2.6 + randn(rng) * 0.3, 4.0 + randn(rng) * 0.4, 1.25 + randn(rng) * 0.2],
      label: -1,
      species: 'versicolor',
    });
  }
  for (let i = 0; i < 50; i++) {
    // virginica cluster
    data.push({
      x: [6.5 + randn(rng) * 0.5, 3.0 + randn(rng) * 0.3, 5.5 + randn(rng) * 0.5, 2.0 + randn(rng) * 0.3],
      label: 1,
      species: 'virginica',
    });
  }
  return data;
}

// ─── Preprocessing ────────────────────────────────────────────────────────────

export function fitScaler(X_train) {
  const n_feat = X_train[0].length;
  const a = new Array(n_feat).fill(Infinity);
  const d = new Array(n_feat).fill(-Infinity);
  for (const x of X_train) {
    for (let j = 0; j < n_feat; j++) {
      a[j] = Math.min(a[j], x[j]);
      d[j] = Math.max(d[j], x[j]);
    }
  }
  return { a, d };
}

export function scaleX(x, scaler) {
  return x.map((xi, j) => {
    const range = scaler.d[j] - scaler.a[j];
    if (range === 0) return Math.PI / 2;
    return Math.PI * Math.min(1, Math.max(0, (xi - scaler.a[j]) / range));
  });
}

// ─── Quantum Circuit Simulation ───────────────────────────────────────────────

/**
 * Simulate a 4-qubit circuit:
 *   |0000⟩ → RY(x̃) → [RY(θ[l,w,0]) RZ(θ[l,w,1]) CNOT ring] × 2 → ⟨Z₀⟩
 *
 * Uses statevector simulation (exact, analytic).
 * Returns { expectation, probabilities, stateVector, quantumActivations }
 */
export function quantumCircuit(theta, x_tilde, shots = null) {
  const n = 4;
  const dim = 1 << n;

  // Initialize state |0000⟩
  let state = new Array(dim).fill(null).map((_, i) => ({ re: i === 0 ? 1 : 0, im: 0 }));

  // Encoding: RY(x̃ᵢ) per wire
  for (let w = 0; w < n; w++) {
    state = applyRY(state, w, n, x_tilde[w]);
  }

  // 2 variational layers
  const quantumActivations = [];
  for (let layer = 0; layer < 2; layer++) {
    const layerActivations = [];
    for (let w = 0; w < n; w++) {
      state = applyRY(state, w, n, theta[layer][w][0]);
      state = applyRZ(state, w, n, theta[layer][w][1]);
      // Compute Z expectation at this wire for visualization
      layerActivations.push(expectZ(state, w, n));
    }
    quantumActivations.push(layerActivations);
    // CNOT ring
    for (let w = 0; w < n; w++) {
      state = applyCNOT(state, w, (w + 1) % n, n);
    }
  }

  // Probabilities |ψ|²
  const probabilities = state.map(s => s.re ** 2 + s.im ** 2);

  // ⟨Z₀⟩ expectation
  let expZ0 = 0;
  for (let s = 0; s < dim; s++) {
    const qubit0 = (s >> (n - 1)) & 1; // bit n-1 = qubit 0
    const sign = qubit0 === 0 ? 1 : -1;
    expZ0 += sign * (state[s].re ** 2 + state[s].im ** 2);
  }

  // Add shot noise if shots specified
  let f = expZ0;
  let shotNoise = 0;
  if (shots !== null) {
    const variance = (1 - expZ0 ** 2) / shots;
    shotNoise = (Math.random() * 2 - 1) * Math.sqrt(variance);
    f = Math.max(-1, Math.min(1, expZ0 + shotNoise));
  }

  return {
    expectation: f,
    analyticalExpectation: expZ0,
    probabilities,
    stateVector: state,
    quantumActivations,
    shotNoise,
    blochVectors: computeBlochVectors(state, n),
  };
}

function expectZ(state, wire, n) {
  const dim = 1 << n;
  let exp = 0;
  for (let s = 0; s < dim; s++) {
    const bit = (s >> (n - 1 - wire)) & 1;
    const sign = bit === 0 ? 1 : -1;
    exp += sign * (state[s].re ** 2 + state[s].im ** 2);
  }
  return exp;
}

function computeBlochVectors(state, n) {
  // Single-qubit reduced density matrix → Bloch vector for each qubit
  const dim = 1 << n;
  return Array.from({ length: n }, (_, w) => {
    let rho00 = 0, rho11 = 0, rho01Re = 0, rho01Im = 0;
    for (let s = 0; s < dim; s++) {
      const bit = (s >> (n - 1 - w)) & 1;
      const prob = state[s].re ** 2 + state[s].im ** 2;
      if (bit === 0) rho00 += prob;
      else rho11 += prob;

      const s1 = bit === 0 ? s | (1 << (n - 1 - w)) : s & ~(1 << (n - 1 - w));
      rho01Re += state[s].re * state[s1].re + state[s].im * state[s1].im;
      rho01Im += state[s].im * state[s1].re - state[s].re * state[s1].im;
    }
    return {
      x: 2 * rho01Re,
      y: 2 * rho01Im,
      z: rho00 - rho11,
    };
  });
}

// ─── Gate implementations (complex arithmetic) ───────────────────────────────

function applyRY(state, wire, n, angle) {
  const dim = 1 << n;
  const c = Math.cos(angle / 2);
  const s = Math.sin(angle / 2);
  const newState = [...state.map(a => ({ ...a }))];
  for (let idx = 0; idx < dim; idx++) {
    const bit = (idx >> (n - 1 - wire)) & 1;
    const partner = idx ^ (1 << (n - 1 - wire));
    if (bit === 0) {
      const a = state[idx], b = state[partner];
      newState[idx]     = { re: c * a.re - s * b.re, im: c * a.im - s * b.im };
      newState[partner] = { re: s * a.re + c * b.re, im: s * a.im + c * b.im };
    }
  }
  return newState;
}

function applyRZ(state, wire, n, angle) {
  const dim = 1 << n;
  const newState = state.map(a => ({ ...a }));
  for (let idx = 0; idx < dim; idx++) {
    const bit = (idx >> (n - 1 - wire)) & 1;
    const phi = bit === 0 ? -angle / 2 : angle / 2;
    const c = Math.cos(phi), s = Math.sin(phi);
    const { re, im } = state[idx];
    newState[idx] = { re: c * re - s * im, im: s * re + c * im };
  }
  return newState;
}

function applyCNOT(state, control, target, n) {
  const dim = 1 << n;
  const newState = state.map(a => ({ ...a }));
  for (let idx = 0; idx < dim; idx++) {
    const ctrl = (idx >> (n - 1 - control)) & 1;
    if (ctrl === 1) {
      const partner = idx ^ (1 << (n - 1 - target));
      if (idx < partner) {
        newState[idx] = state[partner];
        newState[partner] = state[idx];
      }
    }
  }
  return newState;
}

// ─── QNN Model ────────────────────────────────────────────────────────────────

export function initQNNTheta(seed = 0) {
  const rng = seededRng(seed + 12345);
  const theta = [];
  for (let l = 0; l < 2; l++) {
    theta.push([]);
    for (let w = 0; w < 4; w++) {
      theta[l].push([rng() * 2 * Math.PI, rng() * 2 * Math.PI]);
    }
  }
  return { theta, bias: 0.0 };
}

export function qnnForward(params, x_tilde, shots = null) {
  const result = quantumCircuit(params.theta, x_tilde, shots);
  const score = result.expectation + params.bias;
  return { score, prediction: score >= 0 ? 1 : -1, ...result };
}

export function qnnLoss(params, dataset, shots = null) {
  let totalLoss = 0;
  let correct = 0;
  for (const sample of dataset) {
    const { score } = qnnForward(params, sample.x_scaled, shots);
    const residual = score - sample.label;
    totalLoss += residual ** 2;
    if ((score >= 0 ? 1 : -1) === sample.label) correct++;
  }
  return { loss: totalLoss / dataset.length, accuracy: correct / dataset.length };
}

/**
 * Parameter-shift rule gradient (analytic, 2-term):
 * ∂f/∂θₖ = ½[f(θ+π/2 eₖ) - f(θ-π/2 eₖ)]
 */
export function qnnGradient(params, dataset, _shots = null) {
  const S = Math.PI / 2;
  const gradTheta = Array.from({ length: 2 }, () =>
    Array.from({ length: 4 }, () => [0, 0])
  );
  let gradBias = 0;

  for (const sample of dataset) {
    const { x_scaled: x, label: y } = sample;
    const f0 = quantumCircuit(params.theta, x).expectation + params.bias;
    const residual = f0 - y;

    // PSR for each angle parameter
    for (let l = 0; l < 2; l++) {
      for (let w = 0; w < 4; w++) {
        for (let g = 0; g < 2; g++) {
          const thetaPlus = JSON.parse(JSON.stringify(params.theta));
          const thetaMinus = JSON.parse(JSON.stringify(params.theta));
          thetaPlus[l][w][g] += S;
          thetaMinus[l][w][g] -= S;
          const fPlus = quantumCircuit(thetaPlus, x).expectation;
          const fMinus = quantumCircuit(thetaMinus, x).expectation;
          const df = 0.5 * (fPlus - fMinus);
          gradTheta[l][w][g] += (2 / dataset.length) * residual * df;
        }
      }
    }
    gradBias += (2 / dataset.length) * residual;
  }
  return { gradTheta, gradBias };
}

// ─── Classical Neural Network ─────────────────────────────────────────────────

export function initMLPWeights(seed = 0) {
  const rng = seededRng(seed + 99999);
  const architecture = [4, 8, 4, 1];
  const weights = [];
  const biases = [];
  for (let i = 0; i < architecture.length - 1; i++) {
    const fan_in = architecture[i];
    const fan_out = architecture[i + 1];
    const scale = Math.sqrt(2 / fan_in) * 0.6;
    const W = Array.from({ length: fan_out }, () =>
      Array.from({ length: fan_in }, () => randn(rng) * scale)
    );
    const b = new Array(fan_out).fill(0);
    weights.push(W);
    biases.push(b);
  }
  return { weights, biases, architecture };
}

export function mlpForward(model, x) {
  let activation = x;
  const activations = [x];
  const preActivations = [];

  for (let layer = 0; layer < model.weights.length; layer++) {
    const W = model.weights[layer];
    const b = model.biases[layer];
    const pre = W.map((row, i) =>
      row.reduce((sum, w, j) => sum + w * activation[j], 0) + b[i]
    );
    preActivations.push(pre);
    const isLast = layer === model.weights.length - 1;
    activation = isLast ? pre : pre.map(z => Math.max(0, z));
    activations.push(activation);
  }

  const score = activation[0];
  return {
    score,
    prediction: score >= 0 ? 1 : -1,
    activations,
    preActivations,
  };
}

export function mlpLoss(model, dataset) {
  let totalLoss = 0;
  let correct = 0;
  for (const sample of dataset) {
    const { score } = mlpForward(model, sample.x_scaled);
    const residual = score - sample.label;
    totalLoss += residual ** 2;
    if ((score >= 0 ? 1 : -1) === sample.label) correct++;
  }
  return { loss: totalLoss / dataset.length, accuracy: correct / dataset.length };
}

export function mlpGradient(model, dataset) {
  const dW = model.weights.map(W => W.map(row => new Array(row.length).fill(0)));
  const db = model.biases.map(b => new Array(b.length).fill(0));

  for (const sample of dataset) {
    const { activations, preActivations, score } = mlpForward(model, sample.x_scaled);
    const y = sample.label;
    // Linear output derivative is 1: dL/dz = 2 * (score - y) / N
    let delta = [2 * (score - y) / dataset.length];

    for (let l = model.weights.length - 1; l >= 0; l--) {
      const act_prev = activations[l];
      for (let i = 0; i < delta.length; i++) {
        db[l][i] += delta[i];
        for (let j = 0; j < act_prev.length; j++) {
          dW[l][i][j] += delta[i] * act_prev[j];
        }
      }
      if (l > 0) {
        const newDelta = new Array(model.weights[l][0].length).fill(0);
        for (let j = 0; j < newDelta.length; j++) {
          for (let i = 0; i < delta.length; i++) {
            // Hidden layers are ReLU: derivative is 1 if preactivation > 0, else 0
            const deriv = preActivations[l - 1][j] > 0 ? 1 : 0;
            newDelta[j] += model.weights[l][i][j] * delta[i] * deriv;
          }
        }
        delta = newDelta;
      }
    }
  }
  return { dW, db };
}

// ─── Adam optimizer state ────────────────────────────────────────────────────

export function initAdamState(params, isQNN = false) {
  if (isQNN) {
    return {
      mTheta: Array.from({ length: 2 }, () => Array.from({ length: 4 }, () => [0, 0])),
      vTheta: Array.from({ length: 2 }, () => Array.from({ length: 4 }, () => [0, 0])),
      mBias: 0, vBias: 0, t: 0,
    };
  }
  return {
    mW: params.weights.map(W => W.map(row => new Array(row.length).fill(0))),
    vW: params.weights.map(W => W.map(row => new Array(row.length).fill(0))),
    mB: params.biases.map(b => new Array(b.length).fill(0)),
    vB: params.biases.map(b => new Array(b.length).fill(0)),
    t: 0,
  };
}

export function adamUpdateQNN(params, grad, adamState, lr = 0.03, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  adamState.t += 1;
  const t = adamState.t;

  const newTheta = JSON.parse(JSON.stringify(params.theta));

  for (let l = 0; l < 2; l++) {
    for (let w = 0; w < 4; w++) {
      for (let g = 0; g < 2; g++) {
        const g_val = grad.gradTheta[l][w][g];
        adamState.mTheta[l][w][g] = beta1 * adamState.mTheta[l][w][g] + (1 - beta1) * g_val;
        adamState.vTheta[l][w][g] = beta2 * adamState.vTheta[l][w][g] + (1 - beta2) * g_val ** 2;
        const m_hat = adamState.mTheta[l][w][g] / (1 - beta1 ** t);
        const v_hat = adamState.vTheta[l][w][g] / (1 - beta2 ** t);
        newTheta[l][w][g] -= lr * m_hat / (Math.sqrt(v_hat) + eps);
      }
    }
  }

  adamState.mBias = beta1 * adamState.mBias + (1 - beta1) * grad.gradBias;
  adamState.vBias = beta2 * adamState.vBias + (1 - beta2) * grad.gradBias ** 2;
  const mB_hat = adamState.mBias / (1 - beta1 ** t);
  const vB_hat = adamState.vBias / (1 - beta2 ** t);

  return {
    theta: newTheta,
    bias: params.bias - lr * mB_hat / (Math.sqrt(vB_hat) + eps),
  };
}

export function adamUpdateMLP(model, grad, adamState, lr = 0.03, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
  adamState.t += 1;
  const t = adamState.t;
  const newWeights = model.weights.map((W, l) =>
    W.map((row, i) => row.map((w, j) => {
      const g = grad.dW[l][i][j];
      adamState.mW[l][i][j] = beta1 * adamState.mW[l][i][j] + (1 - beta1) * g;
      adamState.vW[l][i][j] = beta2 * adamState.vW[l][i][j] + (1 - beta2) * g ** 2;
      const m_hat = adamState.mW[l][i][j] / (1 - beta1 ** t);
      const v_hat = adamState.vW[l][i][j] / (1 - beta2 ** t);
      return w - lr * m_hat / (Math.sqrt(v_hat) + eps);
    }))
  );
  const newBiases = model.biases.map((b, l) =>
    b.map((bv, i) => {
      const g = grad.db[l][i];
      adamState.mB[l][i] = beta1 * adamState.mB[l][i] + (1 - beta1) * g;
      adamState.vB[l][i] = beta2 * adamState.vB[l][i] + (1 - beta2) * g ** 2;
      const m_hat = adamState.mB[l][i] / (1 - beta1 ** t);
      const v_hat = adamState.vB[l][i] / (1 - beta2 ** t);
      return bv - lr * m_hat / (Math.sqrt(v_hat) + eps);
    })
  );
  return { ...model, weights: newWeights, biases: newBiases };
}

// ─── Pruning ─────────────────────────────────────────────────────────────────

export function pruneWeights(model, threshold = 0.05) {
  const pruned = JSON.parse(JSON.stringify(model));
  let prunedCount = 0;
  let totalCount = 0;
  pruned.weights = model.weights.map(W =>
    W.map(row => row.map(w => {
      totalCount++;
      if (Math.abs(w) < threshold) { prunedCount++; return 0; }
      return w;
    }))
  );
  return { model: pruned, prunedCount, totalCount, ratio: prunedCount / totalCount };
}
