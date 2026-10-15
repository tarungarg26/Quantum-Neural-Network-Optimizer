/**
 * useTrainingEngine.js
 * ─────────────────────────────────────────────────────────────────────────────
 * React hook that manages the full training loop for both QNN and classical MLP.
 * Supports switching across multiple real-world datasets:
 *   1. Formula 1 Grand Prix Telemetry (OpenF1 / FastF1)
 *   2. Iris Flower Benchmark (Fisher)
 *   3. Wine Chemical Cultivars (UCI Machine Learning)
 *   4. Breast Cancer Diagnostic (Wisconsin Clinical)
 * Exposes training state, live metrics, and dataset controls.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import {
  fitScaler, scaleX,
  initQNNTheta, qnnForward, qnnLoss, qnnGradient, adamUpdateQNN, initAdamState,
  initMLPWeights, mlpForward, mlpLoss, mlpGradient, adamUpdateMLP,
  pruneWeights, quantumCircuit,
} from '../engine/quantumEngine';
import { DATASETS, DEFAULT_DATASET_KEY } from '../engine/datasets';

const BATCH_SIZE = 30;
const TRAIN_RATIO = 0.6;
const VAL_RATIO = 0.2;

function prepareDataset(datasetKey = DEFAULT_DATASET_KEY, seed = 7) {
  const meta = DATASETS[datasetKey] || DATASETS[DEFAULT_DATASET_KEY];
  const allData = meta.data;

  // Shuffle with seed
  const shuffled = [...allData];
  let s = seed;
  for (let i = shuffled.length - 1; i > 0; i--) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const j = s % (i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const n = shuffled.length;
  const nTrain = Math.floor(n * TRAIN_RATIO);
  const nVal = Math.floor(n * VAL_RATIO);

  const trainRaw = shuffled.slice(0, nTrain);
  const valRaw = shuffled.slice(nTrain, nTrain + nVal);
  const testRaw = shuffled.slice(nTrain + nVal);

  const scaler = fitScaler(trainRaw.map(d => d.x));

  const scale = (raw) => raw.map(d => ({
    ...d,
    x_scaled: scaleX(d.x, scaler),
  }));

  return {
    key: datasetKey,
    meta,
    train: scale(trainRaw),
    val: scale(valRaw),
    test: scale(testRaw),
    scaler,
  };
}

export function useTrainingEngine() {
  // Dataset state
  const [datasetKey, setDatasetKeyState] = useState(DEFAULT_DATASET_KEY);
  const [dataset, setDataset] = useState(() => prepareDataset(DEFAULT_DATASET_KEY, 7));

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [mode, setMode] = useState('both'); // 'quantum' | 'classical' | 'both' (default 'both' so user sees both live!)
  const [shots, setShots] = useState(null); // null=analytic, int=finite-shot
  const [lr, setLr] = useState(0.02);
  const [pruneThreshold, setPruneThreshold] = useState(0.05);
  const [pruneEnabled, setPruneEnabled] = useState(false);

  // QNN state
  const [qnnParams, setQnnParams] = useState(() => initQNNTheta(0));
  const qnnParamsRef = useRef(qnnParams);
  const qnnAdamRef = useRef(initAdamState(null, true));
  const [qnnMetrics, setQnnMetrics] = useState({ trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] });

  // MLP state
  const [mlpModel, setMlpModel] = useState(() => initMLPWeights(0));
  const mlpModelRef = useRef(mlpModel);
  const mlpAdamRef = useRef(initAdamState(initMLPWeights(0)));
  const [mlpMetrics, setMlpMetrics] = useState({ trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] });

  // Pruning state
  const [pruneStats, setPruneStats] = useState({ prunedCount: 0, totalCount: 0, ratio: 0 });

  // Live circuit visualization (for the currently selected sample)
  const [liveCircuit, setLiveCircuit] = useState(null);
  const [selectedSample, setSelectedSample] = useState(0);

  const trainingRef = useRef(false);
  const frameRef = useRef(null);
  const epochRef = useRef(0);

  // Reset models & metrics
  const reset = useCallback((seed = 0) => {
    const newQNN = initQNNTheta(seed);
    const newMLP = initMLPWeights(seed);
    qnnParamsRef.current = newQNN;
    mlpModelRef.current = newMLP;
    qnnAdamRef.current = initAdamState(null, true);
    mlpAdamRef.current = initAdamState(newMLP);
    setQnnParams(newQNN);
    setMlpModel(newMLP);
    setQnnMetrics({ trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] });
    setMlpMetrics({ trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] });
    setEpoch(0);
    epochRef.current = 0;
    setPruneStats({ prunedCount: 0, totalCount: 0, ratio: 0 });
  }, []);

  // Switch dataset
  const setDatasetKey = useCallback((newKey) => {
    if (!DATASETS[newKey]) return;
    trainingRef.current = false;
    setIsTraining(false);
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    const newDset = prepareDataset(newKey, 7);
    setDatasetKeyState(newKey);
    setDataset(newDset);
    reset(0);
  }, [reset]);

  // Single training step
  const step = useCallback((currentQNN, currentMLP, currentQNNAdam, currentMLPAdam) => {
    let newQNN = currentQNN;
    let newMLP = currentMLP;
    let newQNNAdam = currentQNNAdam;
    let newMLPAdam = currentMLPAdam;

    if (mode === 'quantum' || mode === 'both') {
      const batchStart = (epochRef.current * BATCH_SIZE) % dataset.train.length;
      const batch = dataset.train.slice(batchStart, batchStart + BATCH_SIZE);

      const grad = qnnGradient(newQNN, batch, shots);
      const updated = adamUpdateQNN(newQNN, grad, newQNNAdam, lr);
      newQNN = { theta: updated.theta, bias: updated.bias };

      if (pruneEnabled) {
        const prunedTheta = newQNN.theta.map(l =>
          l.map(w => w.map(a => Math.abs(a) % (2 * Math.PI) < pruneThreshold ? 0 : a))
        );
        newQNN = { ...newQNN, theta: prunedTheta };
      }
    }

    if (mode === 'classical' || mode === 'both') {
      const batchStart = (epochRef.current * BATCH_SIZE) % dataset.train.length;
      const batch = dataset.train.slice(batchStart, batchStart + BATCH_SIZE);

      const grad = mlpGradient(newMLP, batch);
      newMLP = adamUpdateMLP(newMLP, grad, newMLPAdam, lr);

      if (pruneEnabled) {
        const pruned = pruneWeights(newMLP, pruneThreshold);
        newMLP = pruned.model;
        setPruneStats({ prunedCount: pruned.prunedCount, totalCount: pruned.totalCount, ratio: pruned.ratio });
      }
    }

    return { newQNN, newMLP, newQNNAdam, newMLPAdam };
  }, [mode, shots, lr, pruneEnabled, pruneThreshold, dataset]);

  // Training loop
  const trainLoop = useCallback(function runLoop() {
    if (!trainingRef.current) return;

    const { newQNN, newMLP, newQNNAdam, newMLPAdam } = step(
      qnnParamsRef.current,
      mlpModelRef.current,
      qnnAdamRef.current,
      mlpAdamRef.current
    );

    qnnParamsRef.current = newQNN;
    mlpModelRef.current = newMLP;
    qnnAdamRef.current = newQNNAdam;
    mlpAdamRef.current = newMLPAdam;
    setQnnParams(newQNN);
    setMlpModel(newMLP);

    epochRef.current++;
    setEpoch(epochRef.current);

    if (epochRef.current % 5 === 0) {
      if (mode === 'quantum' || mode === 'both') {
        const trainMetrics = qnnLoss(newQNN, dataset.train, shots);
        const valMetrics = qnnLoss(newQNN, dataset.val, shots);
        setQnnMetrics(prev => ({
          trainLoss: [...prev.trainLoss.slice(-100), trainMetrics.loss],
          valLoss: [...prev.valLoss.slice(-100), valMetrics.loss],
          trainAcc: [...prev.trainAcc.slice(-100), trainMetrics.accuracy],
          valAcc: [...prev.valAcc.slice(-100), valMetrics.accuracy],
        }));
      }

      if (mode === 'classical' || mode === 'both') {
        const trainMetrics = mlpLoss(newMLP, dataset.train);
        const valMetrics = mlpLoss(newMLP, dataset.val);
        setMlpMetrics(prev => ({
          trainLoss: [...prev.trainLoss.slice(-100), trainMetrics.loss],
          valLoss: [...prev.valLoss.slice(-100), valMetrics.loss],
          trainAcc: [...prev.trainAcc.slice(-100), trainMetrics.accuracy],
          valAcc: [...prev.valAcc.slice(-100), valMetrics.accuracy],
        }));
      }

      // Live circuit update
      const sample = dataset.train[selectedSample % dataset.train.length];
      if (sample && (mode === 'quantum' || mode === 'both')) {
        const circuit = quantumCircuit(newQNN.theta, sample.x_scaled, shots);
        setLiveCircuit({ ...circuit, x_scaled: sample.x_scaled, label: sample.label });
      }
    }

    if (trainingRef.current) {
      frameRef.current = requestAnimationFrame(runLoop);
    }
  }, [step, mode, shots, dataset, selectedSample]);

  const startTraining = useCallback(() => {
    trainingRef.current = true;
    setIsTraining(true);
    frameRef.current = requestAnimationFrame(trainLoop);
  }, [trainLoop]);

  const stopTraining = useCallback(() => {
    trainingRef.current = false;
    setIsTraining(false);
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
  }, []);

  useEffect(() => {
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, []);

  // Test a custom input
  const testInput = useCallback((rawInput) => {
    const x_scaled = scaleX(rawInput, dataset.scaler);
    const qnnResult = qnnForward(qnnParams, x_scaled, shots);
    const mlpResult = mlpForward(mlpModel, x_scaled);
    return { quantum: qnnResult, classical: mlpResult, x_scaled };
  }, [qnnParams, mlpModel, shots, dataset.scaler]);

  return {
    dataset,
    datasetKey,
    setDatasetKey,
    datasetMeta: dataset.meta,
    allDatasets: Object.values(DATASETS),
    epoch,
    isTraining,
    startTraining,
    stopTraining,
    reset,
    mode, setMode,
    shots, setShots,
    lr, setLr,
    pruneThreshold, setPruneThreshold,
    pruneEnabled, setPruneEnabled,
    qnnParams,
    mlpModel,
    qnnMetrics,
    mlpMetrics,
    pruneStats,
    liveCircuit,
    selectedSample, setSelectedSample,
    testInput,
  };
}
