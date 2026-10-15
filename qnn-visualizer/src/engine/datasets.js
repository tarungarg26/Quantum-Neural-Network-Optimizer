/**
 * datasets.js — Multiple real-world datasets for QNN vs MLP Benchmark
 * Includes:
 *   1. Formula 1 Grand Prix Telemetry (OpenF1 / FastF1)
 *   2. Iris Flower Benchmark (Fisher)
 *   3. Wine Chemical Cultivars (UCI Machine Learning)
 *   4. Breast Cancer Diagnostic (Wisconsin Clinical)
 */

export const DATASETS = {
  "f1": {
    "id": "f1",
    "name": "Formula 1 Grand Prix Telemetry",
    "tag": "FastF1 / OpenF1",
    "icon": "🏎️",
    "description": "Real Formula 1 telemetry: Attack / Push Lap pace vs Tire Degradation & Pit-In pace based on 3 sectors and speed trap.",
    "featureNames": [
      "Sector 1 (s)",
      "Sector 2 (s)",
      "Sector 3 (s)",
      "Speed Trap (km/h)"
    ],
    "classNames": {
      "1": "Tire Preservation Lap (≥98.2s)",
      "-1": "Attack / Push Lap (<98.2s)"
    },
    "defaults": {
      "Push Lap": [
        28.8,
        41.5,
        27.2,
        289
      ],
      "Preservation Lap": [
        33.7,
        43.4,
        29.9,
        218
      ]
    },
    "data": [
      {
        "x": [
          34.365,
          42.758,
          28.08,
          212
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.095,
          41.944,
          27.835,
          288
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.136,
          41.912,
          27.929,
          291
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.118,
          41.341,
          27.937,
          290
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.988,
          41.724,
          27.754,
          289
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.949,
          41.527,
          27.737,
          294
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.895,
          41.548,
          27.722,
          293
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.931,
          41.493,
          35.544,
          292
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          40.194,
          45.228,
          27.595,
          220
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.694,
          41.335,
          27.529,
          303
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.822,
          41.318,
          27.545,
          292
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.102,
          41.311,
          27.561,
          290
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.182,
          41.36,
          27.583,
          289
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.802,
          41.645,
          27.45,
          294
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.912,
          41.594,
          27.607,
          293
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.876,
          41.611,
          27.603,
          291
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.961,
          42.154,
          27.816,
          295
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.136,
          42.136,
          27.727,
          307
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.832,
          41.527,
          27.376,
          290
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.956,
          41.505,
          27.536,
          286
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.07,
          41.325,
          27.482,
          288
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.059,
          41.718,
          27.57,
          289
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.021,
          41.659,
          27.54,
          288
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.109,
          41.562,
          27.406,
          289
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.008,
          41.436,
          27.435,
          285
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.103,
          41.732,
          27.474,
          284
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.089,
          41.835,
          27.467,
          283
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.052,
          41.67,
          27.623,
          285
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.203,
          41.72,
          27.55,
          287
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.181,
          41.939,
          27.616,
          286
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.355,
          42.484,
          27.68,
          283
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.227,
          42.085,
          33.443,
          286
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.854,
          41.236,
          27.28,
          287
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.829,
          41.62,
          27.341,
          289
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.78,
          40.869,
          27.076,
          283
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.675,
          40.881,
          27.078,
          287
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.813,
          57.185,
          36.339,
          288
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          40.568,
          42.407,
          27.121,
          197
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.666,
          40.717,
          27.023,
          292
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.633,
          40.655,
          26.994,
          289
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.639,
          40.846,
          27.067,
          288
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.724,
          40.859,
          27.038,
          287
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.652,
          40.807,
          27.097,
          290
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.728,
          41.212,
          27.32,
          292
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.651,
          40.767,
          27.052,
          289
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          33.694,
          42.406,
          27.934,
          198
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.05,
          42.068,
          27.805,
          286
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.913,
          41.864,
          27.658,
          289
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.895,
          41.83,
          27.702,
          287
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.908,
          41.471,
          27.911,
          283
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.01,
          41.446,
          27.786,
          281
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.035,
          41.368,
          27.65,
          282
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          40.054,
          46.375,
          27.586,
          203
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.888,
          41.171,
          27.54,
          281
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.107,
          41.422,
          27.933,
          280
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.168,
          41.445,
          33.569,
          278
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.731,
          41.829,
          27.718,
          281
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.875,
          40.88,
          27.492,
          277
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.744,
          40.82,
          27.511,
          282
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.884,
          40.858,
          27.512,
          279
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.654,
          41.237,
          27.358,
          288
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.427,
          41.365,
          27.399,
          303
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.477,
          41.123,
          27.322,
          291
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.817,
          41.06,
          27.399,
          286
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.206,
          41.579,
          27.534,
          285
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.802,
          41.301,
          27.435,
          287
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.797,
          41.305,
          27.428,
          286
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.776,
          41.005,
          27.49,
          285
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          39.108,
          47.316,
          27.377,
          262
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.75,
          41.388,
          27.366,
          286
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.729,
          41.592,
          27.636,
          285
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.748,
          41.303,
          27.49,
          286
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.915,
          42.435,
          27.545,
          287
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          28.932,
          41.649,
          27.452,
          289
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.883,
          41.697,
          27.47,
          286
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.813,
          41.254,
          27.44,
          285
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.864,
          41.37,
          27.459,
          284
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.862,
          41.311,
          27.441,
          283
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          28.933,
          41.606,
          27.587,
          282
        ],
        "label": -1,
        "species": "Attack / Push Lap"
      },
      {
        "x": [
          29.081,
          41.989,
          28.022,
          284
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.809,
          42.25,
          28.918,
          283
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.403,
          44.146,
          28.096,
          295
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.157,
          42.065,
          27.873,
          287
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.041,
          41.892,
          27.708,
          288
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.011,
          42.269,
          27.614,
          287
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.311,
          42.207,
          28.236,
          285
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.799,
          43.301,
          28.968,
          286
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      },
      {
        "x": [
          29.507,
          42.853,
          28.241,
          290
        ],
        "label": 1,
        "species": "Tire Preservation Lap"
      }
    ]
  },
  "iris": {
    "id": "iris",
    "name": "Iris Flower Benchmark",
    "tag": "Fisher Benchmark",
    "icon": "🌸",
    "description": "Canonical 4-qubit quantum benchmark: Versicolor vs Virginica petal & sepal morphometrics.",
    "featureNames": [
      "Sepal Length (cm)",
      "Sepal Width (cm)",
      "Petal Length (cm)",
      "Petal Width (cm)"
    ],
    "classNames": {
      "1": "Virginica",
      "-1": "Versicolor"
    },
    "defaults": {
      "Versicolor": [
        5.5,
        2.6,
        4,
        1.3
      ],
      "Virginica": [
        6.5,
        3,
        5.5,
        2
      ]
    },
    "data": [
      {
        "x": [
          5.55,
          2.63,
          4.05,
          1.27
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.38,
          2.93,
          5.38,
          1.93
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.54,
          2.63,
          4.04,
          1.27
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.38,
          2.93,
          5.38,
          1.93
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.54,
          2.63,
          4.04,
          1.27
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.37,
          2.93,
          5.37,
          1.93
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.54,
          2.62,
          4.03,
          1.27
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.37,
          2.93,
          5.37,
          1.92
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.53,
          2.62,
          4.03,
          1.26
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.36,
          2.92,
          5.36,
          1.92
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.53,
          2.62,
          4.02,
          1.26
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.36,
          2.92,
          5.36,
          1.92
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.52,
          2.62,
          4.02,
          1.26
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.35,
          2.92,
          5.35,
          1.92
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.52,
          2.61,
          4.02,
          1.26
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.35,
          2.92,
          5.35,
          1.91
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.51,
          2.61,
          4.01,
          1.26
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.34,
          2.91,
          5.34,
          1.91
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.51,
          2.61,
          4.01,
          1.25
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.34,
          2.91,
          5.34,
          1.91
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.51,
          2.6,
          4,
          1.25
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.34,
          2.91,
          5.33,
          1.91
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.5,
          2.6,
          4,
          1.25
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.33,
          2.9,
          5.33,
          1.9
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.5,
          2.6,
          3.99,
          1.25
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.33,
          2.9,
          5.32,
          1.9
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.49,
          2.59,
          3.99,
          1.24
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.32,
          2.9,
          5.32,
          1.9
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.49,
          2.59,
          3.99,
          1.24
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.32,
          2.9,
          5.31,
          1.9
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.48,
          2.59,
          3.98,
          1.24
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.31,
          2.89,
          5.31,
          1.89
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.48,
          2.58,
          3.98,
          1.24
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.31,
          2.89,
          5.3,
          1.89
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.48,
          2.58,
          3.97,
          1.24
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.3,
          2.89,
          5.3,
          1.89
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.47,
          2.58,
          3.97,
          1.23
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.3,
          2.89,
          5.29,
          1.89
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.47,
          2.58,
          3.96,
          1.23
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.29,
          2.88,
          5.29,
          1.88
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.46,
          2.57,
          3.96,
          1.23
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.29,
          2.88,
          5.29,
          1.88
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.46,
          2.57,
          3.96,
          1.23
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.28,
          2.88,
          5.28,
          1.88
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.46,
          2.57,
          3.95,
          1.22
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.28,
          2.88,
          5.28,
          1.88
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.45,
          2.56,
          3.95,
          1.22
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.28,
          2.87,
          5.27,
          1.87
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.45,
          2.56,
          3.94,
          1.22
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.27,
          2.87,
          5.27,
          1.87
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.44,
          2.56,
          3.94,
          1.22
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.27,
          2.87,
          5.26,
          1.87
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.44,
          2.56,
          3.94,
          1.22
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.26,
          2.87,
          5.26,
          1.87
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.44,
          2.55,
          3.93,
          1.21
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.26,
          2.86,
          5.25,
          1.86
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.43,
          2.55,
          3.93,
          1.21
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.25,
          2.86,
          5.25,
          1.86
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.43,
          2.55,
          3.92,
          1.21
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.25,
          2.86,
          5.25,
          1.86
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.42,
          2.54,
          3.92,
          1.21
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.24,
          2.86,
          5.24,
          1.86
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.42,
          2.54,
          3.92,
          1.21
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.24,
          2.85,
          5.24,
          1.85
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.42,
          2.54,
          3.91,
          1.2
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.24,
          2.85,
          5.23,
          1.85
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.41,
          2.54,
          3.91,
          1.2
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.23,
          2.85,
          5.23,
          1.85
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.41,
          2.53,
          3.9,
          1.2
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.23,
          2.85,
          5.22,
          1.85
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.4,
          2.53,
          3.9,
          1.2
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.22,
          2.85,
          5.22,
          1.84
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.4,
          2.53,
          3.9,
          1.2
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.22,
          2.84,
          5.22,
          1.84
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.4,
          2.52,
          3.89,
          1.19
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.22,
          2.84,
          5.21,
          1.84
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.39,
          2.52,
          3.89,
          1.19
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.21,
          2.84,
          5.21,
          1.84
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.39,
          2.52,
          3.89,
          1.19
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.21,
          2.84,
          5.2,
          1.84
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.38,
          2.52,
          3.88,
          1.19
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.2,
          2.83,
          5.2,
          1.83
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.38,
          2.51,
          3.88,
          1.19
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.2,
          2.83,
          5.2,
          1.83
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.38,
          2.51,
          3.87,
          1.18
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.2,
          2.83,
          5.19,
          1.83
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.37,
          2.51,
          3.87,
          1.18
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.19,
          2.83,
          5.19,
          1.83
        ],
        "label": 1,
        "species": "Virginica"
      },
      {
        "x": [
          5.37,
          2.51,
          3.87,
          1.18
        ],
        "label": -1,
        "species": "Versicolor"
      },
      {
        "x": [
          6.19,
          2.83,
          5.19,
          1.82
        ],
        "label": 1,
        "species": "Virginica"
      }
    ]
  },
  "wine": {
    "id": "wine",
    "name": "Wine Chemical Profiling",
    "tag": "UCI ML Dataset",
    "icon": "🍷",
    "description": "Spectrometric chemical analysis of Italian wines: Barolo vs Grignolino cultivars (MLP excels here with >95% accuracy).",
    "featureNames": [
      "Alcohol (%)",
      "Flavanoids",
      "Color Intensity",
      "Proline (mg/L)"
    ],
    "classNames": {
      "1": "Grignolino (Cultivar 2)",
      "-1": "Barolo (Cultivar 1)"
    },
    "defaults": {
      "Barolo": [
        14.2,
        3.1,
        5.6,
        1065
      ],
      "Grignolino": [
        12.3,
        2,
        3,
        510
      ]
    },
    "data": [
      {
        "x": [
          14.23,
          3.06,
          5.64,
          1065
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.2,
          2.76,
          4.38,
          1050
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.16,
          3.24,
          5.68,
          1185
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.37,
          3.49,
          7.8,
          1480
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.24,
          2.69,
          4.32,
          735
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.2,
          3.39,
          6.75,
          1450
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.39,
          2.52,
          5.25,
          1290
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.06,
          2.51,
          5.05,
          1295
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.83,
          2.98,
          5.2,
          1045
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.86,
          3.15,
          7.22,
          1045
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.1,
          3.32,
          5.75,
          1510
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.12,
          2.43,
          5,
          1280
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.75,
          2.76,
          5.6,
          1320
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.75,
          3.69,
          5.4,
          1150
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.38,
          3.64,
          7.5,
          1547
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.63,
          2.91,
          7.3,
          1310
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.3,
          3.14,
          6.2,
          1280
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.83,
          3.4,
          6.6,
          1130
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.19,
          3.93,
          8.7,
          1680
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.64,
          3.03,
          5.1,
          845
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.06,
          3.17,
          5.65,
          780
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          12.93,
          2.41,
          4.5,
          770
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.71,
          2.88,
          3.8,
          1035
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          12.85,
          2.37,
          3.93,
          1015
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.5,
          2.61,
          3.52,
          845
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.05,
          2.68,
          3.58,
          830
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.39,
          2.94,
          4.8,
          1195
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.3,
          2.19,
          3.95,
          1285
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.87,
          2.97,
          4.5,
          915
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.02,
          2.33,
          4.7,
          1035
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.73,
          3.25,
          5.7,
          1285
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.58,
          3.19,
          6.9,
          1515
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.68,
          2.69,
          3.84,
          990
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.76,
          2.74,
          5.4,
          1235
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.51,
          2.53,
          4.2,
          1095
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.48,
          2.98,
          5.1,
          920
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.28,
          2.68,
          4.6,
          880
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.05,
          2.43,
          4.25,
          1105
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.07,
          2.64,
          3.7,
          1020
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.22,
          3.04,
          5.1,
          760
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.56,
          3.29,
          6.13,
          795
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.41,
          2.68,
          4.28,
          1035
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.88,
          3.56,
          5.43,
          1095
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.24,
          2.63,
          4.36,
          680
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.05,
          3,
          5.04,
          885
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.21,
          2.65,
          5.24,
          1080
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.38,
          3.17,
          4.9,
          1065
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.9,
          3.39,
          6.1,
          985
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.1,
          2.92,
          6.2,
          1060
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.94,
          3.54,
          8.9,
          1260
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.05,
          3.27,
          7.2,
          1150
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.83,
          2.99,
          5.6,
          1265
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.82,
          3.74,
          7.05,
          1190
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.77,
          2.79,
          6.3,
          1375
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.74,
          2.9,
          5.85,
          1060
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.56,
          2.78,
          6.25,
          1120
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          14.22,
          3,
          6.38,
          970
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.29,
          3.23,
          6,
          1270
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          13.72,
          3.67,
          6.8,
          1285
        ],
        "label": -1,
        "species": "Cultivar 1 (Barolo)"
      },
      {
        "x": [
          12.37,
          0.57,
          1.95,
          520
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.33,
          1.09,
          3.27,
          680
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.64,
          1.41,
          5.75,
          450
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.67,
          1.79,
          3.8,
          630
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.37,
          3.1,
          4.45,
          420
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.17,
          1.75,
          2.95,
          355
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.37,
          2.65,
          4.6,
          678
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.11,
          3.18,
          5.3,
          502
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.37,
          2,
          4.68,
          510
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.34,
          1.3,
          3.17,
          750
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.21,
          1.28,
          2.85,
          718
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.29,
          1.02,
          3.05,
          870
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.86,
          2.86,
          3.38,
          410
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.49,
          1.84,
          3.74,
          472
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.99,
          2.89,
          3.35,
          985
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.96,
          2.14,
          3.21,
          886
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.66,
          1.57,
          3.8,
          428
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.03,
          2.03,
          4.6,
          392
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.84,
          1.32,
          2.65,
          500
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.33,
          1.85,
          3.4,
          750
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.7,
          2.55,
          2.57,
          463
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12,
          2.26,
          2.5,
          278
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.72,
          2.53,
          3.9,
          714
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.08,
          1.58,
          2.2,
          630
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.05,
          1.59,
          4.8,
          515
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.84,
          2.21,
          3.05,
          520
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.67,
          1.94,
          2.62,
          450
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.16,
          1.69,
          2.45,
          495
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.65,
          1.61,
          2.6,
          562
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.64,
          1.69,
          2.8,
          680
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.08,
          1.59,
          1.74,
          625
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.08,
          1.5,
          2.4,
          480
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12,
          1.25,
          3.6,
          450
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.69,
          1.46,
          3.05,
          495
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.29,
          2.25,
          2.15,
          290
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.62,
          2.26,
          3.25,
          345
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.47,
          2.27,
          2.6,
          937
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.81,
          0.99,
          2.5,
          625
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.29,
          2.5,
          2.9,
          428
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.37,
          3.75,
          4.5,
          660
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.29,
          2.99,
          2.3,
          406
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.08,
          2.17,
          3.3,
          710
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.6,
          1.36,
          2.45,
          562
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.34,
          2.11,
          2.8,
          438
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.82,
          1.64,
          2.06,
          415
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.51,
          1.92,
          2.94,
          672
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.42,
          1.84,
          2.7,
          315
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.25,
          2.03,
          3.4,
          510
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.72,
          1.76,
          3.3,
          488
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.22,
          2.04,
          2.7,
          312
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.61,
          2.92,
          2.65,
          680
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.46,
          2.58,
          2.9,
          562
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.52,
          2.27,
          2,
          325
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.76,
          2.03,
          3.8,
          607
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.41,
          2.01,
          3.08,
          434
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.08,
          2.29,
          2.9,
          385
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.03,
          2.17,
          1.9,
          407
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.82,
          1.6,
          1.95,
          495
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.42,
          2.09,
          2.06,
          345
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.77,
          1.25,
          3.4,
          372
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12,
          1.64,
          1.28,
          564
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.45,
          2.79,
          3.25,
          625
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.56,
          5.08,
          6,
          465
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.42,
          2.13,
          2.08,
          365
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          13.05,
          2.65,
          2.6,
          380
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.87,
          3.03,
          2.8,
          380
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.07,
          2.65,
          2.76,
          378
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.43,
          3.15,
          3.94,
          352
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          11.79,
          2.24,
          3,
          466
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.37,
          2.45,
          2.12,
          342
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      },
      {
        "x": [
          12.04,
          1.75,
          2.6,
          580
        ],
        "label": 1,
        "species": "Cultivar 2 (Grignolino)"
      }
    ]
  },
  "cancer": {
    "id": "cancer",
    "name": "Breast Cancer Wisconsin",
    "tag": "Clinical Diagnostic",
    "icon": "🧬",
    "description": "Cell nuclear morphometric features from fine needle aspirate biopsy: Malignant vs Benign diagnosis.",
    "featureNames": [
      "Mean Radius",
      "Mean Texture",
      "Mean Perimeter",
      "Mean Smoothness"
    ],
    "classNames": {
      "1": "Benign",
      "-1": "Malignant"
    },
    "defaults": {
      "Malignant": [
        17.9,
        21.6,
        122.8,
        0.12
      ],
      "Benign": [
        13.5,
        14.4,
        87.5,
        0.08
      ]
    },
    "data": [
      {
        "x": [
          17.99,
          10.38,
          122.8,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          20.57,
          17.77,
          132.9,
          0.08
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          19.69,
          21.25,
          130,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          11.42,
          20.38,
          77.58,
          0.14
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          20.29,
          14.34,
          135.1,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          12.45,
          15.7,
          82.57,
          0.13
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          18.25,
          19.98,
          119.6,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.71,
          20.83,
          90.2,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13,
          21.82,
          87.5,
          0.13
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          12.46,
          24.04,
          83.97,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          16.02,
          23.24,
          102.7,
          0.08
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          15.78,
          17.89,
          103.6,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          19.17,
          24.8,
          132.4,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          15.85,
          23.95,
          103.7,
          0.08
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.73,
          22.61,
          93.6,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          14.54,
          27.54,
          96.73,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          14.68,
          20.13,
          94.74,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          16.13,
          20.68,
          108.1,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          19.81,
          22.15,
          130,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.54,
          14.36,
          87.46,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          13.08,
          15.71,
          85.63,
          0.11
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          9.5,
          12.44,
          60.34,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          15.34,
          14.26,
          102.5,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          21.16,
          23.04,
          137.2,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          16.65,
          21.38,
          110,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          17.14,
          16.4,
          116,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          14.58,
          21.53,
          97.41,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          18.61,
          20.25,
          122.1,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          15.3,
          25.27,
          102.4,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          17.57,
          15.05,
          115,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          18.63,
          25.11,
          124.8,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          11.84,
          18.7,
          77.93,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          17.02,
          23.98,
          112.8,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          19.27,
          26.47,
          127.9,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          16.13,
          17.88,
          107,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          16.74,
          21.59,
          110.1,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          14.25,
          21.72,
          93.63,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.03,
          18.42,
          82.61,
          0.09
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          14.99,
          25.2,
          95.54,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.48,
          20.82,
          88.4,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.44,
          21.58,
          86.18,
          0.08
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          10.95,
          21.35,
          71.9,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          19.07,
          24.81,
          128.3,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.28,
          20.28,
          87.32,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.17,
          21.81,
          85.42,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          18.65,
          17.6,
          123.7,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          8.2,
          16.84,
          51.71,
          0.09
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          13.17,
          18.66,
          85.98,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          12.05,
          14.63,
          78.04,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          13.49,
          22.3,
          86.91,
          0.09
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          11.76,
          21.6,
          74.72,
          0.09
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          13.64,
          16.34,
          87.21,
          0.08
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          11.94,
          18.24,
          75.71,
          0.08
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          18.22,
          18.7,
          120.3,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          15.1,
          22.02,
          97.26,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          11.52,
          18.75,
          73.34,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          19.21,
          18.57,
          125.5,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          14.71,
          21.59,
          95.55,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.05,
          19.31,
          82.61,
          0.08
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          8.62,
          11.79,
          54.34,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          10.17,
          14.88,
          64.55,
          0.11
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          8.6,
          20.98,
          54.66,
          0.12
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          14.25,
          22.15,
          96.42,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          9.17,
          13.86,
          59.2,
          0.08
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          12.68,
          23.84,
          82.69,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          14.78,
          23.94,
          97.4,
          0.12
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          9.46,
          21.01,
          60.11,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          11.31,
          19.04,
          71.8,
          0.08
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          9.03,
          17.33,
          58.79,
          0.11
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          12.78,
          16.49,
          81.37,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          18.94,
          21.31,
          123.6,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          8.89,
          14.64,
          58.79,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          17.2,
          24.52,
          114.2,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.8,
          15.79,
          90.43,
          0.1
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          12.31,
          16.52,
          79.19,
          0.09
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          16.07,
          19.65,
          104.1,
          0.09
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          13.53,
          10.94,
          87.91,
          0.13
        ],
        "label": 1,
        "species": "Benign"
      },
      {
        "x": [
          18.05,
          16.15,
          120.2,
          0.11
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          20.18,
          23.97,
          143.7,
          0.13
        ],
        "label": -1,
        "species": "Malignant"
      },
      {
        "x": [
          12.86,
          18,
          83.19,
          0.1
        ],
        "label": 1,
        "species": "Benign"
      }
    ]
  }
};

export const DEFAULT_DATASET_KEY = 'f1';
