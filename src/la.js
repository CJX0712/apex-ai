// =============================================================
// Apex · la.js — 线性代数与数值方法核心（零依赖，可复现）
// 顶级算法来源：
//   · Jacobi 特征值分解  — Jacobi (1846), Golub & Van Loan (2013) §8.4
//   · SVD (Eckart–Young) — Eckart & Young (1936), GVL §2.4 / §8.6
//   · Cholesky           — Cholesky (1924), GVL §4.2
//   · softmax / layerNorm / cross-entropy — Vaswani et al. (2017) Transformer
// 所有随机性由 seeded RNG 控制 → 可复现（reproducible）。
// =============================================================

// ---------- 可复现随机源（mulberry32, mulberry32 由 <https://github.com>bryc 普及）----------
let _rng = Math.random;
export function setSeed(seed) {
  let a = seed >>> 0;
  _rng = function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function randn() {
  // Box–Muller 变换（Box & Muller 1958），数值稳定
  let u = 0, v = 0;
  while (u === 0) u = _rng();
  while (v === 0) v = _rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- 基础矩阵工具 ----------
export const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
export const ones = (r, c) => Array.from({ length: r }, () => new Array(c).fill(1));
export const identity = (n) => Array.from({ length: n }, (_, i) =>
  Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
);
export function randMat(r, c, scale = 1, gauss = true) {
  return Array.from({ length: r }, () =>
    Array.from({ length: c }, () => (gauss ? randn() * scale : (_rng() * 2 - 1) * scale))
  );
}
export function transpose(A) {
  const r = A.length, c = A[0].length;
  const T = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
  return T;
}
export function matmul(A, B) {
  const r = A.length, k = B.length, c = B[0].length;
  const C = zeros(r, c);
  for (let i = 0; i < r; i++) {
    const Ai = A[i];
    const Ci = C[i];
    for (let p = 0; p < k; p++) {
      const a = Ai[p];
      if (a === 0) continue;
      const Bp = B[p];
      for (let j = 0; j < c; j++) Ci[j] += a * Bp[j];
    }
  }
  return C;
}
// 广播加：bias 形状 [1][c] 加到 [r][c]
export function addMat(A, B) {
  const r = A.length, c = A[0].length;
  const C = zeros(r, c);
  const broad = B.length === 1;
  for (let i = 0; i < r; i++)
    for (let j = 0; j < c; j++) C[i][j] = A[i][j] + (broad ? B[0][j] : B[i][j]);
  return C;
}
export function subMat(A, B) {
  const r = A.length, c = A[0].length;
  const C = zeros(r, c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) C[i][j] = A[i][j] - B[i][j];
  return C;
}
export function scaleMat(A, s) {
  return A.map((row) => row.map((v) => v * s));
}
export function hadamard(A, B) {
  return A.map((row, i) => row.map((v, j) => v * B[i][j]));
}
export function sumRows(A) {
  const c = A[0].length;
  const s = new Array(c).fill(0);
  for (const row of A) for (let j = 0; j < c; j++) s[j] += row[j];
  return [s];
}
export function colSum(A) {
  const r = A.length, c = A[0].length;
  const s = new Array(c).fill(0);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) s[j] += A[i][j];
  return s;
}
export function matVec(A, v) {
  return A.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}
export function maxAbsDiff(A, B) {
  let m = 0;
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j]));
  return m;
}
export function frobenius(A) {
  let s = 0;
  for (const row of A) for (const v of row) s += v * v;
  return Math.sqrt(s);
}

// ---------- softmax（行方向，数值稳定，logsumexp 防溢出）----------
export function softmaxRows(M) {
  return M.map((row) => {
    const m = Math.max(...row);
    const ex = row.map((v) => Math.exp(v - m));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / s);
  });
}
export function logSumExpRows(M) {
  return M.map((row) => {
    const m = Math.max(...row);
    return m + Math.log(row.reduce((a, v) => a + Math.exp(v - m), 0));
  });
}

// ---------- LayerNorm（Ba et al. 2016），返回 cache 供反向 ----------
export function layerNormForward(x, gamma, beta, eps = 1e-5) {
  const r = x.length, d = x[0].length;
  const y = zeros(r, d);
  const mu = new Array(r);
  const varr = new Array(r);
  for (let i = 0; i < r; i++) {
    const row = x[i];
    const m = row.reduce((a, b) => a + b, 0) / d;
    let v = 0;
    for (const val of row) v += (val - m) ** 2;
    v /= d;
    mu[i] = m;
    varr[i] = v;
    const inv = 1 / Math.sqrt(v + eps);
    for (let j = 0; j < d; j++) y[i][j] = (row[j] - m) * inv * gamma[j] + beta[j];
  }
  return { y, cache: { x, gamma, beta, eps, mu, varr } };
}
export function layerNormBackward(dY, cache) {
  const { x, gamma, beta, eps, mu, varr } = cache;
  const r = x.length, d = x[0].length;
  const dX = zeros(r, d);
  const dGamma = new Array(d).fill(0);
  const dBeta = new Array(d).fill(0);
  for (let i = 0; i < r; i++) {
    const inv = 1 / Math.sqrt(varr[i] + eps);
    let dVar = 0, dMu = 0;
    for (let j = 0; j < d; j++) {
      const xhat = (x[i][j] - mu[i]) * inv;
      dGamma[j] += dY[i][j] * xhat;
      dBeta[j] += dY[i][j];
      dVar += dY[i][j] * gamma[j] * (x[i][j] - mu[i]) * -0.5 * inv ** 3;
      dMu += -dY[i][j] * gamma[j] * inv;
    }
    for (let j = 0; j < d; j++) {
      dX[i][j] = dY[i][j] * gamma[j] * inv + dVar * (2 * (x[i][j] - mu[i])) / d + dMu / d;
    }
  }
  return { dX, dGamma, dBeta };
}

// ---------- 交叉熵（对 logits 行，target 为类别索引数组）----------
export function crossEntropyLogits(logits, targets) {
  // 返回 { loss, dLogits }；数值稳定：logsumexp
  const probs = softmaxRows(logits);
  let loss = 0;
  const dLogits = zeros(logits.length, logits[0].length);
  for (let i = 0; i < logits.length; i++) {
    const t = targets[i];
    loss += -Math.log(probs[i][t] + 1e-12);
    for (let j = 0; j < logits[0].length; j++) dLogits[i][j] = probs[i][j] - (j === t ? 1 : 0);
  }
  return { loss: loss / logits.length, dLogits: scaleMat(dLogits, 1 / logits.length) };
}

// =============================================================
//  顶级数值：Jacobi 对称特征值分解（Golub & Van Loan §8.4）
// =============================================================
export function jacobiEig(S, maxSweeps = 200, tol = 1e-13) {
  const n = S.length;
  let A = S.map((row) => row.slice());
  let V = identity(n);
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < tol) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
        const c = Math.cos(theta), s = Math.sin(theta);
        // 旋转 A 的第 p,q 列与行
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        // 累积特征向量
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const eig = [];
  for (let i = 0; i < n; i++) eig.push(A[i][i]);
  return { eig, V };
}

// =============================================================
//  顶级数值：SVD（Eckart–Young / GVL §8.6）
//  A (m×n) → U (m×r) Σ (r) V (n×r)，满足 A ≈ U diag(Σ) Vᵀ
// =============================================================
export function svd(A) {
  const m = A.length, n = A[0].length;
  let U, Sigma, V, r;
  if (m >= n) {
    // 右奇异向量 = AᵀA 的特征向量
    const AtA = matmul(transpose(A), A); // n×n
    const { eig, V: Veig } = jacobiEig(AtA);
    const pairs = eig.map((l, i) => ({ s: Math.sqrt(Math.max(l, 0)), v: Veig.map((row) => row[i]) }));
    pairs.sort((a, b) => b.s - a.s);
    r = Math.max(1, pairs.filter((p) => p.s > 1e-9).length);
    Sigma = pairs.slice(0, r).map((p) => p.s);
    const Vr = transpose(pairs.slice(0, r).map((p) => p.v)); // n×r，列为右奇异向量
    const Utmp = zeros(m, r);
    for (let i = 0; i < r; i++) {
      const Av = matVec(A, pairs[i].v); // m×1
      for (let k = 0; k < m; k++) Utmp[k][i] = Av[k] / Sigma[i];
    }
    U = Utmp; V = Vr;
  } else {
    // 左奇异向量 = A Aᵀ 的特征向量
    const AAt = matmul(A, transpose(A)); // m×m
    const { eig, V: Ueig } = jacobiEig(AAt);
    const pairs = eig.map((l, i) => ({ s: Math.sqrt(Math.max(l, 0)), u: Ueig.map((row) => row[i]) }));
    pairs.sort((a, b) => b.s - a.s);
    r = Math.max(1, pairs.filter((p) => p.s > 1e-9).length);
    Sigma = pairs.slice(0, r).map((p) => p.s);
    const Ur = transpose(pairs.slice(0, r).map((p) => p.u)); // m×r，列为左奇异向量
    const Vtmp = zeros(n, r);
    for (let i = 0; i < r; i++) {
      const Atu = matVec(transpose(A), pairs[i].u); // n×1
      for (let k = 0; k < n; k++) Vtmp[k][i] = Atu[k] / Sigma[i];
    }
    U = Ur; V = Vtmp;
  }
  return { U, Sigma, V, r };
}

// =============================================================
//  顶级数值：Cholesky 分解（GVL §4.2）A = L Lᵀ，A 对称正定
// =============================================================
export function cholesky(A) {
  const n = A.length;
  const L = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error('Cholesky: 矩阵非正定');
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}
