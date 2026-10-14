// =============================================================
// Apex · ml.js — 经典顶级算法（集大成，可验证不变量）
//   · PCA   — 基于 SVD（Eckart–Young / GVL §8.6），最优低秩重构
//   · GMM   — EM 算法（Dempster, Laird & Rubin 1977），对数似然单调
//   · kNN   — 惰性学习（Cover & Hart 1967）
// 每个算法暴露可被独立实现交叉验证的「不变量」。
// =============================================================
import { zeros, matmul, transpose, svd, cholesky, setSeed } from './la.js';

// =============================================================
// PCA（SVD 分解，数值最稳定；优于特征分解协方差）
// =============================================================
export function pca(X, k) {
  const n = X.length, d = X[0].length;
  const mean = new Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) mean[j] += row[j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  const Xc = X.map((row) => row.map((v, j) => v - mean[j]));
  const { U, Sigma, V } = svd(Xc); // V 为 d×r，列为右奇异向量（主成分）
  const r = V[0].length;
  const kk = k ?? r;
  const comps = transpose(V).slice(0, kk); // kk×d，每行为一个主成分
  const scores = matmul(Xc, V.slice ? V : V); // 见下：取前 kk 列
  // 取 V 前 kk 列投影
  const Vk = V.map((row) => row.slice(0, kk)); // d×kk
  const Z = matmul(Xc, Vk); // n×kk
  const totalVar = Sigma.reduce((s, v) => s + v * v, 0);
  const explained = Sigma.slice(0, kk).map((v) => (v * v) / totalVar);
  return { mean, components: comps, scores: Z, singularValues: Sigma.slice(0, kk), explainedRatio: explained, V: Vk };
}

// =============================================================
// GMM — EM 算法（Dempster, Laird & Rubin 1977）
// 不变量：① 责任度每行 Σ=1；② 混合权重 Σ=1；③ 对数似然单调非减
// =============================================================
function forwardSolve(L, b) {
  const n = L.length;
  const z = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) s -= L[i][j] * z[j];
    z[i] = s / L[i][i];
  }
  return z;
}
function logMVN(x, mu, L) {
  const d = x.length;
  const diff = x.map((v, i) => v - mu[i]);
  const z = forwardSolve(L, diff);
  let quad = 0;
  for (const v of z) quad += v * v;
  let logdet = 0;
  for (let i = 0; i < d; i++) logdet += Math.log(L[i][i]);
  logdet *= 2;
  return -0.5 * (d * Math.log(2 * Math.PI) + logdet + quad);
}
export function gmmFit(X, k, iters = 50, reg = 1e-6, seed = 42) {
  setSeed(seed);
  const n = X.length, d = X[0].length;
  // 初始化：均值取随机样本点，协方差取各维方差
  const meanAll = new Array(d).fill(0);
  for (const row of X) for (let j = 0; j < d; j++) meanAll[j] += row[j];
  for (let j = 0; j < d; j++) meanAll[j] /= n;
  let varAll = 0;
  for (const row of X) for (let j = 0; j < d; j++) varAll += (row[j] - meanAll[j]) ** 2;
  varAll = (varAll / (n * d)) || 1;
  const idxs = [];
  for (let i = 0; i < k; i++) idxs.push(Math.floor(Math.random() * n));
  const mu = idxs.map((i) => X[i].slice());
  const Sigma = Array.from({ length: k }, () => {
    const S = zeros(d, d);
    for (let i = 0; i < d; i++) S[i][i] = varAll;
    return S;
  });
  let pi = new Array(k).fill(1 / k);
  const llTrace = [];
  const L = Sigma.map((S) => {
    const Sreg = S.map((row, i) => row.map((v, j) => (i === j ? v + reg : v)));
    return cholesky(Sreg);
  });
  for (let it = 0; it < iters; it++) {
    // E-step
    const logR = zeros(n, k);
    let ll = 0;
    for (let i = 0; i < n; i++) {
      const ls = [];
      for (let c = 0; c < k; c++) ls.push(Math.log(pi[c] + 1e-12) + logMVN(X[i], mu[c], L[c]));
      const m = Math.max(...ls);
      const sumExp = ls.reduce((a, v) => a + Math.exp(v - m), 0);
      ll += m + Math.log(sumExp);
      for (let c = 0; c < k; c++) logR[i][c] = Math.exp(ls[c] - m) / sumExp;
    }
    llTrace.push(ll);
    // M-step
    const Nk = new Array(k).fill(0);
    for (let i = 0; i < n; i++) for (let c = 0; c < k; c++) Nk[c] += logR[i][c];
    for (let c = 0; c < k; c++) pi[c] = Nk[c] / n;
    for (let c = 0; c < k; c++) {
      const muC = new Array(d).fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) muC[j] += logR[i][c] * X[i][j];
      for (let j = 0; j < d; j++) muC[j] /= Nk[c] || 1e-12;
      mu[c] = muC;
      const S = zeros(d, d);
      for (let i = 0; i < n; i++) {
        const diff = X[i].map((v, j) => v - muC[j]);
        for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] += logR[i][c] * diff[a] * diff[b];
      }
      for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) S[a][b] /= Nk[c] || 1e-12;
      for (let a = 0; a < d; a++) S[a][a] += reg;
      Sigma[c] = S;
      L[c] = cholesky(S);
    }
  }
  return { pi, mu, Sigma, llTrace };
}

// =============================================================
// kNN — 惰性学习（Cover & Hart 1967），分类
// =============================================================
export class KNN {
  constructor(X, y) {
    this.X = X;
    this.y = y;
  }
  _dist(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s);
  }
  predict(q, k = 3) {
    const dists = this.X.map((x, i) => ({ i, d: this._dist(q, x) }));
    dists.sort((a, b) => a.d - b.d);
    const top = dists.slice(0, k);
    const votes = new Map();
    for (const { i } of top) votes.set(this.y[i], (votes.get(this.y[i]) || 0) + 1);
    let best = null, bestC = -1;
    for (const [cls, c] of votes) if (c > bestC) { bestC = c; best = cls; }
    return best;
  }
}
