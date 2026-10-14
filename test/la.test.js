import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  setSeed, randMat, svd, cholesky, jacobiEig, matmul, transpose, zeros,
  layerNormForward, layerNormBackward, softmaxRows, maxAbsDiff, identity
} from '../src/la.js';
import { relErr, numGradMat } from './helpers.js';

test('SVD 重构误差极小 + 左右奇异向量正交', () => {
  setSeed(1);
  const A = randMat(6, 4, 1);
  const { U, Sigma, V, r } = svd(A);
  const S = zeros(r, r);
  for (let i = 0; i < r; i++) S[i][i] = Sigma[i];
  const recon = matmul(matmul(U, S), transpose(V));
  assert.ok(maxAbsDiff(recon, A) < 1e-7, `重构误差应<1e-7，实际=${maxAbsDiff(recon, A)}`);

  // UᵀU = I_r
  const UtU = matmul(transpose(U), U);
  assert.ok(maxAbsDiff(UtU, identity(r)) < 1e-7, `U 应正交，误差=${maxAbsDiff(UtU, identity(r))}`);
  // VᵀV = I_r
  const VtV = matmul(transpose(V), V);
  assert.ok(maxAbsDiff(VtV, identity(r)) < 1e-7, `V 应正交，误差=${maxAbsDiff(VtV, identity(r))}`);
});

test('SVD 在 m<n（矮矩阵）也正确', () => {
  setSeed(2);
  const A = randMat(3, 7, 1);
  const { U, Sigma, V, r } = svd(A);
  const S = zeros(r, r);
  for (let i = 0; i < r; i++) S[i][i] = Sigma[i];
  const recon = matmul(matmul(U, S), transpose(V));
  assert.ok(maxAbsDiff(recon, A) < 1e-7, `矮矩阵重构误差=${maxAbsDiff(recon, A)}`);
});

test('Cholesky 重构 A = L Lᵀ', () => {
  setSeed(3);
  const M = randMat(4, 4, 1);
  const A = matmul(M, transpose(M));
  for (let i = 0; i < 4; i++) A[i][i] += 0.5; // 保证正定
  const L = cholesky(A);
  const recon = matmul(L, transpose(L));
  assert.ok(maxAbsDiff(recon, A) < 1e-9, `Cholesky 重构误差=${maxAbsDiff(recon, A)}`);
});

test('Jacobi 特征值分解匹配已知矩阵 [[2,1],[1,2]] → {3,1}', () => {
  const S = [[2, 1], [1, 2]];
  const { eig } = jacobiEig(S);
  const sorted = [...eig].sort((a, b) => b - a);
  assert.ok(Math.abs(sorted[0] - 3) < 1e-9, `λ1 应≈3，实际=${sorted[0]}`);
  assert.ok(Math.abs(sorted[1] - 1) < 1e-9, `λ2 应≈1，实际=${sorted[1]}`);
});

test('softmax 行和为 1', () => {
  setSeed(4);
  const M = randMat(5, 3, 2);
  const sm = softmaxRows(M);
  for (const row of sm) {
    const s = row.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(s - 1) < 1e-12, `行和应=1，实际=${s}`);
  }
});

test('LayerNorm 反向通过梯度检验', () => {
  setSeed(5);
  const r = 4, d = 5;
  const x = randMat(r, d, 1);
  const gamma = Array.from({ length: d }, () => 1 + Math.random());
  const beta = Array.from({ length: d }, () => Math.random() - 0.5);
  const G = randMat(r, d, 1); // 上游随机梯度
  const fwd = (xx, gg, bb) => {
    const o = layerNormForward(xx, gg, bb).y;
    let s = 0;
    for (let i = 0; i < r; i++) for (let j = 0; j < d; j++) s += o[i][j] * G[i][j];
    return s;
  };
  const { cache } = layerNormForward(x, gamma, beta);
  const bwd = layerNormBackward(G, cache);
  const numX = numGradMat(() => fwd(x, gamma, beta), x);
  assert.ok(relErr(numX, bwd.dX) < 1e-5, `dX 相对误差=${relErr(numX, bwd.dX)}`);
  // dGamma / dBeta 为 1D，逐元素中心差分
  const numG = new Array(d).fill(0);
  const numB = new Array(d).fill(0);
  const eps = 1e-5;
  for (let j = 0; j < d; j++) {
    const og = gamma[j];
    gamma[j] = og + eps; const lp = fwd(x, gamma, beta);
    gamma[j] = og - eps; const lm = fwd(x, gamma, beta);
    gamma[j] = og; numG[j] = (lp - lm) / (2 * eps);
    const ob = beta[j];
    beta[j] = ob + eps; const lpb = fwd(x, gamma, beta);
    beta[j] = ob - eps; const lmb = fwd(x, gamma, beta);
    beta[j] = ob; numB[j] = (lpb - lmb) / (2 * eps);
  }
  for (let j = 0; j < d; j++) assert.ok(Math.abs(numG[j] - bwd.dGamma[j]) < 1e-5, `dGamma 误差@${j}=${Math.abs(numG[j] - bwd.dGamma[j])}`);
  for (let j = 0; j < d; j++) assert.ok(Math.abs(numB[j] - bwd.dBeta[j]) < 1e-5, `dBeta 误差@${j}=${Math.abs(numB[j] - bwd.dBeta[j])}`);
});
