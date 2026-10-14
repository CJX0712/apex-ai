import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setSeed, randMat, matmul, transpose, addMat, zeros } from '../src/la.js';
import { scaledDotProductAttention, attentionBackward, MultiHeadAttention } from '../src/attention.js';
import { relErr, numGradMat } from './helpers.js';

test('缩放点积注意力反向通过梯度检验', () => {
  setSeed(20);
  const seq = 4, d = 3;
  const Q = randMat(seq, d, 1);
  const K = randMat(seq, d, 1);
  const V = randMat(seq, d, 1);
  const G = randMat(seq, d, 1);
  const { out, cache } = scaledDotProductAttention(Q, K, V, null, d);
  const bwd = attentionBackward(G, cache);
  const f = (qq, kk, vv) => { const o = scaledDotProductAttention(qq, kk, vv, null, d).out; let s = 0; for (let i = 0; i < seq; i++) for (let j = 0; j < d; j++) s += o[i][j] * G[i][j]; return s; };
  const numQ = numGradMat(() => f(Q, K, V), Q);
  const numK = numGradMat(() => f(Q, K, V), K);
  const numV = numGradMat(() => f(Q, K, V), V);
  assert.ok(relErr(numQ, bwd.dQ) < 1e-5, `dQ 误差=${relErr(numQ, bwd.dQ)}`);
  assert.ok(relErr(numK, bwd.dK) < 1e-5, `dK 误差=${relErr(numK, bwd.dK)}`);
  assert.ok(relErr(numV, bwd.dV) < 1e-5, `dV 误差=${relErr(numV, bwd.dV)}`);
});

test('多头注意力反向（输入与 Wq）通过梯度检验', () => {
  setSeed(21);
  const seq = 4, d = 4, heads = 2;
  const mha = new MultiHeadAttention(d, heads, 0.5);
  const X = randMat(seq, d, 1);
  const G = randMat(seq, d, 1);
  const Y = mha.forward(X, X, X, null);
  const { dQp, dKp, dVp } = mha.backward(G);
  mha.paramGrads(X, X, X);
  const dX = addMat(
    matmul(dQp, transpose(mha.Wq)),
    addMat(matmul(dKp, transpose(mha.Wk)), matmul(dVp, transpose(mha.Wv)))
  );
  const f = (xx) => { const o = mha.forward(xx, xx, xx, null); let s = 0; for (let i = 0; i < seq; i++) for (let j = 0; j < d; j++) s += o[i][j] * G[i][j]; return s; };
  const numX = numGradMat(() => f(X), X);
  const numWq = numGradMat(() => f(X), mha.Wq);
  assert.ok(relErr(numX, dX) < 1e-4, `dX 误差=${relErr(numX, dX)}`);
  assert.ok(relErr(numWq, mha.dWq) < 1e-4, `dWq 误差=${relErr(numWq, mha.dWq)}`);
});
