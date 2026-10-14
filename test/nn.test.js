import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setSeed, randMat, matmul, transpose, zeros, addMat } from '../src/la.js';
import { Dense, MLP, relu, mseLoss, Adam } from '../src/nn.js';
import { relErr, numGradMat } from './helpers.js';

test('Dense 层解析梯度通过梯度检验', () => {
  setSeed(10);
  const n = 5, inD = 4, outD = 3;
  const x = randMat(n, inD, 1);
  const layer = new Dense(inD, outD, 0.5);
  const G = randMat(n, outD, 1);
  const y = layer.forward(x);
  const dX = layer.backward(G);
  // 数值梯度
  const numW = numGradMat(() => { const o = addMat(matmul(x, layer.W), layer.b); let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < outD; j++) s += o[i][j] * G[i][j]; return s; }, layer.W);
  const numB = numGradMat(() => { const o = addMat(matmul(x, layer.W), layer.b); let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < outD; j++) s += o[i][j] * G[i][j]; return s; }, layer.b);
  const numX = numGradMat(() => { const o = addMat(matmul(x, layer.W), layer.b); let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < outD; j++) s += o[i][j] * G[i][j]; return s; }, x);
  assert.ok(relErr(numW, layer.dW) < 1e-5, `dW 误差=${relErr(numW, layer.dW)}`);
  assert.ok(relErr(numB, layer.db) < 1e-5, `db 误差=${relErr(numB, layer.db)}`);
  assert.ok(relErr(numX, dX) < 1e-5, `dX 误差=${relErr(numX, dX)}`);
});

test('MLP 学习 XOR：损失显著下降', () => {
  setSeed(11);
  const X = [
    [-1, -1], [-1, 1], [1, -1], [1, 1]
  ];
  const Y = [[0], [1], [1], [0]];
  const mlp = new MLP([2, 8, 1], 0.5);
  const params = mlp.denseLayers().flatMap((L) => [{ ref: L, key: 'W' }, { ref: L, key: 'b' }]);
  const opt = new Adam(params, 0.05);
  const initial = (() => { const p = mlp.forward(X); return mseLoss(p, Y).loss; })();
  for (let e = 0; e < 2000; e++) {
    const pred = mlp.forward(X);
    const { loss, dPred } = mseLoss(pred, Y);
    mlp.backward(dPred);
    opt.step(mlp.denseLayers().flatMap((L) => [L.dW, L.db]));
  }
  const final = mseLoss(mlp.forward(X), Y).loss;
  assert.ok(final < initial * 0.3, `XOR 损失应由 ${initial.toFixed(4)} 降至 ${final.toFixed(4)}`);
  // 验证预测符号正确
  const pred = mlp.forward(X);
  const ok = pred.every((p, i) => (p[0] > 0.5 ? 1 : 0) === Y[i][0]);
  assert.ok(ok, 'XOR 分类应在训练后全对');
});
