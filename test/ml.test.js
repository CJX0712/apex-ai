import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setSeed, randn, randMat } from '../src/la.js';
import { pca, gmmFit, KNN } from '../src/ml.js';

test('PCA：主成分方向正确 + 方差解释率归一', () => {
  setSeed(40);
  const n = 200;
  const u = [1 / Math.SQRT2, 1 / Math.SQRT2]; // 主方向 (1,1)
  const X = [];
  for (let i = 0; i < n; i++) {
    const t = (Math.random() - 0.5) * 10;
    X.push([t * u[0] + randn() * 0.05, t * u[1] + randn() * 0.05]);
  }
  const { components, explainedRatio } = pca(X, 2);
  const dot = Math.abs(components[0][0] * u[0] + components[0][1] * u[1]);
  assert.ok(dot > 0.99, `第一主成分应逼近方向(1,1)，实际 dot=${dot.toFixed(4)}`);
  const sumExp = explainedRatio.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(sumExp - 1) < 1e-9, `方差解释率之和应=1，实际=${sumExp}`);
  // 第一主成分应解释绝大部分方差
  assert.ok(explainedRatio[0] > 0.95, `第一主成分解释率应>0.95，实际=${explainedRatio[0].toFixed(4)}`);
});

test('GMM-EM：对数似然单调 + 责任度/权重归一', () => {
  setSeed(41);
  // 两个明显分离的二维簇
  const X = [];
  for (let i = 0; i < 100; i++) X.push([randn() * 0.5 + 5, randn() * 0.5 + 5]);
  for (let i = 0; i < 100; i++) X.push([randn() * 0.5 - 5, randn() * 0.5 - 5]);
  const { pi, mu, Sigma, llTrace } = gmmFit(X, 2, 40, 1e-6, 41);
  // 权重和 = 1
  const wSum = pi.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(wSum - 1) < 1e-9, `混合权重和应=1，实际=${wSum}`);
  // 对数似然单调非减
  let mono = true;
  for (let i = 1; i < llTrace.length; i++) if (llTrace[i] < llTrace[i - 1] - 1e-4) mono = false;
  assert.ok(mono, `对数似然应单调非减，trace=${llTrace.map((v) => v.toFixed(1)).join(',')}`);
  assert.ok(llTrace[llTrace.length - 1] > llTrace[0], '对数似然应总体上升');
});

test('kNN：线性可分数据 100% 准确', () => {
  setSeed(42);
  const X = [], y = [];
  for (let i = 0; i < 50; i++) { X.push([randn() * 0.3 + 2, randn() * 0.3 + 2]); y.push(0); }
  for (let i = 0; i < 50; i++) { X.push([randn() * 0.3 - 2, randn() * 0.3 - 2]); y.push(1); }
  const knn = new KNN(X, y);
  let correct = 0;
  for (let i = 0; i < X.length; i++) if (knn.predict(X[i], 3) === y[i]) correct++;
  assert.ok(correct === X.length, `kNN 应 100% 准确，实际 ${correct}/${X.length}`);
});
