// =============================================================
// Apex · nn.js — 神经网络原语与优化（零依赖）
// Dense 层前向/反向（解析梯度）、ReLU、MLP、Adam 优化器（Kingma & Ba 2015）、
// MSE 损失。所有反向实现均可通过梯度检验（gradcheck）交叉验证。
// =============================================================
import { zeros, randMat, matmul, transpose, addMat, scaleMat, hadamard } from './la.js';

// ---------- Dense 全连接层：y = x W + b ----------
export class Dense {
  constructor(inDim, outDim, scale = 0.1) {
    this.W = randMat(inDim, outDim, scale);
    this.b = zeros(1, outDim);
  }
  forward(x) {
    this.x = x; // [n][inDim]
    return addMat(matmul(x, this.W), this.b); // 广播 b
  }
  backward(dOut) {
    // dOut: [n][outDim]
    this.dW = matmul(transpose(this.x), dOut); // [inDim][outDim]
    this.db = dOut.reduce((acc, row) => {
      for (let j = 0; j < row.length; j++) acc[j] += row[j];
      return acc;
    }, new Array(dOut[0].length).fill(0));
    this.db = [this.db]; // [1][outDim]
    return matmul(dOut, transpose(this.W)); // dX
  }
  params() {
    return [this.W, this.b];
  }
  grads() {
    return [this.dW, this.db];
  }
  setParams(W, b) {
    this.W = W;
    this.b = b;
  }
}

// ---------- ReLU 激活（无状态）----------
export const relu = (x) => x.map((row) => row.map((v) => (v > 0 ? v : 0)));
export function reluBackward(dY, x) {
  return dY.map((row, i) => row.map((v, j) => (x[i][j] > 0 ? v : 0)));
}

// ---------- MLP（可堆叠 Dense+ReLU），含反向 ----------
export class MLP {
  constructor(layerSizes, scale = 0.1) {
    this.layers = [];
    for (let i = 0; i < layerSizes.length - 1; i++) {
      this.layers.push(new Dense(layerSizes[i], layerSizes[i + 1], scale));
      if (i < layerSizes.length - 2) this.layers.push({ relu: true });
    }
  }
  forward(x) {
    this.caches = [];
    let h = x;
    for (const L of this.layers) {
      if (L.relu) {
        this.caches.push(h);
        h = relu(h);
      } else {
        h = L.forward(h);
      }
    }
    return h;
  }
  backward(dOut) {
    let d = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const L = this.layers[i];
      if (L.relu) {
        d = reluBackward(d, this.caches.pop());
      } else {
        d = L.backward(d);
      }
    }
    return d;
  }
  denseLayers() {
    return this.layers.filter((L) => !L.relu);
  }
}

// ---------- MSE 损失（含梯度）----------
export function mseLoss(pred, target) {
  const n = pred.length, d = pred[0].length;
  let loss = 0;
  const dPred = zeros(n, d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) {
      const e = pred[i][j] - target[i][j];
      loss += e * e;
      dPred[i][j] = (2 * e) / (n * d);
    }
  return { loss: loss / (n * d), dPred };
}

// ---------- Adam 优化器（Kingma & Ba 2015）----------
export class Adam {
  constructor(params, lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.params = params; // [{ref:obj, key:'W'|'b'}]
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.t = 0;
    this.m = params.map(() => null);
    this.v = params.map(() => null);
  }
  step(grads) {
    this.t++;
    const { beta1, beta2, eps, lr } = this;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      const g = grads[i];
      const r = p.ref[p.key];
      const is2D = Array.isArray(r[0]);
      const rows = r.length;
      const cols = is2D ? r[0].length : 1;
      const m = this.m[i] || (is2D ? zeros(rows, cols) : new Array(rows).fill(0));
      const v = this.v[i] || (is2D ? zeros(rows, cols) : new Array(rows).fill(0));
      const nh = is2D ? zeros(rows, cols) : new Array(rows).fill(0);
      for (let a = 0; a < rows; a++)
        for (let b = 0; b < cols; b++) {
          const ga = is2D ? g[a][b] : g[a];
          const ra = is2D ? r[a][b] : r[a];
          const ma = beta1 * (is2D ? m[a][b] : m[a]) + (1 - beta1) * ga;
          const va = beta2 * (is2D ? v[a][b] : v[a]) + (1 - beta2) * ga * ga;
          const mHat = ma / (1 - Math.pow(beta1, this.t));
          const vHat = va / (1 - Math.pow(beta2, this.t));
          const na = ra - (lr * mHat) / (Math.sqrt(vHat) + eps);
          if (is2D) {
            m[a][b] = ma;
            v[a][b] = va;
            nh[a][b] = na;
          } else {
            m[a] = ma;
            v[a] = va;
            nh[a] = na;
          }
        }
      p.ref[p.key] = nh;
      this.m[i] = m;
      this.v[i] = v;
    }
  }
}
