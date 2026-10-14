// =============================================================
// Apex · attention.js — 缩放点积注意力 + 多头注意力（零依赖）
// 算法：Vaswani et al. (2017) "Attention Is All You Need" §3.2
// 前向/反向均为解析梯度，可通过梯度检验交叉验证。
// 反向推导核心（行方向 softmax 的 Jacobian）：
//   dScores = attn ⊙ (dAttn − (attn · dAttn)[:,None])
// =============================================================
import { zeros, randMat, matmul, transpose, addMat, scaleMat, softmaxRows } from './la.js';

// ---------- 缩放点积注意力（单头）----------
export function scaledDotProductAttention(Q, K, V, mask = null, d = null) {
  const seq = Q.length;
  const dh = d ?? Q[0].length;
  const sqrtD = Math.sqrt(dh);
  // scores = Q Kᵀ / √d
  const scores = matmul(Q, transpose(K)).map((row) => row.map((v) => v / sqrtD));
  if (mask) {
    for (let i = 0; i < seq; i++)
      for (let j = 0; j < seq; j++) if (mask[i][j] === 0) scores[i][j] = -1e9;
  }
  const attn = softmaxRows(scores); // [seq][seq] 行归一
  const out = matmul(attn, V); // [seq][dh]
  return { out, cache: { Q, K, V, attn, scores, mask, sqrtD } };
}

export function attentionBackward(dOut, cache) {
  const { Q, K, V, attn, mask, sqrtD } = cache;
  const seq = Q.length;
  // dV = attnᵀ dOut
  const dV = matmul(transpose(attn), dOut);
  // dAttn = dOut Vᵀ
  const dAttn = matmul(dOut, transpose(V));
  // dScores = attn ⊙ (dAttn − rowsum(attn⊙dAttn)[:,None])
  const dScores = zeros(seq, seq);
  for (let i = 0; i < seq; i++) {
    let s = 0;
    for (let j = 0; j < seq; j++) s += attn[i][j] * dAttn[i][j];
    for (let j = 0; j < seq; j++) {
      let v = attn[i][j] * (dAttn[i][j] - s);
      if (mask && mask[i][j] === 0) v = 0;
      dScores[i][j] = v;
    }
  }
  // dQ = dScores K / √d ; dK = dScoresᵀ Q / √d
  const dQ = scaleMat(matmul(dScores, K), 1 / sqrtD);
  const dK = scaleMat(matmul(transpose(dScores), Q), 1 / sqrtD);
  return { dQ, dK, dV };
}

// ---------- 多头注意力（Multi-Head Attention）----------
export class MultiHeadAttention {
  constructor(dModel, heads, scale = 0.1) {
    this.d = dModel;
    this.h = heads;
    this.dh = dModel / heads;
    this.Wq = randMat(dModel, dModel, scale);
    this.Wk = randMat(dModel, dModel, scale);
    this.Wv = randMat(dModel, dModel, scale);
    this.Wo = randMat(dModel, dModel, scale);
    this.bo = zeros(1, dModel);
  }
  forward(Q, K, V, mask = null) {
    const seq = Q.length;
    const dh = this.dh;
    const Qp = matmul(Q, this.Wq);
    const Kp = matmul(K, this.Wk);
    const Vp = matmul(V, this.Wv);
    // 切分为头
    const Qh = [], Kh = [], Vh = [], caches = [];
    for (let k = 0; k < this.h; k++) {
      const c = (i) => i * dh;
      Qh.push(Qp.map((row) => row.slice(c(k), c(k) + dh)));
      Kh.push(Kp.map((row) => row.slice(c(k), c(k) + dh)));
      Vh.push(Vp.map((row) => row.slice(c(k), c(k) + dh)));
    }
    const headOuts = [];
    for (let k = 0; k < this.h; k++) {
      const { out, cache } = scaledDotProductAttention(Qh[k], Kh[k], Vh[k], mask, dh);
      headOuts.push(out);
      caches.push(cache);
    }
    // 拼接头（沿特征维）
    const concat = headOuts[0].map((_, i) => {
      const row = [];
      for (let k = 0; k < this.h; k++) row.push(...headOuts[k][i]);
      return row;
    });
    const Y = addMat(matmul(concat, this.Wo), this.bo);
    this.cache = { Qp, Kp, Vp, concat, caches, mask, seq, dh };
    return Y;
  }
  backward(dY) {
    const { Qp, Kp, Vp, concat, caches, mask, seq, dh } = this.cache;
    const dConcat = matmul(dY, transpose(this.Wo));
    this.dWo = matmul(transpose(concat), dY);
    this.dbo = dY.reduce((acc, row) => {
      for (let j = 0; j < row.length; j++) acc[j] += row[j];
      return acc;
    }, new Array(dY[0].length).fill(0));
    this.dbo = [this.dbo];
    const dQp = zeros(seq, this.d), dKp = zeros(seq, this.d), dVp = zeros(seq, this.d);
    for (let k = 0; k < this.h; k++) {
      const dOutK = dConcat.map((row) => row.slice(k * dh, k * dh + dh));
      const { dQ, dK, dV } = attentionBackward(dOutK, caches[k]);
      for (let i = 0; i < seq; i++) {
        for (let j = 0; j < dh; j++) {
          dQp[i][k * dh + j] += dQ[i][j];
          dKp[i][k * dh + j] += dK[i][j];
          dVp[i][k * dh + j] += dV[i][j];
        }
      }
    }
    // 投影层的反向
    const Q = Qp; // 注意：调用方需提供原始 Q，这里用 Qp 反推需原始输入
    // 返回 dQ,dK,dV 给输入；参数梯度由 dQp 等计算需原始 Q,K,V
    this._dQp = dQp;
    this._dKp = dKp;
    this._dVp = dVp;
    return { dQp, dKp, dVp };
  }
  // 计算投影参数梯度（需原始输入 Xq,Xk,Xv）
  paramGrads(Xq, Xk, Xv) {
    this.dWq = matmul(transpose(Xq), this._dQp);
    this.dWk = matmul(transpose(Xk), this._dKp);
    this.dWv = matmul(transpose(Xv), this._dVp);
  }
}
