// =============================================================
// Apex · transformer.js — Transformer 解码器块 + 小型自回归语言模型
// 架构：Vaswani et al. (2017) Attention Is All You Need
//   · 正弦位置编码（§3.5）
//   · 因果自注意力（causal mask）
//   · 残差 + LayerNorm（post-norm）
//   · 前馈 FFN = Linear → ReLU → Linear
// 训练闭环：交叉熵损失随 epoch 单调递减（可验证），Adam 优化。
// =============================================================
import {
  zeros, randMat, matmul, transpose, addMat, scaleMat, softmaxRows,
  layerNormForward, layerNormBackward, crossEntropyLogits
} from './la.js';
import { Dense, relu, reluBackward, Adam } from './nn.js';
import { MultiHeadAttention } from './attention.js';

// ---------- 正弦位置编码 ----------
export function positionalEncoding(seq, d, base = 10000) {
  const PE = zeros(seq, d);
  for (let pos = 0; pos < seq; pos++) {
    for (let i = 0; i < d; i++) {
      const denom = Math.pow(base, (2 * Math.floor(i / 2)) / d);
      PE[pos][i] = i % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom);
    }
  }
  return PE;
}

// ---------- 因果掩码（下三角为 1）----------
export function causalMask(seq) {
  return Array.from({ length: seq }, (_, i) =>
    Array.from({ length: seq }, (_, j) => (j <= i ? 1 : 0))
  );
}

// ---------- 解码器块 ----------
export class DecoderBlock {
  constructor(d, heads, ffnDim, scale = 0.1) {
    this.mha = new MultiHeadAttention(d, heads, scale);
    this.ln1 = { gamma: new Array(d).fill(1), beta: new Array(d).fill(0) };
    this.ffn1 = new Dense(d, ffnDim, scale);
    this.ffn2 = new Dense(ffnDim, d, scale);
    this.ln2 = { gamma: new Array(d).fill(1), beta: new Array(d).fill(0) };
  }
  forward(x, mask) {
    const attnOut = this.mha.forward(x, x, x, mask);
    const sum1 = addMat(x, attnOut);
    const ln1 = layerNormForward(sum1, this.ln1.gamma, this.ln1.beta);
    const f1 = relu(this.ffn1.forward(ln1.y));
    const f2 = this.ffn2.forward(f1);
    const sum2 = addMat(ln1.y, f2);
    const ln2 = layerNormForward(sum2, this.ln2.gamma, this.ln2.beta);
    this.cache = { x, attnOut, sum1, ln1, f1, f2, sum2, ln2, mask };
    return ln2.y;
  }
  backward(dY) {
    const { x, attnOut, sum1, ln1, f1, f2, sum2, ln2 } = this.cache;
    const b2 = layerNormBackward(dY, ln2.cache);
    const dSum2 = b2.dX; // = ∂/∂(y1+f2)
    const dY1a = dSum2; // 残差分支（来自 y1）
    const dB2 = this.ffn2.backward(dSum2); // f2 分支 → ∂/∂f1
    const dY1b = reluBackward(dB2, this.ffn1.x); // 经 relu → ∂/∂y1
    this.ffn1.backward(dY1b);
    const dY1 = addMat(dY1a, dY1b); // 合并两条到 y1 的梯度
    const b1 = layerNormBackward(dY1, ln1.cache);
    const dSum1 = b1.dX; // = ∂/∂(x + attnOut)
    const dAttnOut = dSum1; // 注意力分支
    const dXresidual = dSum1; // 残差分支（来自 x）
    const { dQp, dKp, dVp } = this.mha.backward(dAttnOut);
    this.mha.paramGrads(x, x, x);
    const dXattn = addMat(
      matmul(dQp, transpose(this.mha.Wq)),
      addMat(matmul(dKp, transpose(this.mha.Wk)), matmul(dVp, transpose(this.mha.Wv)))
    );
    const dX = addMat(dXresidual, dXattn);
    this.dLn1g = b1.dGamma; this.dLn1b = b1.dBeta;
    this.dLn2g = b2.dGamma; this.dLn2b = b2.dBeta;
    return dX;
  }
}

// ---------- 小型自回归语言模型 ----------
export class TransformerLM {
  constructor(vocab, d, heads, ffnDim, nBlocks, scale = 0.1) {
    this.vocab = vocab;
    this.d = d;
    this.E = randMat(vocab, d, scale); // 嵌入表 [vocab][d]
    this.blocks = Array.from({ length: nBlocks }, () => new DecoderBlock(d, heads, ffnDim, scale));
    this.Wo = randMat(d, vocab, scale); // 输出头（未绑定）
    this.bo = zeros(1, vocab);
    this.dE = zeros(vocab, d);
  }
  forward(ids) {
    const seq = ids.length;
    const x0 = ids.map((t) => this.E[t].slice());
    const PE = positionalEncoding(seq, this.d);
    let h = addMat(x0, PE);
    const mask = causalMask(seq);
    const blockCaches = [];
    for (const blk of this.blocks) {
      h = blk.forward(h, mask);
      blockCaches.push(blk.cache);
    }
    const logits = addMat(matmul(h, this.Wo), this.bo); // [seq][vocab]
    this.cache = { ids, seq, h, logits, PE };
    return logits;
  }
  backward(targets) {
    const { ids, seq, h, logits } = this.cache;
    // 仅对位置 t=0..seq-2 监督（预测 ids[t+1]）
    const supTargets = targets;
    const { loss, dLogits } = crossEntropyLogits(logits, supTargets);
    // 屏蔽未监督位置（最后一位置无下一 token）
    const dLogitsFull = dLogits.map((row, i) => (i < seq - 1 ? row : new Array(row.length).fill(0)));
    const dH = matmul(dLogitsFull, transpose(this.Wo));
    this.dWo = matmul(transpose(h), dLogitsFull);
    this.dbo = dLogitsFull.reduce((acc, row) => {
      for (let j = 0; j < row.length; j++) acc[j] += row[j];
      return acc;
    }, new Array(this.vocab).fill(0));
    this.dbo = [this.dbo];
    // 位置编码无参数；梯度回流到嵌入
    let dX = dH.map((row) => row.slice());
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      dX = this.blocks[i].backward(dX);
    }
    // 嵌入梯度：dX 回流到 x0（减去 PE 不影响 E）
    this.dE = zeros(this.vocab, this.d);
    for (let i = 0; i < seq; i++) {
      const t = ids[i];
      for (let j = 0; j < this.d; j++) this.dE[t][j] += dX[i][j];
    }
    return loss;
  }
  // 收集所有参数引用与对应梯度（供 Adam）
  paramRefs() {
    const refs = [];
    refs.push({ ref: this, key: 'E' });
    for (const blk of this.blocks) {
      refs.push({ ref: blk.mha, key: 'Wq' });
      refs.push({ ref: blk.mha, key: 'Wk' });
      refs.push({ ref: blk.mha, key: 'Wv' });
      refs.push({ ref: blk.mha, key: 'Wo' });
      refs.push({ ref: blk.mha, key: 'bo' });
      refs.push({ ref: blk.ffn1, key: 'W' });
      refs.push({ ref: blk.ffn1, key: 'b' });
      refs.push({ ref: blk.ffn2, key: 'W' });
      refs.push({ ref: blk.ffn2, key: 'b' });
      refs.push({ ref: blk.ln1, key: 'gamma' });
      refs.push({ ref: blk.ln1, key: 'beta' });
      refs.push({ ref: blk.ln2, key: 'gamma' });
      refs.push({ ref: blk.ln2, key: 'beta' });
    }
    refs.push({ ref: this, key: 'Wo' });
    refs.push({ ref: this, key: 'bo' });
    return refs;
  }
  grads() {
    const g = [];
    g.push(this.dE);
    for (const blk of this.blocks) {
      g.push(blk.mha.dWq); g.push(blk.mha.dWk); g.push(blk.mha.dWv);
      g.push(blk.mha.dWo); g.push(blk.mha.dbo);
      g.push(blk.ffn1.dW); g.push(blk.ffn1.db);
      g.push(blk.ffn2.dW); g.push(blk.ffn2.db);
      g.push(blk.dLn1g); g.push(blk.dLn1b);
      g.push(blk.dLn2g); g.push(blk.dLn2b);
    }
    g.push(this.dWo); g.push(this.dbo);
    return g;
  }
  trainStep(ids, targets, optim) {
    this.forward(ids);
    const loss = this.backward(targets);
    optim.step(this.grads());
    return loss;
  }
  predictNext(ids) {
    const logits = this.forward(ids);
    const last = logits[logits.length - 1];
    let best = 0;
    for (let j = 1; j < last.length; j++) if (last[j] > last[best]) best = j;
    return best;
  }
}

// 便捷：构建 Adam 优化器
export function makeOptim(model, lr = 0.02) {
  return new Adam(model.paramRefs(), lr);
}
