import { test } from 'node:test';
import assert from 'node:assert/strict';
import { setSeed } from '../src/la.js';
import { TransformerLM, makeOptim } from '../src/transformer.js';

test('Transformer 语言模型：交叉熵损失随训练单调下降', () => {
  setSeed(30);
  const vocab = 4;
  // 强规律语料：循环 0,1,2,3,0,1,2,3...
  const corpus = [
    [0, 1, 2, 3, 0, 1, 2, 3],
    [1, 2, 3, 0, 1, 2, 3, 0],
    [2, 3, 0, 1, 2, 3, 0, 1],
    [3, 0, 1, 2, 3, 0, 1, 2]
  ];
  const model = new TransformerLM(vocab, 16, 2, 32, 2, 0.1);
  const opt = makeOptim(model, 0.05);
  const losses = [];
  for (let step = 0; step < 400; step++) {
    const seq = corpus[step % corpus.length];
    const targets = seq.slice(1).concat(seq[seq.length - 1]); // 预测下一 token；末位自预测
    const loss = model.trainStep(seq, targets, opt);
    if (step % 40 === 0) losses.push(loss);
  }
  const finalLoss = model.trainStep(corpus[0], corpus[0].slice(1).concat(corpus[0][corpus[0].length - 1]), opt);
  losses.push(finalLoss);
  assert.ok(finalLoss < 1.2, `最终损失应明显下降，实际=${finalLoss.toFixed(4)}`);
  assert.ok(losses[losses.length - 1] < losses[0] * 0.6, `损失应由 ${losses[0].toFixed(4)} 降至 ${losses[losses.length - 1].toFixed(4)}`);
  // 学习到规律：给定 [0,1,2] 应预测 3
  const pred = model.predictNext([0, 1, 2]);
  assert.ok(pred === 3, `给定 [0,1,2] 应预测 3，实际=${pred}`);
});
