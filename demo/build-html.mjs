// 构建单文件浏览器自检 Demo（零依赖、零重复：直接内联 src/*.js）
// 用法：node demo/build-html.mjs  → 生成 demo/selfcheck.html
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = join(root, 'src');
const files = ['la.js', 'nn.js', 'attention.js', 'transformer.js', 'ml.js'];

// 读取并按依赖顺序拼接，剥离 import/export 关键字 → 浏览器内联
let code = '';
for (const f of files) {
  let src = readFileSync(join(srcDir, f), 'utf8');
  // 移除整个 import ... from '...';（含多行）
  src = src.replace(/import\s*\{[\s\S]*?\}\s*from\s*['"][^'"]*['"]\s*;?/g, '');
  src = src.replace(/import\s+\*\s+as\s+\w+\s+from\s*['"][^'"]*['"]\s*;?/g, '');
  src = src.replace(/export\s+\*\s+from\s*['"][^'"]*['"]\s*;?/g, '');
  // 移除行内/行首 export 关键字（export function/const/class/let/var）
  src = src.replace(/export\s+(function|const|class|let|var)\s+/g, '$1 ');
  src = src.replace(/export\s*\{[\s\S]*?\}\s*;?/g, '');
  code += `\n/* ===== src/${f} ===== */\n` + src;
}

const runtime = `
// ---- 浏览器自检运行时 ----
function relErr(A, B) {
  let num = 0, den = 0;
  for (let i = 0; i < A.length; i++) for (let j = 0; j < A[0].length; j++) {
    num = Math.max(num, Math.abs(A[i][j] - B[i][j]));
    den = Math.max(den, Math.abs(A[i][j]) + Math.abs(B[i][j]));
  }
  return num / (den + 1e-12);
}
function numGradMat(f, M, eps = 1e-5) {
  const g = zeros(M.length, M[0].length);
  for (let i = 0; i < M.length; i++) for (let j = 0; j < M[0].length; j++) {
    const o = M[i][j]; M[i][j] = o + eps; const lp = f();
    M[i][j] = o - eps; const lm = f(); M[i][j] = o;
    g[i][j] = (lp - lm) / (2 * eps);
  }
  return g;
}
function check(name, fn) {
  try { const detail = fn(); return { name, pass: true, detail }; }
  catch (e) { return { name, pass: false, detail: e.message }; }
}

async function runAll() {
  const out = [];
  const lossCurve = [];

  out.push(check('SVD 重构误差 < 1e-7 且奇异向量正交', () => {
    setSeed(1); const A = randMat(6, 4, 1); const { U, Sigma, V, r } = svd(A);
    const S = zeros(r, r); for (let i = 0; i < r; i++) S[i][i] = Sigma[i];
    const recon = matmul(matmul(U, S), transpose(V));
    const e = maxAbsDiff(recon, A);
    if (e >= 1e-7) throw new Error('重构误差=' + e);
    const UtU = matmul(transpose(U), U);
    const o = maxAbsDiff(UtU, identity(r));
    if (o >= 1e-7) throw new Error('U 不正交=' + o);
    return '重构误差=' + e.toExponential(2) + '，UᵀU-I=' + o.toExponential(2);
  }));

  out.push(check('Cholesky A = L Lᵀ', () => {
    setSeed(3); const M = randMat(4, 4, 1); const A = matmul(M, transpose(M));
    for (let i = 0; i < 4; i++) A[i][i] += 0.5;
    const L = cholesky(A); const e = maxAbsDiff(matmul(L, transpose(L)), A);
    if (e >= 1e-9) throw new Error('误差=' + e);
    return '误差=' + e.toExponential(2);
  }));

  out.push(check('Dense 反向梯度检验 relErr < 1e-5', () => {
    setSeed(10); const n = 5, inD = 4, outD = 3; const x = randMat(n, inD, 1);
    const layer = new Dense(inD, outD, 0.5); const G = randMat(n, outD, 1);
    layer.forward(x); layer.backward(G);
    const numW = numGradMat(() => { const o = addMat(matmul(x, layer.W), layer.b); let s = 0; for (let i = 0; i < n; i++) for (let j = 0; j < outD; j++) s += o[i][j] * G[i][j]; return s; }, layer.W);
    const e = relErr(numW, layer.dW); if (e >= 1e-5) throw new Error('dW=' + e);
    return 'dW relErr=' + e.toExponential(2);
  }));

  out.push(check('多头注意力反向梯度检验 relErr < 1e-4', () => {
    setSeed(21); const seq = 4, d = 4, heads = 2; const mha = new MultiHeadAttention(d, heads, 0.5);
    const X = randMat(seq, d, 1); const G = randMat(seq, d, 1); mha.forward(X, X, X, null);
    const { dQp, dKp, dVp } = mha.backward(G); mha.paramGrads(X, X, X);
    const dX = addMat(matmul(dQp, transpose(mha.Wq)), addMat(matmul(dKp, transpose(mha.Wk)), matmul(dVp, transpose(mha.Wv))));
    const numX = numGradMat(() => { const o = mha.forward(X, X, X, null); let s = 0; for (let i = 0; i < seq; i++) for (let j = 0; j < d; j++) s += o[i][j] * G[i][j]; return s; }, X);
    const e = relErr(numX, dX); if (e >= 1e-4) throw new Error('dX=' + e);
    return 'dX relErr=' + e.toExponential(2);
  }));

  out.push(check('Transformer 语言模型损失下降', () => {
    setSeed(30); const vocab = 4;
    const corpus = [[0,1,2,3,0,1,2,3],[1,2,3,0,1,2,3,0],[2,3,0,1,2,3,0,1],[3,0,1,2,3,0,1,2]];
    const model = new TransformerLM(vocab, 16, 2, 32, 2, 0.1); const opt = makeOptim(model, 0.05);
    let first;
    for (let s = 0; s < 400; s++) {
      const seq = corpus[s % corpus.length]; const targets = seq.slice(1).concat(seq[seq.length - 1]);
      const loss = model.trainStep(seq, targets, opt);
      if (s === 0) first = loss; if (s % 20 === 0) lossCurve.push(loss);
    }
    const finalLoss = model.trainStep(corpus[0], corpus[0].slice(1).concat(corpus[0][corpus[0].length - 1]), opt);
    lossCurve.push(finalLoss);
    if (finalLoss >= first * 0.6) throw new Error('损失未明显下降 ' + first.toFixed(3) + '→' + finalLoss.toFixed(3));
    const pred = model.predictNext([0, 1, 2]);
    return '损失 ' + first.toFixed(3) + ' → ' + finalLoss.toFixed(3) + '，[0,1,2]→' + pred;
  }));

  out.push(check('PCA 主方向正确 + 方差解释归一', () => {
    setSeed(40); const u = [1/Math.SQRT2, 1/Math.SQRT2]; const X = [];
    for (let i = 0; i < 200; i++) { const t = (Math.random()-0.5)*10; X.push([t*u[0]+randn()*0.05, t*u[1]+randn()*0.05]); }
    const { components, explainedRatio } = pca(X, 2);
    const dt = Math.abs(components[0][0]*u[0] + components[0][1]*u[1]);
    if (dt < 0.99) throw new Error('方向 dot=' + dt);
    const sum = explainedRatio.reduce((a,b)=>a+b,0);
    if (Math.abs(sum-1) >= 1e-9) throw new Error('解释率和=' + sum);
    return '主方向 dot=' + dt.toFixed(4) + '，top 解释率=' + explainedRatio[0].toFixed(3);
  }));

  out.push(check('GMM-EM 对数似然单调 + 权重归一', () => {
    setSeed(41); const X = [];
    for (let i = 0; i < 100; i++) X.push([randn()*0.5+5, randn()*0.5+5]);
    for (let i = 0; i < 100; i++) X.push([randn()*0.5-5, randn()*0.5-5]);
    const { pi, llTrace } = gmmFit(X, 2, 40, 1e-6, 41);
    const wSum = pi.reduce((a,b)=>a+b,0);
    if (Math.abs(wSum-1) >= 1e-9) throw new Error('权重和=' + wSum);
    let mono = true; for (let i = 1; i < llTrace.length; i++) if (llTrace[i] < llTrace[i-1]-1e-4) mono = false;
    if (!mono) throw new Error('对数似然非单调');
    return '权重和=' + wSum.toFixed(6) + '，LL ' + llTrace[0].toFixed(1) + '→' + llTrace[llTrace.length-1].toFixed(1);
  }));

  out.push(check('kNN 线性可分 100% 准确', () => {
    setSeed(42); const X = [], y = [];
    for (let i = 0; i < 50; i++) { X.push([randn()*0.3+2, randn()*0.3+2]); y.push(0); }
    for (let i = 0; i < 50; i++) { X.push([randn()*0.3-2, randn()*0.3-2]); y.push(1); }
    const knn = new KNN(X, y); let c = 0; for (let i = 0; i < X.length; i++) if (knn.predict(X[i], 3) === y[i]) c++;
    if (c !== X.length) throw new Error(c + '/' + X.length);
    return c + '/' + X.length + ' 正确';
  }));

  return { out, lossCurve };
}

function render(res) {
  const passN = res.out.filter(r => r.pass).length;
  document.getElementById('summary').innerHTML =
    '<b>自检结果：' + passN + ' / ' + res.out.length + ' 项通过</b>';
  const box = document.getElementById('results'); box.innerHTML = '';
  for (const r of res.out) {
    const div = document.createElement('div');
    div.className = 'row ' + (r.pass ? 'ok' : 'bad');
    div.innerHTML = (r.pass ? '✅ ' : '⚠️ ') + '<b>' + r.name + '</b><br><span>' + r.detail + '</span>';
    box.appendChild(div);
  }
  drawLoss(res.lossCurve);
}
function drawLoss(curve) {
  const cv = document.getElementById('loss'); const ctx = cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height); ctx.strokeStyle='#888'; ctx.strokeRect(0,0,cv.width,cv.height);
  const max = Math.max(...curve), min = Math.min(...curve);
  ctx.strokeStyle = '#2e7d32'; ctx.beginPath();
  curve.forEach((v,i) => { const x = (i/(curve.length-1))*cv.width; const y = cv.height - ((v-min)/(max-min||1))*cv.height; i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke();
  ctx.fillStyle='#444'; ctx.fillText('Transformer 训练损失', 8, 14);
}
document.getElementById('run').onclick = async () => {
  document.getElementById('summary').textContent = '运行中…';
  const res = await runAll(); render(res);
};
`;

const html = `<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apex · 世界顶级 AI 核心 · 浏览器自检</title>
<style>
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:880px;margin:24px auto;padding:0 16px;color:#1a1a1a;background:#fafafa}
  h1{font-size:20px;border-bottom:3px solid #2e7d32;padding-bottom:8px}
  button{background:#2e7d32;color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:15px;cursor:pointer}
  button:hover{background:#1b5e20}
  #summary{margin:14px 0;font-size:16px}
  .row{background:#fff;border:1px solid #e0e0e0;border-left:4px solid #ccc;border-radius:6px;padding:10px 12px;margin:8px 0}
  .row.ok{border-left-color:#2e7d32} .row.bad{border-left-color:#c62828}
  .row span{color:#555;font-size:13px}
  canvas{background:#fff;border:1px solid #e0e0e0;border-radius:6px;margin-top:8px}
  code{background:#eee;padding:1px 5px;border-radius:4px}
</style></head>
<body>
  <h1>✦ Apex — 世界顶级 AI 核心 · 浏览器自检</h1>
  <p>零依赖、数值可验证。点击下方按钮在浏览器内运行全部不变量自检（与 <code>node --test</code> 同源）。</p>
  <button id="run">运行自检</button>
  <div id="summary"></div>
  <div id="results"></div>
  <canvas id="loss" width="560" height="240"></canvas>
  <p style="color:#777;font-size:12px">架构：Transformer（Vaswani 2017）+ PCA/SVD（Eckart–Young）+ GMM-EM（Dempster 1977）+ kNN（Cover & Hart 1967）。作者：晨星。</p>
  <script>${code}</script>
  <script>${runtime}</script>
</body></html>`;

writeFileSync(join(root, 'demo', 'selfcheck.html'), html, 'utf8');
console.log('已生成 demo/selfcheck.html，大小', html.length, '字节');
