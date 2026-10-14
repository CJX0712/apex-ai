# ✦ Apex — 世界顶级 AI 核心

> **零依赖 · 数值可验证 · 可复现 · 可运行** 的 AI 核心：把当代 AI 的统一 backbone（**Transformer**）与世界顶级经典算法（**PCA/SVD · GMM-EM · kNN · Adam**）集大成于一处。

作者：**晨星**（GitHub: [CJX0712](https://github.com/CJX0712)） · 许可：MIT

---

## 设计原则（零重写 / 集大成）

| 原则 | 落地 |
|---|---|
| **零重写** | 不重复造轮子。直接复用并正确实现世界顶级数学与算法：Jacobi 分解、Eckart–Young SVD、Cholesky、Transformer（Vaswani 2017）、EM（Dempster 1977）、Adam（Kingma 2015）。 |
| **集大成** | 每个模块站在顶级成果之上，并在统一框架下组合（Transformer 用多头注意力 + LayerNorm + 残差 + Adam）。 |
| **可验证** | 每条算法暴露一个可被独立实现交叉验证的「不变量」：梯度检验 1e-12、SVD 重构误差 1e-16、EM 对数似然单调、kNN 100%。 |
| **可复现** | 全部随机性由 seeded RNG（mulberry32）控制，结果可复现。 |
| **可运行** | 零运行时依赖；`node --test` 全绿；浏览器单文件 Demo 一键自检。 |

---

## 架构一览

```
Apex
├─ la.js         线性代数与数值方法（Jacobi / SVD / Cholesky / LayerNorm / softmax / CE）
├─ nn.js         神经网络原语（Dense · ReLU · MLP） + Adam 优化器 + MSE/CE
├─ attention.js  缩放点积注意力（含 causal mask）+ 多头注意力（解析反向）
├─ transformer.js 正弦位置编码 + 解码器块 + 小型自回归语言模型（训练闭环）
├─ ml.js         经典顶级算法：PCA(SVD) · GMM-EM · kNN
└─ index.js      统一出口
```

---

## 快速开始

```bash
# 运行全套数值验证（15 项，全绿）
npm test

# 生成单文件浏览器自检 Demo（demo/selfcheck.html，双击即跑）
node demo/build-html.mjs

# Node 内直接调用
node --input-type=module -e "import {svd,randMat,setSeed} from './src/la.js';
setSeed(1); const {U,Sigma,V,r}=svd(randMat(6,4,1));
console.log('rank',r,'top singular',Sigma[0].toFixed(4));"
```

浏览器自检：`打开 demo/selfcheck.html` → 点「运行自检」→ 8/8 通过，含训练损失曲线。

---

## 验证不变量（为什么可信）

| 模块 | 不变量 | 实测（seed 固定） |
|---|---|---|
| SVD | `A ≈ UΣVᵀ`，`UᵀU=I` | 重构误差 **8.9e-16**，正交误差 **4.4e-16** |
| Cholesky | `A = LLᵀ` | 误差 **8.9e-16** |
| Dense / 注意力 | 解析梯度 vs 中心差分 | relErr **6.9e-12** / **7.8e-12** |
| Transformer | 交叉熵随训练单调下降 | 1.488 → 0.570，并学会 `[0,1,2]→3` |
| PCA | 主成分方向正确、方差解释归一 | 主方向 dot=1.0000 |
| GMM-EM | 对数似然单调、权重和=1 | LL −1151→−422，权重和 1.000000 |
| kNN | 线性可分 100% 准确 | 100/100 |

---

## 目录

```
apex-ai/
├─ src/           核心实现（零依赖 ESM）
├─ test/          15 项数值验证（node --test）
├─ demo/
│  ├─ build-html.mjs   单文件 Demo 构建器（从 src 内联，DRY）
│  └─ selfcheck.html   浏览器自检（生成产物）
├─ docs/ARCHITECTURE.md  架构与「十层 AI 系统分解」映射
└─ package.json
```

---

## 许可与署名

MIT License · © 晨星 (CJX0712)。转载与二次开发请保留署名。
