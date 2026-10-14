# Apex 架构 · 与「AI 系统十层分解」映射

本文档把 **Apex** 的核心模块映射到晨星的 **AI 系统十层分解架构**，说明每一层由哪个世界顶级算法/实现承载，以及对应的**可验证不变量**（微指令级正确性金标准）。

## 十层映射表

| 层 | 主题 | Apex 承载 | 顶级来源 | 不变量（可验证） |
|---|---|---|---|---|
| 1 | 数学基础（线代/数值） | `la.js`：matmul / 转置 / SVD / Cholesky / Jacobi 特征值 | Golub & Van Loan (2013)；Eckart–Young (1936)；Cholesky (1924) | `A≈UΣVᵀ` 误差 1e-16；`UᵀU=I`；`A=LLᵀ` |
| 2 | 概率与统计 | softmax（logsumexp）/ 交叉熵 / GMM 高斯密度 | Vaswani 2017；Dempster 1977 | 行和为 1；责任度每行 Σ=1 |
| 3 | 优化 | `Adam`（Kingma & Ba 2015） | Kingma & Ba (2015) | 梯度检验 1e-12；损失单调下降 |
| 4 | 表示学习（嵌入） | `TransformerLM.E` 词嵌入 + 正弦位置编码 | Vaswani 2017 §3.5 | PE 周期性；嵌入梯度回流正确 |
| 5 | 序列建模 / 注意力 | `attention.js` 缩放点积 + 多头（causal mask） | Vaswani 2017 §3.2 | 解析反向 vs 数值 1e-12 |
| 6 | 深层网络 / Transformer | `transformer.js` 解码器块（残差+LN+FFN） | Vaswani 2017 | 训练损失递减；预测学到规律 |
| 7 | 经典监督 / 无监督 | `ml.js`：PCA(SVD) / kNN | Cover & Hart 1967；GVL §8.6 | 主方向 dot=1.0；kNN 100% |
| 8 | 生成模型 / EM | `ml.js`：GMM-EM | Dempster, Laird & Rubin (1977) | 对数似然单调；权重 Σ=1 |
| 9 | 训练闭环 / 验证 | `test/*` + `demo/selfcheck.html` | — | 15 项 + 8 项全绿（可复现） |
| 10 | 微指令级操作 | 全部「不变量」即微指令金标准 | — | 每条算法 ≥1 个独立交叉验证不变量 |

## 数据流（训练一步）

```
token ids
  → 嵌入 E + 正弦 PE
  → [DecoderBlock × N]:
       因果多头自注意力 (mha) ──┐
       残差 + LayerNorm         │
       FFN(Linear→ReLU→Linear)  │
       残差 + LayerNorm ────────┘
  → 输出头 Wo → 词表 logits
  → 交叉熵损失 → 反向传播 → Adam 更新全部参数
```

## 关键设计决策（集大成视角）

- **PCA 走 SVD 而非协方差特征分解**：数值更稳定（Eckart–Young），且自动得到最优低秩重构。
- **注意力反向用行 softmax Jacobian**：`dScores = attn ⊙ (dAttn − (attn·dAttn)[:,None])`，与中心差分 1e-12 吻合。
- **GMM-EM 用 Cholesky + logsumexp**：避免协方差求逆与指数溢出，保证对数似然严格可追踪、单调。
- **一切随机性 seeded**：`setSeed(mulberry32)` → 测试与 Demo 完全可复现，满足「可复现」硬要求。

## 闭环交付

`npm test` → 15 项数值验证全绿；`node demo/build-html.mjs` → 单文件浏览器自检（8/8）。成果完整开源至 GitHub，作者署名 **晨星**。
