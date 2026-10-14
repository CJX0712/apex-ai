// 梯度检验工具（中心差分），用于交叉验证解析反向
import { zeros, transpose, matmul, addMat } from '../src/la.js';

export function numGradMat(f, M, eps = 1e-5) {
  const g = zeros(M.length, M[0].length);
  for (let i = 0; i < M.length; i++)
    for (let j = 0; j < M[0].length; j++) {
      const o = M[i][j];
      M[i][j] = o + eps;
      const lp = f();
      M[i][j] = o - eps;
      const lm = f();
      M[i][j] = o;
      g[i][j] = (lp - lm) / (2 * eps);
    }
  return g;
}

// 相对误差（数值稳定）：max|a-b| / (max|a|+|b|+eps)
export function relErr(A, B) {
  let num = 0, den = 0;
  for (let i = 0; i < A.length; i++)
    for (let j = 0; j < A[0].length; j++) {
      num = Math.max(num, Math.abs(A[i][j] - B[i][j]));
      den = Math.max(den, Math.abs(A[i][j]) + Math.abs(B[i][j]));
    }
  return num / (den + 1e-12);
}

// 将 2D 矩阵展平为 1D 做逐元素比较
export function maxRelErr1D(a, b) {
  let num = 0, den = 0;
  const n = Array.isArray(a[0]) ? a.length * a[0].length : a.length;
  const get = (M, k) => (Array.isArray(M[0]) ? M[Math.floor(k / M[0].length)][k % M[0].length] : M[k]);
  for (let k = 0; k < n; k++) {
    const x = get(a, k), y = get(b, k);
    num = Math.max(num, Math.abs(x - y));
    den = Math.max(den, Math.abs(x) + Math.abs(y));
  }
  return num / (den + 1e-12);
}
