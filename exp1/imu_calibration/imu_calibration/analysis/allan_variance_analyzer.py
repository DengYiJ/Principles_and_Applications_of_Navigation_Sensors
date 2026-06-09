"""
Allan方差分析模块（优化版）
对应 02_algorithm_design.md → Pseudocode 4

优化策略：
1. 对数间隔τ序列（τ = 2^k），高τ区间跳跃采样，减少90%无效循环
2. cumsum向量化：预计算累积和，使任意簇均值计算降为 O(1)
3. 重叠Allan方差改用向量化实现，消除内层for循环
4. tqdm进度条
"""
import numpy as np
from typing import Tuple
from ..common.types import AllanResult
from ..common.constants import PI, RAD2DEG


# 尝试导入tqdm（如不可用则回退到简单打印）
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


class AllanVarianceAnalyzer:
    """
    Allan方差分析器（向量化加速版）
    
    核心优化：利用 cumsum 实现任意长度子段均值的 O(1) 计算。
    
    设原始数据为 x[0..N-1]，预先计算 cumsum[i] = Σ_{j=0}^{i} x[j]。
    对子段 [i, i+m-1] 的均值为：
        mean_i = (cumsum[i+m-1] - cumsum[i-1]) / m
    所有子段均值的计算从 O(N·m) 降为 O(N·(N/m)/m?) 实际操作：
    - 非重叠：reshape + mean(axis=1) 已是最优
    - 重叠：原实现用了for循环逐个计算，现改为 cumsum 向量化
    """

    def __init__(self, tau_base: float = 2.0, min_cluster_count: int = 9,
                 use_overlapping: bool = True, show_progress: bool = True):
        """
        参数:
            tau_base: τ序列倍增基数（默认2, 即τ = 2^k, 对数均匀分布）
            min_cluster_count: 最小子段数（默认9）
            use_overlapping: 是否使用重叠Allan方差（默认True）
            show_progress: 是否显示进度条
        """
        self.tau_base = tau_base
        self.min_cluster_count = min_cluster_count
        self.use_overlapping = use_overlapping
        self.show_progress = show_progress

    def analyze(self, gyro_static: np.ndarray, fs: float = 200.0) -> AllanResult:
        """
        执行Allan方差全部分析

        参数:
            gyro_static: shape=(M,3), 长时间静态陀螺数据(°/s)
            fs: 采样率(Hz), 默认200

        返回:
            AllanResult
        """
        M = len(gyro_static)
        dt = 1.0 / fs

        if M < fs * 3600:
            import warnings
            warnings.warn(f"Static data too short: {M/fs:.0f}s < 3600s (1h), "
                          f"Allan variance confidence reduced")

        n_axes = gyro_static.shape[1]

        # Step 1: 构建τ序列（对数均匀分布，τ = dt · tau_base^k）
        tau_max = (M * dt) / self.min_cluster_count
        k_max = int(np.floor(np.log(tau_max / dt) / np.log(self.tau_base)))
        k_values = np.arange(0, k_max + 1, dtype=int)
        tau = dt * (self.tau_base ** k_values)  # 对数间隔：τ = dt, 2dt, 4dt, 8dt, ...

        print(f"[AllanVariance] {len(tau)} tau values (log spacing, base={self.tau_base}), "
              f"data={M} samples @ {fs}Hz = {M/fs:.0f}s")

        # Step 2: 计算每个轴的Allan方差（已优化）
        sigma = np.zeros((len(tau), n_axes), dtype=np.float64)
        
        # 创建进度条
        total_steps = n_axes * len(tau)
        if self.show_progress and HAVE_TQDM:
            pbar = tqdm(total=total_steps, desc="Allan Var", unit="τ")
        else:
            pbar = None

        for axis in range(n_axes):
            sigma[:, axis] = self._compute_allan_variance(
                gyro_static[:, axis], dt, tau, pbar=pbar
            )

        if pbar is not None:
            pbar.close()

        # Step 3: 拟合ARW
        ARW = np.zeros(n_axes, dtype=np.float64)
        fitted_slopes = np.zeros(n_axes, dtype=np.float64)
        for axis in range(n_axes):
            ARW[axis], fitted_slopes[axis] = self._fit_ARW(tau, sigma[:, axis])

        # Step 4: 提取Bias Instability（曲线最低点）
        BI = np.zeros(n_axes, dtype=np.float64)
        for axis in range(n_axes):
            valid = ~np.isnan(sigma[:, axis])
            if np.any(valid):
                BI[axis] = np.min(sigma[valid, axis])

        # 输出结果
        print("\n[Allan Variance] Results:")
        for axis in range(n_axes):
            print(f"  Axis {['X','Y','Z'][axis]}: "
                  f"ARW={ARW[axis]:.6f} °/√h, "
                  f"BI={BI[axis]:.6f} °/h, "
                  f"log-log slope={fitted_slopes[axis]:.3f}")

        return AllanResult(
            tau=tau,
            sigma=sigma,
            ARW=ARW,
            BI=BI,
            fitted_log_slopes=fitted_slopes,
        )

    def _compute_allan_variance(self, data: np.ndarray, dt: float,
                                 tau: np.ndarray, pbar=None) -> np.ndarray:
        """
        计算单个轴不同τ的Allan标准差（向量化优化版）

        参数:
            data: shape=(M,), 单个轴静态数据
            dt: 采样间隔(s)
            tau: τ序列(s)
            pbar: tqdm进度条对象(可选)
        """
        sigma = np.zeros(len(tau))
        M = len(data)
        
        # 预计算累积和：cumsum[i] = Σ_{j=0}^{i} data[j]
        cumsum = np.zeros(M + 1, dtype=np.float64)
        np.cumsum(data, out=cumsum[1:])

        for k, tau_k in enumerate(tau):
            m = int(round(tau_k / dt))  # 子段长度(样本数)
            if m < 1:
                m = 1
            L = M // m  # 子段数

            if L < self.min_cluster_count:
                sigma[k] = np.nan
                if pbar is not None:
                    pbar.update(1)
                continue

            if self.use_overlapping:
                sigma[k] = self._overlapping_allan_fast(cumsum, m, M)
            else:
                sigma[k] = self._non_overlapping_allan_fast(cumsum, m, L)

            if pbar is not None:
                pbar.update(1)

        # 转换为°/h
        sigma = sigma * 3600.0
        return sigma

    @staticmethod
    def _non_overlapping_allan_fast(cumsum: np.ndarray, m: int, L: int) -> float:
        """
        非重叠Allan方差（向量化）
        
        利用 cumsum 计算 L 个非重叠子段的均值，每个均值 O(1)。
        cumsum[i] 为 data[0..i-1] 的累积和。
        
        子段 j 的均值 = (cumsum[(j+1)*m] - cumsum[j*m]) / m
        """
        # 子段起始和结束索引
        starts = np.arange(L) * m
        ends = starts + m
        # 向量化计算所有子段均值: O(L)
        means = (cumsum[ends] - cumsum[starts]) / m
        # Allan方差
        diff = means[1:] - means[:-1]
        return float(np.sqrt(0.5 * np.mean(diff ** 2)))

    @staticmethod
    def _overlapping_allan_fast(cumsum: np.ndarray, m: int, M: int) -> float:
        """
        重叠Allan方差（向量化加速版）
        
        利用 cumsum，所有重叠子段的均值可在 O(N) 时间内一次性计算。
        
        子段 [i, i+m-1] 的均值 = (cumsum[i+m] - cumsum[i]) / m
        其中 i=0..N-m。
        
        原算法 O(N·m) → 现算法 O(N)。
        """
        n_clusters = M - m + 1  # 重叠子段总数
        
        # 向量化计算所有 n_clusters 个子段的均值: O(N)
        # means[i] = (cumsum[i+m] - cumsum[i]) / m
        starts = np.arange(n_clusters)
        ends = starts + m
        means = (cumsum[ends] - cumsum[starts]) / m

        # Allan方差: (1/2) * E[(y_{k+1} - y_k)²]
        diff = means[1:] - means[:-1]
        return float(np.sqrt(0.5 * np.mean(diff ** 2)))

    @staticmethod
    def _fit_ARW(tau: np.ndarray, sigma: np.ndarray) -> Tuple[float, float]:
        """
        拟合ARW: 在log-log斜率≈-1/2的区间
        ARW = σ(τ) · √τ  (在τ较小处)

        返回:
            ARW值(°/√h), 拟合区间斜率
        """
        # 去除NaN
        valid = ~np.isnan(sigma)
        tau_v = tau[valid]
        sigma_v = sigma[valid]

        if len(tau_v) < 3:
            return 0.0, 0.0

        log_tau = np.log10(tau_v)
        log_sigma = np.log10(sigma_v)

        # 寻找斜率≈-0.5的区间（前1/3数据）
        n_third = max(3, len(tau_v) // 3)
        log_tau_sub = log_tau[:n_third]
        log_sigma_sub = log_sigma[:n_third]

        # 线性拟合求解斜率
        A = np.vstack([log_tau_sub, np.ones_like(log_tau_sub)]).T
        slope, intercept = np.linalg.lstsq(A, log_sigma_sub, rcond=None)[0]

        # 提取ARW: sigma(τ) * sqrt(τ) 在斜率区的均值
        arw_values = sigma_v[:n_third] * np.sqrt(tau_v[:n_third])
        ARW = float(np.median(arw_values))

        return ARW, float(slope)