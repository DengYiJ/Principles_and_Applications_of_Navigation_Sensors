"""
Allan方差曲线绘制模块
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from imu_calibration.common.types import AllanResult

# ====== 禁用Unicode负号（避免\u2212缺失警告） ======
import matplotlib as _mpl
_mpl.rcParams['axes.unicode_minus'] = False
# ======================================================


class AllanPlotter:
    """Allan方差曲线绘图器"""

    def __init__(self, output_dir: str = "./result"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_allan_curve(self, allan_result: AllanResult,
                          filename: str = "allan_variance.png"):
        """
        绘制Allan方差双对数曲线

        包含: 三轴Allan标准差 vs 相关时间τ, ARW/BI标注
        """
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle('Allan Variance Analysis - Gyroscope Noise Characteristics',
                     fontsize=14, fontweight='bold')

        colors = ['C0', 'C1', 'C2']
        axis_labels = ['X-axis', 'Y-axis', 'Z-axis']

        for axis in range(3):
            sigma = allan_result.sigma[:, axis]
            valid = ~np.isnan(sigma)
            if np.sum(valid) < 3:
                continue
            tau_valid = allan_result.tau[valid]
            sigma_valid = sigma[valid]
            ax.loglog(tau_valid, sigma_valid, 'o-', color=colors[axis],
                      label=f'{axis_labels[axis]}', markersize=4, linewidth=1.5)

        # 标注ARW参考线
        if np.any(~np.isnan(allan_result.sigma[:, 0])):
            tau_ref = allan_result.tau[~np.isnan(allan_result.sigma[:, 0])]
            if len(tau_ref) > 5:
                tau_ref = tau_ref[:len(tau_ref)//3]
                arw_ref = allan_result.ARW[0]
                sigma_ref = arw_ref / np.sqrt(tau_ref)
                ax.loglog(tau_ref, sigma_ref, '--', color='gray', alpha=0.7,
                          label='ARW slope (-1/2)')

        # 标注Bias Instability
        for axis in range(3):
            min_idx = np.nanargmin(allan_result.sigma[:, axis])
            ax.plot(allan_result.tau[min_idx], allan_result.sigma[min_idx, axis],
                    'D', color=colors[axis], markersize=8, zorder=5)
            ax.annotate(f'BI={allan_result.BI[axis]:.4f} deg/h',
                        (allan_result.tau[min_idx], allan_result.sigma[min_idx, axis]),
                        textcoords="offset points", xytext=(10, 10),
                        fontsize=9, color=colors[axis])

        ax.set_xlabel('Correlation Time tau (s)', fontsize=12)
        ax.set_ylabel('Allan Standard Deviation sigma(tau) (deg/h)', fontsize=12)
        ax.set_title('Allan Variance Curve (Log-Log)')
        ax.legend(fontsize=10)
        ax.grid(True, which='both', alpha=0.3)

        # 文本结果框
        text_lines = ['Allan Variance Results', '']
        for i, axis_name in enumerate(['X', 'Y', 'Z']):
            text_lines.append(
                f'{axis_name}: ARW={allan_result.ARW[i]:.6f} deg/sqrt(h), '
                f'BI={allan_result.BI[i]:.4f} deg/h'
            )
        ax.text(0.98, 0.02, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        fpath = self.output_dir / filename
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[AllanPlotter] Saved {fpath}")