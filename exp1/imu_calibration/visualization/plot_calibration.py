"""
标定结果可视化模块
生成加速度计/陀螺仪标定曲线和图表（全英文）
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, Optional

from imu_calibration.common.types import (
    AccelCalibResult, GyroRateCalibResult, GyroBiasCalibResult
)


class CalibrationPlotter:
    """标定结果绘图器（全英文）"""

    def __init__(self, output_dir: str = "./result"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_accel_calibration(self, accel_result: AccelCalibResult,
                                acc_means: Dict[str, np.ndarray],
                                pose_table: np.ndarray,
                                filename: str = "accel_calibration.png"):
        """
        (4)+(5) Accelerometer 6-position calibration curve + K_a, D_a results
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Accelerometer 6-Position Calibration', fontsize=14, fontweight='bold')

        pose_ids = sorted(acc_means.keys())
        n_poses = len(pose_ids)

        B_obs = np.zeros((n_poses, 3))
        for i, pid in enumerate(pose_ids):
            B_obs[i] = acc_means[pid]

        A = np.ones((n_poses, 4))
        for i in range(n_poses):
            A[i, :3] = pose_table[i]
        X = np.linalg.lstsq(A, B_obs, rcond=None)[0]
        B_pred = A @ X

        # Plot 1: Measured vs Fitted
        ax = axes[0, 0]
        axis_names = ['X-axis', 'Y-axis', 'Z-axis']
        for axis in range(3):
            ax.scatter(range(n_poses), B_obs[:, axis], marker='o', label=f'{axis_names[axis]}(measured)')
            ax.plot(range(n_poses), B_pred[:, axis], '--', label=f'{axis_names[axis]}(fitted)')
        ax.set_xlabel('Position Index')
        ax.set_ylabel('Acceleration (m/s^2)')
        ax.set_title('Measured vs Fitted (6 positions)')
        ax.legend()
        ax.grid(True)

        # Plot 2: Residuals
        ax = axes[0, 1]
        residuals = B_obs - B_pred
        for axis in range(3):
            ax.bar(np.arange(n_poses) + axis * 0.25, residuals[:, axis],
                   width=0.25, label=f'{axis_names[axis]} residual')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Position Index')
        ax.set_ylabel('Residual (m/s^2)')
        ax.set_title(f'Residual Distribution (RMS={np.sqrt(np.mean(residuals**2)):.4f} m/s^2)')
        ax.legend()
        ax.grid(True)

        # Plot 3: K_a heatmap
        ax = axes[1, 0]
        im = ax.imshow(accel_result.K_a, cmap='RdBu_r', aspect='auto')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['X-axis', 'Y-axis', 'Z-axis'])
        ax.set_yticklabels(['X-axis', 'Y-axis', 'Z-axis'])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{accel_result.K_a[i, j]:.4f}',
                        ha='center', va='center', fontsize=9)
        ax.set_title('Combined Error Matrix K_a')
        plt.colorbar(im, ax=ax)

        # Plot 4: Text results
        ax = axes[1, 1]
        ax.axis('off')
        text_lines = [
            'Accelerometer Calibration Results (5)',
            '',
            'K_a =',
            f'  [{accel_result.K_a[0,0]:.4f}  {accel_result.K_a[0,1]:.4f}  {accel_result.K_a[0,2]:.4f}]',
            f'  [{accel_result.K_a[1,0]:.4f}  {accel_result.K_a[1,1]:.4f}  {accel_result.K_a[1,2]:.4f}]',
            f'  [{accel_result.K_a[2,0]:.4f}  {accel_result.K_a[2,1]:.4f}  {accel_result.K_a[2,2]:.4f}]',
            '',
            f'D_a = [{accel_result.D_a[0]:.4f}, {accel_result.D_a[1]:.4f}, {accel_result.D_a[2]:.4f}] m/s^2',
            '',
            f'Reprojection Error = {accel_result.reprojection_error:.6f} m/s^2',
            f'Condition Number = {accel_result.condition_number:.2e}',
        ]
        ax.text(0.1, 0.9, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        fpath = self.output_dir / filename
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[CalibrationPlotter] Saved (4)+(5): {fpath}")

    def plot_gyro_rate_calibration(self, gyro_rate_result: GyroRateCalibResult,
                                    filename: str = "gyro_rate_calibration.png"):
        """
        (1)+(3) Gyroscope rate calibration curve + K_g results
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Gyroscope Rate Calibration (1)', fontsize=14, fontweight='bold')

        # Plot 1: K_g heatmap
        ax = axes[0]
        im = ax.imshow(gyro_rate_result.K_g, cmap='RdBu_r', aspect='auto')
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(['X-axis', 'Y-axis', 'Z-axis'])
        ax.set_yticklabels(['X-axis', 'Y-axis', 'Z-axis'])
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{gyro_rate_result.K_g[i, j]:.4f}',
                        ha='center', va='center', fontsize=9)
        ax.set_title('Combined Error Matrix K_g (3)')
        plt.colorbar(im, ax=ax)

        # Plot 2: Diagonal vs rotation rate
        ax = axes[1]
        n_rates = gyro_rate_result.K_g_per_rate.shape[0]
        rate_indices = np.arange(n_rates)
        for axis in range(3):
            diag_values = gyro_rate_result.K_g_per_rate[:, axis, axis]
            ax.plot(rate_indices, diag_values, 'o-', label=f'{["X","Y","Z"][axis]}-axis scale factor')
        ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal=1.0')
        ax.set_xlabel('Rate Index (10 to 50 deg/s)')
        ax.set_ylabel('Scale Factor')
        ax.set_title('Scale Factor vs Rotation Rate')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        fpath = self.output_dir / filename
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[CalibrationPlotter] Saved (1)+(3): {fpath}")

    def plot_gyro_bias_calibration(self, gyro_bias_result: GyroBiasCalibResult,
                                    filename: str = "gyro_bias_calibration.png"):
        """
        (2)+(3) Gyroscope 8-position bias calibration curve + D_g results
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Gyroscope 8-Position Bias Calibration (2)', fontsize=14, fontweight='bold')

        # Plot 1: Bias estimates per pose
        ax = axes[0]
        n_poses = gyro_bias_result.bias_per_pose.shape[0]
        x = np.arange(n_poses)
        for axis in range(3):
            ax.plot(x, gyro_bias_result.bias_per_pose[:, axis] * 3600.0,
                    'o-', label=f'{["X","Y","Z"][axis]}-axis')
        ax.axhline(y=gyro_bias_result.D_g_deg_h[0], color='C0', linestyle='--', alpha=0.5)
        ax.axhline(y=gyro_bias_result.D_g_deg_h[1], color='C1', linestyle='--', alpha=0.5)
        ax.axhline(y=gyro_bias_result.D_g_deg_h[2], color='C2', linestyle='--', alpha=0.5)
        ax.set_xlabel('Position Index')
        ax.set_ylabel('Bias Estimate (deg/h)')
        ax.set_title('Bias Estimates at 8 Positions')
        ax.legend()
        ax.grid(True)

        # Plot 2: Final bias values
        ax = axes[1]
        ax.axis('off')
        text_lines = [
            'Gyroscope Bias Calibration Results (3)',
            '',
            'D_g (deg/s):',
            f'  X-axis = {gyro_bias_result.D_g[0]:.6f}',
            f'  Y-axis = {gyro_bias_result.D_g[1]:.6f}',
            f'  Z-axis = {gyro_bias_result.D_g[2]:.6f}',
            '',
            'D_g (deg/h):',
            f'  X-axis = {gyro_bias_result.D_g_deg_h[0]:.2f}',
            f'  Y-axis = {gyro_bias_result.D_g_deg_h[1]:.2f}',
            f'  Z-axis = {gyro_bias_result.D_g_deg_h[2]:.2f}',
            '',
            'Std Dev (deg/h):',
            f'  X-axis = {gyro_bias_result.D_g_std[0]*3600:.4f}',
            f'  Y-axis = {gyro_bias_result.D_g_std[1]*3600:.4f}',
            f'  Z-axis = {gyro_bias_result.D_g_std[2]*3600:.4f}',
        ]
        ax.text(0.1, 0.9, '\n'.join(text_lines), transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        fpath = self.output_dir / filename
        plt.savefig(fpath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[CalibrationPlotter] Saved (2)+(3): {fpath}")