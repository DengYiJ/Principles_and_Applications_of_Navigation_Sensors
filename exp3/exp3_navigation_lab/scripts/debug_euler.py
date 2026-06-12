"""
调试欧拉角互转
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.utils.euler_angles import euler312_to_dcm, dcm_to_euler312

roll, pitch, yaw = 0.1, 0.2, 0.3
C = euler312_to_dcm(roll, pitch, yaw)
r, p, y = dcm_to_euler312(C)
print(f"Input:  roll={roll:.10f}, pitch={pitch:.10f}, yaw={yaw:.10f}")
print(f"DCM:\n{C}")
print(f"Output: roll={r:.10f}, pitch={p:.10f}, yaw={y:.10f}")
print(f"Diff:   roll={abs(r-roll):.2e}, pitch={abs(p-pitch):.2e}, yaw={abs(y-yaw):.2e}")
print(f"det(C)={np.linalg.det(C):.15f}")
print(f"C^T*C:\n{C.T @ C}")