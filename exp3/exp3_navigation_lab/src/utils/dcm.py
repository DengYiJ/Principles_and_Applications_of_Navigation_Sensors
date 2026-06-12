"""
方向余弦矩阵 (DCM) 运算库
==========================
定义方向余弦矩阵基本运算：构造、正交化、与四元数互转
"""

import numpy as np

from .quaternion import quat_normalize


def dcm_identity() -> np.ndarray:
    """返回3×3单位矩阵"""
    return np.eye(3)


def dcm_from_quat(q: np.ndarray) -> np.ndarray:
    """
    四元数 → 方向余弦矩阵
    输入: q — shape (4,) 四元数 [qw, qx, qy, qz]
    输出: C — shape (3,3) 方向余弦矩阵 (正交，det=+1)
    """
    q = quat_normalize(q)  # 从 quaternion.py 导入
    qw, qx, qy, qz = q
    
    C = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qw*qz),       2*(qx*qz + qw*qy)    ],
        [2*(qx*qy + qw*qz),          1 - 2*(qx**2 + qz**2),   2*(qy*qz - qw*qx)    ],
        [2*(qx*qz - qw*qy),          2*(qy*qz + qw*qx),       1 - 2*(qx**2 + qy**2)]
    ])
    return C


def dcm_to_quat(C: np.ndarray) -> np.ndarray:
    """
    方向余弦矩阵 → 四元数
    输入: C — shape (3,3) 方向余弦矩阵
    输出: q — shape (4,) 四元数 [qw, qx, qy, qz]
    """
    # 使用最大迹法避免奇异
    tr = np.trace(C)
    
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (C[2,1] - C[1,2]) / S
        qy = (C[0,2] - C[2,0]) / S
        qz = (C[1,0] - C[0,1]) / S
    elif (C[0,0] > C[1,1]) and (C[0,0] > C[2,2]):
        S = np.sqrt(1.0 + C[0,0] - C[1,1] - C[2,2]) * 2
        qw = (C[2,1] - C[1,2]) / S
        qx = 0.25 * S
        qy = (C[0,1] + C[1,0]) / S
        qz = (C[0,2] + C[2,0]) / S
    elif C[1,1] > C[2,2]:
        S = np.sqrt(1.0 + C[1,1] - C[0,0] - C[2,2]) * 2
        qw = (C[0,2] - C[2,0]) / S
        qx = (C[0,1] + C[1,0]) / S
        qy = 0.25 * S
        qz = (C[1,2] + C[2,1]) / S
    else:
        S = np.sqrt(1.0 + C[2,2] - C[0,0] - C[1,1]) * 2
        qw = (C[1,0] - C[0,1]) / S
        qx = (C[0,2] + C[2,0]) / S
        qy = (C[1,2] + C[2,1]) / S
        qz = 0.25 * S
    
    q = np.array([qw, qx, qy, qz])
    return quat_normalize(q)


def dcm_orthogonalize(C: np.ndarray) -> np.ndarray:
    """
    DCM 正交化（SVD投影到SO(3)）
    输入: C — shape (3,3) 可能含数值误差的矩阵
    输出: C_orth — shape (3,3) 满足 C^T C = I, det(C)=+1
    """
    U, S, Vt = np.linalg.svd(C)
    # 强制行列式为 +1 (避免反射矩阵)
    D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
    C_orth = U @ D @ Vt
    return C_orth


def skew(v: np.ndarray) -> np.ndarray:
    """
    向量 → 反对称矩阵
    输入: v — shape (3,) 向量
    输出: V_skew — shape (3,3) 反对称矩阵
    性质: skew(a) @ b = a × b
    """
    return np.array([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ])


def dcm_is_valid(C: np.ndarray, tol: float = 1e-6) -> bool:
    """
    检查DCM是否有效（正交且行列式为+1）
    输入: C — shape (3,3)
          tol — 容差
    输出: True 如果有效
    """
    if C.shape != (3, 3):
        return False
    orth_check = np.allclose(C.T @ C, np.eye(3), atol=tol)
    det_check = abs(np.linalg.det(C) - 1.0) < tol
    return orth_check and det_check
