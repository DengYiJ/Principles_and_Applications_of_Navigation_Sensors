"""
RANGEA 字段映射调试
====================
验证 RANGEA 日志中每卫星观测块的字段含义，
确认伪距字段是否正确提取。
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import configs.constants as const

with open('exp2/data.txt', 'r') as f:
    raw = f.read()

print("=" * 78)
print("RANGEA FIELD MAPPING DEBUG")
print("=" * 78)

# =====================================================================
# Step 1: 找到 RANGEA 行，打印前几个观测块的所有字段
# =====================================================================
print("\n" + "-" * 78)
print("[STEP 1] RAW RANGEA LINE PARSING")
print("-" * 78)

for line in raw.strip().split('\n'):
    if '#RANGEA' not in line or ';' not in line:
        continue

    hdr_part, data_part = line.split(';', 1)
    print(f"\nFull RANGEA header: {hdr_part[:100]}...")
    
    obs_str = data_part.split('*')[0]
    print(f"\nFirst 300 chars of data: {obs_str[:300]}...")
    
    obs_fields = obs_str.split(',')
    print(f"\nTotal fields in data segment: {len(obs_fields)}")
    
    # 字段格式: 第1个字段是卫星数量，之后每11个字段为一个观测块
    num_sats_field = obs_fields[0]
    print(f"First field (satellite count?): '{num_sats_field}'")
    
    # 尝试按11字段/块解析
    sv_fields = obs_fields[1:]  # 跳过卫星数
    
    print(f"\n--- First 3 observation blocks (11 fields each) ---")
    print(f"{'idx':>4s} {'Field Name':>20s} {'Block1':>20s} {'Block2':>20s} {'Block3':>20s}")
    print("  " + "-" * 70)
    
    block_names = ["PRN", "FreqID", "TrackFlag", "Pseudorange", "PrStdDev", 
                   "CarrierPhase", "PhaseStdDev", "Doppler", "DopStdDev", 
                   "CN0", "SignalMask"]
    
    for idx in range(11):
        b1 = sv_fields[idx] if len(sv_fields) > idx else ''
        b2 = sv_fields[11 + idx] if len(sv_fields) > 11 + idx else ''
        b3 = sv_fields[22 + idx] if len(sv_fields) > 22 + idx else ''
        print(f"{idx:4d} {block_names[idx]:>20s} {b1:>20s} {b2:>20s} {b3:>20s}")
    
    break

# =====================================================================
# Step 2: 按11字段/块解析，提取所有卫星的伪距
# =====================================================================
print("\n" + "-" * 78)
print("[STEP 2] ALL SATELLITES - PSEUDORANGE VALUES (block_size=11)")
print("-" * 78)

for line in raw.strip().split('\n'):
    if '#RANGEA' not in line or ';' not in line:
        continue
    
    hdr_part, data_part = line.split(';', 1)
    hdr = hdr_part.split(']#')[1] if ']#' in hdr_part else hdr_part
    hdr_fields = hdr.split(',')
    week = hdr_fields[5]
    t_obs = hdr_fields[6]
    
    obs_str = data_part.split('*')[0]
    obs_fields = obs_str.split(',')
    
    sv_fields = obs_fields[1:]  # 跳过卫星计数
    
    print(f"\nRANGEA header: week={week}, t_obs={t_obs}")
    print(f"Total satellites in message (from field): {obs_fields[0]}")
    print(f"\n{'PRN':>4s} {'FreqID':>6s} {'TrackFlag':>10s} {'PSEUDORANGE':>18s} {'BlockIdx':>8s}")
    print("  " + "-" * 50)
    
    # 用 11 字段步长
    sat_count = 0
    for j in range(0, len(sv_fields) - 10, 11):
        block = sv_fields[j:j + 11]
        try:
            prn = int(float(block[0]))
            freq = block[1]
            track = block[2]
            pseudorange = float(block[3])  # 字段4 = 伪距 (0-based index 3)
            print(f"{prn:4d} {freq:>6s} {track:>10s} {pseudorange:>18.3f} {j:>8d}")
            sat_count += 1
        except (ValueError, IndexError) as e:
            print(f"  Error at block {j}: {str(e)[:50]}")
    
    print(f"\nTotal satellites parsed: {sat_count}")
    break

# =====================================================================
# Step 3: 用 10 字段步长（当前代码方式）提取对比
# =====================================================================
print("\n" + "-" * 78)
print("[STEP 3] COMPARISON: CURRENT CODE (block_size=10, block[2]=pseudorange)")
print("-" * 78)

for line in raw.strip().split('\n'):
    if '#RANGEA' not in line or ';' not in line:
        continue
    
    hdr_part, data_part = line.split(';', 1)
    obs_str = data_part.split('*')[0]
    obs_fields = obs_str.split(',')[1:]  # 跳过第一个空字段(分号后, 兼容旧版)
    
    print(f"\n{'PRN':>4s} {'block[0]':>10s} {'block[1]':>10s} {'block[2]':>15s} {'block[9]':>15s} {'block[10]':>15s}")
    print("  " + "-" * 73)
    
    sat_count_10 = 0
    for j in range(0, len(obs_fields) - 9, 10):
        block = obs_fields[j:j + 10]
        if len(block) < 10:
            continue
        b0 = block[0]  # 当前代码认为是 PRN
        b1 = block[1]
        b2 = block[2]  # 当前代码认为是 pseudorange
        b9 = block[9]  # 当前代码认为是 signal mask
        try:
            prn = int(float(b0))
            pr_val = float(b2)
            print(f"{prn:4d} {b0:>10s} {b1:>10s} {pr_val:>15.3f} {b9:>15s} {block[10] if len(block) > 10 else '':>15s}")
            sat_count_10 += 1
        except:
            continue
    
    print(f"\nTotal satellites (10-field step): {sat_count_10}")
    break

# =====================================================================
# Step 4: 验证量级
# =====================================================================
print("\n" + "-" * 78)
print("[STEP 4] PSEUDORANGE VALUE RANGE CHECK (11-field method)")
print("-" * 78)

# 用 11 字段法提取所有伪距
pr_values = []
for line in raw.strip().split('\n'):
    if '#RANGEA' not in line or ';' not in line:
        continue
    _, data_part = line.split(';', 1)
    obs_str = data_part.split('*')[0]
    obs_fields = obs_str.split(',')
    sv_fields = obs_fields[1:]
    
    for j in range(0, len(sv_fields) - 10, 11):
        block = sv_fields[j:j + 11]
        try:
            prn = int(float(block[0]))
            pr_val = float(block[3])  # 第4字段
            pr_values.append((prn, pr_val))
        except:
            continue
    break

print(f"{'PRN':>4s} {'Psuedorange(m)':>18s} {'Valid':>8s}")
print("  " + "-" * 32)
for prn, val in pr_values:
    valid = "OK" if 1.5e7 < val < 4e7 else "SUSPICIOUS"
    print(f"{prn:4d} {val:>18.3f} {valid:>8s}")

print(f"\nTotal pseudoranges: {len(pr_values)}")
print(f"Range: {min(v for _,v in pr_values):.1f} ~ {max(v for _,v in pr_values):.1f} m")

# 检查当前代码提取的值
print("\n" + "-" * 78)
print("[STEP 5] CURRENT CODE EXTRACTION CHECK (extract_pseudorange_matrix)")
print("-" * 78)

from algorithms.extractor import extract_pseudorange_matrix
from algorithms.extractor import extract_obs_time, extract_ephemeris_matrix

week, t_obs = extract_obs_time(raw)
eph, vprn = extract_ephemeris_matrix(raw, week, t_obs)
pr = extract_pseudorange_matrix(raw, week, vprn)

print(f"Extracted GPS satellites: {pr.shape[0]}")
print(f"\n{'PRN':>4s} {'Pseudorange(m)':>18s} {'Valid':>8s}")
print("  " + "-" * 32)
for i in range(pr.shape[0]):
    prn = int(pr[i, 0])
    val = pr[i, 1]
    valid = "OK" if 1.5e7 < val < 4e7 else "SUSPICIOUS"
    print(f"{prn:4d} {val:>18.3f} {valid:>8s}")

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
all_ok = all(1.5e7 < pr[i,1] < 4e7 for i in range(pr.shape[0]))
if all_ok:
    print("  ✅ ALL PSEUDORANGES IN NORMAL RANGE (2e7~3e7 m)")
    print("  RANGEA field mapping is CORRECT for our GPS satellites")
else:
    print("  ❌ SUSPICIOUS VALUES DETECTED")
    print("  RANGEA field mapping may be WRONG")