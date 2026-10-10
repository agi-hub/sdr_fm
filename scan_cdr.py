# -*- coding: utf-8 -*-
"""扫描 FM 频段探测 CDR (中国数字音频广播)
CDR 频谱指纹: 100-200kHz 宽平坦 OFDM 平台, 无载波尖峰
模拟 FM: 有载波峰+导频结构, 频谱起伏大
"""
import ctypes, numpy as np

dll = ctypes.CDLL(r'rtlsdr_x64\rtlsdr.dll')
dll.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
dll.rtlsdr_read_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
dev = ctypes.c_void_p(); dll.rtlsdr_open(ctypes.byref(dev), 0)
FS = 2_400_000
dll.rtlsdr_set_sample_rate(dev, FS)
dll.rtlsdr_set_tuner_gain_mode(dev, 1)
dll.rtlsdr_set_tuner_gain(dev, 450)
print("tuner:", dll.rtlsdr_get_tuner_type(dev))

buf = ctypes.create_string_buffer(FS * 2); n = ctypes.c_int()

# 25kHz 频谱槽, 87-108 MHz
SLOT = 0.025
slots = {}   # 频率(MHz) -> 功率dB

fc = 88.2
while fc <= 107.6:
    dll.rtlsdr_set_center_freq(dev, int(fc * 1e6))
    dll.rtlsdr_reset_buffer(dev)
    if dll.rtlsdr_read_sync(dev, buf, FS * 2, ctypes.byref(n)) != 0:
        fc += 1.6; continue
    iq = np.frombuffer(buf.raw[:n.value], dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
    x = (iq[0::2] + 1j * iq[1::2]); x -= x.mean()
    # 高分辨率频谱: 4段平均, 8192点FFT (~293Hz/bin)
    acc = np.zeros(8192)
    seg = len(x) // 4
    for i in range(4):
        w = x[i * seg:(i + 1) * seg] * np.hanning(seg)
        acc += np.abs(np.fft.fftshift(np.fft.fft(w, 8192))) ** 2
    P = 10 * np.log10(acc / 4 + 1e-12)
    fr = np.fft.fftshift(np.fft.fftfreq(8192, 1 / FS)) / 1e6 + fc
    for k in np.arange(fc - 1.05, fc + 1.05, SLOT):
        key = round(k, 3)
        if 87.0 <= key <= 108.5:
            m = (fr > key - SLOT / 2) & (fr < key + SLOT / 2)
            if m.any():
                slots[key] = max(slots.get(key, -999), np.percentile(P[m], 90))
    fc += 1.6

dll.rtlsdr_close(dev)

keys = sorted(slots)
vals = np.array([slots[k] for k in keys])
floor = np.percentile(vals, 20)          # 全频段底噪
print(f"\n全频段底噪: {floor:.1f} dB\n")
print(f"{'频率':>8}  功率   起伏   判定")

candidates = []
for i, k in enumerate(keys):
    if vals[i] < floor + 9: continue      # 必须明显高于底噪
    # 邻域平坦度: ±75kHz 内 90分位-30分位
    lo, hi = max(0, i - 3), min(len(keys), i + 4)
    seg = vals[lo:hi]
    flat = np.percentile(seg, 90) - np.percentile(seg, 30)
    if flat < 5.0:                        # 平坦 → 数字 OFDM 特征
        candidates.append((k, vals[i], flat))

# 合并相邻候选 (±100kHz 内)
merged = []
for k, v, f in candidates:
    if merged and abs(k - merged[-1][0]) < 0.1:
        if v > merged[-1][1]:
            merged[-1] = (k, v, f)
    else:
        merged.append((k, v, f))

for k, v, f in merged:
    print(f"{k:7.3f}M  {v:5.1f}dB  {f:4.1f}dB  CDR候选(平坦OFDM平台)")
if not merged:
    print("  未发现平坦数字平台候选")
    # 辅助诊断: 列出所有强信号及其平坦度
    print("\n辅助: 各强频点频谱特征 (平坦度<5dB 更像数字):")
    seen = []
    for i, k in enumerate(keys):
        if vals[i] < floor + 12: continue
        if seen and abs(k - seen[-1]) < 0.1: continue
        seen.append(k)
        lo, hi = max(0, i - 3), min(len(keys), i + 4)
        seg = vals[lo:hi]
        flat = np.percentile(seg, 90) - np.percentile(seg, 30)
        print(f"  {k:7.3f}M  {vals[i]:5.1f}dB  平坦度{flat:4.1f}dB")
