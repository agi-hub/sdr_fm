# -*- coding: utf-8 -*-
"""扫描 UHF 470-866 MHz, 探测 DTMB 数字电视信号
DTMB 是 OFDM 调制, 频谱表现为 ~7.6MHz 宽的抬升平台(类噪声, 无载波尖峰)
"""
import ctypes, numpy as np

dll = ctypes.CDLL(r'rtlsdr_x64\rtlsdr.dll')
dll.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
dll.rtlsdr_read_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
dev = ctypes.c_void_p(); dll.rtlsdr_open(ctypes.byref(dev), 0)
FS = 2_400_000
dll.rtlsdr_set_sample_rate(dev, FS)
dll.rtlsdr_set_tuner_gain_mode(dev, 1)
dll.rtlsdr_set_tuner_gain(dev, 400)
print("tuner:", dll.rtlsdr_get_tuner_type(dev))

buf = ctypes.create_string_buffer(FS); n = ctypes.c_int()

# DTMB 频道: CH13-48 = 470-782, 8MHz 栅格, 中心 = 474 + 8*k
print("\n扫描 UHF 频道 (DS13-DS48)...")
print(f"{'频道':>6} {'频率':>8}  底噪(dB)  峰值(dB)  抬升(dB)  判定")
hits = []
for k in range(36):
    ch = 13 + k
    fc = 474.0 + 8 * k          # MHz, 频道中心
    dll.rtlsdr_set_center_freq(dev, int(fc * 1e6))
    dll.rtlsdr_reset_buffer(dev)
    if dll.rtlsdr_read_sync(dev, buf, FS, ctypes.byref(n)) != 0:
        continue
    iq = np.frombuffer(buf.raw[:n.value], dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
    x = (iq[0::2] + 1j * iq[1::2]); x -= x.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    PdB = 10 * np.log10(P + 1e-12)
    # 只看中间 80% (去边带衰落)
    m = len(PdB) // 10
    core = PdB[m:-m]
    floor = np.percentile(core, 30)     # 底噪: 30分位
    peak = np.percentile(core, 95)      # 信号: 95分位
    lift = peak - floor
    is_dtmb = lift < 8 and peak > -95   # OFDM: 平台抬升无尖峰
    is_analog = lift >= 8               # 模拟/有载波
    tag = 'DTMB?' if is_dtmb else ('载波' if is_analog else '')
    if tag:
        hits.append((ch, fc, peak, tag))
    print(f"DS{ch:<4} {fc:6.1f}M  {floor:8.1f}  {peak:8.1f}  {lift:7.1f}  {tag}")

dll.rtlsdr_close(dev)
print(f"\n候选: {len(hits)} 个")
for ch, fc, pk, tag in hits:
    print(f"  DS{ch} @ {fc:.1f} MHz  {pk:.1f}dB  {tag}")
