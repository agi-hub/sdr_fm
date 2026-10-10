# -*- coding: utf-8 -*-
"""稳健扫描 85-110 MHz FM 频段
方法: 中等增益(防互调杂散) + 每频点两次测量取大(抗衰落) + 贪心峰值分离
输出: fm_stations.csv (含导频信息列, 仅作参考不作门槛)
"""
import ctypes, numpy as np, csv
from scipy.signal import butter, lfilter

dll = ctypes.CDLL(r'rtlsdr_x64\rtlsdr.dll')
dll.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
dll.rtlsdr_read_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
dev = ctypes.c_void_p(); dll.rtlsdr_open(ctypes.byref(dev), 0)
FS = 2_048_000
dll.rtlsdr_set_sample_rate(dev, FS)
dll.rtlsdr_set_tuner_gain_mode(dev, 1)
dll.rtlsdr_set_tuner_gain(dev, 350)   # 中等增益: 灵敏度与抗互调折中
print("tuner:", dll.rtlsdr_get_tuner_type(dev))

buf = ctypes.create_string_buffer(FS * 2); n = ctypes.c_int()
chan_pw = {}

def scan_pass():
    fc = 85.9
    while fc <= 110.0:
        dll.rtlsdr_set_center_freq(dev, int(fc * 1e6))
        dll.rtlsdr_reset_buffer(dev)
        if dll.rtlsdr_read_sync(dev, buf, FS * 2, ctypes.byref(n)) != 0:
            fc += 1.8; continue
        iq = np.frombuffer(buf.raw[:n.value], dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        x = (iq[0::2] + 1j * iq[1::2]); x -= x.mean()
        P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2
        freqs = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS)) / 1e6 + fc
        for ch10 in range(850, 1101):
            ch = ch10 / 10
            if abs(ch - fc) > 0.88: continue
            m = (freqs > ch - 0.075) & (freqs < ch + 0.075)
            if m.any():
                pw = 10 * np.log10(np.percentile(P[m], 90) + 1e-12)
                chan_pw[ch] = max(chan_pw.get(ch, -999), pw)
        fc += 1.8

print("第1遍扫描..."); scan_pass()
print("第2遍扫描 (取两次最大, 抗衰落)..."); scan_pass()

# 电台判定: 比最强台低14.5dB内, 或有明确导频(>12dB); 贪心选峰抑制±300kHz
pmax = max(chan_pw.values())
stations = []
for ch, pw in sorted(chan_pw.items(), key=lambda kv: -kv[1]):
    if pw < pmax - 14.5: break
    if any(abs(ch - c) < 0.3 for c, _ in stations): continue
    stations.append((ch, pw))

# 导频参考测量 (不作门槛, 只标注)
FS2 = 960_000
dll.rtlsdr_set_sample_rate(dev, FS2)
buf2 = ctypes.create_string_buffer(FS2)
lp_ch = butter(4, 100e3 / (FS2 / 2), btype="low")
pil_db = {}
for ch, pw in stations:
    dll.rtlsdr_set_center_freq(dev, int(ch * 1e6))
    dll.rtlsdr_reset_buffer(dev)
    nb = ctypes.c_int()
    if dll.rtlsdr_read_sync(dev, buf2, FS2, ctypes.byref(nb)) != 0:
        pil_db[ch] = -99; continue
    iq = np.frombuffer(buf2.raw[:nb.value], dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
    x = (iq[0::2] + 1j * iq[1::2]); x -= x.mean()
    x = lfilter(lp_ch[0], lp_ch[1], x)
    mpx = np.angle(x[1:] * np.conj(x[:-1])); mpx -= mpx.mean()
    M = np.abs(np.fft.rfft(mpx * np.hanning(len(mpx))))
    f = np.fft.rfftfreq(len(mpx), 1 / FS2)
    pil_db[ch] = 10 * np.log10(M[(f > 18_800) & (f < 19_200)].max()
                               / max(M[(f > 16_000) & (f < 17_500)].mean(), 1e-9))

dll.rtlsdr_close(dev)
stations.sort()

with open('fm_stations.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['频率(MHz)', '相对强度(dB)', '导频(立体声迹象>10dB)', '强度条'])
    for ch, pw in stations:
        w.writerow([f'{ch:.1f}', f'{pw:.1f}', f'{pil_db.get(ch, -99):.1f}',
                    '#' * int(max(1, (pw - pmax + 40) / 2))])

print(f"\n共检测到 {len(stations)} 个频点, 已保存到 fm_stations.csv\n")
print(f"{'频率':>8}  强度   导频")
for ch, pw in stations:
    stereo = '♪' if pil_db.get(ch, -99) > 10 else ' '
    print(f"{ch:7.1f}M  {pw:5.1f}dB  {pil_db.get(ch,-99):5.1f}dB {stereo}")
