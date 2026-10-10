# -*- coding: utf-8 -*-
"""
航空频段 AM 收音机 + 活跃频道扫描 (118-137 MHz)
用法:
  python airband_am.py            # 先扫描活跃频道并列出
  python airband_am.py 121.5      # 直接收听指定频率 (AM)
"""
import ctypes, os, sys, queue, threading, time, wave
import numpy as np
from scipy.signal import butter, lfilter, resample_poly

dll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtlsdr_x64", "rtlsdr.dll"))
dll.rtlsdr_open.restype = ctypes.c_int
dll.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
dll.rtlsdr_set_center_freq.restype = ctypes.c_int
dll.rtlsdr_set_sample_rate.restype = ctypes.c_int
dll.rtlsdr_set_tuner_gain_mode.restype = ctypes.c_int
dll.rtlsdr_set_tuner_gain.restype = ctypes.c_int
dll.rtlsdr_set_agc_mode.restype = ctypes.c_int
dll.rtlsdr_reset_buffer.restype = ctypes.c_int
dll.rtlsdr_read_sync.restype = ctypes.c_int
dll.rtlsdr_read_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
dll.rtlsdr_close.restype = ctypes.c_int

FS = 960_000
AUDIO_FS = 48_000
dev = ctypes.c_void_p()
assert dll.rtlsdr_open(ctypes.byref(dev), 0) == 0, "设备被占用?"
dll.rtlsdr_set_sample_rate(dev, FS)
dll.rtlsdr_set_tuner_gain_mode(dev, 1)
dll.rtlsdr_set_tuner_gain(dev, 495)
dll.rtlsdr_set_agc_mode(dev, 0)
lp_ch = butter(4, 8e3 / (FS / 2), btype="low")      # AM 信道 ±8k
lp_aud = butter(4, 3e3 / (AUDIO_FS / 2), btype="low")  # 话音 3k

def set_freq(fc):
    dll.rtlsdr_set_center_freq(dev, int(fc * 1e6))
    dll.rtlsdr_reset_buffer(dev)

# ---------- 模式一: 扫描活跃频道 ----------
if len(sys.argv) < 2:
    print("扫描 118.0-137.0 MHz 航空频段 (宽带快扫)...")
    FS_S = 2_400_000
    dll.rtlsdr_set_sample_rate(dev, FS_S)
    sbuf = ctypes.create_string_buffer(FS_S * 2); n = ctypes.c_int()
    slot_pw = {}
    fc = 119.2
    while fc <= 136.0:
        dll.rtlsdr_set_center_freq(dev, int(fc * 1e6))
        dll.rtlsdr_reset_buffer(dev)
        if dll.rtlsdr_read_sync(dev, sbuf, FS_S * 2, ctypes.byref(n)) == 0:
            iq = np.frombuffer(sbuf.raw[:n.value], dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
            x = (iq[0::2] + 1j * iq[1::2]); x -= x.mean()
            P = np.abs(np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))) ** 2
            fr = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / FS_S)) / 1e6 + fc
            for k25 in np.arange(fc - 1.05, fc + 1.05, 0.025):
                key = round(k25, 3)
                if 118.0 <= key <= 137.0:
                    m = (fr > key - 0.008) & (fr < key + 0.008)   # 载波±8kHz
                    if m.any():
                        slot_pw[key] = max(slot_pw.get(key, -999),
                                           10 * np.log10(np.percentile(P[m], 90) + 1e-12))
        fc += 1.6
    dll.rtlsdr_close(dev)
    keys = sorted(slot_pw)
    vals = np.array([slot_pw[k] for k in keys])
    floor = np.percentile(vals, 30)
    act = [(k, v) for k, v in zip(keys, vals) if v > floor + 15]
    # 合并相邻 (±25kHz)
    merged = []
    for k, v in act:
        if merged and abs(k - merged[-1][0]) <= 0.026:
            if v > merged[-1][1]: merged[-1] = (k, v)
        else:
            merged.append((k, v))
    if merged:
        print(f"\n发现 {len(merged)} 个 AM 载波/活跃频道 (高出底噪15dB以上):")
        for k, v in merged:
            print(f"  {k:7.3f} MHz  强度 {v:.1f} dB")
        print("\n收听: python airband_am.py <频率>")
    else:
        print(f"未发现明显载波 (底噪 {floor:.1f} dB)。航空频段安静, 可直接试塔台频率:")
        print("  北京进近 120.6 / 125.2  首都机场塔台 118.1 / 118.5  北京区调 128.85")
    sys.exit(0)

# ---------- 模式二: AM 收听 ----------
FREQ = float(sys.argv[1])
set_freq(FREQ)
print(f"收听 {FREQ:.3f} MHz (AM), Ctrl+C 退出")
z_ch = np.zeros(max(len(lp_ch[0]), len(lp_ch[1])) - 1, dtype=np.complex64)
z_aud = np.zeros(max(len(lp_aud[0]), len(lp_aud[1])) - 1)

def rx_thread():
    global z_ch, z_aud
    buf = ctypes.create_string_buffer(read_len); n = ctypes.c_int()
    while running:
        if dll.rtlsdr_read_sync(dev, buf, read_len, ctypes.byref(n)) != 0 or n.value != read_len:
            continue
        iq = np.frombuffer(buf.raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        x = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64); x -= x.mean()
        xf, z_ch = lfilter(lp_ch[0], lp_ch[1], x, zi=z_ch)
        env = np.abs(xf)
        audio = env - env.mean()                    # AM 包络检波
        audio = resample_poly(audio, 1, FS // AUDIO_FS)
        audio, z_aud = lfilter(lp_aud[0], lp_aud[1], audio, zi=z_aud)
        # 定增益软限幅
        audio = np.tanh(audio * 8.0) * 0.9
        # ASRC
        err = 4 - audio_q.qsize()
        factor = 1.0 + max(-0.10, min(0.10, err * 0.03))
        n_out = int(round(len(audio) * factor))
        idx = np.linspace(0, len(audio) - 1, n_out)
        audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        audio_q.put(audio); rec_q.put(audio)

def rec_thread():
    wf = wave.open(f"airband_{FREQ:g}MHz.wav".replace('.', '_', 1), "wb")
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(AUDIO_FS)
    while running or not rec_q.empty():
        try:
            blk = rec_q.get(timeout=0.5)
        except queue.Empty:
            continue
        wf.writeframes((np.clip(blk, -1, 1) * 32767).astype(np.int16).tobytes())
    wf.close()

rx = threading.Thread(target=rx_thread, daemon=True); rx.start()
rec = threading.Thread(target=rec_thread, daemon=True); rec.start()
while audio_q.qsize() < 4:
    time.sleep(0.1)

import sounddevice as sd
carry = np.zeros(0, dtype=np.float32)
def callback(outdata, frames, ti, st):
    global carry
    while len(carry) < frames:
        try: carry = np.concatenate([carry, audio_q.get_nowait()])
        except queue.Empty: break
    out = carry[:frames]
    if len(out) < frames: out = np.pad(out, (0, frames - len(out)))
    carry = carry[frames:]
    outdata[:, 0] = out
try:
    with sd.OutputStream(samplerate=AUDIO_FS, channels=1, dtype="float32", blocksize=0, callback=callback):
        while True: time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    running = False; time.sleep(0.6)
    dll.rtlsdr_close(dev)
    print("已退出")
