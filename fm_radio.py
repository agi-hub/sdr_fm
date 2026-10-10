# -*- coding: utf-8 -*-
"""
RTL-SDR FM 立体声收音机 (高级DSP版 v3: 全链路滤波器跨块连续, 无块边界咔哒声)
用法: python fm_radio.py [频率MHz]   默认 100.6
信号链: 960k IQ -> DC消除 -> IQ平衡 -> 80k信道滤波 -> 120k基带 -> 正交鉴频
        -> 导频谐振锁相立体声解码 -> 自适应混合降噪 -> 定增益软限幅
        -> 11k低通 -> 75us去加重 -> 维纳谱减降噪 -> ASRC -> 48k立体声
依赖: numpy, scipy, sounddevice, 64位 rtlsdr.dll (./rtlsdr_x64/)
"""
import ctypes, os, sys, queue, threading, time, wave
import numpy as np
from scipy.signal import butter, lfilter, resample_poly

# ---------- RTL-SDR ----------
dll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtlsdr_x64", "rtlsdr.dll"))
dll.rtlsdr_open.restype = ctypes.c_int
dll.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
dll.rtlsdr_get_tuner_type.restype = ctypes.c_int
dll.rtlsdr_set_center_freq.restype = ctypes.c_int
dll.rtlsdr_set_sample_rate.restype = ctypes.c_int
dll.rtlsdr_set_tuner_gain_mode.restype = ctypes.c_int
dll.rtlsdr_set_tuner_gain.restype = ctypes.c_int
dll.rtlsdr_set_agc_mode.restype = ctypes.c_int
dll.rtlsdr_reset_buffer.restype = ctypes.c_int
dll.rtlsdr_read_sync.restype = ctypes.c_int
dll.rtlsdr_read_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
dll.rtlsdr_close.restype = ctypes.c_int

FREQ = int(float(sys.argv[1]) * 1e6) if len(sys.argv) > 1 else 100_600_000
FS = 960_000          # RTL 采样率
IF_FS = 120_000       # 鉴频采样率 (FS/8)
AUDIO_FS = 48_000

dev = ctypes.c_void_p()
assert dll.rtlsdr_open(ctypes.byref(dev), 0) == 0, "rtlsdr_open 失败(设备被占用?)"
print(f"tuner type: {dll.rtlsdr_get_tuner_type(dev)}")
dll.rtlsdr_set_sample_rate(dev, FS)
dll.rtlsdr_set_center_freq(dev, FREQ)
dll.rtlsdr_set_tuner_gain_mode(dev, 1)
dll.rtlsdr_set_tuner_gain(dev, 495)        # 最大增益 (信号弱)
dll.rtlsdr_set_agc_mode(dev, 0)
dll.rtlsdr_reset_buffer(dev)
print(f"已调谐到 {FREQ/1e6:.3f} MHz")

# ---------- 滤波器设计 ----------
def resonator(f0, fs, q=60):
    """窄带谐振器 (带通), 用于导频/副载波恢复"""
    w0 = 2 * np.pi * f0 / fs
    r = min(1 - np.pi * f0 / fs / (q / 2), 0.9995)
    a = np.array([1.0, -2 * r * np.cos(w0), r * r])
    b = np.array([1.0 - r])
    return b, a

pil_b, pil_a = resonator(19_000, IF_FS, q=80)     # 19kHz 导频
sub_b, sub_a = resonator(38_000, IF_FS, q=40)     # 38kHz 副载波
lp15 = butter(4, 15e3 / (IF_FS / 2), btype="low") # 多路信号 15k 低通
lp_if = butter(4, 80e3 / (FS / 2), btype="low")   # 鉴频前信道滤波 (960k域)
nz_b, nz_a = resonator(16_500, IF_FS, q=40)     # 16.5kHz 噪声参考
lp3k = butter(2, 3e3 / (AUDIO_FS / 2), btype="low") # 高频混合分频点
AUDIO_BW = 15_000                                  # 完整音频带宽
NR_EN = os.environ.get("FM_NR", "0") == "1"        # 默认关闭谱减降噪 (FM_NR=1 启用)
lp_aud = butter(6, AUDIO_BW / (AUDIO_FS / 2), btype="low")
TAU = 75e-6
de_a = np.exp(-1 / (TAU * AUDIO_FS))

# ---------- 维纳谱减降噪器 (带跨块历史, 无块边界痕迹) ----------
NR_N = 1024; NR_HOP = 512
nr_win = np.sqrt(np.hanning(NR_N)).astype(np.float64)
noise_lo_bin = int(12_000 / (AUDIO_FS / 2) * (NR_N // 2))

class Denoiser:
    """维纳谱减降噪器: 持久缓冲+绝对帧格, 样本严格守恒, 无块边界痕迹"""
    def __init__(self):
        self.buf = np.zeros(0); self.out = np.zeros(0); self.wsum = np.zeros(0)
        self.noise = None

    def process(self, new):
        self.buf = np.concatenate([self.buf, new.astype(np.float64)])
        s = 0
        while s + NR_N <= len(self.buf):
            frag_src = self.buf[s:s + NR_N] * nr_win
            S = np.fft.rfft(frag_src)
            mag = np.abs(S)
            # 噪声底: 帧内 12kHz 以上纯噪声段的中位数 (标量平坦模型), EMA平滑
            nf = np.median(mag[noise_lo_bin:])
            self.noise = nf if self.noise is None else 0.9 * self.noise + 0.1 * nf
            gain = np.clip(1.0 - 2.6 * self.noise / (mag + 1e-9), 0.04, 1.0)
            frag = np.fft.irfft(S * gain) * nr_win
            need = s + NR_N
            if len(self.out) < need:
                pad = need - len(self.out)
                self.out = np.pad(self.out, (0, pad)); self.wsum = np.pad(self.wsum, (0, pad))
            self.out[s:need] += frag
            self.wsum[s:need] += nr_win ** 2
            s += NR_HOP
        emit_end = s   # 已有完整帧覆盖的区域起点 (其后样本留待下块补帧)
        if emit_end > 0:
            res = self.out[:emit_end] / np.maximum(self.wsum[:emit_end], 0.9)
            self.buf = self.buf[emit_end:]; self.out = self.out[emit_end:]; self.wsum = self.wsum[emit_end:]
            return res
        return np.zeros(0)

# ---------- 采集/DSP 线程 ----------
audio_q = queue.Queue(maxsize=64)
rec_q = queue.Queue(maxsize=256)
read_len = FS // 4 * 2     # 0.25s uint8 IQ
running = True

def rx_thread():
    buf = ctypes.create_string_buffer(read_len)
    n = ctypes.c_int()
    # 全部滤波器跨块状态 (消除块边界瞬态)
    zi_if  = np.zeros(max(len(lp_if[0]), len(lp_if[1])) - 1, dtype=np.complex64)
    zi_pil = np.zeros(max(len(pil_a), len(pil_b)) - 1)
    zi_sub = np.zeros(max(len(sub_a), len(sub_b)) - 1)
    zi_lpbase = np.zeros(max(len(lp15[0]), len(lp15[1])) - 1)
    zi_lplr   = np.zeros(max(len(lp15[0]), len(lp15[1])) - 1)
    zde_l = np.zeros(1); zde_r = np.zeros(1)
    zi_al = np.zeros(max(len(lp_aud[0]), len(lp_aud[1])) - 1)
    zi_ar = np.zeros(max(len(lp_aud[0]), len(lp_aud[1])) - 1)
    den_l = Denoiser(); den_r = Denoiser()
    blend = 0.7
    g_lr = 1.0    # IQ平衡增益 EMA
    q_ema = 1.0   # 信号质量 EMA
    zi_nz = np.zeros(max(len(nz_a), len(nz_b)) - 1)
    zi_hl = np.zeros(len(lp3k[1]) - 1); zi_hr = np.zeros(len(lp3k[1]) - 1)
    while running:
        if dll.rtlsdr_read_sync(dev, buf, read_len, ctypes.byref(n)) != 0 or n.value != read_len:
            continue
        iq = np.frombuffer(buf.raw, dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
        x = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)
        x -= x.mean()                                        # DC偏移消除
        # IQ 幅度平衡 (EMA)
        pr = float(np.mean(x.real ** 2)) + 1e-12; pi_ = float(np.mean(x.imag ** 2)) + 1e-12
        tgt = np.sqrt((pr + pi_) / 2)
        g_lr = 0.95 * g_lr + 0.05 * tgt
        x = (x.real * (g_lr / np.sqrt(pr)) + 1j * x.imag * (g_lr / np.sqrt(pi_))).astype(np.complex64)
        xf, zi_if = lfilter(lp_if[0], lp_if[1], x, zi=zi_if)  # 信道滤波
        xl = resample_poly(xf, 1, 8)                          # -> 120k
        mpx = np.angle(xl[1:] * np.conj(xl[:-1]))             # 鉴频
        pilot, zi_pil = lfilter(pil_b, pil_a, mpx, zi=zi_pil) # 19k 导频
        nz16, zi_nz = lfilter(nz_b, nz_a, mpx, zi=zi_nz)      # 16.5k 噪声参考
        quality = float(np.mean(pilot ** 2)) / (float(np.mean(nz16 ** 2)) + 1e-12)
        q_ema = 0.9 * q_ema + 0.1 * quality                   # 信号质量 EMA
        sub2f, zi_sub = lfilter(sub_b, sub_a, pilot ** 2, zi=zi_sub)
        sub = sub2f / (np.abs(sub2f).mean() + 1e-12)
        lr_raw = mpx * sub * 2.0
        base, zi_lpbase = lfilter(lp15[0], lp15[1], mpx, zi=zi_lpbase)
        lr,  zi_lplr   = lfilter(lp15[0], lp15[1], lr_raw, zi=zi_lplr)
        # 自适应立体声混合
        p_base = np.mean(base ** 2) + 1e-12
        p_lr = np.mean(lr ** 2) + 1e-12
        a = float(np.clip((p_base / p_lr - 1.2) / 2.0, 0.0, 1.0))
        blend = 0.9 * blend + 0.1 * a
        lr_eff = lr * blend
        L = (base + lr_eff) * 0.5
        R = (base - lr_eff) * 0.5
        # 定增益 + 软限幅
        L = np.tanh(L * 0.66) * 0.9
        R = np.tanh(R * 0.66) * 0.9
        # -> 48k, 去加重 (带状态), 音频低通 (带状态)
        La = resample_poly(L, 2, 5); Ra = resample_poly(R, 2, 5)
        La, zde_l = lfilter([de_a], [1, -(1 - de_a)], La, zi=zde_l)
        Ra, zde_r = lfilter([de_a], [1, -(1 - de_a)], Ra, zi=zde_r)
        La, zi_al = lfilter(lp_aud[0], lp_aud[1], La, zi=zi_al)
        Ra, zi_ar = lfilter(lp_aud[0], lp_aud[1], Ra, zi=zi_ar)
        # 高频立体声混合: 3kHz以上部分并单声道 (嘶声主要来自L-R差通道)
        if q_ema > 15:
            lh, zi_hl = lfilter(lp3k[0], lp3k[1], La, zi=zi_hl)
            rh, zi_hr = lfilter(lp3k[0], lp3k[1], Ra, zi=zi_hr)
            lhf, rhf = La - lh, Ra - rh
            mono_hf = 0.5 * (lhf + rhf)
            La = lh + 0.4 * lhf + 0.6 * mono_hf
            Ra = rh + 0.4 * rhf + 0.6 * mono_hf
        if NR_EN or q_ema > 15:   # 自适应: 强台(导频噪比>15)自动启用降噪
            La = den_l.process(La); Ra = den_r.process(Ra)   # 维纳谱减降噪
        # ASRC: 补偿 RTL 时钟偏差
        err = 4 - audio_q.qsize()
        factor = 1.0 + max(-0.10, min(0.10, err * 0.03))
        n_out = int(round(len(La) * factor))
        idx = np.linspace(0, len(La) - 1, n_out)
        xi = np.arange(len(La))
        stereo = np.stack([np.interp(idx, xi, La), np.interp(idx, xi, Ra)], axis=1).astype(np.float32)
        audio_q.put(stereo)
        rec_q.put(stereo)

def rec_thread():
    while running or not rec_q.empty():
        try:
            blk = rec_q.get(timeout=0.5)
        except queue.Empty:
            continue
        wf.writeframes((np.clip(blk, -1, 1) * 32767).astype(np.int16).tobytes())

# ---------- 录音 ----------
wav_name = f"fm_{FREQ/1e6:g}MHz.wav".replace(".", "_", 1)
wf = wave.open(wav_name, "wb"); wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(AUDIO_FS)
rx = threading.Thread(target=rx_thread, daemon=True); rx.start()
rec = threading.Thread(target=rec_thread, daemon=True); rec.start()
while audio_q.qsize() < 4:
    time.sleep(0.1)

# ---------- 播放 ----------
import sounddevice as sd
print("开始收听 (Ctrl+C 退出)...")
carry = np.zeros((0, 2), dtype=np.float32)
underruns = [0]

def callback(outdata, frames, time_info, status):
    global carry
    while len(carry) < frames:
        try:
            carry = np.concatenate([carry, audio_q.get_nowait()])
        except queue.Empty:
            break
    out = carry[:frames]
    if len(out) < frames:
        underruns[0] += 1
        out = np.pad(out, ((0, frames - len(out)), (0, 0)))
    carry = carry[frames:]
    outdata[:] = out

try:
    print("音频设备:", sd.query_devices(kind="output")["name"])
    with sd.OutputStream(samplerate=AUDIO_FS, channels=2, dtype="float32",
                         blocksize=0, callback=callback):
        while True:
            time.sleep(2)
            print(f"队列 {audio_q.qsize():3d}  欠载累计 {underruns[0]:4d}", flush=True)
except KeyboardInterrupt:
    pass
finally:
    running = False
    time.sleep(0.6)
    dll.rtlsdr_close(dev)
    wf.close()
    print(f"\n已退出, 立体声录音已保存到 {wav_name}")
