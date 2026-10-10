# -*- coding: utf-8 -*-
"""
RTL-SDR ADS-B 接收机 (1090 MHz Mode S / DF17)
物理层: 2.4Msps 采样 -> 包络 -> 前导检测 -> PPM 解调 -> CRC24 校验
链路层: DF17 解析: 呼号 / 高度 / CPR 全球定位解算 / 速度
输出: 终端飞机列表 + adsb_aircraft.json + map.html
用法: python adsb.py [运行秒数, 默认60]
"""
import ctypes, os, sys, time, json, html
import numpy as np

RUN_SEC = int(sys.argv[1]) if len(sys.argv) > 1 else 60
FS = 2_400_000
dll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtlsdr_x64", "rtlsdr.dll"))
dll.rtlsdr_open.restype = ctypes.c_int
dll.rtlsdr_open.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
dll.rtlsdr_set_sample_rate.restype = ctypes.c_int
dll.rtlsdr_set_center_freq.restype = ctypes.c_int
dll.rtlsdr_set_tuner_gain_mode.restype = ctypes.c_int
dll.rtlsdr_set_tuner_gain.restype = ctypes.c_int
dll.rtlsdr_set_agc_mode.restype = ctypes.c_int
dll.rtlsdr_reset_buffer.restype = ctypes.c_int
dll.rtlsdr_read_sync.restype = ctypes.c_int
dll.rtlsdr_read_sync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
dll.rtlsdr_close.restype = ctypes.c_int

dev = ctypes.c_void_p()
assert dll.rtlsdr_open(ctypes.byref(dev), 0) == 0, "设备打开失败(被占用?)"
dll.rtlsdr_set_sample_rate(dev, FS)
dll.rtlsdr_set_center_freq(dev, 1_090_000_000)
dll.rtlsdr_set_tuner_gain_mode(dev, 1)
dll.rtlsdr_set_tuner_gain(dev, 495)     # 1090弱信号, 增益拉满
dll.rtlsdr_set_agc_mode(dev, 0)
dll.rtlsdr_reset_buffer(dev)
print("调谐 1090 MHz, 采样率 2.4 Msps")

SAMPLE = 1e6 / FS                       # 每个码元的采样数 (2.4)
BIT_LEN = SAMPLE                        # PPM 每比特 1µs
PREAMBLE_PULSES = [0.0, 1.5, 3.5, 4.5]  # µs, 宽 0.5µs 的 4 个脉冲

def bits_to_bytes(bits):
    b = bytearray()
    for i in range(0, len(bits) - 7, 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[i + j]
        b.append(v)
    return bytes(b)

CRC_POLY = 0xFFF409                     # Mode S CRC-24 生成多项式 (倒序表法用标准值)
def crc24(msg_bytes):
    """Mode S CRC-24, 标准多项式 0x1D40F9 (x24+x23+x22+x21+x20+x19+x18+x17+x16+x15+x14+x13+x12+x10+x3+1)"""
    crc = 0
    for byte in msg_bytes:
        crc ^= byte << 16
        for _ in range(8):
            crc <<= 1
            if crc & 0x1000000:
                crc ^= 0x1D40F9
    return crc & 0xFFFFFF

# 6-bit 字符表 (ICAO)
CH6 = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ#####_###############0123456789######"

def decode_ppm(env, start_idx):
    """从 start_idx (前导起点) 解 112 bit PPM; 返回 bits 或 None"""
    bits = []
    idx = start_idx + 8 * SAMPLE        # 跳过 8µs 前导
    for k in range(112):
        t1 = idx + k * BIT_LEN + 0.25 / 1e-6 / (1 / SAMPLE) * 0  # placeholder
    # 直接浮点索引取样
    bits = []
    for k in range(112):
        base = start_idx + (8 + k) * SAMPLE
        p1 = env[int(round(base + 0.25 * SAMPLE))]   # 前半高 = 1
        p0 = env[int(round(base + 0.75 * SAMPLE))]   # 后半高 = 0
        if p1 > p0: bits.append(1)
        elif p0 > p1: bits.append(0)
        else: return None
    return bits

def find_preambles(env, thresh):
    """粗糙前导检测: 第一个脉冲 + 1.5µs处第二脉冲"""
    hits = []
    n = len(env)
    # 降采样包络平滑 (0.5µs 窗)
    k = max(1, int(0.5 * SAMPLE))
    sm = np.convolve(env, np.ones(k) / k, mode='same')
    high = sm > thresh
    i = 0
    limit = n - 130 * int(SAMPLE)
    step = int(SAMPLE)
    while i < limit:
        if high[i]:
            # 检查脉冲位置 0, 1.5, 3.5, 4.5 µs 处高, 且 1.0/2.5/3.0 低
            p = lambda t: high[int(i + t * SAMPLE)]
            if p(0) and p(1.5) and p(3.5) and p(4.5) and not p(1.0) and not p(2.5) and not p(3.0):
                hits.append(i)
                i += int(120 * SAMPLE)
                continue
            i += 1
        else:
            i += 1
    return hits

# ---------- CPR 全球解算 ----------
NZ = 15
D_LAT_EVEN = 360.0 / (4 * NZ)
D_LAT_ODD = 360.0 / (4 * NZ - 1)

def cpr_global(even, odd):
    """even/odd: (lat_cpr, lon_cpr) 各17bit整数. 返回 (lat, lon) 或 None"""
    lat_e, lon_e = even; lat_o, lon_o = odd
    j = int(np.floor(59 * lat_e - 60 * lat_o + 0.5))
    rlat_e = D_LAT_EVEN * (j % 60 + lat_e / 131072.0)
    rlat_o = D_LAT_ODD * (j % 59 + lat_o / 131072.0)
    if rlat_e >= 270: rlat_e -= 360
    if rlat_o >= 270: rlat_o -= 360
    # 偶帧解算
    ni = lambda lat, is_odd: max(1, int((360 - lat) / D_LAT_EVEN) if not is_odd else (360 - lat) / D_LAT_ODD)
    def lon(lat, cpr_lon, is_odd):
        nl = nl_zone(lat) - (1 if is_odd else 0)
        if nl < 1: return None
        m = int(np.floor(cpr_lon * (nl - 1) - (lon_e if is_odd else lon_o) * nl + 0.5))
        lon_v = (360.0 / nl) * (m % nl + cpr_lon / 131072.0)
        return lon_v - 360 if lon_v >= 180 else lon_v
    lat, lon_v = rlat_e, lon(rlat_e, lat_e and lon_e, False)
    return (rlat_e, lon_v)

def nl_zone(lat):
    """经度带数 NL(lat)"""
    if abs(lat) >= 87: return 1
    if abs(lat) >= 86.53464: return 2
    if abs(lat) >= 85.99564: return 3
    for i, bound in enumerate([85.04606, 83.63813, 82.10162, 80.25976, 78.01525,
                               75.34938, 72.23505, 68.59946, 64.39512, 59.61073,
                               54.11423, 47.86253, 40.78025, 32.48441, 22.81516,
                               10.84724], 4):
        if abs(lat) >= bound: return 59 - (i - 1)
    return 59 - 15

# ---------- 飞机数据库 ----------
aircraft = {}   # icao -> dict(callsign, alt, lat, lon, speed, last_seen, frames)

def process_message(bits, env, start):
    if len(bits) < 112: return
    data = bits_to_bytes(bits[:112])
    df = data[0] >> 3
    icao = data[2:5].hex().upper()
    crc = crc24(data[:11])
    if crc != 0:
        # 尝试 DF11 (56bit): 用11字节含填充不校验; DF17必须校验过
        if df == 17: return
    if df == 17:
        tc = data[4] >> 3
        ac = aircraft.setdefault(icao, dict(icao=icao, frames=0, last_seen=0))
        ac['frames'] += 1
        ac['last_seen'] = time.time()
        if 1 <= tc <= 4:      # 呼号
            cs = ''
            raw = data[5:11]
            v = int.from_bytes(raw, 'big')
            for s in range(6):
                c = (v >> (30 - 6 * s)) & 0x3F
                cs += CH6[c] if c < len(CH6) else ' '
            ac['callsign'] = cs.strip('# ').strip()
        elif 9 <= tc <= 18 or 20 <= tc <= 22:   # 空中位置 (CPR)
            msg = int.from_bytes(data[0:7] + data[7:11], 'big') & ((1 << 88) - 1)
            alt_raw = ((data[5] & 0xFC) << 3) | (data[6] >> 2)
            q = (data[6] >> 4) & 1 if False else (alt_raw & 16) >> 4
            if q:
                alt = ((alt_raw & ~16) >> 4) * 25 - 1000 if False else (((alt_raw & 0x1F80) >> 6) * 25 + (alt_raw & 0x3F) * 25 - 1000)
                alt = (((alt_raw >> 1) & 0xFF0) * 25) + ((alt_raw & 1) << 4) * 0  # 简化见下
                # 标准 25ft: N = (bits 25-1 of 12, Q位置1) 重排
                n12 = ((alt_raw & 0xFFF0) >> 1) | (alt_raw & 1)
                alt = n12 * 25 - 1000
                ac['alt_ft'] = alt
            f = data[6] & 1          # 0=even 1=odd
            lat_cpr = ((data[6] & 1) << 16 | data[7] << 8 | data[8]) & 0x1FFFF
            lat_cpr = ((data[6] & 1) * 131072) | (data[7] << 8) | data[8]
            lon_cpr = (data[9] << 8) | data[10]
            key = 'cpr_even' if f == 0 else 'cpr_odd'
            other = 'cpr_odd' if f == 0 else 'cpr_even'
            ac[key] = (lat_cpr, lon_cpr, time.time())
            if other in ac and time.time() - ac[other][2] < 10:
                pos = cpr_position(ac['cpr_even'][:2], ac['cpr_odd'][:2], ac['cpr_even'][2] > ac['cpr_odd'][2])
                if pos:
                    ac['lat'], ac['lon'] = pos
        elif tc == 19:       # 速度
            ew_dir = (data[5] >> 2) & 1
            ew_v = (((data[5] & 3) << 8) | data[6])
            ns_dir = (data[7] >> 7) & 1
            ns_v = (((data[7] & 0x7F) << 3) | (data[8] >> 5))
            ew_v = ew_v - 1 if ew_dir else ew_v
            ns_v = ns_v - 1 if ns_dir else ns_v
            ac['speed_kt'] = int(round((ew_v ** 2 + ns_v ** 2) ** 0.5))

def cpr_position(even, odd, use_odd):
    lat_e, lon_e = even; lat_o, lon_o = odd
    j = int(np.floor(59 * lat_e - 60 * lat_o + 0.5))
    rlat_e = D_LAT_EVEN * ((j % 60) + lat_e / 131072.0)
    rlat_o = D_LAT_ODD * ((j % 59) + lat_o / 131072.0)
    if rlat_e >= 270: rlat_e -= 360
    if rlat_o >= 270: rlat_o -= 360
    if use_odd:
        lat, cpr_lon, is_odd = rlat_o, lon_o, True
    else:
        lat, cpr_lon, is_odd = rlat_e, lon_e, False
    nl = nl_zone(lat) - (1 if is_odd else 0)
    if nl < 1: return None
    m = int(np.floor(cpr_lon * (nl - 1) - (lon_o if is_odd else lon_e) * nl + 0.5))
    lon_v = (360.0 / nl) * ((m % nl) + cpr_lon / 131072.0)
    if lon_v >= 180: lon_v -= 360
    if not (-90 <= lat <= 90 and -180 <= lon_v <= 180): return None
    return lat, lon_v

# ---------- 主循环 ----------
read_len = FS // 2 * 2     # 0.5s
buf = ctypes.create_string_buffer(read_len)
n_read = ctypes.c_int()
frame_count = 0
crc_ok = 0
t_end = time.time() + RUN_SEC
while time.time() < t_end:
    if dll.rtlsdr_read_sync(dev, buf, read_len, ctypes.byref(n_read)) != 0:
        continue
    iq = np.frombuffer(buf.raw[:n_read.value], dtype=np.uint8).astype(np.float32) / 127.5 - 1.0
    env = np.sqrt(iq[0::2] ** 2 + iq[1::2] ** 2)
    noise = np.percentile(env, 50)
    thresh = noise * 2.2 + 0.02
    for st in find_preambles(env, thresh):
        bits = decode_ppm(env, st)
        if bits is None: continue
        frame_count += 1
        data = bits_to_bytes(bits)
        if crc24(data[:11]) == 0:
            crc_ok += 1
            process_message(bits, env, st)

dll.rtlsdr_close(dev)

# ---------- 输出 ----------
now = time.time()
live = {k: v for k, v in aircraft.items() if now - v.get('last_seen', 0) < 60}
print(f"\n采集 {RUN_SEC}s: 前导检测 {frame_count} 帧, CRC 通过 {crc_ok} 帧")
print(f"跟踪到 {len(live)} 架飞机:\n")
print(f"{'ICAO':>8} {'呼号':>9} {'高度ft':>7} {'纬度':>8} {'经度':>9} {'速度kt':>6} 帧")
rows = []
for k in sorted(live, key=lambda x: -live[x].get('frames', 0)):
    v = live[k]
    lat = v.get('lat'); lon = v.get('lon')
    print(f"{k:>8} {v.get('callsign','—'):>9} {v.get('alt_ft','—'):>7} "
          f"{f'{lat:.4f}' if lat else '—':>8} {f'{lon:.4f}' if lon else '—':>9} "
          f"{v.get('speed_kt','—'):>6} {v['frames']}")
    if lat and lon:
        rows.append(dict(icao=k, cs=v.get('callsign',''), alt=v.get('alt_ft'), lat=lat, lon=lon, spd=v.get('speed_kt')))

json.dump(rows, open('adsb_aircraft.json', 'w'), ensure_ascii=False, indent=1)
# 简单地图页
if rows:
    html_doc = f"""<!doctype html><meta charset="utf-8"><title>ADS-B</title>
<body style="margin:0"><div id="map" style="height:100vh"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script>
var map = L.map('map').setView([{rows[0]['lat']},{rows[0]['lon']}], 8);
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{attribution:'OSM'}}).addTo(map);
{chr(10).join(
    f"L.marker([{r['lat']},{r['lon']}]).addTo(map).bindTooltip('{r['cs'] or r['icao']} {r['alt'] or ''}ft', permanent=true);"
    for r in rows)}
</script>"""
    open('map.html', 'w', encoding='utf-8').write(html_doc)
    print(f"\n地图已生成: map.html ({len(rows)} 架有位置)")
