# -*- coding: utf-8 -*-
"""SDR FM radio algorithm figures (Nature style, engine A).
R1: initial — fig1 system chain, fig2 mpx spectrum + wiener gain.
"""
import sys, math
from pathlib import Path
import numpy as np
SKILL = Path(r'C:\Users\wuzhe\.omp\agent\managed-skills\academic-figure-skill')
sys.path.insert(0, str(SKILL / 'lib'))
import figure_style as fs
from figure_style import C, F, text, box, arrow, elbow, title, export
import matplotlib.pyplot as plt

OUT = Path('figs'); OUT.mkdir(exist_ok=True)

# ================= Figure 1: system chain =================
LANES = [
    ('RF to baseband', 'compute', [
        ('RTL-SDR\n960 ksps IQ', 'route'),
        ('DC offset\nremoval', 'compute'),
        ('IQ amplitude\nbalancing', 'compute'),
        ('Channel filter\n\u00b180 kHz', 'compute'),
        ('Decimate \u00f78\n\u2192 120 ksps', 'memory'),
        ('Quadrature\nFM detector', 'event'),
    ]),
    ('Stereo decoding', 'memory', [
        ('MPX\n(demodulated)', 'event'),
        ('19 kHz pilot\nresonator', 'event'),
        ('Square +\n38 kHz resonator', 'event'),
        ('DSB-SC coherent\ndemod \u2192 L\u2212R', 'event'),
        ('15 kHz LPF\n\u2192 L+R', 'memory'),
        ('Adaptive\nblend', 'route'),
    ]),
    ('Audio back-end', 'control', [
        ('Fixed gain +\nsoft limiter', 'control'),
        ('Resample\n\u2192 48 ksps', 'memory'),
        ('75 \u00b5s\nde-emphasis', 'memory'),
        ('Quality gate\n(pilot/noise)', 'route'),
        ('HF stereo blend\n+ Wiener NR', 'control'),
        ('ASRC +\nplayback', 'compute'),
    ]),
]

BOX_W, BOX_H, GAP = 24.0, 9.0, 4.6
X0 = 8.0
LANE_H = 24.0                      # lane pitch
fig1_h = 10 + 3 * LANE_H + 6
fig = fs.canvas(fig1_h)
titles = ['a', 'b', 'c']
for li, (lane_name, _, boxes) in enumerate(LANES):
    ytop = 8 + li * LANE_H
    title(fig, 5, ytop + 2.2, titles[li], lane_name)
    for bi, (label, kind) in enumerate(boxes):
        x = X0 + bi * (BOX_W + GAP)
        box(fig, x, ytop + 7, BOX_W, BOX_H, label, kind=kind)
        if bi < len(boxes) - 1:
            arrow(fig, (x + BOX_W, ytop + 7 + BOX_H / 2),
                  (x + BOX_W + GAP, ytop + 7 + BOX_H / 2))
    if li < len(LANES) - 1:        # lane-to-lane connector: 底边中点下行折返
        xr = X0 + 5 * (BOX_W + GAP) + BOX_W / 2
        y_bot = ytop + 7 + BOX_H
        elbow(fig, [(xr, y_bot), (xr, y_bot + 5),
                    (X0 + BOX_W / 2, y_bot + 5), (X0 + BOX_W / 2, y_bot + LANE_H - BOX_H - 7)])
export(fig, OUT, 'fig1_chain', panels=['a RF', 'b stereo', 'c audio'])

W2 = 81.0
# ================= Figure 2: spectrum + wiener gain =================
fig2_h = 66.0
fig = fs.canvas(fig2_h)

# ---- panel a: FM multiplex spectrum (schematic) ----
ax = fs.axis(fig, 8, 12, W2, 42)
for a_ in (fig._ax,):
    a_.set_zorder(20); a_.patch.set_visible(False)
rng = np.random.default_rng(7)
fx = np.linspace(0, 60, 1200)
def band(f, lo, hi, n=6, base=0.32):
    m = np.where((f > lo) & (f < hi), 1.0, 0.0)
    bumps = np.zeros_like(f)
    for c, a in zip(rng.uniform(lo + 1, hi - 1, n), rng.uniform(0.5, 1, n)):
        bumps += a * np.exp(-((f - c) / 1.2) ** 2)
    return base * m + 0.55 * bumps * m + 0.045
sig = band(fx, 0.4, 15, 9) + band(fx, 23, 53, 12, base=0.26)
pilot = 0.9 * np.exp(-((fx - 19) / 0.25) ** 2)
noise = 0.05 + 0.012 * np.sin(fx * 3.1)
ax.fill_between(fx, 0, sig, color=F['memory'], edgecolor=C['memory'], lw=.7)
ax.fill_between(fx, 0, noise, color=C['grey'], lw=0)
ax.plot(fx, pilot, color=C['control'], lw=1.1)
ax.set_xlim(0, 60); ax.set_ylim(0, 1.05)
ax.set_xticks([0, 10, 19, 38, 53, 60])
ax.set_xlabel('Frequency (kHz)', labelpad=1.2)
ax.set_ylabel('Magnitude', labelpad=1.2)
ax.set_yticks([])
text(fig, 8 + 10, 16.5, 'L+R  (0\u201315 kHz)', 6, color=C['memory'], ha='center')
text(fig, 8 + 38, 16.5, 'L\u2212R DSB-SC  (23\u201353 kHz)', 6, color=C['memory'], ha='center')
text(fig, 8 + 23.5, 26, 'pilot 19 kHz', 5.5, color=C['control'])
arrow(fig, (8 + 22.0, 25.0), (8 + 19.6, 29.0), color=C['control'], rad=-.25)
text(fig, 8 + 47, 43.5, 'noise ref 16.5 kHz', 5.5, color=C['muted'])
ax.plot([16.5, 16.5], [0, .34], ls='--', lw=.6, color=C['muted'])
text(fig, 8 + 2.5, 40.5, '16.5 kHz resonator \u2192\nnoise floor estimate', 5.5,
     color=C['muted'])
title(fig, 5, 8, 'a', 'FM stereo multiplex (post-discriminator)')

# ---- panel b: Wiener spectral-subtraction gain ----
ax2 = fs.axis(fig, 8 + W2 + 8, 12, W2, 42)
snr = np.linspace(-15, 30, 500)          # dB
g = np.clip(1 - 2.6 * 10 ** (-snr / 20), 0.04, 1.0)
ax2.plot(snr, g, color=C['compute'], lw=1.2)
ax2.axhline(1.0, ls='--', lw=.55, color=C['muted'])
ax2.axhline(0.04, ls='--', lw=.55, color=C['control'])
ax2.set_xlim(-15, 30); ax2.set_ylim(0, 1.15)
ax2.set_xlabel('Per-bin SNR (dB)', labelpad=1.2)
ax2.set_ylabel('Wiener gain G(f)', labelpad=1.2)
ax2.set_yticks([0, 0.5, 1])
text(fig, 8 + W2 + 8 + 58, 12 + 9, 'unity', 5.5, color=C['muted'])
text(fig, 8 + W2 + 8 + 58, 12 + 40.5, 'floor \u221228 dB', 5.5, color=C['control'])
text(fig, 8 + W2 + 8 + 16, 12 + 33, 'hiss bins \u2192 suppressed', 6, color=C['compute'])
arrow(fig, (8 + W2 + 8 + 24, 12 + 31.5), (8 + W2 + 8 + 8, 12 + 22), color=C['compute'], rad=.2)
text(fig, 8 + W2 + 8 + 46, 12 + 16, 'signal bins \u2192 preserved', 6, color=C['compute'])
arrow(fig, (8 + W2 + 8 + 52, 12 + 18.5), (8 + W2 + 8 + 63, 12 + 24.5), color=C['compute'], rad=.2)
title(fig, 8 + W2 + 8 - 3, 8, 'b', 'Adaptive Wiener gain (strong stations only)')
export(fig, OUT, 'fig2_nature', panels=['a mpx', 'b gain'])
print('done')
