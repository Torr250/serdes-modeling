#!/usr/bin/env python3
"""jitter_budget_pcie_gen2.py
Simple numerical example and simulator to compute jitter budget for PCIe Gen2 (5 Gbps).
Models deterministic jitter (DJ) from ISI (via simple channel taps), DCD, and spurs (tones in dBc),
and computes required RJ given TJ target.
"""
import numpy as np

# --- Parameters ---
bit_rate = 5e9               # 5 Gbps
UI = 1.0 / bit_rate         # UI in seconds (200 ps)
TJ_target_UI = 0.40         # target total TJ in UI (peak-to-peak)
TJ_target = TJ_target_UI * UI
BER = 1e-12
# For BER=1e-12 the Gaussian Q ~ 7.034 (precomputed)
Q = 7.034

# ISI/channel model: taps in units of UI (current bit = tap 0, previous bits tap 1..)
channel_taps = [1.0, 0.25, 0.12]   # simple low-order ISI example
osr = 128                          # oversampling per UI for analog waveform

# DCD: fractional deviation of duty around 0.5 (e.g., 0.03 -> 3% duty error)
dcd_delta = 0.03   # fraction of UI (per half period offset)

# Spurs: list of (dBc_relative_to_carrier, spur_frequency_Hz)
spurs = [(-60, 1e6), (-70, 5e6)]   # example tones (dBc, Hz)

# RJ components (example guesses) in ps RMS (will be combined by RSS)
# These are illustrative and can be adjusted.
rj_components_ps = {
    'PLL/VCO': 2.5,
    'Tx_driver': 2.5,
    'Channel/Receiver': 1.5,
}

# --- Helper functions ---

def prbs_sequence(n_bits, seed=1):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n_bits)


def nrz_levels(bits):
    # NRZ: 0 -> -1, 1 -> +1
    return 2*bits - 1


def simulate_waveform(levels, taps, osr):
    # convolve per-bit taps to produce per-sample waveform
    # Upsample levels by osr
    up = np.repeat(levels, osr)
    # Make impulse response in samples: taps spaced by osr
    h_samples = np.zeros(len(taps)*osr)
    for i, a in enumerate(taps):
        h_samples[i*osr] = a
    wf = np.convolve(up, h_samples)[:len(up)]
    return wf


def find_zero_crossings(wf, dt):
    # returns times (s) of rising/falling transitions (linear interp between samples)
    idx = np.where(np.sign(wf[:-1]) != np.sign(wf[1:]))[0]
    times = []
    for i in idx:
        v1, v2 = wf[i], wf[i+1]
        if v2 == v1:
            frac = 0.0
        else:
            frac = -v1 / (v2 - v1)
        t = (i + frac) * dt
        times.append(t)
    return np.array(times)


def compute_isi_dj_pp(t_transitions, ideal_T):
    # compute timing error relative to nearest ideal transition grid (multiples of UI/2)
    # We map each transition to nearest ideal index
    errs = []
    for t in t_transitions:
        k = int(round(t / ideal_T))
        errs.append(t - k*ideal_T)
    if len(errs) == 0:
        return 0.0
    errs = np.array(errs)
    dj_pp = (errs.max() - errs.min())
    return dj_pp


def spur_dbc_to_djpp(dbc, f_carrier):
    # convert spur level in dBc to deterministic peak-to-peak time jitter (approx small-signal)
    R = 10**(dbc/10.0)
    beta = 2.0 * np.sqrt(R)             # rad
    dt_peak = beta / (2.0 * np.pi * f_carrier)
    return 2.0 * dt_peak

# --- Simulation / Numerical example ---

# 1) ISI simulation using PRBS
n_bits = 2000
bits = prbs_sequence(n_bits)
levels = nrz_levels(bits)
wf = simulate_waveform(levels, channel_taps, osr)
dt = UI / osr

# find transitions and compute ISI-induced DJ
t_trans = find_zero_crossings(wf, dt)
# Use ideal transition period = UI/2 (since edges every half-UI for NRZ)
ideal_T = UI / 2.0
dj_pp_isi = compute_isi_dj_pp(t_trans[:1000], ideal_T)  # use first transitions

# convert to ps
dj_pp_isi_ps = dj_pp_isi * 1e12

# 2) DCD
# DCD peak-to-peak in seconds = 2 * delta * UI
dj_pp_dcd = 2.0 * dcd_delta * UI
dj_pp_dcd_ps = dj_pp_dcd * 1e12

# 3) Spurs
dj_pp_spurs = 0.0
dj_pp_spurs_list = []
for dbc, f_sp in spurs:
    # convert using carrier = bit_rate (approx sampling around bit transitions)
    dj = spur_dbc_to_djpp(dbc, bit_rate)
    dj_pp_spurs += dj
    dj_pp_spurs_list.append((dbc, f_sp, dj*1e12))

dj_pp_spurs_ps = dj_pp_spurs * 1e12

# Aggregate deterministic jitter DJ_pp (worst-case add)
dj_pp_total = dj_pp_isi + dj_pp_dcd + dj_pp_spurs
dj_pp_total_ps = dj_pp_total * 1e12

# Compute required RJ (rms) from TJ_target and DJ_pp_total
required_rj = (TJ_target - dj_pp_total) / (2.0 * Q)
required_rj_ps = required_rj * 1e12

# RSS of example RJ components
rss_rj_ps = np.sqrt(sum(v*v for v in rj_components_ps.values()))

# --- Print results ---
print("PCIe Gen2 (5 Gbps) jitter budget example:\n")
print(f"UI = {UI*1e12:.1f} ps")
print(f"Target TJ_pp = {TJ_target_UI:.2f} UI = {TJ_target*1e12:.2f} ps")
print('\nDeterministic jitter (DJ) components:')
print(f"  ISI (simulated) DJ_pp = {dj_pp_isi_ps:.2f} ps")
print(f"  DCD (delta={dcd_delta:.3f}) DJ_pp = {dj_pp_dcd_ps:.2f} ps")
print('  Spurs (sum of DJ_pp contributions):')
for dbc, f_sp, djps in dj_pp_spurs_list:
    print(f"    spur {dbc} dBc -> DJ_pp = {djps:.4f} ps")
print(f"  Total DJ_pp (worst-case sum) = {dj_pp_total_ps:.3f} ps")

print('\nRandom jitter (RJ) requirement:')
print(f"  Required RJ_rms (from target) = {required_rj_ps:.3f} ps RMS")
print('\nExample RJ component RSS:')
for name, val in rj_components_ps.items():
    print(f"  {name}: {val:.2f} ps RMS")
print(f"  RSS total = {rss_rj_ps:.3f} ps RMS")

# Check whether example RJ fits requirement
print('\nVerdict:')
if rss_rj_ps <= required_rj_ps:
    print("  Example RJ components PASS the required RJ budget.")
else:
    print("  Example RJ components EXCEED the required RJ budget — reduce RJ or DJ.")

# Print recommended breakdown in UI and ps
print('\nSummary (ps):')
print(f"  TJ_target_pp = {TJ_target*1e12:.2f} ps")
print(f"  DJ_pp_total = {dj_pp_total_ps:.2f} ps")
print(f"  Required RJ_rms = {required_rj_ps:.3f} ps RMS")

# End
