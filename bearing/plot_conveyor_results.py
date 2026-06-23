import matplotlib.pyplot as plt
import numpy as np
import os

# Data dari Tabel 4.1 sampai 4.4
speeds = ['20%', '40%', '60%', '80%']

# Data Latency (ms)
latency_20 = [35.7, 41.1, 47.8, 26.6, 30.5, 27.7]
latency_40 = [29.6, 26.5, 35.2, 35.1, 30.1, 36.2]
latency_60 = [38.5, 28.5, 29.1, 26.2, 31.5, np.nan] # Item 6 tidak terbaca
latency_80 = [38.5, 28.5, 29.1, 26.2, 31.5, np.nan] # Sesuai gambar, data sama dengan 60%

# Data Confidence
conf_20 = [0.75, 0.81, 0.81, 0.64, 0.62, 0.62]
conf_40 = [0.71, 0.67, 0.75, 0.71, 0.71, 0.65]
conf_60 = [0.75, 0.85, 0.82, 0.78, 0.76, np.nan]
conf_80 = [0.75, 0.85, 0.82, 0.78, 0.76, np.nan]

# Fungsi untuk menghitung rata-rata latency (abaikan NaN)
def calc_avg_latency(data):
    clean_data = [x for x in data if not np.isnan(x)]
    return np.mean(clean_data) if clean_data else 0

# Fungsi untuk menghitung rata-rata confidence (NaN dianggap 0)
def calc_avg_conf(data):
    clean_data = [x if not np.isnan(x) else 0 for x in data]
    return np.mean(clean_data)

avg_latency = [calc_avg_latency(latency_20), calc_avg_latency(latency_40), calc_avg_latency(latency_60), calc_avg_latency(latency_80)]
avg_conf = [calc_avg_conf(conf_20) * 100, calc_avg_conf(conf_40) * 100, calc_avg_conf(conf_60) * 100, calc_avg_conf(conf_80) * 100]

readability = [
    (6 - np.isnan(latency_20).sum()) / 6 * 100,
    (6 - np.isnan(latency_40).sum()) / 6 * 100,
    (6 - np.isnan(latency_60).sum()) / 6 * 100,
    (6 - np.isnan(latency_80).sum()) / 6 * 100,
]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Analisis Performa Sistem Berdasarkan Kecepatan Conveyor', fontsize=18, fontweight='bold', y=1.05)

# --- Plot 1: Rata-rata Latency ---
bars1 = axs[0].bar(speeds, avg_latency, color=['#4fc3f7', '#29b6f6', '#03a9f4', '#0288d1'])
axs[0].set_title('Rata-rata Latency', fontsize=14, pad=10)
axs[0].set_xlabel('Kecepatan Conveyor', fontsize=12)
axs[0].set_ylabel('Latency (ms)', fontsize=12)
axs[0].set_ylim(0, max(avg_latency) + 10)
axs[0].grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f} ms', ha='center', va='bottom', fontweight='bold')

# --- Plot 2: Rata-rata Confidence ---
bars2 = axs[1].bar(speeds, avg_conf, color=['#81c784', '#66bb6a', '#4caf50', '#43a047'])
axs[1].set_title('Rata-rata Confidence', fontsize=14, pad=10)
axs[1].set_xlabel('Kecepatan Conveyor', fontsize=12)
axs[1].set_ylabel('Confidence (%)', fontsize=12)
axs[1].set_ylim(0, 100)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    axs[1].text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

# --- Plot 3: Tingkat Keterbacaan Objek ---
axs[2].plot(speeds, readability, marker='o', linestyle='-', color='#e53935', linewidth=2, markersize=8)
axs[2].set_title('Tingkat Keberhasilan Deteksi', fontsize=14, pad=10)
axs[2].set_xlabel('Kecepatan Conveyor', fontsize=12)
axs[2].set_ylabel('Keberhasilan (%)', fontsize=12)
axs[2].set_ylim(0, 110)
axs[2].grid(True, linestyle='--', alpha=0.7)
for i, v in enumerate(readability):
    axs[2].text(i, v + 3, f'{v:.0f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('grafik_kombinasi_conveyor.png', dpi=300, bbox_inches='tight')
print("Berhasil menyimpan grafik_kombinasi_conveyor.png")
