import matplotlib.pyplot as plt

# Data dari tabel baru
speeds = ['20%', '40%', '60%', '80%']
akurasi = [100, 100, 90, 80]
confidence = [72, 70, 65, 55]
latency = [36.74, 33.29, 31.31, 36.09]

# --- Plot 1: Persentase Akurasi ---
plt.figure(figsize=(8, 5))
bars1 = plt.bar(speeds, akurasi, color=['#e53935', '#e53935', '#ef5350', '#ff8a80'])
plt.title('Persentase Akurasi vs Kecepatan Conveyor (Total 10 Sampel)', fontsize=14, pad=15)
plt.xlabel('Kecepatan Conveyor', fontsize=12)
plt.ylabel('Akurasi (%)', fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('grafik_total_akurasi.png', dpi=300)
plt.close()

# --- Plot 2: Rata-rata Confidence ---
plt.figure(figsize=(8, 5))
bars2 = plt.bar(speeds, confidence, color=['#4caf50', '#66bb6a', '#81c784', '#a5d6a7'])
plt.title('Rata-rata Confidence vs Kecepatan Conveyor (Total 10 Sampel)', fontsize=14, pad=15)
plt.xlabel('Kecepatan Conveyor', fontsize=12)
plt.ylabel('Confidence (%)', fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('grafik_total_confidence.png', dpi=300)
plt.close()

# --- Plot 3: Rata-rata Latency ---
plt.figure(figsize=(8, 5))
bars3 = plt.bar(speeds, latency, color=['#03a9f4', '#29b6f6', '#4fc3f7', '#81d4fa'])
plt.title('Rata-rata Latency vs Kecepatan Conveyor (Total 10 Sampel)', fontsize=14, pad=15)
plt.xlabel('Kecepatan Conveyor', fontsize=12)
plt.ylabel('Latency (ms)', fontsize=12)
plt.ylim(0, max(latency) + 10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f} ms', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('grafik_total_latency.png', dpi=300)
plt.close()

print("Berhasil menyimpan 3 grafik secara terpisah: grafik_total_akurasi.png, grafik_total_confidence.png, grafik_total_latency.png")
