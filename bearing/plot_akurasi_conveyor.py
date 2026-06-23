import matplotlib.pyplot as plt

# Data dari Tabel Pengujian Aktual (6 sampel per kecepatan)
# 20%: 6/6 terbaca (100%)
# 40%: 6/6 terbaca (100%)
# 60%: 5/6 terbaca (83.3%)
# 80%: 5/6 terbaca (83.3%)
kecepatan = ['20%', '40%', '60%', '80%']
akurasi = [100.0, 100.0, 83.3, 83.3] 

# Membuat figur grafik dengan ukuran proporsional
plt.figure(figsize=(9, 6))

# 1. Membuat Grafik Batang (Bar Chart)
# Warna biru untuk yang 100%, warna oranye/merah untuk yang menurun
colors = ['#4C72B0', '#4C72B0', '#DD8452', '#C44E52']
bars = plt.bar(kecepatan, akurasi, color=colors, width=0.5, alpha=0.9, label='Tingkat Akurasi')

# 2. Membuat Garis Tren (Line Chart) sebagai overlay
plt.plot(kecepatan, akurasi, color='#333333', marker='o', linestyle='dashed', 
         linewidth=2.5, markersize=8, label='Tren Penurunan')

# 3. Menambahkan teks angka persentase tepat di atas setiap batang
for bar, acc in zip(bars, akurasi):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, 
             f"{acc:.1f}%" if acc < 100 else f"{int(acc)}%", 
             ha='center', va='bottom', fontweight='bold', fontsize=12, color='black')

# 4. Kustomisasi Teks, Label, dan Judul
plt.title('Pengaruh Kecepatan Conveyor terhadap Akurasi Inspeksi', fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Kecepatan Conveyor (Persentase PWM)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('Tingkat Akurasi / Readability (%)', fontsize=12, fontweight='bold', labelpad=10)

# 5. Mengatur rentang Sumbu Y (Mulai dari 60 agar kurva penurunannya terlihat dramatis & jelas)
plt.ylim(60, 110)

# 6. Menambahkan garis bantu (Grid) horizontal agar rapi
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 7. Menampilkan Legenda
plt.legend(loc='upper right', fontsize=11)

# Merapikan *layout* dan menampilkan grafik
plt.tight_layout()

# Menyimpan gambar
plt.savefig('Grafik_Akurasi_Conveyor.png', dpi=300)
print("Berhasil menyimpan Grafik_Akurasi_Conveyor.png")
