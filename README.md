# ForestWatch — Forest Change Classification

## Deskripsi

ForestWatch adalah proyek Python sederhana untuk mempelajari klasifikasi penggerak perubahan tutupan hutan menggunakan Machine Learning. Proyek ini terinspirasi oleh penelitian yang memanfaatkan citra satelit multi-sensor dan AI untuk mengidentifikasi penyebab deforestasi.

## Fitur

* Membuat dataset simulasi perubahan hutan.
* Melatih model Random Forest.
* Mengevaluasi model menggunakan Accuracy dan Macro F1-score.
* Menampilkan grafik distribusi kategori dan kepentingan fitur.

## Teknologi

* Python
* Pandas
* Scikit-learn
* Matplotlib

## Cara Menjalankan

Instal dependensi:

```bash
pip install -r requirements.txt
```

Jalankan program:

```bash
python src/main.py
```

Jalankan pengujian:

```bash
python -m unittest discover -s tests -v
```

## Kategori Klasifikasi

Pertanian skala kecil, pertanian skala besar, pembangunan jalan, penebangan selektif, penambangan, kebakaran hutan, dan gangguan alami.

## Batasan

Dataset yang digunakan merupakan data sintetis untuk pembelajaran, bukan citra Sentinel-1 atau Sentinel-2 yang sebenarnya. Hasil model tidak dapat digunakan sebagai bukti penyebab deforestasi nyata.

## Pengembangan Selanjutnya

Menggunakan data satelit publik, menambahkan fitur spasial dan temporal, serta membandingkan Random Forest dengan model lain.
