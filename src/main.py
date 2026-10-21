
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = [
    "area_ha",
    "distance_to_road_km",
    "fire_detected",
    "radar_change",
    "vegetation_loss",
]

DRIVERS = [
    "smallholder_agriculture",
    "large_scale_agriculture",
    "road_development",
    "selective_logging",
    "mining",
    "wildfire",
    "natural_disturbance",
]


def create_dataset(n=350, seed=42):
    """Membuat data simulasi untuk latihan klasifikasi."""
    rng = random.Random(seed)
    rows = []

    # Pola angka ini dibuat untuk simulasi, bukan hasil satelit nyata.
    patterns = {
        "smallholder_agriculture": (3, 1.8, 0, 0.40, 0.45),
        "large_scale_agriculture": (130, 0.9, 0, 0.78, 0.85),
        "road_development": (9, 0.2, 0, 0.52, 0.58),
        "selective_logging": (5, 3.0, 0, 0.30, 0.28),
        "mining": (18, 1.0, 0, 0.81, 0.78),
        "wildfire": (32, 2.0, 1, 0.70, 0.90),
        "natural_disturbance": (11, 5.0, 0, 0.55, 0.36),
    }

    for i in range(n):
        driver = DRIVERS[i % len(DRIVERS)]
        area, road, fire, radar, vegetation = patterns[driver]

        rows.append({
            "area_ha": round(max(0.1, rng.gauss(area, max(area * 0.18, 0.5))), 2),
            "distance_to_road_km": round(
                max(0, rng.gauss(road, 0.35)), 2
            ),
            "fire_detected": (
                fire if rng.random() > 0.08 else 1 - fire
            ),
            "radar_change": round(
                min(1, max(0, rng.gauss(radar, 0.08))), 3
            ),
            "vegetation_loss": round(
                min(1, max(0, rng.gauss(vegetation, 0.08))), 3
            ),
            "driver": driver,
        })

    return pd.DataFrame(rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Simpan data sintetis agar dataset dapat diperiksa.
    df = create_dataset()
    dataset_path = OUTPUT_DIR / "synthetic_forest_alerts.csv"
    df.to_csv(dataset_path, index=False)

    X = df[FEATURES]
    y = df["driver"]

    # Pisahkan data latih dan data uji.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Latih model Machine Learning konvensional.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Evaluasi pada data uji yang tidak digunakan saat pelatihan.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    print("\n=== FORESTWATCH: ANALISIS DEFORESTASI ===")
    print("Jenis data: simulasi pembelajaran")
    print(f"Jumlah catatan       : {len(df)}")
    print(f"Data latih           : {len(X_train)}")
    print(f"Data uji             : {len(X_test)}")
    print(f"Accuracy             : {accuracy:.3f}")
    print(f"Macro F1-score       : {macro_f1:.3f}")

    print("\n--- Laporan Klasifikasi ---")
    print(classification_report(
        y_test,
        predictions,
        labels=DRIVERS,
        zero_division=0,
    ))

    # Grafik jumlah catatan per kategori.
    counts = df["driver"].value_counts().reindex(DRIVERS)
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar", color="forestgreen")
    plt.title("Jumlah Catatan Simulasi per Kategori")
    plt.xlabel("Kategori penggerak")
    plt.ylabel("Jumlah catatan")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "driver_distribution.png")
    plt.close()

    # Grafik kepentingan fitur menurut model.
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values()

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color="darkseagreen")
    plt.title("Kepentingan Fitur Menurut Random Forest")
    plt.xlabel("Feature importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()

    print("\nFile hasil tersimpan di folder outputs/")
    print("Catatan: hasil evaluasi hanya menggambarkan data simulasi.")


if __name__ == "__main__":
    main()