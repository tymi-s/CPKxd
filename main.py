import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# KROK 1: WCZYTANIE I PODSTAWOWA EKSPLORACJA
# ============================================

las = laspy.read("Chmura_Zadanie.las")
print(f"Liczba punktów: {len(las.x):,}")
print(f"\nDostępne wymiary (kolumny):")
print(las.point_format.dimension_names)

# ============================================
# KROK 2: WYŚWIETL PIERWSZE 3 WIERSZE
# ============================================

# Stwórz DataFrame z pierwszymi 3 punktami
data_dict = {
    'X': las.X[:3],
    'Y': las.Y[:3],
    'Z': las.Z[:3],
    'intensity': las.intensity[:3],
}

# Dodaj kolory jeśli są dostępne
if hasattr(las, 'red'):
    data_dict['Red'] = las.red[:3]
    data_dict['Green'] = las.green[:3]
    data_dict['Blue'] = las.blue[:3]

# Dodaj inne dostępne wymiary
for dim in las.point_format.dimension_names:
    dim_lower = dim.lower()
    if dim_lower not in ['x', 'y', 'z', 'intensity', 'red', 'green', 'blue']:
        if hasattr(las, dim_lower):
            data_dict[dim] = getattr(las, dim_lower)[:3]

df_sample = pd.DataFrame(data_dict)

# Ustaw pandas do pełnego wyświetlania
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n" + "="*80)
print("PIERWSZE 3 PUNKTY - WSZYSTKIE KOLUMNY:")
print("="*80)
print(df_sample.to_string())
print("\n")

# ============================================
# KROK 3: STATYSTYKI KLUCZOWYCH PARAMETRÓW
# ============================================

print("="*80)
print("STATYSTYKI DANYCH:")
print("="*80)

print("\n📍 Współrzędne XYZ:")
print(f"  X: min={las.x.min():.2f}, max={las.x.max():.2f}")
print(f"  Y: min={las.y.min():.2f}, max={las.y.max():.2f}")
print(f"  Z: min={las.z.min():.2f}, max={las.z.max():.2f}")

print("\n💡 Intensity (współczynnik odbicia):")
print(f"  min={las.intensity.min()}, max={las.intensity.max()}")
print(f"  średnia={las.intensity.mean():.2f}")
print(f"  mediana={np.median(las.intensity):.2f}")

if hasattr(las, 'red'):
    print("\n🎨 Kolory RGB:")
    print(f"  Red: min={las.red.min()}, max={las.red.max()}")
    print(f"  Green: min={las.green.min()}, max={las.green.max()}")
    print(f"  Blue: min={las.blue.min()}, max={las.blue.max()}")

if hasattr(las, 'classification'):
    print("\n🏷️ Classification (jeśli już są klasy):")
    unique_classes = np.unique(las.classification)
    print(f"  Unikalne klasy: {unique_classes}")
    for cls in unique_classes:
        count = np.sum(las.classification == cls)
        print(f"    Klasa {cls}: {count:,} punktów ({count/len(las.classification)*100:.2f}%)")

# ============================================
# KROK 4: HISTOGRAM INTENSITY (KLUCZOWY!)
# ============================================

print("\n" + "="*80)
print("HISTOGRAM INTENSITY - KLUCZOWY DO LABELOWANIA!")
print("="*80)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(las.intensity, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Intensity')
plt.ylabel('Liczba punktów')
plt.title('Rozkład Intensity')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(las.intensity, bins=50, color='steelblue', edgecolor='black', alpha=0.7, log=True)
plt.xlabel('Intensity')
plt.ylabel('Liczba punktów (skala log)')
plt.title('Rozkład Intensity (log scale)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('intensity_histogram.png', dpi=150)
print("\n✅ Zapisano histogram: intensity_histogram.png")
plt.show()

# ============================================
# KROK 5: WIZUALIZACJA 3D (SAMPLE)
# ============================================

print("\n" + "="*80)
print("WIZUALIZACJA 3D (sample 10000 punktów)")
print("="*80)

# Wybierz losowy sample do wizualizacji (bo cała chmura to za dużo)
sample_size = min(10000, len(las.x))
sample_indices = np.random.choice(len(las.x), sample_size, replace=False)

fig = plt.figure(figsize=(15, 5))

# Subplot 1: XYZ kolorowane według Z (wysokości)
ax1 = fig.add_subplot(131, projection='3d')
scatter1 = ax1.scatter(
    las.x[sample_indices],
    las.y[sample_indices],
    las.z[sample_indices],
    c=las.z[sample_indices],
    cmap='terrain',
    s=0.5
)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z (wysokość)')
ax1.set_title('Kolorowanie: wysokość Z')
plt.colorbar(scatter1, ax=ax1, shrink=0.5)

# Subplot 2: Kolorowane według Intensity
ax2 = fig.add_subplot(132, projection='3d')
scatter2 = ax2.scatter(
    las.x[sample_indices],
    las.y[sample_indices],
    las.z[sample_indices],
    c=las.intensity[sample_indices],
    cmap='viridis',
    s=0.5
)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Kolorowanie: Intensity')
plt.colorbar(scatter2, ax=ax2, shrink=0.5)

# Subplot 3: RGB (jeśli dostępne)
ax3 = fig.add_subplot(133, projection='3d')
if hasattr(las, 'red'):
    # Normalizuj RGB do 0-1
    rgb_colors = np.column_stack([
        las.red[sample_indices] / 65535.0,
        las.green[sample_indices] / 65535.0,
        las.blue[sample_indices] / 65535.0
    ])
    ax3.scatter(
        las.x[sample_indices],
        las.y[sample_indices],
        las.z[sample_indices],
        c=rgb_colors,
        s=0.5
    )
    ax3.set_title('Kolorowanie: RGB rzeczywiste')
else:
    ax3.scatter(
        las.x[sample_indices],
        las.y[sample_indices],
        las.z[sample_indices],
        c='gray',
        s=0.5
    )
    ax3.set_title('Brak danych RGB')

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.savefig('point_cloud_3d_visualization.png', dpi=150)
print("✅ Zapisano wizualizację: point_cloud_3d_visualization.png")
plt.show()

# ============================================
# KROK 6: ANALIZA INTENSITY DLA RÓŻNYCH WYSOKOŚCI
# ============================================

print("\n" + "="*80)
print("ANALIZA: Intensity vs Wysokość (kluczowe do rozpoznawania obiektów!)")
print("="*80)

plt.figure(figsize=(12, 6))

# Scatter plot: Z vs Intensity
sample_for_scatter = min(50000, len(las.x))
scatter_idx = np.random.choice(len(las.x), sample_for_scatter, replace=False)

plt.subplot(1, 2, 1)
plt.scatter(las.z[scatter_idx], las.intensity[scatter_idx],
            c=las.intensity[scatter_idx], cmap='plasma', s=1, alpha=0.5)
plt.xlabel('Wysokość Z')
plt.ylabel('Intensity')
plt.title('Intensity vs Wysokość')
plt.colorbar(label='Intensity')
plt.grid(True, alpha=0.3)

# Binned średnie
plt.subplot(1, 2, 2)
z_bins = np.linspace(las.z.min(), las.z.max(), 50)
z_digitized = np.digitize(las.z, z_bins)
intensity_means = [las.intensity[z_digitized == i].mean() if np.sum(z_digitized == i) > 0 else 0
                   for i in range(1, len(z_bins))]
plt.plot(z_bins[:-1], intensity_means, linewidth=2)
plt.xlabel('Wysokość Z')
plt.ylabel('Średnia Intensity')
plt.title('Średnia Intensity dla różnych wysokości')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('intensity_vs_height.png', dpi=150)
print("✅ Zapisano: intensity_vs_height.png")
plt.show()

# ============================================
# KROK 7: PRZYGOTOWANIE DO LABELOWANIA
# ============================================

print("\n" + "="*80)
print("GOTOWE DO LABELOWANIA!")
print("="*80)
print("\n📋 Sugestie klas do labelowania (CPK):")
print("  1. DROGA (road) - niska wysokość, płaska, niska-średnia intensity")
print("  2. KRAWĘŻNIK (curb) - liniowy, ~15cm nad drogą, średnia intensity")
print("  3. SŁUP (pole) - pionowy, wysoki, mała podstawa")
print("  4. ZIELEŃ (vegetation) - nieregularna, zielony kolor")
print("  5. ZABUDOWA (building) - pionowe płaszczyzny, wysoka intensity")

print("\n💡 Wskazówki do labelowania na podstawie danych:")

# Analiza intensity percentiles
percentiles = [10, 25, 50, 75, 90]
intensity_percentiles = np.percentile(las.intensity, percentiles)
print(f"\n  Percentyle Intensity:")
for p, val in zip(percentiles, intensity_percentiles):
    print(f"    {p}%: {val:.0f}")

print("\n  Propozycja progów Intensity:")
print(f"    Bardzo niska (asfalt):     < {intensity_percentiles[1]:.0f}")
print(f"    Niska-średnia (beton):     {intensity_percentiles[1]:.0f} - {intensity_percentiles[3]:.0f}")
print(f"    Wysoka (oznakowanie):      > {intensity_percentiles[3]:.0f}")

# Wysokości
z_min = las.z.min()
print(f"\n  Wysokości względne (od gruntu na Z={z_min:.2f}):")
print(f"    Grunt/droga:        < 0.5m")
print(f"    Krawężniki:         0.1-0.3m")
print(f"    Elementy niskie:    0.5-2m")
print(f"    Słupy/budynki:      > 2m")

print("\n" + "="*80)
print("NASTĘPNY KROK: Wybierz region do labelowania!")
print("="*80)
print("Możesz:")
print("  1. Wybrać mały prostokąt XY zawierający różne obiekty")
print("  2. Oznaczyć ręcznie ~1000 punktów każdej klasy")
print("  3. Użyć progów Intensity + Z jako pierwszego filtra")
print("\nGotowy na labelowanie? 🚀")