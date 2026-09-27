# ======================================================================
# PROFESSIONAL MULTI-PANEL SPATIAL MAP SCRIPT
# Wind Speed (WS), Wind Power Density (WPD), Wind Energy Density (WED)
# 10 Hub Heights (10m – 100m)
# Robust Column Cleaning + Ordinary Kriging + Masking
# Fully Automatic | Publication Ready
# ======================================================================

# =========================
# 1. IMPORT LIBRARIES
# =========================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.vectorized import contains
from pykrige.ok import OrdinaryKriging


# =========================
# 2. SET WORKING DIRECTORY
# =========================
base_dir = os.getcwd()

print("Working Directory:", base_dir)


# =========================
# 3. DEFINE FILE PATHS
# =========================
ws_file  = os.path.join(base_dir, "1-Wind-Speed-at-different-height.csv")
wpd_file = os.path.join(base_dir, "2-Wind-Power-Density-at-different-height.csv")
wed_file = os.path.join(base_dir, "3-Wind-Energy-Density-at-different-height.csv")

shapefile_path = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/Pakistan_shape_file-with-Kashmir/Pakistan_with_Kashmir.shp"

output_folder = os.path.join(base_dir, "FINAL_Spatial_Maps_Output")
os.makedirs(output_folder, exist_ok=True)


# =========================
# 4. LOAD SHAPEFILE
# =========================
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)
pakistan_union = pakistan.unary_union


# =========================
# 5. FUNCTION: CLEAN COLUMN NAMES
# =========================
def clean_columns(df):
    """
    Remove extra spaces, hidden characters, and standardize column names.
    This prevents KeyError due to spacing inconsistencies.
    """
    df.columns = (
        df.columns
        .str.replace('\xa0', ' ', regex=False)   # remove non-breaking space
        .str.replace('\s+', ' ', regex=True)     # remove double spaces
        .str.strip()                             # trim edges
    )
    return df


# =========================
# 6. LOAD & CLEAN CSV FILES
# =========================
ws_df  = clean_columns(pd.read_csv(ws_file))
wpd_df = clean_columns(pd.read_csv(wpd_file))
wed_df = clean_columns(pd.read_csv(wed_file))

print("✔ CSV files loaded and cleaned successfully")


# =========================
# 7. DEFINE HUB HEIGHTS
# =========================
heights = [10,20,30,40,50,60,70,80,90,100]


# =========================
# 8. CREATE INTERPOLATION GRID
# =========================
minx, miny, maxx, maxy = pakistan.total_bounds

grid_x = np.linspace(minx, maxx, 200)
grid_y = np.linspace(miny, maxy, 200)

gridx_mesh, gridy_mesh = np.meshgrid(grid_x, grid_y)


# =========================
# 9. KRIGING FUNCTION
# =========================
def perform_kriging(df, column_name):

    if column_name not in df.columns:
        print(f"⚠ Column not found: {column_name}")
        return np.full_like(gridx_mesh, np.nan)

    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    df_clean = df.dropna(subset=[column_name, "Latitude", "Longitude"])

    if len(df_clean) < 3:
        print(f"⚠ Not enough stations for {column_name}")
        return np.full_like(gridx_mesh, np.nan)

    lons = df_clean["Longitude"].values
    lats = df_clean["Latitude"].values
    values = df_clean[column_name].values

    OK = OrdinaryKriging(
        lons,
        lats,
        values,
        variogram_model="spherical",
        coordinates_type="geographic",
        verbose=False
    )

    z, ss = OK.execute("grid", grid_x, grid_y)

    mask = contains(pakistan_union, gridx_mesh, gridy_mesh)
    z_masked = np.where(mask, z, np.nan)

    return z_masked


# =========================
# 10. CREATE MULTI-PANEL FIGURE
# =========================
fig, axes = plt.subplots(10, 3, figsize=(18, 38), sharex=True, sharey=True)

plt.rcParams['font.family'] = 'Times New Roman'


# =========================
# 11. LOOP THROUGH HEIGHTS
# =========================
for row, h in enumerate(heights):

    print(f"Processing {h}m")

    # ---- WIND SPEED ----
    ws_col = f"WS at {h}m"
    z_ws = perform_kriging(ws_df, ws_col)

    ax = axes[row, 0]
    ax.contourf(grid_x, grid_y, z_ws, cmap="viridis", levels=15)
    pakistan.boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_title(f"WS {h}m", fontsize=8)


    # ---- WIND POWER DENSITY ----
    wpd_col = f"WPD at {h}m"
    z_wpd = perform_kriging(wpd_df, wpd_col)

    ax = axes[row, 1]
    ax.contourf(grid_x, grid_y, z_wpd, cmap="plasma", levels=15)
    pakistan.boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_title(f"WPD {h}m", fontsize=8)


    # ---- WIND ENERGY DENSITY ----
    wed_col = f"WED at {h}m"
    z_wed = perform_kriging(wed_df, wed_col)

    ax = axes[row, 2]
    ax.contourf(grid_x, grid_y, z_wed, cmap="inferno", levels=15)
    pakistan.boundary.plot(ax=ax, color="black", linewidth=0.4)
    ax.set_title(f"WED {h}m", fontsize=8)


# Remove ticks for clean journal layout
for ax in axes.flat:
    ax.set_xticks([])
    ax.set_yticks([])


# Column Titles
axes[0,0].set_title("Wind Speed (WS)", fontsize=14, fontweight="bold")
axes[0,1].set_title("Wind Power Density (WPD)", fontsize=14, fontweight="bold")
axes[0,2].set_title("Wind Energy Density (WED)", fontsize=14, fontweight="bold")


plt.tight_layout()


# =========================
# 12. SAVE OUTPUT
# =========================
output_path = os.path.join(output_folder, "Pakistan_MultiPanel_Spatial_Maps.png")

plt.savefig(output_path, dpi=600, bbox_inches="tight")
plt.close()

print("--------------------------------------------------")
print("✔ SUCCESS: 30 Spatial Maps Generated")
print("✔ Saved at:", output_path)
print("--------------------------------------------------")
