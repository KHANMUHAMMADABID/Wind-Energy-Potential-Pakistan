# ======================================================================
# PROFESSIONAL MULTI-PANEL SPATIAL MAPS
# Wind Speed (WS), Wind Power Density (WPD), Wind Energy Density (WED)
# 10 Heights (10m–100m)
# Ordinary Kriging + Pakistan Mask
# Separate Professional Colorbars (Outside Panels)
# ======================================================================


# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.vectorized import contains
from pykrige.ok import OrdinaryKriging


# ===============================
# 2. SET WORKING DIRECTORY
# ===============================
base_dir = os.getcwd()
print("Working Directory:", base_dir)


# ===============================
# 3. FILE PATHS
# ===============================
ws_file  = os.path.join(base_dir, "1-Wind-Speed-at-different-height.csv")
wpd_file = os.path.join(base_dir, "2-Wind-Power-Density-at-different-height.csv")
wed_file = os.path.join(base_dir, "3-Wind-Energy-Density-at-different-height.csv")

shapefile_path = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/Pakistan_shape_file-with-Kashmir/Pakistan_with_Kashmir.shp"

output_folder = os.path.join(base_dir, "FINAL_Spatial_Maps_Output")
os.makedirs(output_folder, exist_ok=True)


# ===============================
# 4. LOAD SHAPEFILE
# ===============================
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)
pakistan_union = pakistan.unary_union


# ===============================
# 5. CLEAN COLUMN NAMES
# ===============================
def clean_columns(df):
    df.columns = (
        df.columns
        .str.replace('\xa0', ' ', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df


# ===============================
# 6. LOAD DATA
# ===============================
ws_df  = clean_columns(pd.read_csv(ws_file))
wpd_df = clean_columns(pd.read_csv(wpd_file))
wed_df = clean_columns(pd.read_csv(wed_file))

print("✔ CSV files loaded successfully")


# ===============================
# 7. DEFINE HEIGHTS
# ===============================
heights = [10,20,30,40,50,60,70,80,90,100]


# ===============================
# 8. CREATE INTERPOLATION GRID
# ===============================
minx, miny, maxx, maxy = pakistan.total_bounds
grid_x = np.linspace(minx, maxx, 200)
grid_y = np.linspace(miny, maxy, 200)
gridx_mesh, gridy_mesh = np.meshgrid(grid_x, grid_y)


# ===============================
# 9. KRIGING FUNCTION
# ===============================
def perform_kriging(df, column_name):

    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    df_clean = df.dropna(subset=[column_name, "Latitude", "Longitude"])

    lons = df_clean["Longitude"].values
    lats = df_clean["Latitude"].values
    values = df_clean[column_name].values

    OK = OrdinaryKriging(
        lons, lats, values,
        variogram_model="spherical",
        coordinates_type="geographic",
        verbose=False
    )

    z, ss = OK.execute("grid", grid_x, grid_y)

    mask = contains(pakistan_union, gridx_mesh, gridy_mesh)
    return np.where(mask, z, np.nan)


# ===============================
# 10. COMPUTE GLOBAL COLOR LIMITS
# (Scientific Consistency)
# ===============================
ws_min  = ws_df.filter(like="WS").min().min()
ws_max  = ws_df.filter(like="WS").max().max()

wpd_min = wpd_df.filter(like="WPD").min().min()
wpd_max = wpd_df.filter(like="WPD").max().max()

wed_min = wed_df.filter(like="WED").min().min()
wed_max = wed_df.filter(like="WED").max().max()


# ===============================
# 11. CREATE FIGURE WITH EXTRA SPACE
# ===============================
fig = plt.figure(figsize=(22, 40))

# GridSpec: 10 rows, 4 columns (last column reserved for colorbars)
gs = fig.add_gridspec(10, 4, width_ratios=[1,1,1,0.05], wspace=0.05)

plt.rcParams['font.family'] = 'Times New Roman'


# ===============================
# 12. PLOT MAPS
# ===============================
for i, h in enumerate(heights):

    print(f"Processing {h}m")

    # ---- WS ----
    ax_ws = fig.add_subplot(gs[i, 0])
    z_ws = perform_kriging(ws_df, f"WS at {h}m")
    im_ws = ax_ws.contourf(grid_x, grid_y, z_ws,
                           levels=20, cmap="viridis",
                           vmin=ws_min, vmax=ws_max)
    pakistan.boundary.plot(ax=ax_ws, color="black", linewidth=0.4)
    ax_ws.set_xticks([]); ax_ws.set_yticks([])
    ax_ws.set_title(f"{h}m", fontsize=8)

    # ---- WPD ----
    ax_wpd = fig.add_subplot(gs[i, 1])
    z_wpd = perform_kriging(wpd_df, f"WPD at {h}m")
    im_wpd = ax_wpd.contourf(grid_x, grid_y, z_wpd,
                             levels=20, cmap="plasma",
                             vmin=wpd_min, vmax=wpd_max)
    pakistan.boundary.plot(ax=ax_wpd, color="black", linewidth=0.4)
    ax_wpd.set_xticks([]); ax_wpd.set_yticks([])
    ax_wpd.set_title(f"{h}m", fontsize=8)

    # ---- WED ----
    ax_wed = fig.add_subplot(gs[i, 2])
    z_wed = perform_kriging(wed_df, f"WED at {h}m")
    im_wed = ax_wed.contourf(grid_x, grid_y, z_wed,
                             levels=20, cmap="inferno",
                             vmin=wed_min, vmax=wed_max)
    pakistan.boundary.plot(ax=ax_wed, color="black", linewidth=0.4)
    ax_wed.set_xticks([]); ax_wed.set_yticks([])
    ax_wed.set_title(f"{h}m", fontsize=8)


# ===============================
# 13. ADD SEPARATE COLORBARS (RIGHT SIDE)
# ===============================
cax_ws  = fig.add_subplot(gs[:,3])
cbar_ws = fig.colorbar(im_ws, cax=cax_ws)
cbar_ws.set_label("Wind Speed (m/s)")

# Create second and third colorbars manually using figure coordinates
cbar_wpd = fig.colorbar(im_wpd, ax=fig.axes, fraction=0.015, pad=0.02)
cbar_wpd.set_label("Wind Power Density (W/m²)")

cbar_wed = fig.colorbar(im_wed, ax=fig.axes, fraction=0.015, pad=0.08)
cbar_wed.set_label("Wind Energy Density (kWh/m²)")


plt.tight_layout()


# ===============================
# 14. SAVE FIGURE
# ===============================
output_path = os.path.join(output_folder,
                           "Pakistan_MultiPanel_Professional.png")

plt.savefig(output_path, dpi=600, bbox_inches="tight")
plt.close()

print("--------------------------------------------------")
print("✔ PROFESSIONAL FIGURE GENERATED")
print("✔ Saved at:", output_path)
print("--------------------------------------------------")
