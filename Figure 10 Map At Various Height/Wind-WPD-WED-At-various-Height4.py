# ======================================================================
# PROFESSIONAL MULTI-PANEL SPATIAL MAPS
# Wind Speed (WS), Wind Power Density (WPD), Wind Energy Density (WED)
# 10 Heights (10m–100m)
# Ordinary Kriging + Pakistan Mask
# Optimized Scientific Layout (Publication Quality)
# ======================================================================


# ===============================
# 1. IMPORT REQUIRED LIBRARIES
# ===============================
import os
import warnings
warnings.filterwarnings("ignore")  # Suppress non-critical warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from shapely.vectorized import contains
from pykrige.ok import OrdinaryKriging


# ===============================
# 2. SET WORKING DIRECTORY
# ===============================
# Automatically detect where the script is executed
base_dir = os.getcwd()
print("Working Directory:", base_dir)


# ===============================
# 3. DEFINE FILE PATHS
# ===============================
ws_file  = os.path.join(base_dir, "1-Wind-Speed-at-different-height.csv")
wpd_file = os.path.join(base_dir, "2-Wind-Power-Density-at-different-height.csv")
wed_file = os.path.join(base_dir, "3-Wind-Energy-Density-at-different-height.csv")

# 👉 UPDATE THIS PATH IF NEEDED
shapefile_path = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/Pakistan_shape_file-with-Kashmir/Pakistan_with_Kashmir.shp"

output_folder = os.path.join(base_dir, "FINAL_Spatial_Maps_Output")
os.makedirs(output_folder, exist_ok=True)


# ===============================
# 4. LOAD SHAPEFILE
# ===============================
# Read Pakistan boundary and convert to WGS84
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)

# Merge multiple geometries into one polygon
pakistan_union = pakistan.unary_union


# ===============================
# 5. CLEAN COLUMN NAMES FUNCTION
# ===============================
def clean_columns(df):
    """
    Remove hidden spaces and formatting problems
    from column names.
    """
    df.columns = (
        df.columns
        .str.replace('\xa0', ' ', regex=False)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df


# ===============================
# 6. LOAD DATA FILES
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

# Increase grid resolution for smoother maps
grid_x = np.linspace(minx, maxx, 250)
grid_y = np.linspace(miny, maxy, 250)

gridx_mesh, gridy_mesh = np.meshgrid(grid_x, grid_y)


# ===============================
# 9. KRIGING FUNCTION
# ===============================
def perform_kriging(df, column_name):
    """
    Perform Ordinary Kriging interpolation.
    Mask values outside Pakistan boundary.
    Returns interpolated grid.
    """

    # Ensure numeric conversion
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    # Remove missing coordinates or values
    df_clean = df.dropna(subset=[column_name, "Latitude", "Longitude"])

    lons = df_clean["Longitude"].values
    lats = df_clean["Latitude"].values
    values = df_clean[column_name].values

    # Ordinary Kriging with spherical model
    OK = OrdinaryKriging(
        lons, lats, values,
        variogram_model="spherical",
        coordinates_type="geographic",
        verbose=False
    )

    # Execute interpolation on grid
    z, ss = OK.execute("grid", grid_x, grid_y)

    # Mask outside Pakistan boundary
    mask = contains(pakistan_union, gridx_mesh, gridy_mesh)

    return np.where(mask, z, np.nan)


# ===============================
# 10. COMPUTE GLOBAL COLOR LIMITS
# ===============================
ws_min, ws_max   = ws_df.filter(like="WS").min().min(), ws_df.filter(like="WS").max().max()
wpd_min, wpd_max = wpd_df.filter(like="WPD").min().min(), wpd_df.filter(like="WPD").max().max()
wed_min, wed_max = wed_df.filter(like="WED").min().min(), wed_df.filter(like="WED").max().max()


# ===============================
# 11. CREATE FIGURE LAYOUT
# ===============================
fig = plt.figure(figsize=(22, 38))

# Reduced vertical spacing (hspace)
# Reduced horizontal spacing (wspace)
gs = fig.add_gridspec(
    10, 3,
    wspace=0.02,   # space between WS-WPD-WED columns
    hspace=0.04    # space between heights
)

plt.rcParams['font.family'] = 'Times New Roman'


# ===============================
# 12. PLOT MAPS (COLUMN STRUCTURE)
# ===============================
for i, h in enumerate(heights):

    print(f"Processing {h}m")

    # ----- COLUMN 1: WIND SPEED -----
    ax_ws = fig.add_subplot(gs[i, 0])
    z_ws = perform_kriging(ws_df, f"WS at {h}m")

    im_ws = ax_ws.contourf(
        grid_x, grid_y, z_ws,
        levels=20,
        cmap="viridis",
        vmin=ws_min, vmax=ws_max
    )

    pakistan.boundary.plot(ax=ax_ws, color="black", linewidth=0.4)
    ax_ws.set_xticks([]); ax_ws.set_yticks([])

    if i == 0:
        ax_ws.set_title("Wind Speed (m/s)", fontsize=11, weight="bold")

    ax_ws.text(0.02, 0.92, f"{h}m",
               transform=ax_ws.transAxes,
               fontsize=8, weight="bold")


    # ----- COLUMN 2: WIND POWER DENSITY -----
    ax_wpd = fig.add_subplot(gs[i, 1])
    z_wpd = perform_kriging(wpd_df, f"WPD at {h}m")

    im_wpd = ax_wpd.contourf(
        grid_x, grid_y, z_wpd,
        levels=20,
        cmap="plasma",
        vmin=wpd_min, vmax=wpd_max
    )

    pakistan.boundary.plot(ax=ax_wpd, color="black", linewidth=0.4)
    ax_wpd.set_xticks([]); ax_wpd.set_yticks([])

    if i == 0:
        ax_wpd.set_title("Wind Power Density (W/m²)", fontsize=11, weight="bold")

    ax_wpd.text(0.02, 0.92, f"{h}m",
                transform=ax_wpd.transAxes,
                fontsize=8, weight="bold")


    # ----- COLUMN 3: WIND ENERGY DENSITY -----
    ax_wed = fig.add_subplot(gs[i, 2])
    z_wed = perform_kriging(wed_df, f"WED at {h}m")

    im_wed = ax_wed.contourf(
        grid_x, grid_y, z_wed,
        levels=20,
        cmap="inferno",
        vmin=wed_min, vmax=wed_max
    )

    pakistan.boundary.plot(ax=ax_wed, color="black", linewidth=0.4)
    ax_wed.set_xticks([]); ax_wed.set_yticks([])

    if i == 0:
        ax_wed.set_title("Wind Energy Density (kWh/m²)", fontsize=11, weight="bold")

    ax_wed.text(0.02, 0.92, f"{h}m",
                transform=ax_wed.transAxes,
                fontsize=8, weight="bold")


# ===============================
# 13. ADD VERTICAL COLORBARS
# ===============================
cbar_width  = 0.015
cbar_height = 0.25

fig.colorbar(im_ws,
             cax=fig.add_axes([0.92, 0.65, cbar_width, cbar_height]))

fig.colorbar(im_wpd,
             cax=fig.add_axes([0.92, 0.37, cbar_width, cbar_height]))

fig.colorbar(im_wed,
             cax=fig.add_axes([0.92, 0.09, cbar_width, cbar_height]))


# ===============================
# 14. SAVE HIGH-RESOLUTION FIGURE
# ===============================
output_path = os.path.join(
    output_folder,
    "Pakistan_MultiPanel_WS_WPD_WED_Professional.png"
)

plt.savefig(output_path, dpi=600, bbox_inches="tight")
plt.close()

print("--------------------------------------------------")
print("✔ PROFESSIONAL MULTI-COLUMN FIGURE GENERATED")
print("✔ Optimized Vertical & Horizontal Spacing")
print("✔ Saved at:", output_path)
print("--------------------------------------------------")
