# ======================================================================
# PROFESSIONAL SPATIAL MAPS GENERATION
# 3 SEPARATE FIGURES:
#   1) Wind Speed (WS)
#   2) Wind Power Density (WPD)
#   3) Wind Energy Density (WED)
#
# Each figure contains:
#   → 10 heights (10m–100m)
#   → Layout: 2 rows × 5 columns
#   → Ordinary Kriging interpolation
#   → Pakistan boundary masking
#
# Fully Automatic | Robust | Beginner Friendly | Publication Quality
# ======================================================================


# ===============================
# 1. IMPORT REQUIRED LIBRARIES
# ===============================
import os
import warnings
warnings.filterwarnings("ignore")  # Suppress unnecessary warnings

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
# 3. DEFINE INPUT FILE PATHS
# ===============================
ws_file  = os.path.join(base_dir, "1-Wind-Speed-at-different-height.csv")
wpd_file = os.path.join(base_dir, "2-Wind-Power-Density-at-different-height.csv")
wed_file = os.path.join(base_dir, "3-Wind-Energy-Density-at-different-height.csv")

# 👉 UPDATE THIS PATH IF NEEDED
shapefile_path = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/Pakistan_shape_file-with-Kashmir/Pakistan_with_Kashmir.shp"

# Output folder
output_folder = os.path.join(base_dir, "FINAL_Spatial_Maps_Output")
os.makedirs(output_folder, exist_ok=True)


# ===============================
# 4. LOAD SHAPEFILE
# ===============================
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)

# Merge geometries into one polygon
pakistan_union = pakistan.unary_union


# ===============================
# 5. CLEAN COLUMN NAMES FUNCTION
# ===============================
def clean_columns(df):
    """
    Remove hidden spaces or formatting issues
    from CSV column names.
    """
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

# High resolution grid
grid_x = np.linspace(minx, maxx, 250)
grid_y = np.linspace(miny, maxy, 250)

gridx_mesh, gridy_mesh = np.meshgrid(grid_x, grid_y)


# ===============================
# 9. KRIGING FUNCTION
# ===============================
def perform_kriging(df, column_name):
    """
    Perform Ordinary Kriging interpolation
    and apply Pakistan mask.
    """

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
# 10. FUNCTION TO GENERATE 2×5 FIGURE
# ===============================
def generate_figure(df, variable_prefix, cmap, unit_label, output_name):
    """
    Generate a 2 rows × 5 columns spatial figure
    for 10 heights.
    """

    print(f"\nGenerating figure for {variable_prefix}")

    # Global color limits
    vmin = df.filter(like=variable_prefix).min().min()
    vmax = df.filter(like=variable_prefix).max().max()

    # Create 2x5 layout
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    plt.rcParams['font.family'] = 'Times New Roman'

    # Flatten axes for easy looping
    axes = axes.flatten()

    for i, h in enumerate(heights):

        print(f"Processing {h}m")

        column_name = f"{variable_prefix} at {h}m"

        z = perform_kriging(df, column_name)

        im = axes[i].contourf(
            grid_x, grid_y, z,
            levels=20,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        pakistan.boundary.plot(ax=axes[i], color="black", linewidth=0.4)

        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # Add height label
        axes[i].set_title(f"{h}m", fontsize=10, weight="bold")

    # Reduce spacing between panels
    plt.subplots_adjust(wspace=0.05, hspace=0.08)

    # Add shared colorbar
    cbar = fig.colorbar(
        im,
        ax=axes,
        orientation="vertical",
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label(unit_label, fontsize=11)

    # Save figure
    output_path = os.path.join(output_folder, output_name)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()

    print("✔ Figure saved at:", output_path)


# ===============================
# 11. GENERATE THREE FIGURES
# ===============================

generate_figure(
    ws_df,
    variable_prefix="WS",
    cmap="viridis",
    unit_label="Wind Speed (m/s)",
    output_name="Figure_WS_2x5_Heights.png"
)

generate_figure(
    wpd_df,
    variable_prefix="WPD",
    cmap="plasma",
    unit_label="Wind Power Density (W/m²)",
    output_name="Figure_WPD_2x5_Heights.png"
)

generate_figure(
    wed_df,
    variable_prefix="WED",
    cmap="inferno",
    unit_label="Wind Energy Density (kWh/m²)",
    output_name="Figure_WED_2x5_Heights.png"
)

print("\n--------------------------------------------------")
print("✔ ALL THREE 2×5 PROFESSIONAL FIGURES GENERATED")
print("✔ Each figure contains 10 panels (2 rows × 5 columns)")
print("✔ Publication-ready layout")
print("--------------------------------------------------")
