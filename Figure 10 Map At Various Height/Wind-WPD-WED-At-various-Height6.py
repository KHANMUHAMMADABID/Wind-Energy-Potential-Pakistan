# ======================================================================
# PROFESSIONAL SPATIAL MAPS GENERATION
# 3 SEPARATE FIGURES:
#   1) Wind Speed (WS)
#   2) Wind Power Density (WPD)
#   3) Wind Energy Density (WED)
#
# Each figure contains:
#   → 10 heights (10m–100m)
#   → Layout: 5 rows × 2 columns
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
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)

# Merge all geometries into single polygon
pakistan_union = pakistan.unary_union


# ===============================
# 5. CLEAN COLUMN NAMES FUNCTION
# ===============================
def clean_columns(df):
    """
    Clean hidden spaces or formatting issues
    in column names.
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

# High resolution grid for smooth interpolation
grid_x = np.linspace(minx, maxx, 250)
grid_y = np.linspace(miny, maxy, 250)

gridx_mesh, gridy_mesh = np.meshgrid(grid_x, grid_y)


# ===============================
# 9. KRIGING FUNCTION
# ===============================
def perform_kriging(df, column_name):
    """
    Perform Ordinary Kriging interpolation
    and mask outside Pakistan boundary.
    """

    # Ensure numeric type
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    # Remove missing rows
    df_clean = df.dropna(subset=[column_name, "Latitude", "Longitude"])

    lons = df_clean["Longitude"].values
    lats = df_clean["Latitude"].values
    values = df_clean[column_name].values

    # Ordinary Kriging (spherical model)
    OK = OrdinaryKriging(
        lons, lats, values,
        variogram_model="spherical",
        coordinates_type="geographic",
        verbose=False
    )

    # Execute interpolation
    z, ss = OK.execute("grid", grid_x, grid_y)

    # Apply Pakistan boundary mask
    mask = contains(pakistan_union, gridx_mesh, gridy_mesh)

    return np.where(mask, z, np.nan)


# ===============================
# 10. GENERATE 5×2 FIGURE FUNCTION
# ===============================
def generate_figure(df, variable_prefix, cmap, unit_label, output_name):
    """
    Generate a figure with:
        5 rows × 2 columns (10 panels)
    """

    print(f"\nGenerating figure for {variable_prefix}")

    # Compute consistent color limits
    vmin = df.filter(like=variable_prefix).min().min()
    vmax = df.filter(like=variable_prefix).max().max()

    # Create 5 rows × 2 columns layout
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))

    plt.rcParams['font.family'] = 'Times New Roman'

    axes = axes.flatten()  # Convert to 1D array for easy looping

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

    # Reduce spacing for professional look
    plt.subplots_adjust(wspace=0.05, hspace=0.08)

    # Shared colorbar
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
# 11. GENERATE THREE SEPARATE FIGURES
# ===============================

generate_figure(
    ws_df,
    variable_prefix="WS",
    cmap="viridis",
    unit_label="Wind Speed (m/s)",
    output_name="Figure_WS_5x2_Heights.png"
)

generate_figure(
    wpd_df,
    variable_prefix="WPD",
    cmap="plasma",
    unit_label="Wind Power Density (W/m²)",
    output_name="Figure_WPD_5x2_Heights.png"
)

generate_figure(
    wed_df,
    variable_prefix="WED",
    cmap="inferno",
    unit_label="Wind Energy Density (kWh/m²)",
    output_name="Figure_WED_5x2_Heights.png"
)


print("\n--------------------------------------------------")
print("✔ ALL THREE 5×2 PROFESSIONAL FIGURES GENERATED")
print("✔ Each figure contains 10 panels (5 rows × 2 columns)")
print("✔ High-resolution (600 DPI) publication ready")
print("--------------------------------------------------")
