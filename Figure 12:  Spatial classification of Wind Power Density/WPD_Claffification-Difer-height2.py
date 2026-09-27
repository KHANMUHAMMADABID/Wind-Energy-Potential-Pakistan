# -------------------------------------------------------------------------
# COMBINED WIND POWER DENSITY CLASSIFICATION MAPS AT MULTIPLE HUB HEIGHTS - PAKISTAN
# -------------------------------------------------------------------------

# Required Libraries
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from pykrige.ok import OrdinaryKriging
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text  # For non-overlapping text labels
import os
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

# -------------------------------------------------------------------------
# 1. Define Paths and Classification Parameters
# -------------------------------------------------------------------------

# Base directories
base_dir = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM"
shapefile_path = os.path.join(base_dir, "Pakistan_shape_file-with-Kashmir", "Pakistan_with_Kashmir.shp")
csv_dir = os.path.join(base_dir, "WPD-Clafficitation")
output_dir = csv_dir  # Output directory for saving figures

# Classification parameters for each hub height
classification_params = {
    30: {
        "csv": "Classification_at_30.csv",
        "wpd_column": "WPD30m",
        "bins": [0, 160, 240, 320, 400, 480, 640, 1600],
        "labels": ['Poor', 'Marginal', 'Moderate', 'Good', 'Excellent', 'Outstanding', 'Superb'],
        "colors": ['#f7fbff', '#d2e3f3', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d', '#023858']
    },
    50: {
        "csv": "Classification_at_50.csv",
        "wpd_column": "WPD50m",
        "bins": [0, 200, 300, 400, 500, 600, 800, 2000],
        "labels": ['Poor', 'Marginal', 'Moderate', 'Good', 'Excellent', 'Outstanding', 'Superb'],
        "colors": ['#f7fbff', '#d2e3f3', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d', '#023858']
    },
    80: {
        "csv": "Classification_at_80.csv",
        "wpd_column": "WPD80m",
        "bins": [0, 250, 380, 500, 600, 750, 980, 2400],
        "labels": ['Poor', 'Marginal', 'Moderate', 'Good', 'Excellent', 'Outstanding', 'Superb'],
        "colors": ['#f7fbff', '#d2e3f3', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d', '#023858']
    },
    100: {
        "csv": "Classification_at_100.csv",
        "wpd_column": "WPD100m",
        "bins": [0, 146.9, 228.7, 302.8, 372.5, 452.0, 591.5, 3000],
        "labels": ['Poor', 'Marginal', 'Moderate', 'Good', 'Excellent', 'Outstanding', 'Superb'],
        "colors": ['#f7fbff', '#d2e3f3', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d', '#023858']
    }
}

# -------------------------------------------------------------------------
# 2. Load Pakistan Boundary Shape with Kashmir Region
# -------------------------------------------------------------------------
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)
pakistan_union = unary_union(pakistan.geometry)

# -------------------------------------------------------------------------
# 3. Check if Point is in Boundary
# -------------------------------------------------------------------------
def point_in_boundary(point, boundary):
    shapely_point = Point(point)
    if isinstance(boundary, Polygon):
        return boundary.contains(shapely_point)
    elif isinstance(boundary, MultiPolygon):
        return any(poly.contains(shapely_point) for poly in boundary.geoms)
    return False

# -------------------------------------------------------------------------
# 4. Plot Wind Power Density Maps for Multiple Hub Heights
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (height, params) in enumerate(classification_params.items()):
    print(f"Processing {height}m hub height...")
    csv_path = os.path.join(csv_dir, params["csv"])
    df = pd.read_csv(csv_path)
    df[params["wpd_column"]] = pd.to_numeric(df[params["wpd_column"]], errors="coerce")
    df = df.dropna(subset=[params["wpd_column"], "Latitude", "Longitude"])

    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Interpolation Grid
    lons, lats = df["Longitude"].values, df["Latitude"].values
    wpd = df[params["wpd_column"]].values
    minx, miny, maxx, maxy = pakistan.total_bounds
    gridx = np.linspace(minx, maxx, 300)
    gridy = np.linspace(miny, maxy, 300)

    OK = OrdinaryKriging(
        lons, lats, wpd,
        variogram_model='spherical',
        verbose=False, enable_plotting=False,
        coordinates_type='geographic'
    )
    z, ss = OK.execute("grid", gridx, gridy)

    gx, gy = np.meshgrid(gridx, gridy)
    grid_points = np.vstack((gx.flatten(), gy.flatten())).T
    mask = np.array([point_in_boundary(pt, pakistan_union) for pt in grid_points])
    mask = mask.reshape(gx.shape)
    z_masked = np.where(mask, z, np.nan)

    # Plotting
    ax = axes[idx]
    bins, labels, colors = params["bins"], params["labels"], params["colors"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bins, ncolors=len(colors))
    contour = ax.contourf(gridx, gridy, z_masked, levels=bins, cmap=cmap, norm=norm, alpha=0.95)
    pakistan.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)
    gdf.plot(ax=ax, color='black', markersize=30, edgecolor='white', label='Stations', zorder=10)

    # Station Labels
    texts = []
    for _, row in gdf.iterrows():
        text = ax.text(row["Longitude"], row["Latitude"], row["Station Name"],
                       fontsize=5, ha='left', va='bottom',
                       fontname='Times New Roman', fontweight='bold')
        texts.append(text)
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

    # Axis & Title
    ax.set_title(f"Wind Power Density Classification at {height}m", fontsize=14, fontweight='bold', fontname='Times New Roman')
    ax.set_xlabel("Longitude", fontsize=11, fontname='Times New Roman')
    ax.set_ylabel("Latitude", fontsize=11, fontname='Times New Roman')
    ax.tick_params(labelsize=9)

    # ---------------------------------------------
    # CUSTOM LEGEND POSITION (Lowered Slightly)
    # ---------------------------------------------
    # `bbox_to_anchor` controls the inset position; adjust y value to move downward
    cax = inset_axes(ax, width="3%", height="35%", loc='lower right',
                     bbox_to_anchor=(0.03, -0.20, 1, 1),  # <- lowered legend by adjusting y to -0.08
                     bbox_transform=ax.transAxes, borderpad=8.5)

    cb = fig.colorbar(contour, cax=cax, orientation='vertical')
    tick_positions = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
    cb.set_ticks(tick_positions)
    cb.set_ticklabels(labels)
    cb.set_label("WPD Class (W/m²)", fontsize=10, fontname='Times New Roman')
    cb.ax.tick_params(labelsize=8)

# -------------------------------------------------------------------------
# 5. Final Layout & Save
# -------------------------------------------------------------------------
plt.suptitle('Wind Power Density Classification at Different Hub Heights (Pakistan)',
             fontsize=16, fontweight='bold', fontname='Times New Roman')

output_filename = "Combined_WPD_Classification_Pakistan.png"
output_path = os.path.join(output_dir, output_filename)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ensure room for title
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved combined map to: {output_path}")
