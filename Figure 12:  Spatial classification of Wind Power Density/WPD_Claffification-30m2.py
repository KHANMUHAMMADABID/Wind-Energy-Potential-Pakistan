# -------------------------------------------------------------------------
# WIND POWER DENSITY CLASSIFICATION MAP AT 30m HUB HEIGHT - PAKISTAN
# -------------------------------------------------------------------------

# --------------------
# 0. Required Libraries
# --------------------
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
import warnings
warnings.filterwarnings("ignore")  # Suppress pykrige and shapely warnings

# ----------------------------
# 1. Load Pakistan Boundary Shape with Kashmir Region
# ----------------------------
shapefile_path = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/Pakistan_shape_file-with-Kashmir/Pakistan_with_Kashmir.shp"
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)  # Project to WGS84 for consistency

# ----------------------------
# 2. Load Wind Power Density Station Data (CSV)
# ----------------------------
csv_path = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/WPD-Clafficitation/Classification_at_30.csv"
df = pd.read_csv(csv_path)

# Debug: Show column names
print("Column Names:", df.columns.tolist())

# Convert WPD column to numeric and drop rows with missing critical values
df["WPD30m"] = pd.to_numeric(df["WPD30m"], errors="coerce")
df = df.dropna(subset=["WPD30m", "Latitude", "Longitude"])

# ----------------------------
# 3. Convert DataFrame to GeoDataFrame
# ----------------------------
geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# ----------------------------
# 4. Apply Ordinary Kriging for Interpolation
# ----------------------------
# Extract station coordinates and WPD values
lons = df["Longitude"].values
lats = df["Latitude"].values
wpd = df["WPD30m"].values

# Define interpolation grid
minx, miny, maxx, maxy = pakistan.total_bounds
gridx = np.linspace(minx, maxx, 300)
gridy = np.linspace(miny, maxy, 300)

# Perform Kriging interpolation
OK = OrdinaryKriging(
    lons, lats, wpd,
    variogram_model='spherical',
    verbose=False,
    enable_plotting=False,
    coordinates_type='geographic'
)
z, ss = OK.execute("grid", gridx, gridy)

# ----------------------------
# 5. Mask Interpolated Data Outside Pakistan Border
# ----------------------------
gx, gy = np.meshgrid(gridx, gridy)
grid_points = np.vstack((gx.flatten(), gy.flatten())).T
pakistan_union = unary_union(pakistan.geometry)

def point_in_boundary(point, boundary):
    """Check if point lies inside the given boundary polygon or multipolygon."""
    shapely_point = Point(point)
    if isinstance(boundary, Polygon):
        return boundary.contains(shapely_point)
    elif isinstance(boundary, MultiPolygon):
        return any(poly.contains(shapely_point) for poly in boundary.geoms)
    return False

# Apply mask
mask = np.array([point_in_boundary(pt, pakistan_union) for pt in grid_points])
mask = mask.reshape(gx.shape)
z_masked = np.where(mask, z, np.nan)

# ----------------------------
# 6. Define WPD Classification (International Standard)
# ----------------------------
bins = [0, 160, 240, 320, 400, 480, 640, 1600]
labels = ['Poor', 'Marginal', 'Moderate', 'Good', 'Excellent', 'Outstanding', 'Superb']
colors = ['#f7fbff', '#d2e3f3', '#a6bddb', '#74a9cf', '#2b8cbe', '#045a8d', '#023858']
cmap = ListedColormap(colors)
norm = BoundaryNorm(bins, ncolors=len(colors))

# ----------------------------
# 7. Plot the WPD Classification Map with Station Labels
# ----------------------------
fig, ax = plt.subplots(figsize=(10, 12))

# Filled contour of interpolated WPD values
contour = ax.contourf(gridx, gridy, z_masked, levels=bins, cmap=cmap, norm=norm, alpha=0.95)

# Overlay Pakistan border
pakistan.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)

# Plot station points
gdf.plot(ax=ax, color='black', markersize=30, edgecolor='white', label='Stations', zorder=10)

# Adjust station labels using adjustText to prevent overlapping
texts = []
for idx, row in gdf.iterrows():
    text = ax.text(row["Longitude"], row["Latitude"], row["Station Name"],
                   fontsize=7, ha='left', va='bottom',
                   fontname='Times New Roman', fontweight='bold')
    texts.append(text)

adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))

# Title and axis labels
ax.set_title("Wind Power Density Classification at 30m in Pakistan", fontsize=14,
             fontweight='bold', fontname='Times New Roman')
ax.set_xlabel("Longitude", fontsize=11, fontname='Times New Roman')
ax.set_ylabel("Latitude", fontsize=11, fontname='Times New Roman')
ax.tick_params(labelsize=9)

# ----------------------------
# 8. Add Custom Colorbar with Classification Labels
# ----------------------------
cax = inset_axes(ax, width="3%", height="35%", loc='lower right',
                 bbox_to_anchor=(0, 0, 1, 1),
                 bbox_transform=ax.transAxes, borderpad=8)

cb = fig.colorbar(contour, cax=cax, orientation='vertical')
tick_positions = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
cb.set_ticks(tick_positions)
cb.set_ticklabels(labels)
cb.set_label("WPD Class (W/m²)", fontsize=10, fontname='Times New Roman')
cb.ax.tick_params(labelsize=8)

# ----------------------------
# 9. Save and Display the Final Output
# ----------------------------
plt.tight_layout()
output_path = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/WPD_30m_Map_Classification_Masked.png"
plt.savefig(output_path, dpi=600)
plt.show()
