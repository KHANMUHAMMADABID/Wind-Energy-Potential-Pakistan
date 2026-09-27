# ===============================================================
# Professional Map of Pakistan (with Kashmir) + DEM + Red Stations
# ===============================================================

import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from shapely.geometry import mapping, Point
import rioxarray
import os
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects  # For text outline effect

# ------------------------------
# 1. File Paths
# ------------------------------
base_path = os.path.expanduser("~/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM")
shapefile_path = os.path.join(base_path, "Pakistan_shape_file-with-Kashmir", "Pakistan_with_Kashmir.shp")
dem_nc_path = os.path.join(base_path, "Word_DEM-data", "ETOPO_2022_v1_30s_N90W180_bed.nc")
stations_csv_path = os.path.join(base_path, "Meteorological_Observatories_PAK.csv")
output_dir = os.path.join(base_path, "Pakistan_Maps")
os.makedirs(output_dir, exist_ok=True)

# Output file names
cropped_dem_netcdf_path = os.path.join(output_dir, "Pakistan_DEM_Cropped.nc")
cropped_dem_tif_path = os.path.join(output_dir, "Pakistan_DEM_Cropped.tif")
map_output_path = os.path.join(output_dir, "Pakistan_with_Kashmir_DEM_Stations.png")

# ------------------------------
# 2. Load Pakistan Shapefile
# ------------------------------
print("🔹 Loading Pakistan shapefile...")
pakistan = gpd.read_file(shapefile_path).to_crs(epsg=4326)

# ------------------------------
# 3. Load and Prepare DEM from NetCDF
# ------------------------------
print("🔹 Loading DEM from NetCDF...")
ds = xr.open_dataset(dem_nc_path)
elev_var = "z" if "z" in ds else "elevation" if "elevation" in ds else None
if not elev_var:
    raise KeyError("❌ DEM file must contain variable 'z' or 'elevation'.")

dem = ds[elev_var]
dem.rio.write_crs("EPSG:4326", inplace=True)
dem.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

# ------------------------------
# 4. Clip DEM to Pakistan Boundary
# ------------------------------
print("🔹 Clipping DEM to Pakistan boundary...")
geometry = [mapping(geom) for geom in pakistan.geometry]
dem_cropped = dem.rio.clip(geometry, pakistan.crs, drop=True)

if np.isnan(dem_cropped.values).all():
    raise ValueError("❌ Cropped DEM is empty. Check overlap with shapefile.")

# ------------------------------
# 5. Save Cropped DEM Files
# ------------------------------
print("💾 Saving cropped DEM to NetCDF and GeoTIFF...")
dem_cropped.to_netcdf(cropped_dem_netcdf_path)
dem_cropped.rio.to_raster(cropped_dem_tif_path)

# ------------------------------
# 6. Load Meteorological Station Data
# ------------------------------
print("🔹 Loading meteorological stations...")
stations_df = pd.read_csv(stations_csv_path, sep=r'\s*,\s*|\t+', engine='python')
stations_df.columns = [col.strip() for col in stations_df.columns]

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(stations_df["Longitude"], stations_df["Latitude"])]
stations_gdf = gpd.GeoDataFrame(stations_df, geometry=geometry, crs="EPSG:4326")

# ------------------------------
# 7. Plotting Map
# ------------------------------
print("🔹 Generating map...")

fig, ax = plt.subplots(figsize=(10, 12))

# DEM terrain plot
img = dem_cropped.plot(
    ax=ax,
    cmap="terrain",
    robust=True,
    add_colorbar=False
)

# Pakistan boundary
pakistan.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)

# Plot red station points
stations_gdf.plot(ax=ax, color='red', markersize=35, edgecolor='black', linewidth=0.3, label='Stations')

# Add station labels in red with black outline
texts = []
for idx, row in stations_gdf.iterrows():
    text = ax.text(
        row["Longitude"], row["Latitude"], row["Station Name"],
        fontsize=8, fontname="Times New Roman", color="yellow",  # Red text
        ha='left', va='bottom'
    )
    text.set_path_effects([
        PathEffects.Stroke(linewidth=1.2, foreground='black'),  # Black outline
        PathEffects.Normal()
    ])
    texts.append(text)

# Adjust overlapping labels
adjust_text(texts,
            only_move={'points': 'y', 'texts': 'y'},
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5))

# Axis titles
ax.set_title("Pakistan Digital Elevation Map with Meteorological Stations",
             fontsize=14, fontweight="bold", pad=12, fontname="Times New Roman")
ax.set_xlabel("Longitude", fontsize=11, fontname="Times New Roman")
ax.set_ylabel("Latitude", fontsize=11, fontname="Times New Roman")
ax.tick_params(labelsize=9)

# Set limits based on DEM bounds
ax.set_xlim(dem_cropped.lon.min().item(), dem_cropped.lon.max().item())
ax.set_ylim(dem_cropped.lat.min().item(), dem_cropped.lat.max().item())

# Add colorbar for DEM
cax = inset_axes(ax, width="3%", height="35%", loc='lower right',
                 bbox_to_anchor=(0, 0, 1, 1),
                 bbox_transform=ax.transAxes, borderpad=6.0)
cbar = fig.colorbar(img, cax=cax, orientation='vertical')
cbar.set_label("Elevation (m)", fontsize=10, fontname="Times New Roman")
cbar.ax.tick_params(labelsize=8)

# ------------------------------
# 8. Save and Show Map
# ------------------------------
plt.savefig(map_output_path, dpi=600, bbox_inches="tight", pad_inches=1.0)
plt.show()

# ------------------------------
# 9. Completion
# ------------------------------
print(f"\n✅ Map saved to: {map_output_path}")
print(f"✅ DEM NetCDF saved to: {cropped_dem_netcdf_path}")
print(f"✅ DEM GeoTIFF saved to: {cropped_dem_tif_path}")
