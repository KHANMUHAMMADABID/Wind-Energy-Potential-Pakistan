# ============================================================================
# SECTION 1: IMPORT LIBRARIES
# ============================================================================
"""
This script generates professional elevation maps of Pakistan in different regional contexts:
1. Globe view of Pakistan (Orthographic projection)
2. Pakistan in Asia context (Robinson projection)
3. Pakistan in South Asia context (Robinson projection)

Key Features:
1. Enhanced elevation classification with detailed coastal zones
2. Single-column legend for clear readability
3. Three different map projections for different regional contexts
4. Professional cartographic output with multiple formats
5. Comprehensive educational comments for GIS beginners
6. Bottom legend for South Asia map for better layout
"""

# Standard library imports for file and system operations
import os  # For file and directory operations
import sys  # For system-level operations
import warnings  # To suppress non-critical warnings
from datetime import datetime  # For timestamping output files

# Scientific computing libraries
import numpy as np  # For numerical operations on elevation data

# Visualization libraries
import matplotlib.pyplot as plt  # For creating visualizations
import matplotlib.patches as mpatches  # For creating legend patches
from matplotlib.colors import ListedColormap, BoundaryNorm  # For color mapping

# Geographic data processing libraries
import cartopy.crs as ccrs  # Map projections (creates different map views)
import cartopy.feature as cfeature  # Map features (coastlines, borders)
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER  # Grid formatting

# Raster and vector data handling
import rasterio  # For reading Digital Elevation Model (DEM) .tif files
from rasterio.mask import mask  # For cropping raster to shapefile boundaries
import geopandas as gpd  # For working with shapefiles (vector data)

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 2: DEFINE ABSOLUTE PATHS
# ============================================================================
"""
Define absolute file paths for input data and output directories.
Using absolute paths ensures the script runs regardless of working directory.
"""

# User-specific directory paths (modify these according to your system)
SHAPEFILE_DIR = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/fwdpakistanshapefilewithkashmir"
DEM_DIR = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/Pakistan_Maps"
SCRIPT_DIR = "/Users/newpostdoc/Documents/1-NARO_Pakistan_Paper/1-Pakistan-DEM/CoverMap-Pakistan"

# Output directory for generated maps
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

# Define complete file paths
SHAPEFILE_PATH = os.path.join(SHAPEFILE_DIR, "Pakistan_with_Kashmir.shp")
DEM_TIFF_PATH = os.path.join(DEM_DIR, "Pakistan_DEM_Cropped.tif")

# ============================================================================
# SECTION 3: COORDINATE ALIGNMENT CLASS
# ============================================================================
class CoordinateAligner:
    """
    Handles coordinate system alignment between DEM and shapefile.
    Fixes common coordinate inversion issues that occur in GIS data processing.
    
    Key Responsibilities:
    1. Check coordinate reference systems (CRS) of input datasets
    2. Transform shapefile to match DEM's CRS if needed
    3. Crop DEM to shapefile boundaries with coordinate correction
    
    Educational Note:
    Coordinate systems define how geographic locations are represented on Earth.
    Different datasets must be in the same coordinate system for accurate overlay.
    """
    
    @staticmethod
    def check_coordinate_systems(shapefile_path, dem_path):
        """
        Check and compare coordinate reference systems (CRS) of both datasets.
        
        Parameters:
        shapefile_path (str): Path to shapefile
        dem_path (str): Path to DEM file
        
        Returns:
        dict: Dictionary containing CRS information and bounds
        """
        print("   🔍 Checking coordinate systems...")
        
        crs_info = {}
        
        try:
            # Read shapefile and extract CRS information
            # GeoPandas reads shapefiles and stores CRS information
            gdf = gpd.read_file(shapefile_path)
            crs_info['shapefile_crs'] = gdf.crs  # Coordinate Reference System
            crs_info['shapefile_bounds'] = gdf.total_bounds  # Spatial extent
            print(f"   ✅ Shapefile CRS: {gdf.crs}")
        except Exception as e:
            print(f"   ❌ Failed to read shapefile CRS: {e}")
            return None
        
        try:
            # Read DEM file and extract CRS information
            # Rasterio is used for reading raster files like GeoTIFF
            with rasterio.open(dem_path) as src:
                crs_info['dem_crs'] = src.crs
                crs_info['dem_bounds'] = src.bounds
                crs_info['dem_transform'] = src.transform  # Geo-transform matrix
                crs_info['dem_height'] = src.height  # Number of rows
                crs_info['dem_width'] = src.width  # Number of columns
            print(f"   ✅ DEM CRS: {crs_info['dem_crs']}")
        except Exception as e:
            print(f"   ❌ Failed to read DEM CRS: {e}")
            return None
        
        return crs_info
    
    @staticmethod
    def transform_shapefile_to_dem_crs(shapefile_path, dem_crs):
        """
        Transform shapefile to match DEM's coordinate system if needed.
        
        Parameters:
        shapefile_path (str): Path to shapefile
        dem_crs (CRS): Coordinate Reference System of DEM
        
        Returns:
        GeoDataFrame: Transformed shapefile or original if already matching
        """
        print("   🔄 Transforming shapefile to DEM CRS...")
        
        try:
            # Load shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Check if transformation is needed
            # Different CRS can cause misalignment between datasets
            if gdf.crs != dem_crs:
                print(f"   ⚠️  CRS mismatch detected.")
                print(f"     Shapefile CRS: {gdf.crs}")
                print(f"     DEM CRS: {dem_crs}")
                
                # Perform coordinate transformation
                # to_crs() method transforms coordinates to new CRS
                gdf_transformed = gdf.to_crs(dem_crs)
                print(f"   ✅ Shapefile transformed to: {gdf_transformed.crs}")
                return gdf_transformed
            else:
                print("   ✅ Shapefile already in DEM CRS")
                return gdf
                
        except Exception as e:
            print(f"   ❌ Failed to transform shapefile: {e}")
            return None
    
    @staticmethod
    def crop_dem_to_shapefile(dem_path, shapefile_gdf):
        """
        Crop DEM raster to shapefile boundaries.
        Includes fix for coordinate inversion (north-south flip).
        
        Parameters:
        dem_path (str): Path to DEM file
        shapefile_gdf (GeoDataFrame): Shapefile geometry
        
        Returns:
        tuple: (cropped_data, transform, bounds) of cropped DEM
        """
        print("   ✂️  Cropping DEM to shapefile boundaries...")
        
        try:
            # Open DEM file using rasterio context manager
            with rasterio.open(dem_path) as src:
                # Extract geometry from shapefile for masking
                shapes = [feature["geometry"] for feature in shapefile_gdf.__geo_interface__['features']]
                
                # Crop DEM using shapefile geometry as mask
                # mask() function extracts raster data within the polygon
                out_image, out_transform = mask(src, shapes, crop=True)
                
                # Update metadata for cropped raster
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                
                # Calculate bounds of cropped raster
                out_bounds = rasterio.transform.array_bounds(
                    out_image.shape[1], out_image.shape[2], out_transform
                )
                
                print(f"   ✅ DEM cropped. New shape: {out_image.shape}")
                
                # FIX: Check for coordinate inversion (common GIS issue)
                # Sometimes DEMs have inverted latitude values
                left, bottom, right, top = out_bounds
                if bottom > top:
                    print(f"   ⚠️  Latitude inversion detected (bottom > top)")
                    print(f"     Original: bottom={bottom:.6f}, top={top:.6f}")
                    
                    # Correct inversion by swapping coordinates
                    corrected_bounds = (left, top, right, bottom)
                    print(f"     Corrected: bottom={top:.6f}, top={bottom:.6f}")
                    return out_image[0], out_transform, corrected_bounds
                
                return out_image[0], out_transform, out_bounds
                
        except Exception as e:
            print(f"   ❌ Failed to crop DEM: {e}")
            return None, None, None

# ============================================================================
# SECTION 4: MAIN MAP GENERATOR CLASS
# ============================================================================
class PakistanMultiMapGenerator:
    """
    Main class orchestrating the generation of multiple maps of Pakistan.
    Creates three different professional elevation maps:
    1. Globe view of Pakistan (Orthographic projection)
    2. Pakistan in Asia context (Robinson projection)
    3. Pakistan in South Asia context (Robinson projection)
    
    Workflow:
    1. Validate input files
    2. Check and align coordinate systems
    3. Load and preprocess data
    4. Classify elevation
    5. Create three different visualizations
    6. Save output files
    
    Educational Purpose:
    This class is designed to be educational for GIS beginners.
    Each method includes detailed comments explaining GIS concepts.
    """
    
    def __init__(self):
        """
        Initialize the map generator with configuration settings.
        """
        # Print startup banner
        self.print_banner()
        
        # Store file paths
        self.shapefile_path = SHAPEFILE_PATH
        self.dem_path = DEM_TIFF_PATH
        self.output_dir = OUTPUT_DIR
        
        # Configuration dictionary for map settings
        self.config = {
            'dpi': 300,  # Resolution for output images (dots per inch)
            'figsize': (16, 12),  # Figure dimensions (width, height) in inches
            'center_lon': 70.0,  # Center longitude for globe projection
            'center_lat': 30.0,  # Center latitude for globe projection
            'ocean_color': '#a6cee3',  # Color for ocean areas
            'land_color': '#f0f0f0',  # Color for land background
            'border_color': '#333333',  # Color for country borders
            'border_width': 1.0,  # Line width for borders
            'coastline_width': 1.0,  # Line width for coastlines
            'title_fontsize': 22,  # Font size for main title
            'legend_fontsize': 8,  # Font size for legend text
            'legend_title_fontsize': 10,  # Font size for legend title
            'grid_fontsize': 10,  # Font size for grid labels
            'legend_columns': 1,  # SINGLE COLUMN LEGEND
        }
        
        # Regional map configurations
        # UPDATED: Removed Pakistan labels from Asia and South Asia maps
        # UPDATED: South Asia map legend moved to bottom position
        self.regional_configs = {
            'asia': {
                'name': 'Asia',
                'extent': [25, 150, -15, 60],  # [lon_min, lon_max, lat_min, lat_max]
                'center_lon': 87.5,  # Center longitude for Asia
                'title': 'Pakistan in Asia Region',
                'show_pakistan_label': False,  # UPDATED: No Pakistan label on map
                'legend_position': 'top',  # Legend at top for Asia map
                'highlight_pakistan': True,  # Highlight Pakistan with fill
            },
            'south_asia': {
                'name': 'South Asia',
                'extent': [60, 100, 0, 40],  # [lon_min, lon_max, lat_min, lat_max]
                'center_lon': 80.0,  # Center longitude for South Asia
                'title': 'Pakistan in South Asia Region',
                'show_pakistan_label': False,  # UPDATED: No Pakistan label on map
                'legend_position': 'bottom',  # UPDATED: Legend at bottom for South Asia map
                'highlight_pakistan': True,  # Highlight Pakistan with fill
            }
        }
        
        # Initialize data storage variables
        self.boundary = None  # Country boundary data
        self.dem_data = None  # Digital Elevation Model data
        self.dem_bounds = None  # Bounds of DEM data
        self.dem_transform = None  # Geo-transform matrix
        self.elevation_classes = None  # Classified elevation data
        self.elevation_stats = {}  # Statistics about elevation data
        self.valid_dem_mask = None  # Mask for valid elevation values
        self.cropped_dem_data = None  # Cropped DEM data
        self.transformed_boundary = None  # Boundary in DEM coordinate system
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up enhanced elevation classification with detailed coastal zones
        self.setup_enhanced_elevation_classes()
        
        # Initialize coordinate aligner
        self.aligner = CoordinateAligner()
    
    # ------------------------------------------------------------------------
    # HELPER METHODS FOR USER FEEDBACK
    # ------------------------------------------------------------------------
    def print_banner(self):
        """Print a professional startup banner."""
        print("\n" + "="*70)
        print("🌍 PAKISTAN MULTI-MAP GENERATOR")
        print("="*70)
        print("Version 6.1 - Professional GIS Visualization Tool")
        print("="*70)
        print("Author: Scientific Python Developer")
        print("Purpose: Educational - For beginner GIS students")
        print("="*70)
        print("Generating 3 maps:")
        print("  1. Globe view of Pakistan (Orthographic projection)")
        print("  2. Pakistan in Asia context (Robinson projection)")
        print("  3. Pakistan in South Asia context (Robinson projection)")
        print("="*70)
        print("UPDATES:")
        print("  • No Pakistan labels on regional maps (title is sufficient)")
        print("  • South Asia map legend moved to bottom position")
        print("="*70)
    
    def print_step(self, number, name):
        """Print formatted step header for progress tracking."""
        print(f"\n[STEP {number}/9] {name}")
    
    def print_success(self, message):
        """Print success message with checkmark icon."""
        print(f"   ✅ {message}")
    
    def print_error(self, message):
        """Print error message with cross icon."""
        print(f"   ❌ {message}")
    
    def print_info(self, message):
        """Print informational message."""
        print(f"   ℹ️  {message}")
    
    def print_warning(self, message):
        """Print warning message."""
        print(f"   ⚠️  {message}")
    
    def print_debug(self, message):
        """Print debug message (only for development)."""
        print(f"   🐛 DEBUG: {message}")
    
    # ------------------------------------------------------------------------
    # ENHANCED ELEVATION CLASSIFICATION SETUP
    # ------------------------------------------------------------------------
    def setup_enhanced_elevation_classes(self):
        """
        Define enhanced color scheme and classification for elevation data.
        Includes detailed coastal zone classification with descriptive names.
        Uses scientifically appropriate color gradient from low (blue/green)
        to high (brown/white) elevation.
        
        Educational Note:
        Elevation classification is important for understanding topography.
        Coastal zones (0-300m) are divided into 7 sub-classes for detailed analysis.
        """
        print("   🎨 Setting up enhanced elevation classification...")
        
        # Enhanced color dictionary for each elevation range
        # Colors are chosen to create a natural progression from sea to mountains
        self.elevation_colors = {
            'Below Sea Level': '#1f78b4',  # Deep blue for areas below sea level
            '0m to 50m': '#e0f3f8',  # Very light blue for very low coastal areas
            '50m to 100m': '#bae4bc',  # Light green for low coastal areas
            '100m to 150m': '#7bccc4',  # Teal for mid coastal areas
            '150m to 200m': '#43a2ca',  # Light blue for high coastal areas
            '200m to 250m': '#0868ac',  # Blue for coastal plains
            '250m to 300m': '#084081',  # Dark blue for upper coastal plains
            '300m to 600m': '#b2df8a',  # Light green for lowlands
            '600m to 1000m': '#33a02c',  # Green for foothills
            '1000m to 1500m': '#fb9a99',  # Light red for hills
            '1500m to 2000m': '#e31a1c',  # Red for high hills
            '2000m to 3000m': '#fdbf6f',  # Orange for lower mountains
            '3000m to 4000m': '#ff7f00',  # Dark orange for mountains
            '400m to 5000m': '#cab2d6',  # Light purple for high mountains
            '500m to 6000m': '#6a3d9a',  # Purple for very high mountains
            '600m to 7000m': '#ffff99',  # Light yellow for extreme mountains
            'Above 7000m': '#ffffff'  # White for peaks (snow-covered)
        }
        
        # Enhanced elevation boundaries with detailed coastal zones
        # These values define the breakpoints between elevation classes
        self.elevation_bounds = [
            -float('inf'),  # Below sea level (negative infinity to 0)
            0, 50, 100, 150, 200, 250, 300,  # Coastal zones (7 sub-classes)
            600, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000,  # Inland zones
            float('inf')  # Above 7000m (7000 to positive infinity)
        ]
        
        # Enhanced class names with detailed coastal zone descriptions
        # Each class has a descriptive name for educational purposes
        self.elevation_class_names = [
            'Below Sea Level',
            '0m to 50m (Very Low Coastal)',
            '50m to 100m (Low Coastal)',
            '100m to 150m (Mid Coastal)',
            '150m to 200m (High Coastal)',
            '200m to 250m (Coastal Plains)',
            '250m to 300m (Upper Coastal Plains)',
            '300m to 600m (Lowlands)',
            '600m to 1000m (Foothills)',
            '1000m to 1500m (Hills)',
            '1500m to 2000m (High Hills)',
            '2000m to 3000m (Lower Mountains)',
            '3000m to 4000m (Mountains)',
            '400m to 5000m (High Mountains)',
            '500m to 6000m (Very High Mountains)',
            '600m to 7000m (Extreme Mountains)',
            'Above 7000m (Peaks)'
        ]
        
        # Validate that all class names have a corresponding color
        print("   🔍 Validating enhanced color map...")
        for full_name in self.elevation_class_names:
            # Extract base name (part before parenthesis for color lookup)
            # Some names have descriptive text in parentheses
            if '(' in full_name:
                base_name = full_name.split(' (')[0]
            else:
                base_name = full_name
            
            if base_name not in self.elevation_colors:
                raise KeyError(f"Color not found for elevation class: '{base_name}'.")
        
        self.print_success(f"Enhanced color map validated ({len(self.elevation_class_names)} classes)")
        
        # Create a matplotlib colormap from the enhanced colors
        # ListedColormap creates a discrete colormap from a list of colors
        colors_list = []
        for full_name in self.elevation_class_names:
            if '(' in full_name:
                base_name = full_name.split(' (')[0]
            else:
                base_name = full_name
            colors_list.append(self.elevation_colors[base_name])
        
        self.cmap = ListedColormap(colors_list)
        
        # Create a normalization object for mapping data to colors
        # BoundaryNorm maps data values to discrete colors based on boundaries
        self.norm = BoundaryNorm(self.elevation_bounds, self.cmap.N)
        
        self.print_info(f"Total elevation classes: {len(self.elevation_class_names)}")
        self.print_info(f"Detailed coastal zone classification: 7 sub-classes (0-300m)")
        
        # Print class information for educational purposes
        print("\n   📋 Elevation Classification System:")
        for i, class_name in enumerate(self.elevation_class_names):
            if i < len(self.elevation_bounds) - 1:
                # Format the bounds display nicely
                if self.elevation_bounds[i] == -float('inf'):
                    lower_bound = "-∞"
                else:
                    lower_bound = f"{self.elevation_bounds[i]:>6.0f}"
                
                if self.elevation_bounds[i+1] == float('inf'):
                    upper_bound = "∞"
                else:
                    upper_bound = f"{self.elevation_bounds[i+1]:<6.0f}"
                
                print(f"     {i+1:2d}. {class_name:<35s} [{lower_bound}m - {upper_bound}m]")
    
    # ------------------------------------------------------------------------
    # MAIN PROCESSING PIPELINE
    # ------------------------------------------------------------------------
    def validate_inputs(self):
        """
        STEP 1: Validate that all input files exist.
        
        Returns:
        bool: True if all files exist, False otherwise
        
        Educational Note:
        Always validate input files before processing to avoid runtime errors.
        Shapefiles consist of multiple files (.shp, .shx, .dbf, .prj).
        """
        self.print_step(1, "Validating Input Files")
        self.print_info(f"Shapefile: {os.path.basename(self.shapefile_path)}")
        self.print_info(f"DEM File: {os.path.basename(self.dem_path)}")
        
        # Check shapefile existence
        if not os.path.exists(self.shapefile_path):
            self.print_error(f"Shapefile not found: {self.shapefile_path}")
            return False
        
        # Check DEM file existence
        if not os.path.exists(self.dem_path):
            self.print_error(f"DEM file not found: {self.dem_path}")
            return False
        
        # Check for required shapefile components
        # Shapefiles consist of multiple files with different extensions
        shapefile_extensions = ['.shp', '.shx', '.dbf', '.prj']
        base_path = os.path.splitext(self.shapefile_path)[0]
        missing_files = []
        
        for ext in shapefile_extensions:
            if not os.path.exists(base_path + ext):
                missing_files.append(ext)
        
        if missing_files:
            self.print_error(f"Missing shapefile components: {missing_files}")
            return False
        
        self.print_success("All input files validated")
        return True
    
    def check_coordinate_systems(self):
        """
        STEP 2: Check and compare coordinate reference systems.
        
        Returns:
        bool: True if coordinate systems are compatible
        
        Educational Note:
        Coordinate Reference Systems (CRS) define how geographic data is projected.
        Different datasets must be in the same CRS for accurate alignment.
        """
        self.print_step(2, "Checking Coordinate Systems")
        
        # Get CRS information from both datasets
        crs_info = self.aligner.check_coordinate_systems(self.shapefile_path, self.dem_path)
        
        if not crs_info:
            self.print_error("Failed to get coordinate system information")
            return False
        
        # Extract CRS from both datasets
        shapefile_crs = crs_info.get('shapefile_crs')
        dem_crs = crs_info.get('dem_crs')
        
        # Check if transformation is needed
        if shapefile_crs != dem_crs:
            self.print_warning("Coordinate systems don't match. Transformation will be applied.")
        else:
            self.print_success("Coordinate systems match")
        
        return True
    
    def load_and_align_data(self):
        """
        STEP 3 & 4: Load and align boundary and elevation data.
        
        Returns:
        bool: True if data loaded successfully
        
        Educational Note:
        This step combines loading both vector (shapefile) and raster (DEM) data.
        The data must be aligned in the same coordinate system for accurate overlay.
        """
        # STEP 3: Load Pakistan boundary
        self.print_step(3, "Loading and Aligning Pakistan Boundary")
        try:
            # Get DEM coordinate system by opening DEM file
            with rasterio.open(self.dem_path) as src:
                dem_crs = src.crs
            
            # Transform shapefile to match DEM CRS if needed
            self.transformed_boundary = self.aligner.transform_shapefile_to_dem_crs(
                self.shapefile_path, dem_crs
            )
            
            if self.transformed_boundary is None:
                self.print_error("Failed to transform shapefile")
                return False
            
            # Load original boundary for visualization
            self.boundary = gpd.read_file(self.shapefile_path)
            bounds = self.boundary.total_bounds
            
            self.print_success(f"Loaded boundary with {len(self.boundary)} region(s)")
            self.print_info(f"Boundary extent: {bounds[0]:.2f}°E to {bounds[2]:.2f}°E, "
                          f"{bounds[1]:.2f}°N to {bounds[3]:.2f}°N")
            
        except Exception as e:
            self.print_error(f"Failed to load shapefile: {e}")
            return False
        
        # STEP 4: Load and crop Digital Elevation Model
        self.print_step(4, "Loading and Cropping Elevation Data")
        try:
            # Crop DEM to boundary using transformed shapefile
            self.cropped_dem_data, self.dem_transform, self.dem_bounds = (
                self.aligner.crop_dem_to_shapefile(self.dem_path, self.transformed_boundary)
            )
            
            if self.cropped_dem_data is None:
                self.print_error("Failed to crop DEM")
                return False
            
            self.dem_data = self.cropped_dem_data
            
            # Identify valid data (non-null, non-no-data values)
            # DEM files often use specific values to represent no-data areas
            nodata = -99999.0  # Common no-data value in DEMs
            self.valid_dem_mask = (self.dem_data != nodata) & (~np.isnan(self.dem_data))
            
            # Calculate elevation statistics
            if np.any(self.valid_dem_mask):
                valid_data = self.dem_data[self.valid_dem_mask]
                
                # Basic statistics
                self.elevation_stats = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'valid_pixels': int(np.sum(self.valid_dem_mask))
                }
                
                # Calculate detailed coastal zone statistics
                coastal_zones = [
                    (0, 50), (50, 100), (100, 150), (150, 200),
                    (200, 250), (250, 300)
                ]
                total_coastal = 0
                
                for lower, upper in coastal_zones:
                    mask = (valid_data >= lower) & (valid_data < upper)
                    count = np.sum(mask)
                    total_coastal += count
                
                self.elevation_stats['coastal_pixels'] = total_coastal
                self.elevation_stats['coastal_percentage'] = (
                    total_coastal / self.elevation_stats['valid_pixels']
                ) * 100
                
                self.print_success(f"DEM loaded: {self.dem_data.shape[1]}×{self.dem_data.shape[0]} pixels")
                self.print_info(f"Valid pixels: {self.elevation_stats['valid_pixels']:,}")
                self.print_info(f"Elevation range: {self.elevation_stats['min']:.1f}m to "
                              f"{self.elevation_stats['max']:.1f}m")
                self.print_info(f"Mean elevation: {self.elevation_stats['mean']:.1f}m")
                self.print_info(f"Coastal zone pixels (0-300m): {total_coastal:,} "
                              f"({self.elevation_stats['coastal_percentage']:.1f}%)")
                
                # Debug information for coordinate bounds
                self.print_debug(f"DEM bounds: {self.dem_bounds}")
                
            else:
                self.print_error("No valid elevation data found!")
                return False
                
        except Exception as e:
            self.print_error(f"Failed to load DEM: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def classify_elevation(self):
        """
        STEP 5: Classify each pixel into enhanced elevation categories.
        
        Returns:
        bool: True if classification successful
        
        Educational Note:
        Elevation classification assigns each pixel to a specific elevation class.
        This allows for visualization and analysis of different elevation zones.
        """
        self.print_step(5, "Classifying Elevation (Enhanced Coastal Zones)")
        try:
            # Create array for classification results with same shape as DEM
            # Initialize with -1 (unclassified)
            self.elevation_classes = np.full_like(self.dem_data, -1, dtype=int)
            
            if self.valid_dem_mask is None:
                self.print_error("Valid DEM mask not created")
                return False
            
            # Classify each elevation range using enhanced elevation bounds
            # Loop through each elevation class defined in bounds
            for i in range(len(self.elevation_bounds) - 1):
                lower = self.elevation_bounds[i]
                upper = self.elevation_bounds[i + 1]
                
                # Handle infinite bounds for first and last classes
                if lower == -float('inf'):
                    # Below sea level class
                    mask = (self.dem_data < upper) & self.valid_dem_mask
                elif upper == float('inf'):
                    # Above highest elevation class
                    mask = (self.dem_data >= lower) & self.valid_dem_mask
                else:
                    # Regular elevation range classes
                    mask = (self.dem_data >= lower) & (self.dem_data < upper) & self.valid_dem_mask
                
                # Assign class index to pixels in this range
                self.elevation_classes[mask] = i
            
            total_valid = np.sum(self.valid_dem_mask)
            self.print_success(f"Classified {total_valid:,} pixels into "
                             f"{len(self.elevation_class_names)} enhanced categories")
            
            # Count pixels in each class for verification
            class_counts = []
            for i in range(len(self.elevation_class_names)):
                count = np.sum(self.elevation_classes == i)
                class_counts.append((self.elevation_class_names[i], count))
            
            # Print summary of classification
            print("\n   📊 Classification Summary:")
            for class_name, count in class_counts:
                if count > 0:
                    percentage = (count / total_valid) * 100
                    print(f"     {class_name:<40s}: {count:>10,} pixels ({percentage:>5.1f}%)")
            
            return True
        except Exception as e:
            self.print_error(f"Classification failed: {e}")
            return False
    
    # ------------------------------------------------------------------------
    # MAP CREATION METHODS
    # ------------------------------------------------------------------------
    def create_globe_map(self):
        """
        STEP 6: Create globe visualization of Pakistan (Orthographic projection).
        
        Returns:
        tuple: (figure, axis) matplotlib objects
        
        Educational Note:
        Orthographic projection creates a globe-like view as seen from space.
        This is useful for showing Pakistan's position on Earth.
        """
        self.print_step(6, "Creating Globe Map (Orthographic Projection)")
        try:
            # Create figure with specified size and resolution
            fig = plt.figure(figsize=self.config['figsize'], dpi=self.config['dpi'])
            
            # Define orthographic projection (globe view)
            # Orthographic projection shows Earth as it appears from space
            projection = ccrs.Orthographic(
                central_longitude=self.config['center_lon'],
                central_latitude=self.config['center_lat']
            )
            
            # Add subplot with globe projection
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            
            # Set map extent based on boundary with buffer
            if self.boundary is not None:
                bounds = self.boundary.total_bounds
                buffer = 2.0  # Degrees buffer around boundary
                ax.set_extent([bounds[0] - buffer, bounds[2] + buffer, 
                              bounds[1] - buffer, bounds[3] + buffer], 
                             crs=ccrs.PlateCarree())
            else:
                # Default extent for Pakistan if boundary not loaded
                ax.set_extent([60, 80, 20, 38], crs=ccrs.PlateCarree())
            
            # Add background features in order of increasing zorder
            # zorder controls drawing order (higher = drawn later/on top)
            ax.add_feature(cfeature.OCEAN, color=self.config['ocean_color'], zorder=0)
            ax.add_feature(cfeature.LAND, color=self.config['land_color'], zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=self.config['coastline_width'],
                          edgecolor=self.config['border_color'], zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=self.config['border_width'],
                          edgecolor=self.config['border_color'], zorder=1)
            
            # Plot the DEM data
            if self.dem_data is not None and self.dem_bounds is not None:
                self.print_info("Plotting enhanced elevation data...")
                
                # Extract bounds (left, bottom, right, top)
                left, bottom, right, top = self.dem_bounds
                
                # Create coordinate grids for plotting
                nrows, ncols = self.dem_data.shape
                
                # Longitude increases from left to right
                lon = np.linspace(left, right, ncols)
                
                # Latitude increases from bottom to top (south to north)
                # Check for coordinate inversion and correct if needed
                if bottom < top:
                    # Normal case: bottom is south, top is north
                    lat = np.linspace(bottom, top, nrows)
                else:
                    # Inverted case: bottom is north, top is south
                    # Flip the data vertically to correct coordinate order
                    lat = np.linspace(top, bottom, nrows)
                    self.dem_data = np.flipud(self.dem_data)
                    if self.elevation_classes is not None:
                        self.elevation_classes = np.flipud(self.elevation_classes)
                    if self.valid_dem_mask is not None:
                        self.valid_dem_mask = np.flipud(self.valid_dem_mask)
                    self.print_warning("Flipped DEM data vertically to correct coordinate order")
                
                # Create 2D coordinate grids
                lon_grid, lat_grid = np.meshgrid(lon, lat)
                
                # Create masked array for plotting (hide no-data values)
                dem_to_plot = np.where(self.valid_dem_mask, self.dem_data, np.nan)
                
                # Plot elevation data using pcolormesh
                # pcolormesh handles irregular grids better than imshow
                im = ax.pcolormesh(lon_grid, lat_grid, dem_to_plot,
                                  transform=ccrs.PlateCarree(),
                                  cmap=self.cmap, norm=self.norm,
                                  alpha=0.85, zorder=2)
                
                self.print_success(f"Elevation data plotted with {np.sum(self.valid_dem_mask):,} valid pixels")
            
            # Add Pakistan boundary outline
            if self.boundary is not None:
                # Plot boundary with thick black line (for visibility)
                self.boundary.boundary.plot(ax=ax, transform=ccrs.PlateCarree(),
                                           linewidth=2.5, color='black', zorder=3)
                # Plot boundary with colored line (for style)
                self.boundary.boundary.plot(ax=ax, transform=ccrs.PlateCarree(),
                                           linewidth=1.5, color=self.config['border_color'],
                                           zorder=4)
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.5, color='gray', alpha=0.5,
                             linestyle='--', zorder=5)
            gl.top_labels = gl.right_labels = False  # Show only left and bottom labels
            gl.xlabel_style = {'size': self.config['grid_fontsize']}
            gl.ylabel_style = {'size': self.config['grid_fontsize']}
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            
            # Add main title
            ax.set_title('Elevation Map of Pakistan (Globe View)',
                        fontsize=self.config['title_fontsize'],
                        fontweight='bold', pad=25)
            
            # Add enhanced legend in single column at top-left (default)
            self.add_enhanced_legend(ax, position='top')
            
            self.print_success("Globe map created successfully")
            return fig, ax
        except Exception as e:
            self.print_error(f"Failed to create globe map: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def create_regional_map(self, region_name):
        """
        STEP 7: Create regional map showing Pakistan in regional context.
        
        Parameters:
        region_name (str): Name of region ('asia' or 'south_asia')
        
        Returns:
        tuple: (figure, axis) matplotlib objects
        
        Educational Note:
        Regional maps show Pakistan in context with neighboring countries.
        Robinson projection is good for showing large areas with minimal distortion.
        """
        self.print_step(7, f"Creating {region_name.upper()} Regional Map (Robinson Projection)")
        
        # Get region configuration
        region_config = self.regional_configs[region_name]
        
        try:
            # Create figure with specified size and resolution
            fig = plt.figure(figsize=self.config['figsize'], dpi=self.config['dpi'])
            
            # Define Robinson projection for regional map
            # Robinson projection is good for showing large areas with minimal distortion
            projection = ccrs.Robinson(central_longitude=region_config['center_lon'])
            
            # Add subplot with Robinson projection
            ax = fig.add_subplot(1, 1, 1, projection=projection)
            
            # Set map extent for the specific region
            # Extent format: [lon_min, lon_max, lat_min, lat_max]
            ax.set_extent(region_config['extent'], crs=ccrs.PlateCarree())
            
            # Add background features for the entire region
            ax.add_feature(cfeature.OCEAN, color=self.config['ocean_color'], zorder=0)
            ax.add_feature(cfeature.LAND, color=self.config['land_color'], zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=self.config['coastline_width'],
                          edgecolor=self.config['border_color'], zorder=1)
            ax.add_feature(cfeature.BORDERS, linewidth=self.config['border_width'],
                          edgecolor=self.config['border_color'], zorder=1)
            
            # Plot the Pakistan DEM data
            if self.dem_data is not None and self.dem_bounds is not None:
                self.print_info(f"Plotting Pakistan elevation data on {region_name} map...")
                
                # Extract bounds (left, bottom, right, top)
                left, bottom, right, top = self.dem_bounds
                
                # Create coordinate grids for plotting
                nrows, ncols = self.dem_data.shape
                
                # Longitude increases from left to right
                lon = np.linspace(left, right, ncols)
                
                # Latitude increases from bottom to top (south to north)
                # Check for coordinate inversion and correct if needed
                if bottom < top:
                    # Normal case: bottom is south, top is north
                    lat = np.linspace(bottom, top, nrows)
                else:
                    # Inverted case: bottom is north, top is south
                    lat = np.linspace(top, bottom, nrows)
                
                # Create 2D coordinate grids
                lon_grid, lat_grid = np.meshgrid(lon, lat)
                
                # Create masked array for plotting (hide no-data values)
                dem_to_plot = np.where(self.valid_dem_mask, self.dem_data, np.nan)
                
                # Plot elevation data using pcolormesh
                im = ax.pcolormesh(lon_grid, lat_grid, dem_to_plot,
                                  transform=ccrs.PlateCarree(),
                                  cmap=self.cmap, norm=self.norm,
                                  alpha=0.85, zorder=2)
                
                self.print_success(f"Pakistan elevation data plotted on {region_name} map")
            
            # Add Pakistan boundary outline (highlighted)
            if self.boundary is not None and region_config['highlight_pakistan']:
                # Fill Pakistan with a semi-transparent color for highlighting
                self.boundary.plot(ax=ax, transform=ccrs.PlateCarree(),
                                 color='yellow', alpha=0.2, zorder=3)
                
                # Plot boundary with thick colored line
                self.boundary.boundary.plot(ax=ax, transform=ccrs.PlateCarree(),
                                           linewidth=3.0, color='red', zorder=4)
                
                # UPDATED: Only add Pakistan label if explicitly configured
                # The title already indicates it's about Pakistan, so no need for label
                if region_config.get('show_pakistan_label', False):
                    # Calculate centroid for label placement
                    centroid = self.boundary.geometry.unary_union.centroid
                    ax.text(centroid.x, centroid.y, 'PAKISTAN',
                           transform=ccrs.PlateCarree(),
                           fontsize=14, fontweight='bold',
                           color='red', ha='center', va='center',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                           zorder=5)
                    self.print_info("Pakistan label added to map")
            
            # Add gridlines with labels
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                             linewidth=0.5, color='gray', alpha=0.5,
                             linestyle='--', zorder=6)
            gl.top_labels = gl.right_labels = False
            gl.xlabel_style = {'size': self.config['grid_fontsize']}
            gl.ylabel_style = {'size': self.config['grid_fontsize']}
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            
            # Add main title
            ax.set_title(region_config['title'],
                        fontsize=self.config['title_fontsize'],
                        fontweight='bold', pad=25)
            
            # Add enhanced legend with position based on region configuration
            # UPDATED: South Asia map legend at bottom, others at top
            legend_position = region_config.get('legend_position', 'top')
            self.add_enhanced_legend(ax, position=legend_position)
            
            self.print_success(f"{region_name.upper()} regional map created successfully")
            return fig, ax
        except Exception as e:
            self.print_error(f"Failed to create {region_name} map: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def add_enhanced_legend(self, ax, position='top'):
        """
        Add enhanced elevation legend to the map at specified position.
        Uses a SINGLE COLUMN layout for clear readability.
        
        Parameters:
        ax: matplotlib axis to add legend to
        position: 'top' or 'bottom' - position of legend on map
        
        Educational Note:
        Legends are crucial for map interpretation.
        Position can be adjusted based on map layout and available space.
        """
        try:
            # Create legend patches (colored rectangles with labels)
            patches = []
            for i, class_name in enumerate(self.elevation_class_names):
                # Extract base name for color lookup
                if '(' in class_name:
                    base_name = class_name.split(' (')[0]
                else:
                    base_name = class_name
                
                color = self.elevation_colors[base_name]
                # Create a patch (colored rectangle) for each legend item
                patch = mpatches.Patch(color=color, label=class_name, alpha=0.85)
                patches.append(patch)
            
            # Set position parameters based on requested position
            if position == 'bottom':
                # UPDATED: Bottom position for South Asia map
                bbox_x = 0.02  # 2% from left
                bbox_y = 0.02  # 2% from bottom (bottom-left corner)
                loc = 'lower left'
                self.print_info(f"Adding legend at BOTTOM position ({bbox_x}, {bbox_y})")
            else:
                # Default top position for other maps
                bbox_x = 0.02  # 2% from left
                bbox_y = 0.98  # 98% from bottom (top-left corner)
                loc = 'upper left'
                self.print_info(f"Adding legend at TOP position ({bbox_x}, {bbox_y})")
            
            # Create legend with specified position
            # bbox_to_anchor positions the legend box in axes coordinates
            legend = ax.legend(handles=patches, 
                             loc=loc,  # Anchor point
                             bbox_to_anchor=(bbox_x, bbox_y),  # Position in axes coordinates
                             fontsize=self.config['legend_fontsize'],
                             title='Elevation Classification\n(with Detailed Coastal Zones)',
                             title_fontsize=self.config['legend_title_fontsize'],
                             frameon=True,  # Draw frame around legend
                             framealpha=0.95,  # Slight transparency
                             edgecolor='black',  # Border color
                             ncol=1)  # SINGLE COLUMN
            
            # Style the legend box
            legend.get_frame().set_facecolor('#f9f9f9')  # Light gray background
            legend.get_frame().set_linewidth(1.5)  # Border thickness
            legend.get_frame().set_boxstyle('round,pad=0.5')  # Rounded corners
            
            self.print_success(f"Enhanced legend added in SINGLE COLUMN layout at {position} position")
        except Exception as e:
            self.print_error(f"Could not add enhanced legend: {e}")
    
    def create_all_maps(self):
        """
        STEP 8: Create all three maps (globe, Asia, South Asia).
        
        Returns:
        list: List of (figure, axis, map_type) tuples for all created maps
        """
        self.print_step(8, "Creating All Three Maps")
        
        all_maps = []
        
        # 1. Create globe map
        self.print_info("Creating Globe Map...")
        globe_fig, globe_ax = self.create_globe_map()
        if globe_fig is not None:
            all_maps.append((globe_fig, globe_ax, 'globe'))
        
        # 2. Create Asia regional map
        self.print_info("Creating Asia Regional Map...")
        asia_fig, asia_ax = self.create_regional_map('asia')
        if asia_fig is not None:
            all_maps.append((asia_fig, asia_ax, 'asia'))
        
        # 3. Create South Asia regional map
        self.print_info("Creating South Asia Regional Map...")
        south_asia_fig, south_asia_ax = self.create_regional_map('south_asia')
        if south_asia_fig is not None:
            all_maps.append((south_asia_fig, south_asia_ax, 'south_asia'))
        
        if len(all_maps) == 3:
            self.print_success("All three maps created successfully")
        else:
            self.print_warning(f"Only {len(all_maps)} out of 3 maps were created")
        
        return all_maps
    
    def save_outputs(self, all_maps):
        """
        STEP 9: Save all generated maps to multiple file formats.
        
        Parameters:
        all_maps: List of (figure, axis, map_type) tuples
        
        Returns:
        dict: Dictionary mapping map types to lists of saved file paths
        
        Educational Note:
        Saving maps in multiple formats ensures compatibility with different uses:
        - PNG for web and presentations
        - PDF for printing and publications
        - High-res PNG for detailed analysis
        - Transparent PNG for overlays
        """
        self.print_step(9, "Saving All Output Files")
        
        saved_files = {}
        
        try:
            # Create timestamp for unique filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for fig, ax, map_type in all_maps:
                map_type_display = map_type.replace('_', ' ').title()
                base_filename = f"Pakistan_{map_type_display.replace(' ', '_')}_Map_{timestamp}"
                
                # Define output paths for different formats
                png_path = os.path.join(self.output_dir, f"{base_filename}.png")
                pdf_path = os.path.join(self.output_dir, f"{base_filename}.pdf")
                highres_path = os.path.join(self.output_dir, f"{base_filename}_HighRes.png")
                transparent_path = os.path.join(self.output_dir, f"{base_filename}_Transparent.png")
                
                # Save as PNG (standard resolution)
                fig.savefig(png_path, dpi=self.config['dpi'], bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                self.print_success(f"{map_type_display} PNG saved: {os.path.basename(png_path)}")
                
                # Save as PDF (vector format, scalable)
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                self.print_success(f"{map_type_display} PDF saved: {os.path.basename(pdf_path)}")
                
                # Save as high-resolution PNG for printing/publication
                fig.savefig(highres_path, dpi=600, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                self.print_success(f"{map_type_display} High-res PNG saved: {os.path.basename(highres_path)}")
                
                # Also save a version with transparent background
                fig.savefig(transparent_path, dpi=self.config['dpi'], bbox_inches='tight',
                           facecolor='none', edgecolor='none', transparent=True)
                self.print_success(f"{map_type_display} Transparent PNG saved: {os.path.basename(transparent_path)}")
                
                # Store saved files
                saved_files[map_type] = [png_path, pdf_path, highres_path, transparent_path]
            
            return saved_files
        except Exception as e:
            self.print_error(f"Failed to save files: {e}")
            return {}
    
    def print_summary(self):
        """
        Print comprehensive summary of the processing.
        
        Educational Note:
        Summarizing the process helps users understand what was done.
        It also provides valuable information for documentation and reproducibility.
        """
        self.print_step(10, "Generating Summary Report")
        print("\n" + "="*70)
        print("📊 PROCESS SUMMARY")
        print("="*70)
        print(f"📍 Script Directory: {SCRIPT_DIR}")
        print(f"🗺️  Shapefile: {os.path.basename(self.shapefile_path)}")
        print(f"🏔️  DEM File: {os.path.basename(self.dem_path)}")
        print(f"💾 Output Directory: {self.output_dir}")
        
        if self.elevation_stats:
            print(f"\n📈 ELEVATION STATISTICS:")
            print(f"   Minimum: {self.elevation_stats.get('min', 'N/A'):.1f} m")
            print(f"   Maximum: {self.elevation_stats.get('max', 'N/A'):.1f} m")
            print(f"   Mean: {self.elevation_stats.get('mean', 'N/A'):.1f} m")
            print(f"   Standard Deviation: {self.elevation_stats.get('std', 'N/A'):.1f} m")
            print(f"   Valid Pixels: {self.elevation_stats.get('valid_pixels', 0):,}")
            
            if 'coastal_pixels' in self.elevation_stats:
                print(f"   Coastal Zone Pixels (0-300m): {self.elevation_stats['coastal_pixels']:,}")
                print(f"   Coastal Zone Percentage: {self.elevation_stats.get('coastal_percentage', 0):.1f}%")
        
        print(f"\n🗺️  MAP TYPES GENERATED:")
        print(f"   1. Globe Map (Orthographic Projection)")
        print(f"   2. Asia Regional Map (Robinson Projection)")
        print(f"   3. South Asia Regional Map (Robinson Projection)")
        
        print(f"\n🎨 VISUALIZATION SETTINGS:")
        print(f"   Figure Size: {self.config['figsize'][0]} × {self.config['figsize'][1]} inches")
        print(f"   Resolution: {self.config['dpi']} DPI")
        print(f"   Legend Layout: SINGLE COLUMN")
        print(f"   Asia Map Legend Position: Top")
        print(f"   South Asia Map Legend Position: Bottom")  # UPDATED
        
        print(f"\n📋 ELEVATION CLASSIFICATION:")
        print(f"   Total Classes: {len(self.elevation_class_names)}")
        print(f"   Coastal Zone Sub-Classes (0-300m): 7")
        print(f"   Highest Class: {self.elevation_class_names[-1]}")
        
        print(f"\n🔧 TECHNICAL FEATURES:")
        print(f"   • Automatic coordinate alignment")
        print(f"   • Coordinate inversion correction")
        print(f"   • Enhanced coastal zone classification")
        print(f"   • No-data value handling")
        print(f"   • Multi-format output (PNG, PDF, High-res, Transparent)")
        print(f"   • SINGLE COLUMN LEGEND for better readability")
        print(f"   • Three different map projections for regional context")
        print(f"   • No Pakistan labels on regional maps (cleaner appearance)")
        print(f"   • South Asia map legend at bottom for better layout")
        
        print("="*70)
    
    def run(self):
        """
        Main method that runs the complete 10-step pipeline.
        
        Returns:
        bool: True if successful, False otherwise
        
        Educational Note:
        This is the main workflow of the GIS processing pipeline.
        Each step builds upon the previous one, creating a logical flow.
        """
        try:
            # Step 1: Validate inputs
            if not self.validate_inputs():
                return False
            
            # Step 2: Check coordinate systems
            if not self.check_coordinate_systems():
                return False
            
            # Step 3 & 4: Load and align data
            if not self.load_and_align_data():
                return False
            
            # Step 5: Classify elevation with enhanced coastal zones
            if not self.classify_elevation():
                return False
            
            # Step 6-8: Create all three maps
            self.print_info("Creating all three map visualizations...")
            all_maps = self.create_all_maps()
            
            if not all_maps:
                self.print_error("No maps were created")
                return False
            
            # Step 9: Save outputs
            saved_files = self.save_outputs(all_maps)
            
            # Step 10: Print summary
            self.print_summary()
            
            # Final success message
            print("\n🎉 MAP GENERATION SUCCESSFUL!")
            print("="*70)
            print("Generated files:")
            
            for map_type, files in saved_files.items():
                map_type_display = map_type.replace('_', ' ').title()
                print(f"\n  📍 {map_type_display}:")
                for file_path in files:
                    print(f"     • {os.path.basename(file_path)}")
            
            print(f"\n📁 Output location: {self.output_dir}")
            print("\nThe maps will display shortly. Close each window to continue.")
            print("="*70)
            
            # Display all maps
            for fig, ax, map_type in all_maps:
                plt.figure(fig.number)
                plt.tight_layout()
            
            plt.show()
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Process interrupted by user.")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

# ============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    """
    Main execution block - entry point of the script.
    This is where the script starts executing when run directly.
    
    Educational Note:
    The __name__ == "__main__" check allows the script to be imported
    as a module or run as a standalone program.
    """
    # Display ASCII art for visual appeal
    print("""
╔══════════════════════════════════════════════════════════╗
║      PAKISTAN MULTI-MAP GENERATOR                       ║
║   Enhanced Coastal Zone Classification                  ║
║           SINGLE COLUMN LEGEND                          ║
║       Three Maps: Globe, Asia, South Asia               ║
║          UPDATED: Cleaner Regional Maps                 ║
║          South Asia Legend at Bottom                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # Verify script location
    current_dir = os.getcwd()
    if current_dir != SCRIPT_DIR:
        print(f"⚠️  Note: Running from: {current_dir}")
        print(f"   Expected: {SCRIPT_DIR}")
        print("   The script will attempt to run with current directory...\n")
    
    # Create and run the generator
    print("Starting multi-map generation process...")
    print("="*70)
    
    generator = PakistanMultiMapGenerator()
    success = generator.run()
    
    # Exit with appropriate status code
    # 0 = success, 1 = error (standard Unix convention)
    sys.exit(0 if success else 1)
