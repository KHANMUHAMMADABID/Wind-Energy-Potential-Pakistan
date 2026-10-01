# Wind Resources and Techno-Economic Potential over Pakistan

[![Status](https://img.shields.io/badge/status-accepted-success)]()
[![License: MIT](https://img.shields.io/badge/Code%20license-MIT-green.svg)](LICENSE)

This repository contains the code, derived datasets, metadata, diagnostic
outputs, result tables, and selected figures associated with the article:

> **Wind resources and techno-economic potential over Pakistan from
> observations and reanalysis**

Accepted for publication in *Modeling Earth Systems and Environment*,
Springer Nature.

## Overview

This study provides an integrated assessment of Pakistan's wind resources
using ground-based observations and multiple reanalysis products. The
workflow includes:

- observational-data screening and missing-data reconstruction;
- validation of ERA5, JRA-55, NCEP/NCAR, and ensemble wind-speed data;
- multi-metric reanalysis ranking using error, bias, agreement,
  correlation, variability, and Perkins PDF skill measures;
- HAC/Newey-West piecewise trend analysis and breakpoint-stability
  assessment;
- wind-speed extrapolation from 10 m to multiple hub heights;
- wind-power-density and wind-energy-density calculations;
- Ordinary Kriging spatial mapping and cross-validation diagnostics;
- turbine power-curve calculations;
- annual energy production, capacity factor, avoided CO2 emissions, and
  levelized-cost-of-electricity screening analysis; and
- province-level comparison of modeled wind-energy potential.

## Repository structure

```text
Dataset_observational_reanalysis/
    Data-processing materials and reanalysis extraction workflow.

Elevation_Map_Pakistan_Global_View/
    Elevation, topographic context, and associated map materials.

Figure_4_Multi_Reanalysis_Composite_Ranking/
    Multi-metric validation data, code, composite ranking, and figures.

Figure_10_Map_At_Various_Height/
    Wind speed, WPD, WED spatial mapping code, inputs, and outputs.

Figure_12_Spatial_Classification_WPD/
    WPD classification code, thresholds, and spatial maps.

Observational_Piecewise_Trend_Analysis/
    HAC-robust piecewise trend-analysis scripts, outputs, and diagnostics.

PDF_Skill_Score/
    Perkins PDF skill-score calculations and distributional comparison.

Performance_of_Designated_Wind_Turbine/
    Turbine performance, AEP, capacity factor, CO2 reduction, and LCoE
    calculations.

Pakistan_Shape_File_and_DEM_Datasets/
    Geospatial boundary and DEM inputs, subject to the stated source licenses.

Figures/
    Final main-text and supplementary figures.

docs/
    Data dictionary, workflow description, external-data access instructions,
    and reproducibility documentation.
```

## Data availability

This repository provides code, metadata, derived datasets, result tables,
diagnostic outputs, and selected figures necessary to reproduce the reported
analyses.

### Source data

The original meteorological station observations were obtained from the
Pakistan Meteorological Department (PMD). Raw PMD observations are not
redistributed in this repository where provider access or redistribution
conditions apply.

The reanalysis products were obtained from their official providers:

- ERA5: Copernicus Climate Change Service,
  https://climate.copernicus.eu/
- JRA-55: Japan Meteorological Agency,
  https://jra.kishou.go.jp/JRA-55/index_en.html
- NCEP/NCAR Reanalysis: NOAA/NCEI,
  https://www.ncei.noaa.gov/

Users should obtain third-party source data directly from the official
providers and comply with their applicable terms of use and attribution
requirements.

### Derived data

Derived datasets, model-evaluation outputs, trend-analysis results, kriging
diagnostics, and techno-economic results are provided in the relevant
analysis folders, subject to the license information stated in this
repository.

## Reproducibility

The analysis workflow is described in the README files within the relevant
folders. Before running any script:

1. Download required third-party source data from the official providers.
2. Review the input-file paths at the top of each script.
3. Use relative paths where possible.
4. Install the dependencies listed in `requirements.txt`.
5. Run the scripts in the order described in `docs/reproducibility_guide.md`.

## Software environment

The main analysis scripts were developed using Python and R.

Python dependencies are listed in `requirements.txt`.

R packages used in selected components include:

```text
dplyr
ggplot2
tidyr
fitdistrplus
Metrics
reshape2
mgr
```

## Citation

Please cite the associated article:

> Khan, M. A. et al. *Wind resources and techno-economic potential over
> Pakistan from observations and reanalysis*. Modeling Earth Systems and
> Environment. [Insert final DOI after publication.]

Please also cite the archived software and data release:

> Khan, M. A. et al. (2026). Wind resources and techno-economic potential
> over Pakistan: Code and derived data (Version 1.0.0). Zenodo.
> [Insert Zenodo DOI after release.]

## License

- Code is released under the MIT License.
- Author-generated derived data are released under CC BY 4.0 unless otherwise stated.
- Third-party datasets, boundary files, DEMs, and reference materials remain
  subject to their original licenses and access conditions.

## Contact

**Muhammad Abid Khan, Ph.D.**  
Institute of Advanced Academic Research and  
Center for Environmental Remote Sensing, Chiba University, Japan  
Email: khan.muhammad.sa@alumni.tsukuba.ac.jp  
ORCID: https://orcid.org/0000-0001-8387-1044
