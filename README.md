# Glacial Lake Outburst Flood (GLOF) Risk Prediction Pipeline

This repository accompanies a study presented at the 2025 Midstates Research Symposium examining **Glacial Lake Outburst Flood (GLOF)** risks in the Himalayas using satellite remote sensing and machine‑learning.  The goal is to forecast which glacial lakes are most susceptible to outburst flooding.  Below is a high‑level explanation of the pipeline, major datasets and notebook structure, and the key functions used throughout the project.

## Project objectives

- **Identify potentially dangerous glacial lakes** in the Himalayas using Landsat satellite imagery, digital elevation models (DEMs) and glacier inventories.
- **Extract lake‑centric features** such as area, elevation, expansion rate, glacier contact and slope.
- **Impute missing data** using Multiple Imputation by Chained Equations (MICE).
- **Train machine‑learning models** (logistic regression, random forest, gradient boosting and SVM) to classify whether a lake has previously experienced a GLOF.
- **Develop an interactive hazard map** using `folium` and `streamlit` that highlights lakes with elevated predicted risk.

## Repository structure

| Directory/File                   | Description |
|----------------------------------|-------------|
| **`Notebooks/`**                | Jupyter notebooks containing data preprocessing, feature engineering, exploratory analysis, model training and deployment.  The main notebooks are `Final_LakeFeatures.ipynb`, `ML_implementation.ipynb` (with a lighter copy in `Experimental_notebooks/ML_implementation-Copy1.ipynb`), and `Model_deployment.ipynb` which demonstrates how to build the Streamlit app. |
| **`CSVs/`**                     | Cleaned and engineered datasets used for modelling.  Examples include `df_final_with_laketypesorted_pre.csv` (final features and labels), `ml_pos_y_drop.csv`/`ml_neg_y_drop.csv` (positive/negative samples with the response removed for imputation), and `IHR_GlacialLake_Atlas table_cleaned.csv`. |
| **`ExcelSheets/`**              | Original Excel versions of the International Himalaya Region glacial lake database and other supporting data. |
| **`model.pkl`**                 | Serialised logistic regression model used in the Streamlit app. |
| **`Midstates Research Symposium Poster.png`** | The poster summarising the project methodology and results. |
| **`requirements.txt`**          | List of Python packages and versions required to run the notebooks and Streamlit app (e.g., `scikit‑learn 1.6.1`, `pandas 2.2.3`, `miceforest 6.0.3`, `folium 0.20.0`, `streamlit 1.45.1`):contentReference[oaicite:0]{index=0}. |

## Data sources

The project integrates multiple data products:

1. **Landsat 7 and 8 Surface Reflectance**: raw spectral bands used to delineate water bodies and compute the **Normalized Difference Water Index (NDWI)**.  Gaps in the Landsat archive (notably due to the SLC failure from ~2003–2009) create missing data issues:contentReference[oaicite:1]{index=1}.
2. **SRTM and other DEMs**: digital terrain models used to calculate lake elevation and slope.
3. **ICIMOD Glacier Inventory**: outlines glaciers near each lake and provides glacier elevation information.
4. **IHR Glacial Lake Atlas**: database of ~5,000 Himalayan glacial lakes used to create the final positive (historical GLOF) and negative training sets.  `df_final_with_laketypesorted_pre.csv` merges attributes such as lake area, expansion rates and lake type for each sample.
5. **GLOF event catalogue**: historical records of known GLOF events compiled from literature and the IHR atlas; positive labels in the training data correspond to lakes with recorded GLOFs:contentReference[oaicite:2]{index=2}.

## Pipeline overview

### 1. Data preprocessing

Data preprocessing is performed in early notebooks (e.g. `Data Cleaning.ipynb`):

- **Cloud, shadow and snow masking** – contaminated pixels are removed using the Landsat QA band and simple threshold rules.
- **Surface reflectance scaling** – digital numbers are scaled to top‑of‑atmosphere reflectance.
- **NDWI calculation** – NDWI is computed using green and near‑infrared bands: `NDWI = (X_green – X_nir) / (X_green + X_nir)`:contentReference[oaicite:3]{index=3}.  A threshold (NDWI ≥ 0.3) is then applied to delineate water pixels (lake surfaces).
- **Temporal compositing** – multiple dates are aggregated to reduce noise: for non‑GLOF lakes, the median of Year−1, Year and Year+1 is taken; for pre‑GLOF lakes, the median of Year−3, Year−2 and Year−1 is used.  This minimises clouds and seasonal variation.
- **Vectorisation** – lake rasters are converted to polygons and intersected with glacier polygons to derive shape metrics and perform spatial joins.  Attributes such as area (in hectares) and centroid coordinates are computed.


### 2. Feature engineering

Comprehensive feature extraction is performed in `Feature Engineering.ipynb` and `Final_LakeFeatures.ipynb`. Key engineered features include:

| Feature | Description |
|--------|-------------|
| **`Longitude, Latitude`** | Coordinates of the lake centroid. |
| **`Year_final`** | Reference year used for expansion rate calculations. |
| **`Lake_area_calculated_ha`** | Lake area (ha) derived from the vectorised mask. |
| **`Elevation_m`** | Lake elevation from DEMs. |
| **`glacier_area_ha`** | Area of the nearest glacier (ha). |
| **`slope_glac_to_lake`** | Average slope between the lake and the adjacent glacier. |
| **`glacier_touch_count`** | Number of glacier polygons touching the lake. |
| **`nearest_glacier_dist_m`** | Horizontal distance from lake polygon to the nearest glacier. |
| **`glacier_elev_m`** | Elevation of the neighbouring glacier. |
| **`5y_expansion_rate`, `10y_expansion_rate`** | Percentage change in lake area over 5‑ and 10‑year windows. |
| **`Lake_type_simplified`** | Categorical label (ice‑contact, moraine‑dammed, or other); one‑hot encoded using `pd.get_dummies()`:contentReference[oaicite:4]{index=4}. |
| **`is_supraglacial`** | Boolean flag indicating if the lake sits on a glacier. |
| **`glacier_contact`** | Binary indicator showing whether the lake directly touches a glacier. |
| **`GLOF`** | Target label: 1 if the lake has a recorded GLOF, 0 otherwise. |

Feature engineering functions in the notebooks compute distances using geodesic formulas, count glacier contacts, and calculate expansion rates:

```python
def compute_expansion_rate(area_t0, area_t1, years):
    """Calculate percentage area change per year over a given period."""
    return ((area_t1 - area_t0) / area_t0) / years * 100

def count_glacier_contacts(lake_poly, glacier_polys):
    """Count how many glacier polygons intersect the lake polygon."""
    return sum(lake_poly.intersects(g) for g in glacier_polys)




