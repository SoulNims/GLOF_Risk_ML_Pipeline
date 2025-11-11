# Glacial Lake Outburst Flood (GLOF) Risk Prediction Pipeline

This repository accompanies a study presented at the 2025 Midstates Research Symposium examining **Glacial Lake Outburst Flood (GLOF)** risks in the Himalayas using satellite remote sensing and machine‑learning.  The goal is to forecast which glacial lakes are most susceptible to outburst flooding.  Below is a detailed description of the pipeline, including the pre‑ML geospatial processing routines from the `Final_LakeFeatures.ipynb` notebook, and key functions used throughout the project.

## Project objectives

- **Identify potentially dangerous glacial lakes** in the Himalayas using Landsat imagery, SRTM DEMs and the GLIMS glacier inventory.
- **Extract lake‑centric features** such as lake area, elevation, expansion rate, glacier contact, distance to glaciers and slope.
- **Impute missing data** using Multiple Imputation by Chained Equations (MICE) to deal with gaps in remote‑sensing archives.
- **Train machine‑learning models** to classify whether a lake has previously experienced a GLOF.
- **Develop an interactive hazard map** that allows users to visualise predicted risk for individual lakes.

## Repository structure

| Directory/File | Description |
|---------------|-------------|
| **`Notebooks/`** | Jupyter notebooks for data preprocessing, feature engineering, exploratory analysis, model training and deployment.  Key notebooks include `Final_LakeFeatures.ipynb` (detailed geospatial processing), `ML_implementation.ipynb` (model training), and `Model_deployment.ipynb` (building a Streamlit app). |
| **`CSVs/`** | Cleaned and engineered datasets used for modelling (e.g., `df_final_with_laketypesorted_pre.csv`, `ml_pos_y_drop.csv`). |
| **`model.pkl`** | Serialised logistic regression model trained on the engineered features. |
| **`requirements.txt`** | List of Python packages needed to run the notebooks (e.g., scikit‑learn, pandas, miceforest, folium, streamlit):contentReference[oaicite:0]{index=0}. |
| **`Midstates Research Symposium Poster.png`** | Poster summarising the project methodology and results. |

## Data sources

The pipeline uses several geospatial data products:

1. **Landsat 5/7/8/9 Surface Reflectance**: used to compute the **Normalized Difference Water Index (NDWI)** for lake detection.  Gaps in Landsat 7 caused by the SLC failure (~2003–2009) introduce missing data:contentReference[oaicite:1]{index=1}.
2. **SRTM (USGS/SRTMGL1_003) DEM**: provides elevation data for lakes and glaciers.
3. **GLIMS Glacier Inventory** (`GLIMS/current`): glacier polygons used to determine glacier contact, distance, elevation difference and slope metrics.
4. **IHR Glacial Lake Atlas**: database of ~5,000 glacial lakes across the Himalayas.  Merged with GLOF event catalogues to create positive (historical GLOF) and negative samples.
5. **GLOF event records**: compiled from literature and the IHR atlas; positive labels correspond to lakes with recorded GLOFs:contentReference[oaicite:2]{index=2}.

## Geospatial preprocessing and feature extraction

### Lake detection via Google Earth Engine

The `Final_LakeFeatures.ipynb` notebook implements Earth Engine (EE) functions that detect lake polygons around specified coordinates and years.  These functions operate entirely on the EE server, maximising efficiency when applied to thousands of lakes.

- **`detect_lake_from_point(point, year, search_radius=3000, ndwi_thresh=0.3)`**: For each non‑GLOF lake, this function buffers a point (`ee.Geometry.Point`) by a search radius (default 3 km) to define an area of interest, then:
  1. Retrieves a fall‑season Landsat image collection for the year ± 1 year, selecting Landsat 5, 7, 8 or 9 depending on the date.
  2. Applies a cloud/shadow/snow mask (`mask_landsat_clouds`), scales surface reflectance bands (`scale_sr`) and adds NDWI (`add_ndwi`).  NDWI is computed as `(green − nir)/(green + nir)`:contentReference[oaicite:3]{index=3}.
  3. Computes a median composite, thresholds NDWI at `ndwi_thresh` to identify water, and converts the water mask to polygons using `reduceToVectors`.
  4. Attaches an attribute `Lake_area_calculated_ha` (polygon area in hectares) and returns the largest polygon; if no polygons are found, returns the original point with area `None`.

- **`detect_lake_pre_glof(point, year, search_radius=3000, ndwi_thresh=0.3)`**: Similar to `detect_lake_from_point` but used for positive lakes.  It composes Landsat images over the years `year-3` to `year-1` to capture the lake before the GLOF event, then performs NDWI thresholding and vectorisation.

- **`df_to_fc(df)`**: Converts a pandas DataFrame of lakes (with longitude, latitude and other properties) into an EE `FeatureCollection` of points.

- **`detect_area_nonglof(feature)` / `detect_area_glof(feature)`**: Wrapper functions applied to each point in a `FeatureCollection`.  They call `detect_lake_from_point` or `detect_lake_pre_glof`, copy original properties and attach `Lake_area_calculated_ha`.

### Expansion rates

Expansion rates measure how quickly lake area changes, expressed as hectares per year.  Functions defined in the notebook compute 5‑year and 10‑year expansion rates for GLOF and non‑GLOF lakes:

- **`calc_expansion_glof(feature)`** (5 y and 10 y versions): For a positive lake, detect lake polygons at `year` and at `year − 5` (or `year − 10`) using `detect_lake_pre_glof`, retrieve their areas, and compute `(area_t2 − area_t1) / years`.  The function returns a new feature containing the point coordinates, areas and `expansion_ha_peryr`.

- **`calc_expansion_nonglof(feature)`** (5 y and 10 y versions): Same as above but uses `detect_lake_from_point` to compute expansion rates for negative lakes.

### Glacier metrics

Glacier proximity and elevation are key predictors of GLOF risk.  The notebook defines server‑side functions to extract these metrics:

- **`get_nearest_glacier(lake_geom, buffer_km=50)`**: Given a lake polygon, identifies glaciers within a search radius (default 50 km) from the GLIMS inventory.  If glaciers intersect the lake, it sums their area and sets `nearest_glacier_dist_m=0`; otherwise, it finds the nearest glacier by distance.  It returns a dictionary containing:
  - `glacier_contact` (boolean),
  - `nearest_glacier_dist_m` (distance in metres),
  - `lake_elev_m` and `glacier_elev_m` (elevations from SRTM),
  - `slope_glac_to_lake` (rise/run slope between the lake and glacier),
  - `glacier_area_km2` (area in square kilometres), and
  - `glacier_ids` (list of GLIMS IDs).

- **`_glacier_metrics_for_lake_polygon(lake, buffer_km=50)`**: A more robust version used to annotate many lakes.  It computes the nearest glacier, counts the number of touching glaciers (`glacier_touch_count`), sums their areas (`glacier_area_ha`), and calculates slope using the DEM.  If no glaciers are found, it returns null‑like fields.

- **`annotate_glacier_metrics_from_polygons(lakes_fc, buffer_km=50)`**: Applies `_glacier_metrics_for_lake_polygon` to each feature in a `FeatureCollection` of lake polygons, returning a new collection with glacier metrics attached.

These functions enable automated extraction of complex spatial relationships for each lake, eliminating the need for manual GIS processing.

## Constructed feature set

After running the EE functions, the resulting datasets (e.g., `df_final_with_laketypesorted_pre.csv`) include the following engineered features:

| Feature | Description |
|--------|-------------|
| `Longitude`, `Latitude` | Coordinates of the lake centroid. |
| `Year_final` | Reference year for each record (typically the most recent observation for negative lakes or the GLOF event year for positive lakes). |
| `Lake_area_calculated_ha` | Lake area computed from NDWI‑derived polygons. |
| `Elevation_m` / `lake_elev_m` | Elevation of the lake (from SRTM). |
| `glacier_area_ha` / `glacier_area_km2` | Combined area of touching glaciers (ha / km²). |
| `slope_glac_to_lake` | Rise/run slope between the lake and the nearest glacier. |
| `glacier_touch_count` | Number of glacier polygons intersecting the lake. |
| `nearest_glacier_dist_m` | Distance to the closest glacier if not touching (metres). |
| `glacier_elev_m` | Elevation of the nearest glacier (SRTM). |
| `5y_expansion_rate`, `10y_expansion_rate` | Expansion rate in hectares per year over 5 y and 10 y periods. |
| `Lake_type` / `Lake_type_simplified` | Categorical labels describing lake dam type (moraine, ice, other).  One‑hot encoding is applied using `pandas.get_dummies()`:contentReference[oaicite:16]{index=16}. |
| `is_supraglacial` | Boolean flag for lakes on glacier surfaces. |
| `glacier_contact` | Binary indicator: `True` if the lake touches a glacier, `False` otherwise. |
| `GLOF` | Target label: 1 if a historical GLOF occurred, 0 otherwise. |

## Handling missing data

Due to Landsat gaps and variable DEM coverage, many features contain `NaN`.  The machine‑learning notebook (`ML_implementation.ipynb`) uses **MICE** from the `miceforest` library to impute missing values.

## Model training and evaluation

The model training follows these steps:

1. **Prepare the dataset**: One‑hot encode categorical variables (`Lake_type_simplified`, `is_supraglacial`, `glacier_contact`) and split the imputed dataset into training and validation sets (80/20).  Stratified 5‑fold cross‑validation ensures balanced evaluation.
2. **Baseline models**: Train logistic regression (LR), random forest (RF), gradient boosting (GB) and support vector machine (SVM).  LR exhibits the highest recall (~0.92) but lower precision (~0.50):contentReference[oaicite:17]{index=17}.
3. **Tune the logistic regression**: Adjust class weights to prioritise recall.  The tuned LR achieves recall ≈ 0.92 and F1 ≈ 0.76 on the validation set.
4. **Save the model**: The final LR model is serialised to `model.pkl` and used in the Streamlit app.

## Interactive hazard map

`Model_deployment.ipynb` demonstrates how to build an interactive hazard map:

- Load `model.pkl` and compute predicted probabilities for each lake.
- Use `folium.Map` to create a map centred on the Himalayas.  Add circle markers where colour reflects predicted GLOF risk (e.g. green for low risk, yellow for moderate, red for high risk).
- Build a simple `streamlit` app (`app.py`) that displays the map and allows users to filter lakes or view details via tooltips.

## Results and limitations

- **Model performance**: The tuned logistic regression provides a balanced trade‑off between recall and precision (recall ≈ 0.92, F1 ≈ 0.76) and was chosen for deployment:contentReference[oaicite:18]{index=18}.
- **Key predictors**: Glacier contact count, distance to the nearest glacier, glacier elevation, lake expansion rates and slope are among the most informative features.  Rapidly expanding lakes that touch or closely approach glaciers are predicted to be higher risk.
- **Limitations**: Missing satellite data (e.g., Landsat 7 SLC failure) and NDWI misclassification can lead to under‑detected small lakes and label uncertainty.  Historical GLOF records are incomplete, DEM resolution limits slope estimation in rugged terrain, and the annual temporal resolution may miss short‑term lake changes:contentReference[oaicite:19]{index=19}.

## Getting started

1. **Install dependencies**: Create an environment and run `pip install -r requirements.txt`.
2. **Run the geospatial notebook**: To reproduce the engineered features, execute `Final_LakeFeatures.ipynb` in a Jupyter environment connected to Earth Engine.  You will need to authenticate with Google Earth Engine (`ee.Authenticate()` and `ee.Initialize()`) and may need to adjust `project` IDs for access.
3. **Train the model**: Run `ML_implementation.ipynb` to impute missing data, perform one‑hot encoding, train and evaluate models, and save the final logistic regression model.
4. **Launch the app**: Use `streamlit run app.py` (after creating `app.py` based on `Model_deployment.ipynb`) to view the interactive hazard map.

## Acknowledgements

This research was supported by the **Richter Memorial Fund at Knox College**. We thank **Dr. Andrew Leahy** for mentorship and the developers of `earthengine-api`, `geemap`, `miceforest`, `scikit-learn`, `folium` and `streamlit`.  Please refer to the attached poster for a visual overview of the methods and results.
