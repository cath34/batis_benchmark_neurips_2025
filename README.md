# BATIS: Benchmarking Bayesian Approaches for Improving Species Distribution Models

This repository contains the necessary code to reproduce the experiments shown in **BATIS: Benchmarking Bayesian Approaches for Improving Species Distribution Models**. To download the associated dataset, please see our [Hugging Face repository](https://huggingface.co/datasets/cathv/batis_benchmark_2025). 

**Authors :** Catherine Villeneuve, Melisande Teng, Benjamin Akera and David Rolnick.

## ⚠️ !!! ERRATUM IN THE MAIN PAPER !!! ⚠️

We would like to apologize to the reviewers for a typo in Table 1 of the main paper. The table incorrectly suggests that hundreds of thousands of species can be observed in the United States during summer, and nearly 50,000 in winter. While many birders would surely dream of such an extraordinary high avian biodiversity, these numbers are clearly far from the reality. The values intended for the `number_of_hotspots` column were unfortunately placed in the `number_of_species` column. The first table of the Appendix reports the appropriate numbers, but we also include it here to avoid any confusion : 

| **Region**           | **Date Range**              | **Number of Checklists** | **Number of Hotspots** | **Number of Species** | **Species List** |
|----------------------|-----------------------------|---------------------------|-------------------------|------------------------|------------------|
| Kenya (KE)           | 2010-01-01 to 2023-12-31    | 44,852                    | 8,551                   | 1,054                  | Avibase          |
| South Africa (ZA)    | 2018-01-01 to 2024-06-17    | 498,867                   | 6,643                   | 755                    | BirdLife         |
| USA-Winter (US-W)    | 2022-12-01 to 2023-01-31    | 3,673,742                 | 45,882                  | 670                    | ABA 1-2          |
| USA-Summer (US-S)    | 2022-06-01 to 2022-07-31    | 3,920,846                 | 98,443                  | 670                    | ABA 1-2          |

## Installation

Please run the following code to install the requirements

```
conda env create -f requirements.txt
```

## eBird Data

The code to process eBird data can be found in the `ebird_data/` folder :
* **Step 1 :** Using raw sampling and metadata files from the ebird Database, use the R Script `ebird_data/data_eBird.R` to extract only the observations associated with complete checklists. This R script leverages [auk](https://cornelllabofornithology.github.io/auk/), an R package specifically designed for eBird Data Extraction and Processing. 
* **Step 2 :** Once the observations associated with complete checklists are extracted, auk will output a single file containing multiple thousands of sightings. You can use the script `ebird_data/extract_checklist.py` to extract each checklist into a single CSV file with the following columns :
  * `ebird_code` : Code corresponding to a single species in the eBird database
  * `is_observed` : 1 if the species was observed in a given checklist, and 0 otherwise
Note that the script `ebird_data/extract_checklist.py` requires a species list input file, because it will discard any observation of species that aren't among that list. 

## Data Splits

Code for splitting data can be found in the `Splits/` folder. In order to get train/test/splits according to the methodology described in our paper, you need to run the script `combined_script.py` . 

## Input Variables

The code to generate bounding boxes centered around given (lat, lon) coordinates is available in `worldclim/create_squares.py` . Once these bounding boxes are generated, you can : 
* **Extract Sentinel-2 Rasters :** By using the Colab notebook `Splits/Sentinel2_EE.ipynb`
* **Extract WorldClim Rasters :** By using the script `worldclim/extract_env_rasters.py`. Once these rasters are extracted, you can fill the NaN values through bilinear interpolation by running `worldclim/fill_env_nans.py`, then `worldclim/filter_env.py`.

For baselines that only require an input vector instead of a matrix (i.e. Multi-Layer Perceptron and Random Forest), you can simply run `worldclim/get_env_vector.py` to extract the input variables. 

## Models

Models implementations can be found in the `Models/` folder. Examples of all config files for different baselines and subdatasets are available in `configs`. The training scripts also contain the code to test the model and save the individual predictions for each hotspot to a pre-determined folder. 

## Bayesian Updating Framework 

The code for our Bayesian Updating Framework can be found in the `Bayesian_Updates/` folder. Use : 
* `Bayesian_Updates/Updating_Scripts/bayesian_exploration_fv.py` : For our Fixed-Variance approach
* `Bayesian_Updates/Updating_Scripts/bayesian_updates_mean_var.py` : For any other approach

## Metrics

Our metrics can be computed through the following scripts : 
* `Bayesian_Updates/Evaluation_Scripts/evaluate_metrics_hotspots.py` : To evaluate our metrics for each hotspot, and for each bird, on a given number of bayesian updates.
* `Bayesian_Updates/Updating_Scripts/evaluate_metrics_wholedataset_za.py` : To evaluate the average performance of any approach across a whole dataset

## License 

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).


