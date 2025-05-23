# BATIS: Benchmarking Bayesian Approaches for Improving Species Distribution Models

This repository contains the necessary code to reproduce the experiments shown in **BATIS: Benchmarking Bayesian Approaches for Improving Species Distribution Models**. 

**Authors :** Catherine Villeneuve, Melisande Teng, Benjamin Akera and David Rolnick.

## Installation

Please run the following code to install the requirements

```
conda env create -f requirements.txt
```

## Models

Models implementations can be found in the `Models/` folder.

## Data Splits

Code for splitting data can be found in the `Splits/` folder. In order to get train/test/splits according to the methodology described in our paper, you need to run the script `combined_script.py` . 

## Input Variables

The code to generate bounding boxes centered around given (lat, lon) coordinates is available in `worldclim/create_squares.py` . Once these bounding boxes are generated, you can : 
* **Extract Sentinel-2 Rasters :** By using the Colab notebook `Splits/Sentinel2_EE.ipynb`
* **Extract WorldClim Rasters :** By using the script `worldclim/extract_env_rasters.py`. Once these rasters are extracted, you can fill the NaN values through bilinear interpolation by running `worldclim/fill_env_nans.py`, then `worldclim/filter_env.py`.

For baselines that only require an input vector instead of a matrix (i.e. Multi-Layer Perceptron and Random Forest), you can simply run `worldclim/get_env_vector.py` to extract the input variables. 


