
# Noninvasive megapixel fluorescence microscopy through scattering layers by a virtual reflection-matrix

This repository contains the implementation of the I-CLASS algorithm for incoherent data, such as fluorescence, as described in "Noninvasive megapixel fluorescence microscopy through scattering layers by a virtual reflection-matrix" ([DOI: 10.1126/sciadv.adl5218](https://doi.org/10.1126/sciadv.adl5218)). The algorithm is applied to example measurement data, which can be downloaded from the provided link. The repository includes improved memory efficiency implementation of the CLASS algorithm as described in the paper.


![](fig2.png)
## Content of the Repository

- `ICLASS.py`: The implementation of the I-CLASS algorithm with improved memory efficiency.
- `visualize.py`: A script for visualizing the results obtained from the algorithm.
- `main.py`: A script to run the whole pipeline, from loading the measurement files to results and visualization.
- `requirements.txt`: Lists all the Python dependencies required to run the code.

## Getting Started


### 1. Install Requirements

```sh
pip install -r requirements.txt
```

### 2. Download the Data

Download the example measurement data from [this link](https://drive.google.com/drive/folders/18A_W_JemctYMtloonCAh3RTWsZfOJyJN?usp=sharing) and place it in a folder of your choice.

### 3. Run the Code

1. Open `main.py` and insert the path to the data folder (`DATA_PATH`) and specify the ground truth and measurements indices (`ground_truth_idx` and `meas_idx`).
2. Run `main.py`:

```sh
python main.py
```

If you have already run the above steps, you can visualize the results by running the `showResults` function from `visualize.py`:

```python
from visualize import showResults

data_path = "path_to_your_data"
meas_idx = index of measurement
showResults(data_path, meas_idx)
```



## Acknowledgements

- If you use this code or the associated paper, please cite:

```
@article{
doi:10.1126/sciadv.adl5218,
author = {Gil Weinberg  and Elad Sunray  and Ori Katz },
title = {Noninvasive megapixel fluorescence microscopy through scattering layers by a virtual incoherent reflection matrix},
journal = {Science Advances},
volume = {10},
number = {47},
pages = {eadl5218},
year = {2024},
doi = {10.1126/sciadv.adl5218},
URL = {https://www.science.org/doi/abs/10.1126/sciadv.adl5218}}
