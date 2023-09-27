
# CTR-CLASS for Incoherent Data

This repository contains the implementation of the CTR-CLASS algorithm for incoherent data, such as fluorescence, as described in [title of your paper]. The algorithm is applied to example measurement data, which can be downloaded from the provided link. The repository includes improved memory efficiency enhancements as described in the paper.

## Content of the Repository

- `CTRCLASS.py`: The implementation of the CTR-CLASS algorithm with improved memory efficiency.
- `visualize.py`: A script for visualizing the results obtained from the algorithm.
- `main.py`: A script to run the whole pipeline, from loading the measurements files to results and visualization.
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
@article{your_paper,
  title={title_of_your_paper},
  author={author_names},
  journal={journal_name},
  year={year_of_publication},
}
```

```
CLASS Algorithm
@article{kang2017high,
  title={High-resolution adaptive optical imaging within thick scattering media using closed-loop accumulation of single scattering},
  author={Kang, Sungsam and Kang, Pilsung and Jeong, Seungwon and Kwon, Yongwoo and Yang, Taeseok D and Hong, Jin Hee and Kim, Moonseok and Song, Kyung--Deok and Park, Jin Hyoung and Lee, Jun Ho and others},
  journal={Nature communications},
  volume={8},
  number={1},
  pages={2157},
  year={2017},
  publisher={Nature Publishing Group UK London}
}
```

```
CTR-CLASS
@article{lee2022high,
  title={High-throughput volumetric adaptive optical imaging using compressed time-reversal matrix},
  author={Lee, Hojun and Yoon, Seokchan and Loohuis, Pascal and Hong, Jin Hee and Kang, Sungsam and Choi, Wonshik},
  journal={Light: Science \& Applications},
  volume={11},
  number={1},
  pages={16},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```