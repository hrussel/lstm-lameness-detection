# Lameness detection in dairy cows using pose estimation and bidirectional LSTMs

This repository provides the code and data for the paper:

Russello, H., van der Tol, R., van Henten, E. J., & Kootstra, G. (2025). **Lameness detection in dairy cows using pose estimation and bidirectional LSTMs.** [arXiv preprint arXiv:2508.10643.](https://arxiv.org/abs/2508.10643)


## Dataset

The dataset consists of 272 keypoint trajectories extracted with [T-LEAP](https://doi.org/10.1016/j.compag.2021.106559).
The keypoint trajectories are located in the folder `data/videos_keypoints`.

Each keypoint trajectory has an associated lameness score, present in `data/videos_lameness_scores.csv`.

## Code

### Requirements

To run the code, install the following python packages in a virtual environment.

Create the virtual environment with virtual env. Make sure you run the commands from the root of the repo's directory (e.g., `/home/myuser/lstm-lameness-detection`)

```shell
# Create a virtual env in the folder .venv
python3 -m venv .venv

# Activate the venv
source .venv/bin/activate

# Install pip
python3 -m pip install --upgrade pip
```

Install the required packages:
* Automatically:

```shell
# Install the packages listed in requirements.txt
pip install -r ./requirements.txt
```

* OR Manually:

```shell
# Pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Other packages
pip install scikit-learn pandas PyYAML

# (Optional) Weights and Biases
pip install wandb
```

### Configuration

The configuration file `cfg/config.yml` contains the default parameters to run the code.
We recommend leaving the parameters to the default value when running the code for the first time.

### Training the model

The model training and evaluation are run from the `main.py` file.
Make sure the venv is activated before running the python script.

```shell
python main.py
```

## Citing

If you're using the code or data for your work, please consider citing the relevant papers associtated with this repo:

Code:
```text
@article{russello2025lameness,
  title={Lameness detection in dairy cows using pose estimation and bidirectional LSTMs},
  author={Russello, Helena and van der Tol, Rik and van Henten, Eldert J and Kootstra, Gert},
  journal={arXiv preprint arXiv:2508.10643},
  year={2025}
}
```

Keypoint trajectories:

```text
@article{russello2022t,
  title={T-LEAP: Occlusion-robust pose estimation of walking cows using temporal information},
  author={Russello, Helena and van der Tol, Rik and Kootstra, Gert},
  journal={Computers and Electronics in Agriculture},
  volume={192},
  pages={106559},
  year={2022},
  publisher={Elsevier}
}
```

Lameness scores:

```text
@article{russello2024video,
  title={Video-based automatic lameness detection of dairy cows using pose estimation and multiple locomotion traits},
  author={Russello, Helena and van der Tol, Rik and Holzhauer, Menno and van Henten, Eldert J and Kootstra, Gert},
  journal={Computers and Electronics in Agriculture},
  volume={223},
  pages={109040},
  year={2024},
  publisher={Elsevier}
}
```

