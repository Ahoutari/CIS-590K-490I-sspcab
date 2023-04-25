# CIS 590K / 490I Paper Review

## Paper: [Self-Supervised Predictive Convolutional Attentive Block for Anomaly Detection](https://arxiv.org/pdf/2111.09099.pdf)

## Winter 2023

---

### Getting Started

-   Pull the repository
-   Install the requirements using `pip install -r requirements.txt`
    -   make sure to do this in a virtual environment

### Code Structure

-   `app/` contains the code for the paper

    -   [`app/download_datasets.py`](app/download_datasets.py) contains the code for downloading the datasets used in the paper `python download_datasets.py --url <dataset_url ending with .tar.xz>`
    -   [`app/sspcab`](app/sspcab/__init__.py) contains the code for the sspcab layer from the paper

### Training

-   `app/train_mvtec.py` contains the code for training the model on the MVTec AD dataset

-   Example usage:

    ```bash
    python train_mvtec.py --model simple_model \
                    --dataset <path to dataset folder> \
                    --epochs 20 \
                    --batch_size 32 \
                    --out <path to save files> \
                    --num_workers 4
    ```

-   Output:

    -   `model.h5` contains the trained model
    -   `feature_extractor.h5` contains the trained feature extractor
    -   `kde.pickle` contains the trained KDE model
    -   `history.json` contains the training history
    -   `metrics.json` contains the metrics used for anomaly detection
