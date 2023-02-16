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

    -   [`app/download_datasets.py`]('app/download_datasets.py') contains the code for downloading the datasets used in the paper `python download_datasets.py --url <dataset_url ending with .tar.xz>`
    -   [`app/sspcab`]('app/sspcab/__init__.py') contains the code for the sspcab layer from the paper

-   `mvtec_ad_evaluation/` contains the code for the evaluation of the paper on the MVTec AD dataset
