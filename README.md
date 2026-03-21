# Body Pose Predictor: CSE 586 MidTerm Project 1

This repository contains the implementation of an autoregressive forecasting model designed to predict future human body poses based on a sequence of past poses. This project was developed as part of the **CSE 586: Computer Vision II** course at Penn State University.

## Project Overview
The goal is to predict future human body poses using the **SMPL** (Skinned Multi-Person Linear) parameterized body model. To facilitate sequence modeling, poses are encoded into a 32-dimensional latent vector space using **VPoser**, a variational autoencoder (VAE). The model performs predictions within this latent space before decoding them back into SMPL parameters for evaluation and visualization.

### Key Features
* **Autoregressive Forecasting**: Implements a sequence prediction method where each predicted frame is used to predict the next.
* **VPoser Integration**: Utilizes a pretrained VAE to map 63-parameter body rotations into a regularized 32D latent space.
* **Transformer-Based Architecture**: Chosen for its superior ability to handle long-range dependencies in time-series data compared to traditional CNNs.
* **Long-Horizon Prediction**: Designed to challenge the model by predicting poses up to 5 seconds into the future.

## Network Architecture
The core of the prediction engine is a **Transformer** network modified for regression.
* **Positional Encoding**: Incorporated to preserve the temporal order of the pose sequences.
* **Regression Head**: The traditional softmax layer is replaced with a fully connected (FC) layer that outputs 32-dimensional continuous vectors.
* **Input**: 32D VPoser latent vectors are fed directly into the model as real-valued input.
* **Loss Function**: Employs Mean Squared Error (MSE) or Mean Absolute Error (MAE) to minimize the distance between predicted and actual latent vectors.

## Evaluation Metrics
The model's performance is quantitatively measured using:
* **MPJPE (Mean Per Joint Position Error)**: The primary metric, calculating the Euclidean distance between predicted 3D joint positions and the ground truth.
* **Baselines**: Accuracy is compared against two standard trajectory prediction methods:
    * **Zero-Velocity**: Predicts the last seen pose for all future steps.
    * **Constant-Velocity**: Predicts the next position based on the velocity between the two previous frames.

## Dataset
We use the **AMASS** (Archive of motion capture as surface shapes) dataset.
* **Data Format**: Unified SMPL format.
* **Preprocessing**: To handle long time horizons, the original 120 fps data is subsampled to 30 fps or 24 fps.

## Requirements
The project requires the following dependencies (as seen in `testVPoser.ipynb`):
* `torch`
* `numpy`
* `human_body_prior` (VPoser)
* `trimesh` (for visualization)
* `omegaconf`
* `loguru`

## Getting Started
1. **Environment Setup**: Use the provided `testVPoser.ipynb` to install necessary libraries and mount Google Drive for model storage.
2. **Model Files**: Ensure you have the VPoser v2.0 model directory (`vposer_v2_05`) and the neutral SMPLX body model file (`smplx_neutral_model.npz`).
3. **Training**: The project is optimized for Google Colab to leverage GPU acceleration.

## Team Members
* Anupama Srikanthan
* Saurin Rajesh Savla
* Surya Prasad Senthilkumaran
* Vaishnavi Gatla

## References
* [1] Nima Ghorbani. human_body_prior.
* [2] Andrej Karpathy. makemore.
* [3] Matthew Loper, et al. SMPL: a skinned multi-person linear model.
* [4] Julieta Martinez, et al. On human motion prediction using recurrent neural networks.
