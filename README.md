# Airplane Image Classification using Keras (CNN - Deep Learning)

This project implements a Convolutional Neural Network (CNN) using Keras for classifying images of military airplanes. This work was completed as a 7th Semester Industrial Training project at the Institute of System Study and Analyses (ISSA), Defence R&D Organisation (DRDO), Ministry of Defence, Delhi, India, between May 2019 and August 2019.

## Project Overview
The primary goal is to develop an image classification model capable of distinguishing various military aircraft (e.g., Chinook, Mikoyan MiG-21, Boeing 737) from other structures like bridges, buildings, and airbases. This model aims to assist in identifying military-related structures and potentially enemy military bases.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [CNN Architecture](#cnn-architecture)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Project Report](#project-report)
- [Future Work](#future-work)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

## Technologies Used
* **Python**
* **Keras** (with TensorFlow as the backend)
* **NumPy**
* **Pillow** (for image manipulation, a Keras dependency)
* **Matplotlib** (optional, for visualizing training progress or images)
* **Anaconda** (recommended for environment management)

## Dataset

The model was trained and tested using a confidential dataset of aerial images provided during an industrial training at the Defence Research and Development Organisation (DRDO), India. This dataset includes labeled images of various military aircraft (e.g., Chinook, Mikoyan MiG-21, Boeing 737) and non-aircraft infrastructure (e.g., bridges, buildings, airbases).

Due to the sensitive nature of the source and its use in a defense context, the dataset cannot be shared publicly.

*Note: The specific dataset used in the original project is not included in this repository. Users would need to prepare their own dataset following this structure.*

## CNN Architecture
The Convolutional Neural Network architecture includes:
1.  **Input Layer:** Expects images of size (64, 64, 3).
2.  **Convolutional Layer 1:** 32 filters, kernel size 3x3, ReLU activation.
3.  **Max Pooling Layer 1:** Pool size 2x2.
4.  **Convolutional Layer 2:** 32 filters, kernel size 3x3, ReLU activation.
5.  **Max Pooling Layer 2:** Pool size 2x2.
6.  **Flatten Layer:** Converts the 2D feature maps into a 1D vector.
7.  **Fully Connected (Dense) Layer:** 128 units, ReLU activation.
8.  **Output Layer (Dense):** 1 unit, sigmoid activation (for binary classification between 'Airplane' and 'Unrecognized').

The model was compiled using the 'adam' optimizer and 'binary_crossentropy' loss function, tracking 'accuracy' as a metric.

## Setup and Installation
1.  **Prerequisites:**
    * Anaconda (recommended for managing Python environments).
    * Python (3.6+ recommended)

2.  **Create a Conda Environment (Recommended):**
    ```bash
    conda create -n cnn_project python=3.8
    conda activate cnn_project
    ```

3.  **Install Libraries:**
    ```bash
    pip install tensorflow
    pip install keras
    pip install numpy
    pip install Pillow
    # pip install matplotlib  # Optional, for visualization
    ```

## How to Run
1.  **Prepare your dataset:** Create a `Dataset` folder in the project's root directory with `Training`, `Test`, and `single_prediction` subfolders as described under the [Dataset](#dataset) section. Populate them with your images.

2.  **Place the Python script(s):** Ensure the Python script containing the CNN model and training logic (e.g., `airplane_classifier.py`) is in the project's root directory.

3.  **Execute the script:**
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of your Python file.)

    The script will:
    * Initialize and compile the CNN.
    * Set up data generators for training and testing.
    * Train the model using `fit_generator`.
    * Include a prediction pipeline for new single images.

## Results
The model, as per the report (page 24), achieved the following performance after 30 epochs:
* **Training Loss/Accuracy:** `loss: 0.0454 - acc: 0.9922` (Epoch 30/30)
* **Validation Loss/Accuracy:** `val_loss: 0.7899 - val_acc: 0.8929` (Epoch 30/30)

*Note: Actual performance may vary based on the dataset used.*

## Project Report
The detailed project report submitted for the 7th Semester Industrial Training can be found here:
[📄 Airplane_Image_Classification_using_CNN_Deep_Learning.docx.pdf](https://github.com/Deepikag8/Airplane-Image-Classification/blob/main/Airplane_Image_Classification_using_CNN_Deep_Learning.docx.pdf)

## Future Work
Based on the conclusions in the report:
* Explore more complex CNN architectures and hyperparameter tuning for better performance.
* Investigate different feature visualization techniques for various layers.
* Implement advanced data augmentation strategies.
* Experiment with transfer learning using pre-trained models.

## Author
**Deepika Ghotra**  
*Computer Science Engineering*  
[GitHub Profile](https://github.com/Deepikag8)

## Acknowledgments
* Guidance provided by **Mr. Babloo Saha** (Scientist 'D'), ISSA, DRDO.
* Opportunity provided by the **Institute of System Study and Analyses (ISSA), Defence R&D Organisation (DRDO), Ministry of Defence, Delhi**.
* Special thanks to **Mr. Anurag Pathak**, Head HRD, ISSA, DRDO.

---

*This README was generated based on the project report "Airplane Image Classification using Keras (CNN - Deep Learning)" by Deepika Ghotra.*
