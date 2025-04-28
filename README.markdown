# Diabetic Retinopathy Detection with TensorFlow

This project implements a Convolutional Neural Network (CNN) using TensorFlow to classify retinal images for diabetic retinopathy severity. The model is trained on the APTOS 2019 Blindness Detection dataset from Kaggle, which contains retinal images labeled with severity levels from 0 (No DR) to 4 (Proliferative DR).

## Project Overview

The goal is to develop a deep learning model that can assist in the early detection of diabetic retinopathy by classifying retinal images into one of five severity categories:
- No DR (0)
- Mild (1)
- Moderate (2)
- Severe (3)
- Proliferative (4)

The project includes data preprocessing, model training, validation, and visualization of results, all implemented in a Jupyter Notebook.

## Dataset

The dataset used is the [APTOS 2019 Blindness Detection dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection) available on Kaggle. It includes:
- **train_images/**: Directory containing retinal images in PNG format.
- **train_1.csv**: CSV file with image filenames (`id_code`) and corresponding severity labels (`diagnosis`).

The dataset is downloaded automatically using the `kagglehub` library in the notebook.

## Requirements

To run the notebook, you need the following dependencies:
- Python 3.10
- TensorFlow 2.16.1
- kagglehub
- numpy
- pandas
- opencv-python
- scikit-learn
- matplotlib
- seaborn
- torch (optional, not used in core functionality)

Install the required packages using:
```bash
pip install tensorflow==2.16.1 kagglehub numpy pandas opencv-python scikit-learn matplotlib seaborn torch
```

## Project Structure

- **Untitled(1).ipynb**: Main Jupyter Notebook containing the complete workflow, including:
  - Dataset downloading and preprocessing
  - Image loading and normalization
  - Data augmentation using `ImageDataGenerator`
  - CNN model definition and training
  - Model evaluation and visualization
- **README.md**: This file, providing an overview and instructions for the project.

## Usage

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up Kaggle API**:
   - Ensure you have a Kaggle account and API token.
   - Place your `kaggle.json` file in `~/.kaggle/` (or the appropriate directory for your system).
   - Run the notebook cell to download the dataset using `kagglehub`.

3. **Run the Notebook**:
   - Open `Untitled(1).ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells sequentially to:
     - Install dependencies
     - Download and preprocess the dataset
     - Train the CNN model
     - Evaluate the model and visualize results

4. **Model Details**:
   - The CNN model uses a `Sequential` architecture with:
     - Convolutional layers (`Conv2D`)
     - MaxPooling layers (`MaxPooling2D`)
     - BatchNormalization
     - Dropout for regularization
     - Dense layers for classification
   - Data augmentation is applied to the training set to improve generalization.
   - Early stopping is used to prevent overfitting.

5. **Output**:
   - The notebook generates a plot of a random test case, showing the true and predicted labels alongside the retinal image.
   - Additional metrics (e.g., accuracy, loss) can be visualized by extending the notebook.

## Results

The model is trained on 80% of the dataset and validated on 20%. The notebook includes a function to visualize a random test case, displaying the true and predicted severity labels. Further evaluation metrics (e.g., confusion matrix, classification report) can be added for a comprehensive analysis.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate documentation.



## Acknowledgments

- [APTOS 2019 Blindness Detection dataset](https://www.kaggle.com/competitions/aptos2019-blindness-detection) for providing the data.
- TensorFlow and Keras for the deep learning framework.
- Kaggle for hosting the dataset.

For any questions or issues, please open an issue on GitHub.
</x-theoryArtifact>