
```markdown
# Leaf Classification using Convolutional Neural Networks (CNNs)

This project uses a Convolutional Neural Network (CNN) to classify different types of leaves. The dataset consists of images of various leaf types stored in labeled folders, and the model is trained to predict the type of leaf in a given image.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Notes](#notes)

## Dataset
The dataset should be organized into folders for training and testing, with each class having a dedicated folder under `veggie_heap_training` and `veggie_heap_testing`, respectively. Each folder should contain images of a single leaf type in `.jpg`, `.jpeg`, or `.png` format.

Directory structure:
```
.
├── veggie_heap_training
│   ├── Class_1
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── Class_2
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── veggie_heap_testing
    ├── Class_1
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── Class_2
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

## Model Architecture
This CNN model is built using TensorFlow and has the following layers:
1. Convolutional Layer with 32 filters and ReLU activation.
2. MaxPooling Layer to reduce dimensionality.
3. Convolutional Layer with 64 filters and ReLU activation.
4. MaxPooling Layer.
5. Convolutional Layer with 64 filters and ReLU activation.
6. Flattening Layer to convert the 2D output to a 1D vector.
7. Dense Layer with 64 neurons and ReLU activation.
8. Dense Output Layer with softmax activation for multi-class classification.

The model uses `sparse_categorical_crossentropy` as the loss function and the `adam` optimizer.

## Requirements
Install the necessary dependencies using pip:
```bash
pip install tensorflow numpy scikit-learn
```

## Usage
1. **Prepare the Dataset**: Organize your dataset in folders as described above.
2. **Update Paths**: In the script, update `train_folder_path` and `test_folder_path` with the paths to your dataset.
3. **Run the Script**:
   ```bash
   python leaf_classification.py
   ```

The script will:
- Load images from the folders and resize them to 28x28 pixels.
- Encode the class labels.
- Normalize the image pixel values to the range [0, 1].
- Train the CNN model on the training dataset with a 20% validation split.
- Print training accuracy after each epoch.
- Evaluate and print the accuracy on the test dataset.

## Results
After training, the model will output:
- **Test accuracy**: The accuracy of the model on the testing dataset.
- **Train accuracy**: The accuracy of the model on the training dataset.

## Notes
- **Hyperparameters**: You can adjust the `epochs` and `batch_size` parameters in the `model.fit` function to improve training.
- **Image Size**: The images are resized to 28x28 pixels to match the input shape of the model.
- **Callback**: A custom callback, `PrintTrainAccuracy`, is included to print training accuracy after each epoch.
- **File Paths**: Update the folder paths in the script to match the locations of your dataset.

Happy training!
```
