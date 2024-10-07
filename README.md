# alzheimer-detection-cnn

This code is a comprehensive implementation of a convolutional neural network (CNN) model to classify brain MRI images into four categories of dementia severity: Non-Demented, Very Mild Demented, Mild Demented, and Moderate Demented. Here's a detailed breakdown:

### 1. **Importing Libraries**
   The necessary libraries for deep learning, data manipulation, image processing, and visualization are imported.
   
### 2. **Data Loading and Preprocessing**
   - The dataset is located in `/kaggle/input/imagesoasis/Data`, which contains images categorized by dementia levels.
   - **Image Categorization:** Images are sorted into four categories (`non_demented`, `very_mild_demented`, `mild_demented`, `moderate_demented`) based on the directory names. The `os.walk()` function traverses the dataset directory, and images are stored in their respective lists.
   - **Sample Size Adjustment:** To balance the dataset, only 488 images are retained from each category.
   - **OneHotEncoder:** A `OneHotEncoder` is initialized to encode labels into one-hot vectors, which is necessary for multi-class classification.

### 3. **Image Processing**
   - A helper function `process_images()` resizes each image to `128x128` pixels and appends it to the `data` list if it has three channels (RGB).
   - Labels are one-hot encoded using the `encoder` and stored in the `result` list.
   - **Data and Label Conversion:** The processed images (`data`) and corresponding labels (`result`) are converted to NumPy arrays (`X` and `y`). The labels (`y`) are reshaped and simplified by using `np.argmax()` to get the category index.

### 4. **Train-Test Split**
   - The dataset is split into training and testing sets using `train_test_split()`, with 20% of the data reserved for testing (`X_test`, `y_test`).

### 5. **CNN Model Architecture**
   The CNN model is created using the `Sequential` API from Keras:
   - **Convolutional Layers (`Conv2D`):** Extracts features from images using filters. The model has multiple convolutional layers with increasing filter sizes (32, 64, 128, 256).
   - **Batch Normalization (`BatchNormalization`):** Normalizes layer outputs to accelerate training and reduce sensitivity to initialization.
   - **MaxPooling (`MaxPooling2D`):** Downsamples the feature maps to reduce dimensionality and computation.
   - **Dropout (`Dropout`):** Reduces overfitting by randomly dropping units during training.
   - **Flatten:** Converts 2D feature maps into a 1D vector for the fully connected layers.
   - **Dense Layers:** The fully connected layers, with the final layer having 4 output units (one for each class) and a `sigmoid` activation function (multiclass classification).

### 6. **Model Compilation**
   - **Loss Function:** The model uses `sparse_categorical_crossentropy`, which is suitable for multi-class classification when labels are integers.
   - **Optimizer:** Adam optimizer is used to update weights based on the loss function.
   - **Metrics:** Accuracy is used to evaluate the model’s performance during training.

### 7. **Model Training**
   - **Early Stopping:** Training is monitored using the `EarlyStopping` callback to prevent overfitting by stopping early if validation loss doesn’t improve for 20 epochs.
   - **Training:** The model is trained for up to 50 epochs on the training set (`X_train`, `y_train`) with a batch size of 32. Validation data is split from the training set (20% of the training data is used for validation).

### 8. **Model Evaluation**
   After training, the model is evaluated on the test set (`X_test`, `y_test`) to assess its performance.

### 9. **Visualization of Training Accuracy**
   A plot is generated to show the training and validation accuracy over the epochs.

### 10. **Prediction and Visualization**
   - The model predicts the labels of the test images.
   - **Correct Predictions:** A set of correctly predicted images is displayed using Matplotlib. Each image shows the predicted and true labels.
   - **Incorrect Predictions:** Similarly, incorrect predictions are visualized.
