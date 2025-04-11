# **Disaster Classification Using Deep Learning (CNN Models)**

## **Overview**
Natural disasters such as cyclones, earthquakes, floods, and wildfires cause significant damage to life and property. This project leverages **Deep Learning** techniques, specifically **Convolutional Neural Networks (CNNs)**, to classify disaster images into four categories: Cyclone, Earthquake, Flood, and Wildfire.

The project implements two models:
1. A **Custom CNN** model built from scratch.
2. A **VGG16-based Transfer Learning** model.

Both models are trained on a dataset of 4,500 disaster images to demonstrate the effectiveness of deep learning in disaster identification and response.

---

## **Features**
- **Custom CNN Model**: Lightweight architecture for efficient classification.
- **VGG16 Transfer Learning**: Pre-trained on ImageNet for superior feature extraction.
- **Data Augmentation**: Enhances generalization by applying transformations like rotation, zooming, shearing, and flipping.
- **Evaluation Metrics**: Accuracy, loss curves, confusion matrices, and classification reports.
- **Real-world Deployment**: Final trained model saved as `vgg16_disaster_model.h5`.

---

## **Dataset**
The dataset consists of 4,500 images categorized into four disaster types:
1. Cyclone
2. Earthquake
3. Flood
4. Wildfire

### Preprocessing:
- Images resized to:
  - `128x128` pixels for the Custom CNN model.
  - `224x224` pixels for the VGG16 model.
- Data augmentation applied to improve robustness.

---

## **Model Architectures**

### 1. **Custom CNN Model**
The custom CNN model includes:
- Three convolutional layers with ReLU activation and increasing filter sizes (32, 64, 128).
- MaxPooling layers (`2x2`) for dimensionality reduction.
- Fully connected dense layers with dropout (0.5) to prevent overfitting.
- Final softmax layer for classification into four categories.

### 2. **VGG16 Transfer Learning**
The VGG16 model is pre-trained on ImageNet and fine-tuned for disaster classification:
- Fully connected layers removed (`include_top=False`).
- Convolutional layers frozen to retain pre-trained features.
- Added dense layers with ReLU activation and dropout (0.5).
- Final softmax layer for classification.

---

## **Installation**

### Prerequisites
Ensure you have Python 3.x installed along with the following libraries:
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/disaster-classification.git
   cd disaster-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset and place it in the `data/` folder.

4. Run the training script:
   ```bash
   python train.py
   ```

5. Evaluate the models:
   ```bash
   python evaluate.py
   ```

---

## **Usage**
### Training the Models
To train the models, execute the following command:
```bash
python train.py --model [custom|vgg16]
```

### Evaluating the Models
To evaluate a trained model on the validation set:
```bash
python evaluate.py --model [custom|vgg16]
```

### Predicting Disaster Types
To use a trained model for prediction on new images:
```bash
python predict.py --image  --model [custom|vgg16]
```

---

## **Results**

### Custom CNN Model Performance:
| Metric          | Value |
|------------------|-------|
| Precision        | 0.81  |
| Recall           | 0.80  |
| F1-score         | 0.80  |
| Accuracy         | 81%   |

### VGG16 Model Performance:
| Metric          | Value |
|------------------|-------|
| Precision        | 0.89  |
| Recall           | 0.88  |
| F1-score         | 0.88  |
| Accuracy         | 88%   |

Both models demonstrated high accuracy in classifying disaster images, with VGG16 outperforming the custom CNN due to its superior feature extraction capabilities.

---

## **Project Structure**
```
disaster-classification/
├── dataset/                     # Dataset folder (not included in repo)
├── models/                   # Saved trained models (.h5 files)
├── custom_cnn.py         # Custom CNN model implementation
├── vgg16_model.py        # VGG16 transfer learning implementation
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation (this file)

```

---

## **Future Work**
1. Expand disaster categories to include tsunamis, landslides, volcanic eruptions, etc.
2. Implement real-time video-based disaster detection from surveillance footage.
3. Explore Vision Transformers (ViTs) for improved feature extraction.
4. Optimize models for deployment on edge devices or mobile platforms.
5. Integrate geospatial data and weather forecasting with image classification.

---

## **Contributors**
1. **Debopriyo Choudhury**  
   - Email: sridebopriyo@gmail.com  
   - VITAP University  


