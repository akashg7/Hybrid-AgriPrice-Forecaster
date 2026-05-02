# Module Technical Deep-Dive: Disease_Radar
## 🔬 Intelligence Profile: Computer Vision Diagnosis

### 1. The Neural Architecture
The Disease Radar module utilizes a **MobileNetV2** backbone fine-tuned on the PlantVillage dataset.
- **Model Storage**: `models/best_model.keras` (~21MB).
- **Format**: Keras H5/Native.
- **Input Tensor**: 224x224x3 (RGB).

### 2. Pathological Scope (38 Actual Classes)
The model is trained to recognize 38 distinct plant-pathology states across 14 crop species:
- **Tomato**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, and Healthy.
- **Potato**: Early Blight, Late Blight, and Healthy.
- **Apple**: Apple Scab, Black Rot, Cedar Apple Rust, and Healthy.
- **Corn**: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, and Healthy.
- **Others**: Grapes (Black Rot, Esca), Peach (Bacterial Spot), Pepper (Bacterial Spot), etc.

### 3. Reasoning & Implementation
- **Depthwise Separable Convolutions**: MobileNetV2 was selected for its balance between accuracy and inference speed (approx. 85ms on CPU). 
- **Transfer Learning**: The model leverages features learned from ImageNet, allowing it to detect complex leaf textures and lesion patterns with a relatively small training footprint.
- **Softmax Probabilities**: The final layer provides a probability distribution across all 38 classes, which the UI uses to provide both a primary diagnosis and differential (alternative) considerations.

### 4. Technical Specs
- **Preprocessing**: Rescaling (1./255) and normalization to ImageNet mean/std.
- **Optimizer**: Adam with a learning rate of 0.0001.
- **Output**: 38-way Softmax.
