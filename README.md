# Deep Learning
This repository explores different deep learning architectures. It currently includes these projects:
## Transformer for Text Classification

**Objective:**
Implement a Transformer model from scratch using **PyTorch**, closely following the original paper. Train and evaluate the model on a multi-class text classification dataset, with the goal of achieving **>80% test accuracy**.

**Key Features:**

* **Pre-trained GloVe embeddings** initialize the embedding layer. Weights are frozen to preserve semantic meaning and reduce training complexity.
* **Sinusoidal positional encoding** adds token position information to embeddings.
* **Transformer Encoder:** 4 stacked layers, each with:

  * Multi-head self-attention
  * Feedforward networks
  * Layer normalization
  * Dropout (set to zero here)
* **Pooling:** Mean pooling converts variable-length token sequences into fixed-size representations.
* **Classification Head:** Fully connected linear layer.
* **Model Parameters:**

  * Total parameters: **12,012,196**
  * Trainable parameters: **1,810,596**

**Highlights:**
This design balances **performance** and **efficiency**, leveraging pre-trained embeddings and Transformer’s contextual learning power.

##  Autoencoders for Anomaly Detection

**Objective:**
Use a **convolutional autoencoder** for detecting anomalies in time series data.

**Dataset:**

* NYC Taxi Rides dataset from the **Numenta Anomaly Benchmark (NAB)**
* [Dataset Link](https://www.kaggle.com/datasets/boltzmannbrain/nab)
* Data consists of taxi rides recorded every **30 minutes**.
* Features:

  * `timestamp`: temporal context
  * `value`: number of rides per interval

**Model Architecture:**

* **Encoder:**

  * 2 Conv1d layers with output channels (16 → 8)
  * ReLU activations after each layer
* **Decoder:**

  * 2 ConvTranspose1d layers with output channels (8 → 16 → 1)
  * ReLU applied after the first transposed convolution
* **Hidden Representation:** Captures compressed features for reconstruction.

**Highlights:**

* Learns to reconstruct normal patterns in taxi rides.
* Deviations in reconstruction error signal anomalies.
* Compact convolutional architecture makes it efficient for time series anomaly detection.

## VGG-16 (Version C) vs ResNet-18 for Image Classification

**Objective:**
Implement and compare **VGG-16 (Version C)** and **ResNet-18** on a custom dataset, exploring the evolution from deep CNNs to residual networks. The task demonstrates how residual connections improve training stability and generalization.

**Dataset:**

* Total: **30,000 images**
* 3 Classes: **Dogs, Food, Vehicles**
* Ideal for **multi-class classification**

**Baseline Goal:**

* Accuracy > **75%** for base models
* Accuracy > **80%** for improved models

#### VGG-16 (Version C)

* **Final Accuracy:** 89.4%
* **Training Losses:**
  `[0.0889, 0.0537, 0.0631, 0.0701, 0.0704, 0.0496, 0.0458, 0.0311, 0.0352, 0.0215]`
* **Validation Losses:**
  `[1.0225, 0.8865, 0.9923, 1.0166, 1.1482, 1.1022, 1.1557, 1.1291, 1.2872, 1.1538]`
* **Training Accuracies:**
  `[98.31, 98.87, 98.31, 97.19, 98.31, 98.31, 98.87, 99.43, 99.43, 98.87]`
* **Validation Accuracies:**
  `[80.1, 81.33, 82.95, 85.91, 84.06, 84.96, 84.14, 81.26, 89.4, 87.52]`

#### ResNet-18

* **Final Accuracy:** 89.89%
* **Training Losses:**
  `[0.4167, 0.4212, 0.4164, 0.3940, 0.3863, 0.3913, 0.3867, 0.3913, 0.3443, 0.3835]`
* **Validation Losses:**
  `[0.5373, 0.5344, 0.5454, 0.5620, 0.5827, 0.5475, 0.5502, 0.5428, 0.5450, 0.5462]`
* **Training Accuracies:**
  `[84.34, 80.97, 86.45, 83.84, 82.72, 81.22, 89.12, 81.14, 80.21, 83.27]`
* **Validation Accuracies:**
  `[80.01, 89.89, 82.74, 80.47, 85.1, 89.61, 88.1, 81.68, 84.23, 87.67]`


### Insights

* **VGG-16** achieves strong accuracy but shows higher **validation loss fluctuations**, suggesting overfitting due to its large parameter count.
* **ResNet-18** maintains **more stable validation loss** and achieves slightly better final accuracy by leveraging **residual connections** that improve gradient flow.
* This project highlights the **transition from plain deep CNNs to residual architectures**, demonstrating why ResNets became state-of-the-art.

## Vanishing Gradient Problem in Deep CNNs

**Objective:**
Experimentally demonstrate the **vanishing gradient problem** in very deep convolutional networks (e.g., VGG-Deep) and understand how **ResNet’s residual connections** mitigate it.

### Observations

1. In **very deep networks** like VGG-Deep, **gradients vanish**, making training increasingly difficult.
2. **ResNet** introduces **residual (skip) connections**, allowing gradients to **bypass layers** and flow more easily.

   * This stabilizes training
   * Prevents vanishing gradients
   * Enables deeper architectures to converge effectively


### Experimental Results

* **Gradient Norms in VGG-Deep**

  * Layer 3: `10⁻⁸` → **100× smaller** than Layer 15: `10⁻⁶`
  * Layer 27: `10⁻⁴` → **100× smaller** than Layer 38: `10⁻²`

These results clearly show the **vanishing gradient effect** as depth increases.

* **ResNet Comparison**

  * Residual connections significantly improved **gradient propagation**.
  * Training became **more stable** with stronger gradient flow even in deeper layers.

###  Insights

* **VGG-Deep** struggles with vanishing gradients, causing poor convergence in very deep setups.
* **ResNet** overcomes this by allowing gradients to **flow directly** through skip connections.
* This experiment highlights **why ResNets became the foundation** for modern deep learning architectures.


## Comparing ResNeXt, ResNet, and VGG

**Objective:**
Compare the performance, efficiency, and generalization of **ResNeXt**, **ResNet**, and **VGG** on image classification tasks, highlighting the impact of architectural innovations like **cardinality**.


### Model Overview

1. **ResNeXt**

   * Uses **cardinality**: multiple parallel paths within each block
   * Learns **more complex features** efficiently
   * Outperforms ResNet and VGG in both accuracy and generalization

2. **ResNet**

   * Focuses on depth with **residual connections**
   * Avoids vanishing gradients but lacks cardinality

3. **VGG**

   * Deep stack of layers
   * No residual connections or cardinality, making it less efficient for complex feature learning


### Key Challenges

1. **Selecting Cardinality**

   * Optimal cardinality value is crucial for balancing performance and computational cost

2. **Learning Rate Adjustments**

   * Complex architectures require careful tuning for stable convergence

3. **Modifying Architecture**

   * Adding depth and cardinality increases training time and optimization difficulty


### Experimental Results

| Model     | Train Accuracy | Validation Accuracy | Train Loss | Validation Loss | Notes                                                          |
| --------- | -------------- | ------------------- | ---------- | --------------- | -------------------------------------------------------------- |
| VGG-16    | 99.44%         | 89.40%              | 0.0352     | 1.2873          | Overfitting (high train accuracy, high val loss)               |
| ResNet-18 | 80.97%         | 89.89%              | 0.4212     | 0.5344          | Lower train accuracy, stable validation                        |
| ResNeXt   | 87%            | 90.61%              | —          | 0.4661          | Best balance of accuracy, generalization, and train efficiency |


### Insights

* **ResNeXt** achieves the **highest validation accuracy (90.61%)** and best generalization due to cardinality.
* **VGG-16** overfits the training data despite low train loss.
* **ResNet-18** trains efficiently but has slightly lower validation performance compared to ResNeXt.
* ResNeXt demonstrates how **parallel paths (cardinality)** improve feature learning and model efficiency over traditional deep CNNs.



## Time-Series Forecasting with LSTM

**Objective:**
Forecast **air quality features** (pollutants + weather variables) using sequential modeling with LSTM, capturing temporal dependencies and feature correlations.


### Dataset Characteristics

* Hourly **air quality data**, including pollutant concentrations (CO, NOx, etc.) and weather variables (temperature, humidity).
* **Autocorrelation analysis** shows pollutants depend on past values.
* **Correlation matrix** reveals strong dependencies between pollutants and weather conditions, indicating interrelated dynamics.


### Model Architecture

* **Single-layer LSTM** with **64 hidden units**.
* Predicts **13 air quality features** simultaneously.
* Uses **ReLU activation** and **dropout** to reduce overfitting.
* Requires careful **hyperparameter tuning** for best performance.


### Results

* **Best R²:** 76.16% → model explains a substantial portion of variance.
* **Learning Rate:** Decreasing from `0.01 → 0.001` improved R² from **0.6191 → 0.7616**.
* **Bidirectional LSTM:** Enabled → R² improved from **0.6885 → 0.7616**.
* **Hidden size/Iterations:** Increasing them alone did **not guarantee improvements**.


### Limitations

* Limited hyperparameter tuning → may have missed optimal configurations.
* Model is only **single-layer LSTM**, limiting its ability to capture more **complex temporal dependencies**.


### Improvements

* Expand **hyperparameter search space** (learning rate schedules, dropout values, optimizers).
* Explore **deeper architectures** (multi-layer LSTM, GRU, or Transformer-based time-series models).

## Sentiment Analysis using LSTM

**Objective:**
Perform **binary sentiment classification** on movie reviews from the **Stanford Large Movie Review Dataset**, predicting whether a review is **positive (1)** or **negative (0)** based on its text.

**Dataset:**

* Source: [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
* Total Reviews: **25,000**

  * Positive: 12,500
  * Negative: 12,500 (balanced dataset)
* Preprocessing Notes:

  * Some reviews contain **redundant HTML tags** that need to be cleaned.

###  Model Architectures

#### BaseLSTM

* 3 **unidirectional LSTM layers**.
* Embedding layer converts word indices into dense vectors.
* **Dropout** applied after each LSTM layer for regularization.
* Final hidden state passed through a **fully connected layer**.
* **Sigmoid** activation for binary classification.

#### ImprovedLSTM

* Uses **3 bidirectional LSTM layers**, processing sequences forward and backward.
* Embedding layer can use **pretrained word embeddings** for better initialization.
* Dropout applied after each LSTM layer.
* Final hidden state from the last bidirectional LSTM passed through a fully connected layer.
* Sigmoid activation for binary classification.


### Observations & Limitations

* Slight improvement observed with **pretrained embeddings** and **bidirectional LSTMs**.
* The current model is **not performing optimally**.
* Improvements needed:

  * Explore **different tokenizers** and pretrained embeddings.
  * Conduct **extensive hyperparameter tuning** (learning rate, hidden size, dropout, batch size).
  * Potentially experiment with **more advanced architectures** (e.g., attention-based LSTM or Transformer).


## Transfer Learning with Pre-trained Models

**Objective:**
Leverage **pre-trained CNN models** for image classification on the **Food-11 dataset** and compare their performance.

**Dataset:**

* **Food-11 image dataset**
* Multi-class classification problem

### Models Evaluated

1. **EfficientNet-B0**

   * Accuracy: **90.43%**
   * Efficient scaling balances depth, width, and resolution
   * Computationally efficient but slightly less capable of capturing intricate features due to its smaller depth

2. **MobileNetV3 Large**

   * Accuracy: **90.55%**
   * Uses **depthwise separable convolutions**, reducing parameters and computation
   * Slightly lower accuracy because of limited depth for complex feature learning

3. **ResNet-50**

   * Accuracy: **92.15% (validation)**, **93.6% (testing)**
   * Deep architecture with **residual connections**
   * Learns complex hierarchical features efficiently
   * Avoids vanishing gradient problem, making it suitable for challenging datasets


### Observations

* All models performed well, with **ResNet-50 achieving the best accuracy**.
* Slight differences in performance are explained by:

  * Depth and architectural complexity
  * Ability to capture hierarchical and intricate features
  * Residual connections facilitating better gradient flow








