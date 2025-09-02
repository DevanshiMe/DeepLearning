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

