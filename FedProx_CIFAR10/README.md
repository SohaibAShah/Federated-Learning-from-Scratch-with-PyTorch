# Federated Learning with FedProx on CIFAR-10

This repository contains a comprehensive tutorial on implementing an advanced federated learning system from scratch using PyTorch. It demonstrates how to use the **FedProx** algorithm to train a **Convolutional Neural Network (CNN)** on the **CIFAR-10** dataset under challenging **non-IID (non-independently and identically distributed)** data conditions.

The entire project is self-contained within a single Jupyter Notebook, providing a clear, step-by-step guide for understanding and running the code.

## 🌟 Key Features

-   **FedProx Implementation**: Learn how to implement the FedProx algorithm, which adds a proximal term to the client's local loss function to stabilize training and improve performance on non-IID data.
-   **CNN Model**: Use a Convolutional Neural Network (CNN) architecture, which is highly effective for image classification tasks.
-   **Non-IID Data Simulation**: The tutorial includes code to deliberately partition the CIFAR-10 dataset into non-IID subsets, simulating a realistic federated learning scenario where client data is heterogeneous.
-   **Federated Training Loop**: Understand the full federated learning cycle, from server-side model dispatch to client-side training and final aggregation.

## 🚀 Getting Started

To run the tutorial, you'll need Python and the necessary libraries installed.

### Prerequisites

Make sure you have `git` installed to clone the repository.

```sh
git clone [https://github.com/SohaibAShah/Federated-Learning-from-Scratch-with-PyTorch.git](https://github.com/SohaibAShah/Federated-Learning-from-Scratch-with-PyTorch.git)

```

## 🧠 Core Concepts Covered

This tutorial will guide you through implementing the following fundamental components of a Federated Learning system:

Data Partitioning: Simulating a decentralized data environment by splitting a dataset (MNIST) among multiple "clients."

Local Training: Implementing a standard PyTorch training loop for each client on their private data.

Model Aggregation: Implementing the Federated Averaging (FedAvg) algorithm on the server to combine model updates from clients.

Federated Training Loop: Orchestrating the entire process over several communication rounds, where the server sends the global model, clients train locally, and the server aggregates the results.