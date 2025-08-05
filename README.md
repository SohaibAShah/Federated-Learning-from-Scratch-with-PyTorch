# Federated Learning from Scratch with PyTorch

This repository contains a hands-on, step-by-step tutorial on how to implement a basic Federated Learning (FL) system from scratch using only the PyTorch framework. The goal is to provide a clear understanding of the core mechanics of FL, including client-side training and server-side aggregation, without relying on high-level libraries like Flower or TensorFlow Federated.

The tutorial is structured as a single Jupyter Notebook, making it easy to follow along, run the code, and experiment with the concepts.

## 🚀 Getting Started

To run the tutorial, you'll need to have Python and the necessary libraries installed.

### Prerequisites

Make sure you have `git` installed to clone the repository.

```sh
git clone [https://github.com/SohaibAShah/Federated-Learning-from-Scratch-with-PyTorch.git](https://github.com/SohaibAShah/Federated-Learning-from-Scratch-with-PyTorch.git)


## 🧠 Core Concepts Covered

This tutorial will guide you through implementing the following fundamental components of a Federated Learning system:

Data Partitioning: Simulating a decentralized data environment by splitting a dataset (MNIST) among multiple "clients."

Local Training: Implementing a standard PyTorch training loop for each client on their private data.

Model Aggregation: Implementing the Federated Averaging (FedAvg) algorithm on the server to combine model updates from clients.

Federated Training Loop: Orchestrating the entire process over several communication rounds, where the server sends the global model, clients train locally, and the server aggregates the results.