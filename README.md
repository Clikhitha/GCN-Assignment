# Two-Layer and Three-Layer Vanilla Graph Convolutional Network (GCN)

## Overview

This project implements two Vanilla Graph Convolutional Networks (GCNs) in Java for semi-supervised node classification on citation networks. The first experiment uses a two-layer GCN on the Cora dataset, while the second uses a three-layer GCN on the CiteSeer dataset.

Each research paper is represented as a graph node, its attributes are used as node features, and citation relationships define the graph edges. The models learn from both node features and graph connectivity to predict the subject category of each paper.

The implementations include graph construction, adjacency normalization, forward propagation, backpropagation, model training, validation monitoring, and test-set evaluation.

## Objectives

- Represent citation networks using nodes, edges, feature vectors, and class labels.
- Implement Vanilla GCN architectures directly in Java.
- Develop a two-layer GCN using the Cora citation dataset.
- Develop a three-layer GCN using the CiteSeer citation dataset.
- Evaluate predictions using accuracy, macro precision, macro recall, macro F1-score, per-class metrics, and confusion matrices.
- Analyze model performance and identify areas for further improvement.

## Datasets

### 1. Cora

The Cora citation network is used for the two-layer GCN experiment.

- **Input files:** `cora.content` and `cora.cites`
- **Node features:** 1,433 features per paper
- **Number of classes:** 7
- **Test nodes evaluated:** 548
- **Classification task:** Predict the subject category of research papers.

### 2. CiteSeer

The CiteSeer citation network is used for the three-layer GCN experiment.

- **Input files:** `citeseer.content` and `citeseer.cites`
- **Number of nodes:** 3,312
- **Node features:** 3,703 features per paper
- **Number of classes:** 6
- **Training nodes:** 1,984
- **Validation nodes:** 660
- **Test nodes:** 668
- **Classification task:** Predict the subject category of research papers.

The implementations use stratified training, validation, and test splits with a nominal 60:20:20 ratio and a fixed random seed of 42.

## Technologies Used

- **Programming Language:** Java
- **Data Processing:** Java file handling and collection utilities
- **Model Implementation:** Custom graph convolution, forward propagation, backpropagation, and gradient updates
- **Activation Functions:** ReLU for hidden layers and Softmax for output probabilities
- **Optimization:** Gradient descent with a configured learning rate and L2 regularization
- **Evaluation:** Accuracy, macro precision, macro recall, macro F1-score, per-class metrics, and confusion matrices
- **Visualization and Analysis:** Supporting Python scripts and saved plots, where included in the project files

## Methodology

1. **Dataset Loading:** Read paper identifiers, node features, class labels, and citation relationships from the dataset files.
2. **Graph Construction:** Map paper identifiers to node indices, create an undirected graph, and add self-loops.
3. **Adjacency Normalization:** Apply symmetric graph normalization to scale information exchanged between connected nodes.
4. **Model Architecture:** Apply graph convolutional transformations with trainable weights and biases.
5. **Training:** Calculate cross-entropy loss on training nodes and update model parameters through backpropagation and gradient descent.
6. **Validation:** Monitor validation accuracy and macro F1-score and retain the model checkpoint with the best validation macro F1-score.
7. **Evaluation:** Generate test predictions and evaluate the models using aggregate and class-wise metrics.
8. **Result Analysis:** Examine training logs, prediction distributions, and confusion matrices to understand classification performance.

## Model Architectures

### Two-Layer GCN — Cora

The two-layer model is implemented in `TwoLayerGCN.java`.

- Input dimension: 1,433 features
- Hidden dimension: 32
- Output dimension: 7 classes
- Hidden activation: ReLU
- Output activation: Softmax
- Training epochs: 300
- Learning rate: 0.005
- L2 coefficient: 0.0005
- Random seed: 42

The first graph convolution transforms input features into a 32-dimensional hidden representation. The second graph convolution produces class scores for the seven paper categories.

### Three-Layer GCN — CiteSeer

The three-layer model is implemented in `ThreeLayerGCN.java`.

- Input dimension: 3,703 features
- First hidden dimension: 32
- Second hidden dimension: 16
- Output dimension: 6 classes
- Hidden activation: ReLU
- Output activation: Softmax
- Training epochs: 200
- Learning rate: 0.01
- L2 coefficient: 0.0005
- Random seed: 42

The network applies two hidden graph-convolution transformations followed by an output transformation to predict the six paper categories.

## Results

The following results are taken from the saved prediction files and model metrics documented in the project report.

| Evaluation Metric | Two-Layer GCN (Cora) | Three-Layer GCN (CiteSeer) |
|---|---:|---:|
| Test nodes | 548 | 668 |
| Accuracy | 25.73% | 45.81% |
| Macro Precision | 8.52% | 27.31% |
| Macro Recall | 15.79% | 38.38% |
| Macro F1-score | 10.48% | 30.39% |

### Key Observations

- The two-layer GCN on Cora achieved a test accuracy of 25.73%, with uneven classification performance across the seven classes.
- The three-layer GCN on CiteSeer achieved a test accuracy of 45.81%.
- For the Cora experiment, Neural_Networks and Probabilistic_Methods obtained the strongest class-wise F1-scores in the saved predictions.
- For the CiteSeer experiment, Agents and DB achieved the strongest class-wise F1-scores, while AI, HCI, and ML had zero F1-scores in the saved test predictions.
- Confusion matrices and per-class metrics reveal classification errors that overall accuracy alone cannot capture.

**Note:** The two experiments use different datasets and configurations. Their results do not establish that adding an extra GCN layer directly improves performance.

## Limitations

- Classification performance is uneven across several subject categories.
- The saved Cora predictions show weak overall test performance.
- The CiteSeer experiment has zero F1-scores for three classes in the saved test predictions.
- The experiments differ in datasets, feature dimensions, class counts, and hyperparameters, preventing a controlled comparison of model depth.
- Further verification of gradient calculations, label mappings, graph normalization, and training stability is recommended.

## Future Work

- Verify forward and backward propagation using small synthetic graphs.
- Investigate class-wise prediction errors and label mappings.
- Examine training stability and the effects of hyperparameter choices.
- Conduct controlled experiments using comparable configurations and repeated runs.
- Improve reproducibility by documenting software versions and saving model checkpoints.

## Conclusion

This project demonstrates the implementation of two Vanilla Graph Convolutional Networks in Java for semi-supervised node classification on citation graphs. The two-layer Cora model and three-layer CiteSeer model use graph structure and paper attributes to predict research-paper categories.

The saved results demonstrate the complete workflow from dataset loading and graph construction to training, validation, and evaluation. Although the results reveal limitations in class-wise generalization, they provide a foundation for investigating graph-based learning, model implementation, and evaluation.

## Author

**Chittaloori Likhitha**

M.Tech — Artificial Intelligence

Amrita Vishwa Vidyapeetham, Coimbatore
