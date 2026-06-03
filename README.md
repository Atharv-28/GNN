# HGCM-Net: Hybrid Graph-based Contrastive Model for Fake News & Hate Speech Detection

This repository contains a full implementation, training/evaluation pipeline, and demo utilities for **HGCM-Net** (Hybrid Graph-based Contrastive Model), a heterogeneous graph neural network that unifies semantic encoding, social interaction modeling, and behavior-guided contrastive learning to detect fake news and hate speech.

---

## 🚀 Quick Start & Usage

### 📦 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🏋️ Training the Model
To run the full data pipeline (HuggingFace download, SentenceTransformer embedding, clustering, graph builder) and train the model:
```bash
python train.py --data-dir data
```
*Note: The first run will download the WELFake dataset and cache all embeddings and graph structures under `cached_data/` for future runs.*

**Optional Synthetic Mode** (runs instantly without SentenceTransformer downloads or HuggingFace dependency):
```bash
python train.py --use-synthetic
```

### 📊 Testing and Evaluation
To evaluate a trained checkpoint on the test split:
```bash
python evaluate.py --data-dir data --checkpoint checkpoints/best_model.pt
```

### 🔮 Inference and Predictions
To run inference and predict labels for content posts (higher scores = more likely to be fake/harmful):
```bash
python predict.py --data-dir data --checkpoint checkpoints/best_model.pt --top-k 5
```

---

## 📁 Demo Dataset Structure
For rapid local testing and development, CSV files are provided in the `data/` directory:
- **`data/contents.csv`**: Contains posts with columns: `content_id`, `text`, and `label` (0 = Real/Genuine, 1 = Fake/Harmful).
- **`data/users.csv`**: Contains users with columns: `user_id` and 10 numerical style features (`f0` to `f9`).
- **`data/interactions.csv`**: Contains links between users and posts with columns: `user_id`, `content_id`, `rho` (credibility/strength), and `tau_hours` (time delay).

---

## 💡 Notes for 4GB VRAM (e.g., RTX 3050)
If you encounter CUDA Out-Of-Memory (OOM) errors during full-batch graph propagation, adjust these parameters in [config.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/config.py):
- `batch_size`: Reduce to `32` or `16`
- `num_neighbors`: Reduce to `[10, 5]` or `[5, 5]`
- `gat_hidden_dim`: Reduce to `64`
- `gat_out_dim`: Reduce to `32`
- `gat_heads_1`: Reduce to `2`
- `gat_heads_2`: Reduce to `1`

---

## ❓ Comprehensive Q&A (Project & Model Details)

Here is a detailed guide addressing every question you might be asked regarding the theory, architecture, dataset, data pipeline, and implementation details of this project.

<details>
<summary><b>1. What is the core objective and motivation of this project?</b></summary>
<br/>

*   **Objective:** To build an end-to-end classification system that accurately detects fake news and hate speech.
*   **Motivation:** Existing systems analyze textual content in isolation or study user sharing behaviors separately. They miss the crucial interplay between *what* is written and *who* interacts with it over *time*. By modeling news posts and users as nodes in a heterogeneous graph, this project captures structural, semantic, and behavioral dimensions simultaneously.
</details>

<details>
<summary><b>2. What is the base model and what paper is this based on?</b></summary>
<br/>

*   **Model:** **HGCM-Net** (Hybrid Graph-based Contrastive Model Network).
*   **Reference Paper:** *"HGCM-Net: Hybrid Graph-Based Contrastive Model for Fake News and Hate Speech Detection"* by **Nivedita V. S.** and **Bazila Banu**, published in **Nature Scientific Reports (2026)**.
*   **Paper Performance:** The authors report **97.12% accuracy**, **96.72% precision**, and **96.43% F1-score** on the benchmark FakeNewsNet dataset (PolitiFact & GossipCop).
</details>

<details>
<summary><b>3. What are the key technical innovations of HGCM-Net?</b></summary>
<br/>

1.  **Temporally-Decayed Heterogeneous Graph:** Represents users and posts as distinct node types connected by directed, time-sensitive edges.
2.  **Heterogeneous Graph Attention Backbone (HeteroGAT):** Uses multi-head GATConv layers to selectively weight and propagate information across user and content neighborhoods.
3.  **In-Batch Contrastive Learning (NT-Xent):** A joint training objective that aligns the user embedding space and content embedding space, pushing deceptive interactions apart.
4.  **Skip-Connection Classifier Head:** Integrates raw SentenceTransformer text embeddings directly with GNN output to prevent feature over-smoothing.
</details>

<details>
<summary><b>4. Which datasets are used in this implementation?</b></summary>
<br/>

*   **Full Dataset:** The pipeline downloads the **WELFake dataset** from HuggingFace (`davanstrien/WELFake`), which consists of **72,094 news articles** with balanced labels:
    *   **Real news (label 0):** 35,028 articles
    *   **Fake news (label 1):** 37,066 articles
*   **Splits:** Evaluated using a randomized split:
    *   **Train:** 70% (50,466 nodes)
    *   **Validation:** 15% (10,814 nodes)
    *   **Test:** 15% (10,814 nodes)
*   **Demo Dataset:** A small 3-file CSV schema under `data/` used for local testing and validation of scripts without downloading the full dataset.
</details>

<details>
<summary><b>5. How are user nodes and interactions generated if standard text datasets don't have them?</b></summary>
<br/>

To run GNNs on text-only datasets like WELFake, a synthetic social simulation pipeline is executed:
1.  **User Node Synthesis:** Article text embeddings are clustered using **MiniBatchKMeans ($K=800$ clusters)**. Each cluster represents a synthetic "user archetype" or publisher source sharing similar styles of news.
2.  **User Style Features:** For each cluster, we extract **10 text-statistic features** based on the average writing metrics of its articles (details in Question 6), representing the writing style of the cluster.
3.  **Interaction Construction:**
    *   **Primary Edges:** Each content node is linked to its assigned user cluster (simulating publication), with strong intensity $\rho \sim U(0.7, 1.0)$ and fast reaction time $\tau \sim U(0, 48)$ hours.
    *   **Secondary Edges (Cross-links):** 20% random cross-links are added across clusters (simulating social sharing/retweeting) with lower intensity $\rho \sim U(0.3, 0.7)$ and longer delays $\tau \sim U(24, 720)$ hours.
</details>

<details>
<summary><b>6. What raw features represent the user and content nodes?</b></summary>
<br/>

*   **Content Nodes (`content`):** Dense **384-dimensional** embeddings generated by a pre-trained SentenceTransformer model (`sentence-transformers/all-MiniLM-L6-v2`) encoding the combined title and body of the articles.
*   **User Nodes (`user`):** A **10-dimensional** style vector representing writing statistics:
    1.  Word Count
    2.  Character Count
    3.  Sentence Count
    4.  Average Words per Sentence
    5.  Average Word Length
    6.  Vocabulary Richness (Unique words / total words)
    7.  Uppercase Character Ratio (Capitalization intensity)
    8.  Digit Ratio (Frequency of numerical values)
    9.  Punctuation Ratio (Exclamation marks, question marks, etc.)
    10. Has-Content Flag (Binary indicator)
    *User features are normalized to zero-mean and unit variance during graph construction.*
</details>

<details>
<summary><b>7. How are the temporal edge weights computed in the graph?</b></summary>
<br/>

Social interactions decay in influence over time. The directed edges between users and posts are weighted using a logarithmic temporal decay function:

$$w_{ij} = \frac{\rho_{ij}}{1 + \log(1 + \tau_{ij})}$$

*   $w_{ij}$: Final weight of the edge between user $i$ and content $j$.
*   $\rho_{ij}$: Interaction strength (engagement frequency / credibility).
*   $\tau_{ij}$: Time gap in hours between the publication of the content and the user interaction.
</details>

<details>
<summary><b>8. Can you explain the detailed architectural layers of HGCM-Net?</b></summary>
<br/>

The model consists of four primary components:
1.  **Metadata Projector (`MetadataProjector`):** An MLP (Linear $\to$ BatchNorm1d $\to$ ReLU $\to$ Dropout $\to$ Linear) that projects the raw 10-dimensional user features into a $384$-dimensional embedding space to match the dimension of content embeddings.
2.  **Heterogeneous GAT Backbone (`HeteroGATBackbone`):** A 2-layer HeteroConv neural network.
    *   **Layer 1:** Message passing with multi-head attention (4 heads, output dimension 128) over both edge directions: `("user", "interacts_with", "content")` and `("content", "rev_interacts_with", "user")`.
    *   **Layer 2:** Message passing with 2 attention heads projecting to the output space (64 dimensions). Uses LeakyReLU attention activations and dropout.
3.  **Classification Head (`ClassificationHead`):** Takes the content node's graph-aggregated representation and outputs a binary logit. It projects the raw sentence embedding using a linear projection layer and concatenates it with the GAT output (skip-connection).
4.  **Full Model Wrapper (`HGCMNet`):** Manages data routing, projection, graph propagation, and logits generation.
</details>

<details>
<summary><b>9. Why is there a skip-connection in the Classification Head?</b></summary>
<br/>

In Deep GNNs, repeatedly averaging neighbor representations across multiple convolutions causes **over-smoothing** (node features become structurally similar, washing out unique text semantic details).
The skip-connection projects the raw content text embedding and concatenates it directly with the final GAT-convolved output:

$$\mathbf{h}_{\text{combined}} = \left[ \mathbf{h}_{\text{GAT}} \,\|\, \mathbf{W}_{\text{proj}}\mathbf{h}_{\text{raw}} \right]$$

This ensures the classifier retains local, fine-grained semantic details from the SentenceTransformer alongside structural behavior features from the GNN.
</details>

<details>
<summary><b>10. What is the training loss formulation?</b></summary>
<br/>

HGCM-Net is trained end-to-end using a multi-task joint objective:

$$\mathcal{L}_{\text{final}} = \mathcal{L}_{\text{BCE}} + \lambda \mathcal{L}_{\text{contrastive}} + \alpha \|\theta\|_2^2$$

*   $\mathcal{L}_{\text{BCE}}$: Binary Cross-Entropy with Logits loss over classification predictions.
*   $\mathcal{L}_{\text{contrastive}}$: In-batch temperature-scaled contrastive loss (NT-Xent style) that pulls interacting user and content representations together and pushes non-interacting ones apart.
*   $\lambda$: Contrastive weight coefficient (set to $0.05$ to ensure classification remains the dominant task).
*   $\alpha \|\theta\|_2^2$: L2 regularization (weight decay set to $10^{-4}$ to prevent overfitting).
</details>

<details>
<summary><b>11. How does the model avoid GPU Out-Of-Memory (OOM) errors during training?</b></summary>
<br/>

Training GNNs on large graphs (72K+ nodes, 80K+ edges) is VRAM intensive. To run on 4GB VRAM cards:
1.  **Full-Batch Forward GAT Pass:** The entire graph is loaded into GPU memory to execute structural message passing, which is highly efficient.
2.  **Sub-Sampled Contrastive Loss:** Standard contrastive loss requires calculating $O(N^2)$ comparisons. Instead of computing this over all 72K nodes, the loss is computed on a **randomized mini-batch of 512 training nodes** at each epoch. This maintains training quality while capping peak VRAM usage.
3.  **Neighbor Sampling:** The graph configuration restricts message passing to the top neighbors during data loading (`[15, 10]` by default).
</details>

<details>
<summary><b>12. What results did this implementation achieve?</b></summary>
<br/>

When trained and tested on the 72,094 articles in the WELFake dataset (with early stopping triggered at epoch 276):
*   **Test Accuracy:** **91.38%**
*   **Test Precision:** **91.44%**
*   **Test Recall:** **92.16%**
*   **Test F1-Score:** **91.80%**
</details>

<details>
<summary><b>13. What libraries are used to build this codebase?</b></summary>
<br/>

*   **`torch` (PyTorch):** Deep learning tensor operations and optimization.
*   **`torch-geometric` (PyG):** Heterogeneous graph construction (`HeteroData`), heterogeneous convolutions (`HeteroConv`), and graph attention networks (`GATConv`).
*   **`sentence-transformers`:** Pre-trained transformer text embedding extraction.
*   **`scikit-learn`:** K-Means clustering algorithm for synthetic user generation.
</details>

---

## 🛠️ Project Structure
*   [config.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/config.py): Master configuration class containing all hyperparameters, paths, and training dimensions.
*   [model.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/model.py): HGCMNet, MetadataProjector, HeteroGATBackbone, and ClassificationHead module definitions.
*   [losses.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/losses.py): Custom multi-task loss classes, including the temperature-scaled in-batch contrastive loss.
*   [graph_builder.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/graph_builder.py): Logic to assemble `HeteroData` graphs, including temporal edge decay computation.
*   [data_loader.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/data_loader.py): Reader scripts for the demo CSV tables under `data/`.
*   [data_pipeline.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/data_pipeline.py): Full pipeline for WELFake HuggingFace dataset downloading, clustering, style feature generation, and caching.
*   [train.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/train.py): Training loop with learning rate schedules, metrics evaluation, and checkpoint saving.
*   [evaluate.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/evaluate.py): Metrics computation on hold-out splits.
*   [predict.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/predict.py): Command-line inference utility.
#   G N N  
 