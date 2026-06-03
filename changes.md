# Recommended Updates for the GNN Synopsis Report (PDF)

This document maps the mismatches between the current synopsis report (`GNN_for_Fake_News_and_Hate_Speech.pdf`) and the actual project implementation. Use this guide to update your project report so that it aligns perfectly with your working codebase and results.

---

## 📊 Summary of Major Pipeline & Implementation Differences

| Feature | Synopsis Report (PDF) | Actual Project Implementation |
| :--- | :--- | :--- |
| **Dataset** | **FakeNewsNet** (PolitiFact + GossipCop)<br/>• 11,758 records | **WELFake** (HuggingFace)<br/>• **72,094** content articles |
| **User Node Source** | Native users from social media logs | **Synthetic User Nodes**<br/>• Generated via **K-Means clustering** ($K=800$) |
| **User Features** | **12 metadata dimensions** (followers, verification, age, etc.) | **10 style-statistics features** (`f0` to `f9`) averaged per cluster |
| **Graph Library** | NetworkX | **PyTorch Geometric (`HeteroData`)** directly |
| **Software Modules** | `embeddings.py`, `visualize.py` | `data_pipeline.py`, `losses.py`, `predict.py` |
| **Hardware Focus** | Kaggle/Colab Cloud GPUs (Tesla T4) | Local 4GB VRAM optimization (RTX 3050) |
| **Results** | Targets paper results (~97% Acc on FakeNewsNet) | Achieved **91.38% Acc, 91.80% F1** on 72K WELFake dataset |

---

## 🔍 Detailed Updates by Chapter

### Chapter 1: Introduction (Abstract & Introduction)
*   **Location:** Page 6 (Abstract), Page 11 (Section 1.1)
*   **Current PDF Text:**
    *   *Abstract:* "The model is evaluated on the FakeNewsNet benchmark dataset comprising 11,758 records... Our implementation targets equivalent performance..."
    *   *Section 1.1:* "...Our implementation will be considered successful if it achieves classification accuracy of $\ge 95\%$, F1-score of $\ge 94\%$... on the FakeNewsNet test set."
*   **Actual Project Code:**
    *   The model runs on the **WELFake dataset** (72,094 content nodes), yielding a **91.38% Accuracy** and **91.80% F1-score**.
*   **Recommended Update:**
    *   Rewrite the abstract and success criteria to reflect that while the reference paper evaluated on the 11.7K FakeNewsNet dataset, this project scales the architecture to the **72,094-article WELFake dataset**.
    *   Update the success criteria to reflect actual achieved metrics: *"...achieved a classification accuracy of 91.38% and an F1-score of 91.80% on the 72K article split, demonstrating solid scalability."*

---

### Chapter 3: Proposed System & Scope
*   **Location:** Page 12 (Section 3.2.1), Page 13 (Section 3.2.1)
*   **Current PDF Text:**
    *   "Data Preprocessing Pipeline: Implement a complete data preprocessing pipeline for the FakeNewsNet dataset..."
    *   "Evaluation and Baseline Comparison: Evaluate the model on the FakeNewsNet benchmark dataset comprising 11,758 records..."
*   **Actual Project Code:**
    *   The preprocessing pipeline ([data_pipeline.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/data_pipeline.py)) fetches WELFake from HuggingFace, extracts text-statistics, clusters content into user nodes, and constructs temporal edges.
*   **Recommended Update:**
    *   Replace all references to "FakeNewsNet dataset comprising 11,758 records" with "WELFake dataset comprising 72,094 records".
    *   Add a paragraph explaining the **K-Means user clustering** step as a core preprocessing module to simulate user-content graphs.

---

### Chapter 5: Methodology
#### Section 5.1: Data Acquisition and Preprocessing
*   **Location:** Page 29–30
*   **Current PDF Text:**
    *   "The filtered dataset contains 11,758 user records (5,976 harmful, 5,782 normal) with 12 metadata dimensions..."
    *   Lists tokenization, min-max normalization, and feature-wise median imputation of native user profiles.
*   **Actual Project Code:**
    *   Loads **72,094 content articles** (35,028 real, 37,066 fake).
    *   Clusters article embeddings into **800 user clusters**.
    *   Computes **10 text-statistic features** (`f0` to `f9`) derived from user writing styles (word count, char count, uppercase ratio, digits ratio, punctuation ratio, etc.).
*   **Recommended Update:**
    *   Update the text to detail the WELFake statistics: **72,094 articles** balanced between real and fake.
    *   Document the **User Node Synthesis Pipeline**:
        *   Extract dense text representations using the `all-MiniLM-L6-v2` SentenceTransformer.
        *   Cluster these embeddings via `MiniBatchKMeans` with $K=800$ to create user nodes representing source archetypes.
        *   Formulate the 10 style features extracted from textual structure, rather than 12 user profile metadata attributes.

#### Section 5.2: Graph Construction
*   **Location:** Page 30–31
*   **Current PDF Text:**
    *   Suggests graph construction is managed via NetworkX.
*   **Actual Project Code:**
    *   Constructs a heterogeneous PyTorch Geometric `HeteroData` graph directly using vectorised tensor operations in [graph_builder.py](file:///d:/Coding/Machine%20Learning/GNN%20Mini-pro/graph_builder.py).
*   **Recommended Update:**
    *   Remove all references to "NetworkX" in graph assembly. Explain that the graph is directly instantiated using PyTorch Geometric's `HeteroData` node and edge dictionaries on the GPU to maximize performance and avoid data transfer latency.

#### Section 5.6: Modules of the Software System
*   **Location:** Page 33–34
*   **Current PDF Text:**
    *   Lists: `data_loader.py`, `graph_builder.py`, `embeddings.py`, `model.py`, `train.py`, `evaluate.py`, and `visualize.py`.
*   **Actual Project Code:**
    *   The file list consists of:
        *   `data_loader.py`: Utility to read the local demo CSV files (for quick smoke tests).
        *   `data_pipeline.py`: **[NEW]** Orchestrates the full pipeline (HuggingFace download, K-Means clustering, user feature extraction, edge generation, and caching).
        *   `graph_builder.py`: Assembles PyTorch Geometric `HeteroData` graph objects.
        *   `model.py`: Architecture classes (MLP projector, 2-layer HeteroGAT, Skip-Connection classifier).
        *   `losses.py`: **[NEW]** BCE and NT-Xent in-batch contrastive loss functions.
        *   `train.py`: Main training loop with cosine learning rate schedule, VRAM optimization, and checkpointing.
        *   `evaluate.py`: Model test-split evaluator.
        *   `predict.py`: **[NEW]** Inference script to run top-k predictions on input content.
    *   *Note: `embeddings.py` and `visualize.py` do not exist in the codebase.*
*   **Recommended Update:**
    *   Replace the module list on pages 33–34 with the actual files listed above, providing their correct purposes.

---

### Chapter 6: Requirements
#### Section 6.1.1: Frontend & Hardware Specs
*   **Location:** Page 36 (Section 6.2.1), Page 38 (Table 6.1)
*   **Current PDF Text:**
    *   Lists Tesla T4 GPUs (Kaggle/Colab) as the primary training frontend.
    *   States that a GPU is needed to process the graph of 11,758 nodes.
    *   Table 6.1 lists `NetworkX 3.1+` and `Hugging Face Transformers 4.35+` (no mention of PyTorch Geometric versions).
*   **Actual Project Code:**
    *   The code is optimized for local development on consumer-grade hardware with **4GB VRAM** (e.g., NVIDIA RTX 3050) by restricting contrastive loss calculation to a **512-node mini-batch** per step.
    *   It uses PyTorch Geometric $\ge$ 2.5.0, PyTorch $\ge$ 2.1.0, and SentenceTransformers $\ge$ 2.2.0.
*   **Recommended Update:**
    *   Add local development specs: *"...optimized to train on local GPUs with 4GB VRAM (e.g., RTX 3050) using in-batch contrastive sub-sampling (512 nodes per batch) to avoid CUDA Out-Of-Memory (OOM) failures."*
    *   Update Table 6.1: Remove `NetworkX`, and verify the library versions align with your `requirements.txt` (`torch >= 2.1.0`, `torch-geometric >= 2.5.0`).

---

### Chapter 10: Conclusions & Future Work
*   **Location:** Page 49–50 (Section 10.1)
*   **Current PDF Text:**
    *   Describes user metadata projector mapping native profile attributes (followers, verification status, account age).
    *   Targets FakeNewsNet benchmark results ($\ge$ 95% accuracy, $\ge$ 94% F1).
*   **Actual Project Code:**
    *   The metadata projector maps the 10 text-statistic style features.
    *   Achieved **91.38% test accuracy** and **91.80% F1-score** on the 72K WELFake dataset.
*   **Recommended Update:**
    *   Update Section 10.1 to clarify that the user metadata projector processes **10 text-style statistics** (aggregated from user's posts) to form user embeddings.
    *   Update the performance results: State that the project successfully scaled to **72,094 nodes** (WELFake), achieving a highly competitive **91.38% test accuracy** and **91.80% F1-score**, proving the viability of the contrastive graph neural network architecture on large-scale datasets.
