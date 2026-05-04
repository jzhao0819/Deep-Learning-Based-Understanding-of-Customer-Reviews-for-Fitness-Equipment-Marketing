# Deep Learning–Based Understanding of Customer Reviews for Fitness Equipment Marketing

This project applies deep learning (BERT/DistilBERT) to analyze Amazon fitness equipment reviews, comparing its performance against traditional machine learning while extracting actionable marketing insights and analyzing model limitations on neutral sentiment.

## Project Overview

We address three research questions:

- **RQ1 – Model Comparison:** Does DistilBERT outperform TF‑IDF + Logistic Regression?
- **RQ2 – Marketing Insights:** What customer themes (selling points vs. pain points) can be discovered via semantic clustering?
- **RQ3 – Error Analysis:** Which linguistic patterns cause DistilBERT to fail on neutral (3‑star) reviews?

## Dataset

- **Source:** Amazon Reviews 2023 – Sports & Outdoors category ([McAuley-Lab](https://amazon-reviews-2023.github.io/))
- **Raw size:** 19.6M reviews, 2.63 GB
- **Two subsets created:**
  - *Balanced dataset* (50,000 reviews): 15k negative, 15k neutral, 20k positive – for training & validation
  - *Random dataset* (46,352 reviews): real‑world distribution (78% positive, 15% negative, 7% neutral) – for final test evaluation

## Methodology

### RQ1 – Model Comparison
- **Baseline:** TF‑IDF vectorization + Logistic Regression (GridSearchCV)
- **Deep learning:** DistilBERT (`distilbert-base-uncased`, 2 epochs, batch size 8)
- **Evaluation:** Accuracy, Precision, Recall, F1‑score (reported on balanced validation & random test)

### RQ2 – Marketing Insights (Semantic Clustering)
- **Process:**  
  Sample 5k positive + 5k negative reviews → BERT embeddings (384‑dim) → UMAP → K‑Means (k=3) → bigram keyword extraction
- **Key design:** Cluster positive and negative reviews separately to separate selling points from pain points

### RQ3 – Error Analysis on Neutral Reviews
- Compare linguistic markers (negation, contrastive, conditional) in neutral reviews that DistilBERT correctly vs. incorrectly classified
- Statistical test: Chi‑square independence

## Key Results

### RQ1 – Model Comparison
| Model | Balanced Val Accuracy | Balanced Val F1 | Random Test Accuracy |
|-------|----------------------|-----------------|---------------------|
| TF‑IDF + LR | ~76% | ~0.75 | ~76% |
| DistilBERT | **+0.93%** | **+1.31%** | ~76% |

- Class imbalance on random test (78% positive) masks DistilBERT’s true advantage.

### RQ2 – Marketing Insights
| Cluster | Size | Theme | Key Bigrams |
|---------|------|-------|--------------|
| **Pain – Cluster 1** (highest) | 2,267 | Durability + Customer Service | "waste money" (102×), "stopped working", "customer service" |
| **Pain – Cluster 0** | 1,679 | Sizing / Perception gap | "doesn't fit", "looks like" |
| **Pain – Cluster 2** | 826 | Sealing / Air retention | "hold air", "started leaking" |
| **Selling points** | 4,818 | Lightweight, value, comfort | "light quality", "great gift", "comfortable fit" |

### RQ3 – Linguistic Error Analysis on Neutral Reviews
| Linguistic Pattern | With Pattern | Without Pattern | Effect | p‑value |
|--------------------|--------------|----------------|--------|---------|
| Negation (not, never, no) | 55.4% error | 45.3% error | **+10.1%** errors | 0.016 |
| Contrastive (but, however) | 31.6% error | 47.6% error | **−16.1%** errors | <0.001 |
| Conditional (if, would) | 40.4% error | 47.6% error | **−7.2%** errors | 0.0004 |

- Negation systematically increases misclassification; contrastive cues reduce errors.


