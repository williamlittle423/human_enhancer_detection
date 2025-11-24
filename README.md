# Enhancer Classification with Self-Attention

This project investigates whether a lightweight self-attention model can learn biologically meaningful patterns from genomic sequences to classify human enhancer regions. The model operates on k-mer tokenized DNA sequences (4-mers) embedded using pretrained dna2vec vectors, followed by a single-head attention block and a binary classification head.

The research question is whether a compact attention-based architecture can capture both local motifs and long-range contextual dependencies strongly enough to classify enhancer sequences above random chance (ideally much better, but I have just started).

## Dataset

The project uses the **human_enhancers_ensembl** dataset from *Genomic Benchmarks* (Grešová et al., 2023).  
It contains 154,640 sequences with a 50/50 split of positive (FANTOM5 enhancer regions) and negative (length-matched random genomic regions) examples.  

The mean sequence length is 270 bp with a standard deviation of 170 bp.

## Model Overview

The pipeline consists of:

1. **k-mer tokenization** (4-mers, stride 1)
2. **Embedding lookup** using pretrained dna2vec vectors (100-dimensional)
3. **Single-head self-attention block**
4. **Learned [CLS] token for classification** 
5. **Binary classifier head**
6. **Training with BCEWithLogitsLoss**

## Objectives

- Implement a minimal attention-based model for genomic sequence classification.
- Evaluate performance on the human_enhancers_ensembl benchmark.
- Measure accuracy, ROC-AUC, precision, and recall.
- Assess whether learned attention patterns correspond to meaningful biological features.

## References

Grešová et al. *Genomic benchmarks: a collection of datasets for genomic sequence classification.* BMC Genomic Data, 24(1):25, 2023.

Ji et al. *DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language.* Bioinformatics, 37(15):2112–2120, 2021.

Vaswani et al. *Attention is All You Need.* NeurIPS 2017.
