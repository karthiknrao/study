# LLMs for DNA/RNA Modeling: A Deep Dive

## 1. Core Idea: DNA/RNA as Language

The genome is a sequence of four nucleotides (A, T/C, G, C) — analogous to a language with a 4-letter alphabet. Just as NLP models learn grammar and semantics from text, genomic LLMs learn the "grammar of life" — regulatory codes, functional motifs, splice sites, expression patterns — by training on massive corpora of DNA/RNA sequences using the same self-supervised techniques: **masked language modeling** (BERT-style) and **causal/autoregressive language modeling** (GPT-style).

---

## 2. Key Models (The Landscape)

### DNA Foundation Models

| Model | Params | Key Innovation | Training Data |
|-------|--------|----------------|---------------|
| **[Evo 2](https://arcinstitute.org/tools/evo)** (Arc Institute) | 40B | Largest bio AI model to date; single-nucleotide resolution; 1M bp context length; generative design capability | 9.3 trillion nucleotides from 128K+ genomes across all domains of life |
| **[DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2)** | 117M | BPE tokenization (replacing k-mers), ALiBi positional encoding; 21x fewer params than competitors | Multi-species genomes |
| **[Nucleotide Transformer](https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1.full-text) v2/v3** | up to 2.5B | Strong general-purpose representations across species; v3 adds joint sequence-function learning | 3,202 human genomes + multi-species data |
| **[HyenaDNA](https://github.com/HazyResearch/hyena-dna)** | up to 1.6M | Attention-free architecture (Hyena operators); sub-quadratic scaling; context lengths up to 1M tokens | Human reference genome |
| **[Caduceus](https://caduceus-dna.github.io/)** | up to 1.5B | Bi-directional, equivariant long-range modeling; Mamba-based SSM architecture | Multi-species |
| **[AlphaGenome](https://www.alphagenomedocs.com/)** (DeepMind) | — | Multi-modal (RNA-seq, DNA, Hi-C contact maps); unified regulatory code prediction | — |

### RNA-Specific Models

| Model | Focus |
|-------|-------|
| **[RiNALMo](https://www.nature.com/articles/s41467-025-60872-5)** | General-purpose RNA LM; state-of-the-art secondary structure prediction; generalizes to unseen RNA families |
| **[RNA-FM](https://github.com/ml4bio/RNA-FM)** | RNA foundation model producing general-purpose embeddings for structure/function prediction |
| **LncRNA-BERT** | Classifies coding vs. non-coding RNA |
| **NuFold** | RNA 3D structure prediction with unprecedented accuracy |

---

## 3. Architecture Innovations

Standard Transformers struggle with genomics because genomes are *extremely* long (human genome: ~3.2 billion bp). Key architectural adaptations:

- **Hyena operators** (HyenaDNA): Replace attention with long convolutions, enabling sub-quadratic scaling and context lengths up to 1M tokens. Training 160x faster than vanilla Transformers.
- **StripedHyena 2** (Evo 2): Near-linear scaling of compute/memory with context length. Enables modeling at single-nucleotide resolution.
- **State Space Models / Mamba** (Caduceus): Bi-directional equivariant SSMs that capture long-range dependencies efficiently.
- **BPE tokenization** (DNABERT-2): Replaces fixed k-mer tokenization with learned byte-pair encoding, adapting vocabulary to genomic patterns.
- **Reversible residual layers**: Reduce memory footprint for very deep models on long sequences.

---

## 4. Major Applications

### A. Variant Effect Prediction (VEP)
Predict whether a genetic mutation (SNP, indel) is benign or pathogenic. Evo 2 can zero-shot predict the functional impact of variants across the human genome, including clinically relevant mutations like BRCA1 variants. This has direct applications in **precision medicine** and **genetic diagnostics**.

### B. Gene Expression Prediction
Models like Enformer and AlphaGenome predict how DNA sequence regulates gene expression (which genes are turned on/off in different cell types). This connects genotype to phenotype and is crucial for understanding **gene regulation** and **disease mechanisms**.

### C. RNA Structure Prediction
RNA folds into complex 2D and 3D structures that determine its function. RiNALMo and related models predict:
- **Secondary structure** (base-pairing patterns) — generalizing to unseen RNA families
- **3D structure** (NuFold) — critical for understanding function and drug binding
- Enables **RNA-targeted drug design** by identifying binding pockets

### D. Synthetic Biology & Genome Design
Evo 2 can *generate* novel DNA sequences that are biologically functional — designing synthetic genetic elements, regulatory sequences, and even guiding the design of entire bacterial genomes. This is the generative frontier: **writing** new biology, not just reading it.

### E. Protein Design
Since DNA encodes proteins, models like Evo naturally extend to protein-level predictions. Evo 2 can predict protein function from the underlying DNA and even guide protein engineering.

### F. Regulatory Element Identification
Detecting enhancers, promoters, splice sites, and other functional elements in non-coding DNA (which comprises ~98% of the human genome). This was one of the original breakthroughs of DNABERT.

### G. Drug Discovery
- **RNA-targeted therapeutics**: LLMs identify drug-binding sites on RNA molecules
- **Virtual screening**: LLM embeddings enable rapid search of chemical space
- **Target identification**: Predicting which genetic variants are druggable

### H. Comparative Genomics & Evolution
Evo 2, trained across all domains of life, can identify evolutionary conserved elements and predict cross-species functional equivalence — shedding light on evolutionary constraints and innovations.

---

## 5. Training Paradigm

```
Raw DNA/RNA sequences (trillions of nucleotides)
        ↓
   Tokenization (single-nucleotide, k-mer, or BPE)
        ↓
   Self-supervised pretraining:
     - Masked LM: predict masked nucleotides (BERT-style)
     - Causal LM: predict next nucleotide (GPT-style)
        ↓
   Pretrained foundation model (universal genomic representations)
        ↓
   Fine-tuning / probing for downstream tasks:
     - Classification (promoter, enhancer, splice site)
     - Regression (expression level, binding affinity)
     - Generation (novel sequence design)
     - Structure prediction (2D/3D folding)
```

---

## 6. Benchmarks & Current Limitations

The [GUE benchmark](https://github.com/MAGICS-LAB/DNABERT_2) (Genome Understanding Evaluation) and the [Nucleotide Transformer benchmark](https://www.nature.com/articles/s41467-025-65823-8) provide standardized evaluations across 28+ tasks. A recent [comprehensive benchmarking study](https://www.nature.com/articles/s41467-025-65823-8) of five major models found:

- No single model dominates across all tasks
- Larger models generally perform better but with diminishing returns
- **Biological blind spots remain**: a [2026 study](https://www.biorxiv.org/content/10.64898/2026.03.10.710786v1) showed Evo 2 and DNABERT-2 still struggle with clinically relevant variant prediction for well-characterized biological constraints
- RNA structure prediction still lags behind protein structure prediction (no "AlphaFold moment" for RNA yet)
- Training data bias: most models are trained on reference genomes, underrepresenting population-level diversity

---

## 7. The Frontier: What's Coming

- **Multi-modal genomic models**: Integrating DNA sequence with epigenomic data (ATAC-seq, ChIP-seq), expression data, and 3D chromatin structure
- **Whole-genome generation**: Designing synthetic organisms from scratch
- **Clinical integration**: Real-time variant interpretation in clinical genomics pipelines
- **RNA therapeutics design**: LLM-guided design of mRNA vaccines, antisense oligonucleotides, and RNA-targeting small molecules
- **Personalized medicine**: Predicting individual-specific drug responses from genomic sequence

---

## Sources

- [Evo 2 - Nature (2026)](https://www.nature.com/articles/s41586-026-10176-5)
- [Arc Institute - Evo 2](https://arcinstitute.org/news/evo2)
- [DNABERT-2 - GitHub](https://github.com/MAGICS-LAB/DNABERT_2)
- [RiNALMo - Nature Communications (2025)](https://www.nature.com/articles/s41467-025-60872-5)
- [HyenaDNA - GitHub](https://github.com/HazyResearch/hyena-dna)
- [Caduceus Project Page](https://caduceus-dna.github.io/)
- [Benchmarking DNA Foundation Models - Nature Comms (2025)](https://www.nature.com/articles/s41467-025-65823-8)
- [AlphaGenome - DeepMind](https://www.alphagenomedocs.com/)
- [LLMs in Bioinformatics Survey - arXiv (2025)](https://arxiv.org/abs/2503.04490)
- [Gene-LLMs Survey - Frontiers in Genetics (2025)](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2025.1634882/full)
- [DNA Dialect: Guide to Genomic LMs - EMBO (2025)](https://link.springer.com/article/10.1038/s44320-025-00184-4)
- [Biological Blind Spots in Evo2 - bioRxiv (2026)](https://www.biorxiv.org/content/10.64898/2026.03.10.710786v1)
- [Nucleotide Transformer v3 - bioRxiv (2025)](https://www.biorxiv.org/content/10.64898/2025.12.22.695963v1.full-text)
- [RNA-FM - GitHub](https://github.com/ml4bio/RNA-FM)
- [NuFold - SynBioBeta](https://www.synbiobeta.com/read/new-ai-model-predicts-rna-structures-with-unprecedented-accuracy)
