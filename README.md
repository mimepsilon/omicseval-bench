An open-source multi-modal PyTorch validation framework engineered to audit deep learning models under strict biological out-of-distribution (OOD) domain shifts. 

Inspired by the evaluation philosophies established in recent foundation model benchmarks like *LongevityBench (Cell)*, this repository implements zero-leakage homology-separated splitting pipelines alongside structural context-token embedding mechanics.

## Key Architectural Principles
* **Homology-Aware Splitting:** Prevents data leakage by ensuring train/test boundaries split strictly along whole biological context layers (e.g., holding out entire tissue categories).
* **Multi-Modal Feature Attention:** Maps separate continuous gene-expressions and discrete categorical sequencing batch or metadata items into unified sequence arrays using a `PyTorch` Transformer Encoder block to handle context dependency natively.

## Getting Started
```bash
# Clone the repository
git clone https://github.com
cd omicseval-bench

# Install required tools
pip install -r requirements.txt

# Run the raw data collection pipeline and the benchmark evaluation loops
python fetch_gex_data.py
python main.py
```

