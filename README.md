## Active Autonomous Discovery Matrix Results (Human Gene Vectors)
The multi-agent loop evaluates biological hypotheses sequentially by constraining the model to targeted genetic networks. Below are the verified metrics generated when holding out an entire tissue archetype (**"Brain"**) as a strict out-of-distribution target:

| Discovery Iteration | Target Human Gene Pathways | Proposed Biological Hypothesis | OOD C-Index (↑) | OOD MAE (↓) |
| :--- | :--- | :--- | :--- | :--- |
| **Iteration 1** | `MTOR, RPTOR, RICTOR, AKT1, PIK3CA` | Evaluating if active metabolic networks contain structural aging signatures conserved across organs. | `0.5064` | `16.4812 years` |
| **Iteration 2** | `CDKN2A, TP53, SIRT1, SIRT6, PARP1, ATM` | Testing if cell-cycle checkpoint and genomic stability markers maintain an invariant clock. | `0.5083` | `16.4875 years` |
| **Iteration 3** | `Full Cascade Matrix (All 14 Genes)` | Integrating metabolic flux variables with genomic degradation markers to stabilize trajectory predictions. | **`0.5125`** | **`16.4425 years`** |

