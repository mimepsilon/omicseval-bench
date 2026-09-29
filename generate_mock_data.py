import pandas as pd
import numpy as np

# Create 300 mock multiomics profiles
np.random.seed(42)
n_samples = 300

data = {
    'sample_id': [f"SMP_{i}" for i in range(n_samples)],
    'target_age': np.random.uniform(20, 80, size=n_samples),
    'tissue_type': np.random.choice(['Blood', 'Brain', 'Liver'], size=n_samples),
    'batch_id': np.random.choice(['Batch_A', 'Batch_B'], size=n_samples)
}

# Add 10 mock gene expression columns
for g in range(1, 11):
    data[f'gene_{g}'] = np.random.normal(0, 1, size=n_samples)

df = pd.DataFrame(data)
df.to_csv("mock_multiomics.csv", index=False)
print("✅ Created 'mock_multiomics.csv' with balanced structural context metadata.")
