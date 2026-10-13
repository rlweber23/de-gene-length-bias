# gene-length

`gene-length` provides tools for testing whether gene length is associated with
differential expression. It includes input checks, gene-length tables from GTF
files, analysis, and plots.

## Installation

Using a clean Conda environment is recommended:

```bash
conda create -n genelength python=3.11
conda activate genelength
pip install git+https://github.com/rlweber23/de-gene-length-bias.git
```

Install Jupyter only if you plan to use the notebook tutorial:

```bash
pip install jupyter
```

## Quick start

In a notebook or Python script, import the package and prepare:

- a counts table with genes as rows and samples as columns;
- a sample metadata table with one row per sample; and
- a gene-length table with `geneID` and `gene_length` columns.

```python
import pandas as pd
import gene_length as glt

gene_lengths = pd.read_csv("gene_lengths.csv")
gene_length_map = glt.make_gene_length_map(
    gene_lengths,
    id_col="geneID",
    length_col="gene_length",
)

meta = pd.read_csv("sample_table.csv", index_col="sampleName")
counts = pd.read_csv("counts.tsv", sep="\t", index_col="feature")

results = glt.run_gene_length_analysis(
    counts,
    meta,
    gene_length_map,
    group_col="Condition",
    reference="Control",
    tested="Treatment",
)
```

To create a gene-length table from a GTF file:

```python
gene_lengths = glt.generate_gene_length_table(
    "genes.gtf",
    output_file="gene_lengths.csv",
)
```

For a complete example, see
[`gene_length_analysis_tutorial.ipynb`](gene_length_analysis_tutorial.ipynb).