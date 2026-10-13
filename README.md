# gene-length

`gene-length` provides tools for testing whether gene length is associated with
differential expression. It includes input checks, gene-length tables from GTF
files, analysis, and plots.

## Installation

Using a clean Conda environment is recommended:

```bash
conda create -n genelength python=3.11
conda activate genelength
```
Install:
```bash
pip install git+https://github.com/rlweber23/de-gene-length-bias.git
```

Install Jupyter if you plan to use the notebook tutorials:

```bash
pip install jupyter
```

## Tutorials

The tutorials walk through the workflow in two steps:

1. [Generate a gene-length CSV from a GTF](tutorials/step1_get_gene_lengths_from_gtf.ipynb)
2. [Run the gene-length analysis](tutorials/step2_gene_length_analysis_tutorial.ipynb)

Use the same GTF annotation that was used during RNA-seq read mapping. The
annotation version should match as well, since gene IDs and gene coordinates
need to be consistent with the mapped data. For example, a GENCODE mouse M22
annotation can be downloaded with:

```bash
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M22/gencode.vM22.annotation.gtf.gz
gzip -d gencode.vM22.annotation.gtf.gz
```