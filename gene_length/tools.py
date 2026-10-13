"""
Gene-length bias analysis for differential expression (edgePython).

Question answered: are longer genes more (or less) likely to go UP in the
tested group vs the reference group, after adjusting for expression level?

Workflow (2 steps)
------------------
    results = glt.run_gene_length_analysis(counts, meta, gene_length_map, ...)  # 1. analyse
    glt.plot_gene_length_bins(results)                            # 2. plot
    glt.plot_gene_length_forest(results)
    glt.plot_gene_length_sensitivity(results)

All functions work on in-memory objects (no file reading/writing).
"""

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import edgepython as ep


# ============================================================
# Helpers
# ============================================================

def strip_version(ids):
    """'ENSMUSG000001.5' -> 'ENSMUSG000001' (works on Index or Series)."""
    return pd.Index(ids).astype(str).str.split(".").str[0]


def make_gene_length_map(df, id_col="geneID", length_col="gene_length"):
    """Turn a table with gene IDs and lengths into {stripped_gene_id: length}."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"make_gene_length_map expects a DataFrame, got {type(df).__name__}.")
    for col in (id_col, length_col):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. Columns available: {list(df.columns)}. "
                "Set id_col= and length_col= to the right names."
            )
    ids = strip_version(df[id_col])
    if ids.duplicated().any():
        warnings.warn(
            f"{ids.duplicated().sum()} duplicated gene IDs in '{id_col}' (after removing "
            "version suffixes); the last length listed for each is used."
        )
    return dict(zip(ids, df[length_col]))


def _norm(x):
    return str(x).strip().lower()


def _preview(items, n=5):
    items = list(items)
    text = ", ".join(repr(str(i)) for i in items[:n])
    return text + (f", ... (+{len(items) - n} more)" if len(items) > n else "")


def _raise_if(errors):
    if errors:
        body = "\n\n".join(f"{i}. {e}" for i, e in enumerate(errors, 1))
        raise ValueError(f"\nInput check failed with {len(errors)} problem(s):\n\n{body}\n")


def check_inputs(
    counts,
    meta,
    gene_length_map,
    group_col,
    reference,
    tested,
    split_col=None,
    min_mean_cpm=1,
):
    """
    Check that counts, meta and gene_length_map are correctly formatted.

    All problems found are reported together in one error message, each with a
    suggested fix. Called automatically by run_gene_length_analysis, but you can
    also run it on its own first.

    Returns (counts aligned to meta's sample order, gene lengths Series).
    """
    errors = []
    reference, tested = str(reference), str(tested)

    # ---- 0. Object types -------------------------------------------------
    if not isinstance(counts, pd.DataFrame):
        errors.append(f"`counts` must be a pandas DataFrame (genes x samples), got {type(counts).__name__}.")
    if not isinstance(meta, pd.DataFrame):
        errors.append(f"`meta` must be a pandas DataFrame (one row per sample), got {type(meta).__name__}.")
    if isinstance(gene_length_map, pd.DataFrame):
        errors.append(
            "`gene_length_map` is a DataFrame. Convert it first with "
            "glt.make_gene_length_map(df, id_col='...', length_col='...')."
        )
    elif isinstance(gene_length_map, pd.Series):
        gene_length_map = gene_length_map.to_dict()
    elif not isinstance(gene_length_map, Mapping):
        errors.append(f"`gene_length_map` must be a dict {{gene_id: length}}, got {type(gene_length_map).__name__}.")
    if not isinstance(min_mean_cpm, (int, float, np.number)) or min_mean_cpm < 0:
        errors.append(f"min_mean_cpm must be a number >= 0, got {min_mean_cpm!r}.")
    _raise_if(errors)  # nothing else can be checked if types are wrong

    # ---- 1. counts ---------------------------------------------------------
    index_ok = True
    if counts.empty:
        errors.append("`counts` is empty (0 genes or 0 samples). Check that the file was read correctly.")
        index_ok = False
    else:
        if pd.api.types.is_integer_dtype(counts.index):
            index_ok = False
            errors.append(
                "The index of `counts` is numeric (0, 1, 2, ...), so gene IDs are probably in a "
                "column. Read the file with index_col='<gene ID column>' or use "
                "counts = counts.set_index('<gene ID column>')."
            )
        dup = counts.index[counts.index.duplicated()]
        if len(dup):
            errors.append(
                f"`counts` has {len(dup)} duplicated gene IDs (e.g. {_preview(dup.unique())}). "
                "Each gene must appear once; sum or drop the duplicates."
            )
        dupc = counts.columns[counts.columns.duplicated()]
        if len(dupc):
            errors.append(f"`counts` has duplicated sample columns: {_preview(dupc.unique())}.")
        if counts.shape[1] < 2:
            errors.append(
                f"`counts` has only {counts.shape[1]} sample column(s). If this was read from a "
                "file, the separator is probably wrong (sep='\\t' for .tsv, sep=',' for .csv)."
            )

        non_numeric = counts.columns[~counts.dtypes.map(pd.api.types.is_numeric_dtype)]
        if len(non_numeric):
            errors.append(
                f"{len(non_numeric)} column(s) of `counts` are not numeric: {_preview(non_numeric)}. "
                "Likely causes: gene IDs/annotations left as a column (move to the index), "
                "comment or footer lines in the file, or numbers stored as text."
            )
        else:
            special = counts.index[counts.index.astype(str).str.startswith("_")]
            if len(special):
                errors.append(
                    f"`counts` has {len(special)} rows whose ID starts with '_' "
                    f"(e.g. {_preview(special)}). These look like HTSeq summary rows, not genes, and "
                    "would inflate library sizes. Remove them with: "
                    "counts = counts[~counts.index.astype(str).str.startswith('_')]"
                )

            X = counts.to_numpy(dtype=float)
            nonfinite = ~np.isfinite(X)
            if nonfinite.any():
                errors.append(
                    f"`counts` contains {nonfinite.sum()} NaN/infinite values, in samples: "
                    f"{_preview(counts.columns[nonfinite.any(axis=0)])}. Remove those genes/samples, "
                    "or fillna(0) only if missing truly means zero."
                )
            elif (X < 0).any():
                errors.append(
                    f"`counts` contains negative values, in samples: "
                    f"{_preview(counts.columns[(X < 0).any(axis=0)])}. Counts must be non-negative; "
                    "log-transformed or centered/scaled data cannot be used."
                )
            else:
                empty_lib = counts.columns[X.sum(axis=0) == 0]
                if len(empty_lib):
                    errors.append(
                        f"These samples have zero total counts: {_preview(empty_lib)}. "
                        "Check for failed libraries or a misaligned table, and remove them."
                    )
                frac_nonint = np.mean(~np.isclose(X, np.round(X)))
                if frac_nonint > 0.01:
                    warnings.warn(
                        f"{frac_nonint:.1%} of values in `counts` are non-integer. This is fine for "
                        "count-like estimates (e.g. RSEM/Salmon), but if the data are log-transformed "
                        "or normalized (CPM/TPM/FPKM), results will be invalid. Use raw counts."
                    )

    # ---- 2. meta ---------------------------------------------------------------
    samples_ok = False
    if meta.empty:
        errors.append("`meta` is empty. Check that the sample table was read correctly.")
    else:
        dup = meta.index[meta.index.duplicated()]
        if len(dup):
            errors.append(f"`meta` index has duplicated sample names: {_preview(dup.unique())}.")

        if not counts.empty:
            missing = meta.index.difference(counts.columns)
            if len(missing):
                msg = (
                    f"{len(missing)} of {len(meta)} samples in the `meta` index are not columns of "
                    f"`counts`: {_preview(missing)}."
                )
                if len(meta.index.intersection(counts.index)):
                    msg += " Your `counts` looks transposed (samples in rows); use counts.T."
                    index_ok = False  # gene-ID matching can't be checked until this is fixed
                else:
                    cols = [c for c in meta.columns
                            if meta[c].astype(str).isin(counts.columns.astype(str)).mean() > 0.5]
                    if cols and isinstance(meta.index, pd.RangeIndex):
                        msg += (f" Sample names appear to be in meta column '{cols[0]}', not the index; "
                                f"use meta = meta.set_index('{cols[0]}').")
                    elif set(map(_norm, missing)) <= set(map(_norm, counts.columns)):
                        msg += " Names differ only by case or whitespace; make them identical."
                    else:
                        msg += (f" Example meta names: {_preview(meta.index, 3)}; "
                                f"example counts columns: {_preview(counts.columns, 3)}.")
                errors.append(msg)
            else:
                samples_ok = True

    # ---- 3. group / split columns ------------------------------------------------
    group_ok = False
    if group_col not in meta.columns:
        close = [c for c in meta.columns if _norm(c) == _norm(group_col)]
        errors.append(
            f"group_col='{group_col}' is not a column of `meta`. Columns available: "
            f"{list(meta.columns)}."
            + (f" Did you mean '{close[0]}'? (names are case-sensitive)" if close else "")
        )
    else:
        g = meta[group_col]
        if g.isna().any():
            errors.append(
                f"'{group_col}' has {g.isna().sum()} missing values (samples: "
                f"{_preview(g.index[g.isna()])}). Fill them or remove those samples from `meta`."
            )
        present = g.dropna().astype(str).value_counts().to_dict()
        group_ok = True
        if reference == tested:
            group_ok = False
            errors.append(f"reference and tested are both '{reference}'. They must be different groups.")
        for name, val in (("reference", reference), ("tested", tested)):
            if val not in present:
                group_ok = False
                close = [k for k in present if _norm(k) == _norm(val)]
                errors.append(
                    f"{name}='{val}' is not a value in meta['{group_col}']. Values present (with "
                    f"sample counts): {present}."
                    + (f" Did you mean '{close[0]}'? (values are case/space-sensitive)" if close else "")
                )

    split_ok = True
    if split_col is not None:
        if split_col not in meta.columns:
            split_ok = False
            errors.append(f"split_col='{split_col}' is not a column of `meta`. Columns available: {list(meta.columns)}.")
        elif split_col == group_col:
            split_ok = False
            errors.append("split_col and group_col are the same column; they must differ.")
        elif meta[split_col].isna().any():
            split_ok = False
            errors.append(
                f"'{split_col}' has {meta[split_col].isna().sum()} missing values "
                f"(samples: {_preview(meta.index[meta[split_col].isna()])})."
            )

    # ---- 4. replicates per group (per subset) ---------------------------------------
    if group_ok and split_ok and group_col in meta.columns:
        grp = meta[group_col].astype(str)
        keys = meta[split_col].astype(str) if split_col else pd.Series("all_data", index=meta.index)
        usable, skipped = [], []
        for name in keys.unique():
            sel = grp[keys == name]
            n_ref, n_tst = int((sel == reference).sum()), int((sel == tested).sum())
            (usable if n_ref >= 2 and n_tst >= 2 else skipped).append(
                f"{name} ({reference}={n_ref}, {tested}={n_tst})"
            )
        if not usable:
            errors.append(
                "No subset has at least 2 samples in both groups, which edgePython needs to estimate "
                f"dispersion. Sample counts: {_preview(skipped, 20)}."
            )
        elif skipped:
            warnings.warn(f"Subsets with fewer than 2 samples in a group will be skipped: {_preview(skipped, 20)}.")

    # ---- 5. gene_length_map ------------------------------------------------------------
    lengths = None
    if len(gene_length_map) == 0:
        errors.append("`gene_length_map` is empty.")
    else:
        s = pd.to_numeric(pd.Series(gene_length_map), errors="coerce")
        bad = ~np.isfinite(s) | (s <= 0)
        if bad.all():
            errors.append(
                "`gene_length_map` has no valid lengths (all missing, non-numeric, or <= 0). "
                f"Example entries: {dict(list(gene_length_map.items())[:3])}. Check that the right "
                "column was used in make_gene_length_map (length_col=...)."
            )
        else:
            if bad.any():
                warnings.warn(
                    f"{bad.sum()} of {len(s)} entries in gene_length_map have missing, non-numeric or "
                    f"non-positive lengths and will be ignored (e.g. {_preview(s.index[bad], 3)})."
                )
            s = s[~bad]
            s.index = strip_version(s.index)
            dups = s.index.duplicated()
            if dups.any():
                if (s.groupby(level=0).nunique() > 1).any():
                    warnings.warn(
                        "Some gene IDs have different lengths in gene_length_map once version suffixes "
                        "(.5) are removed; the first one listed is used."
                    )
                s = s[~dups]

            if s.median() < 100:
                warnings.warn(
                    f"Median gene length is only {s.median():.0f}. Lengths should be in bp, not kb."
                )

            if index_ok and not counts.empty:
                lengths = pd.Series(strip_version(counts.index).map(s).to_numpy(), index=counts.index)
                n_found = int(lengths.notna().sum())
                frac = n_found / len(lengths)
                if n_found == 0:
                    errors.append(
                        "No gene IDs in `counts` match `gene_length_map`. Example counts IDs: "
                        f"{_preview(counts.index, 3)}; example map IDs: {_preview(s.index, 3)}. "
                        "They must use the same ID type (e.g. Ensembl IDs vs gene symbols) and species."
                    )
                elif frac < 0.5:
                    warnings.warn(
                        f"Only {frac:.0%} of genes in `counts` ({n_found:,}/{len(lengths):,}) have a "
                        f"gene length. Unmatched examples: {_preview(lengths.index[lengths.isna()], 3)}. "
                        "Check the ID type and annotation version."
                    )

    _raise_if(errors)

    # ---- passed: align counts to meta ----------------------------------------------------
    counts = counts.loc[:, meta.index]
    n_unused = counts.shape[1] - int(meta[group_col].astype(str).isin([reference, tested]).sum())
    print(
        f"Input checks passed: {counts.shape[0]:,} genes, {counts.shape[1]} samples in meta"
        + (f" ({n_unused} not in {reference}/{tested}; ignored)" if n_unused else "")
        + f", {int(lengths.notna().sum()):,} genes with lengths."
    )
    return counts, lengths


def _run_de(counts, meta, group_col, reference, tested, lengths, min_mean_cpm):
    """Two-group edgePython LRT. logFC > 0 means higher in `tested`."""
    groups = meta[group_col].astype(str)
    groups = groups[groups.isin([reference, tested])]
    counts = counts[groups.index]

    # Keep genes with mean CPM above threshold in BOTH groups
    cpm = counts.div(counts.sum(axis=0), axis=1) * 1e6
    keep_genes = pd.Series(True, index=counts.index)
    for g in (reference, tested):
        keep_genes &= cpm.loc[:, groups == g].mean(axis=1) > min_mean_cpm
    filtered = counts.loc[keep_genes]
    if len(filtered) < 50:
        raise ValueError(
            f"Only {len(filtered)} genes pass the filter (mean CPM > {min_mean_cpm} in both groups). "
            "Lower min_mean_cpm, or check that counts are raw counts and samples are correct."
        )

    design = np.column_stack([
        (groups == reference).astype(int),
        (groups == tested).astype(int),
    ])

    y = ep.make_dgelist(counts=filtered.to_numpy(), group=groups.to_numpy())
    y = ep.calc_norm_factors(y, method="TMM")
    y = ep.estimate_disp(y, design=design)
    fit = ep.glm_fit(y, design=design)
    lrt = ep.glm_lrt(fit, contrast=np.array([-1.0, 1.0]))

    res = lrt["table"].copy()
    if len(res) != len(filtered):
        raise RuntimeError("edgePython output does not match the filtered genes.")
    res.index = filtered.index
    res.index.name = "gene"
    res = res.reset_index()

    res["FDR"] = multipletests(res["PValue"], method="fdr_bh")[1]
    res["mean_logCPM"] = np.asarray(ep.cpm(y, log=True)).mean(axis=1)
    res["gene_length"] = res["gene"].map(lengths)

    print(
        f"  {reference} n={(groups == reference).sum()}, {tested} n={(groups == tested).sum()} | "
        f"genes kept {len(res):,}/{len(counts):,} | DEGs (FDR<0.05): {(res['FDR'] < 0.05).sum():,}"
    )
    return res


def _prepare_genes(de, min_abs_logfc=0.0):
    """Genes with valid length/logFC; adds log10 length and the 0/1 'up' outcome."""
    d = de[["gene", "gene_length", "logFC", "mean_logCPM"]]
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[(d["gene_length"] > 0) & (d["logFC"].abs() > min_abs_logfc)].copy()
    d["log10_gene_length"] = np.log10(d["gene_length"])
    d["up"] = (d["logFC"] > 0).astype(int)
    return d


def _fit_length_model(genes):
    """Logistic regression: P(logFC > 0) ~ log10(gene_length) + mean_logCPM."""
    if len(genes) < 50:
        raise ValueError(
            f"Only {len(genes)} genes have a valid gene length and logFC (need >= 50). "
            "Check that gene IDs matched the gene-length map and that min_abs_logfc is not too strict."
        )
    if genes["up"].nunique() < 2:
        raise ValueError("All genes have the same logFC direction.")

    model = smf.glm(
        "up ~ log10_gene_length + mean_logCPM",
        data=genes,
        family=sm.families.Binomial(),
    ).fit()

    name = "log10_gene_length"
    ci = model.conf_int().loc[name]
    return {
        "n_genes": len(genes),
        "fraction_up": genes["up"].mean(),
        "beta": model.params[name],
        "odds_ratio": np.exp(model.params[name]),
        "ci_low": np.exp(ci.iloc[0]),
        "ci_high": np.exp(ci.iloc[1]),
        "pvalue": model.pvalues[name],
    }


def _add_fdr(table, by=None):
    """Benjamini-Hochberg FDR on 'pvalue', optionally within groups of column `by`."""
    table = table.copy()
    table["FDR"] = np.nan
    parts = table.groupby(by).groups.values() if by else [table.index]
    for idx in parts:
        table.loc[idx, "FDR"] = multipletests(table.loc[idx, "pvalue"], method="fdr_bh")[1]
    return table


# ============================================================
# 1. Analysis (the one function to run)
# ============================================================

def run_gene_length_analysis(
    counts,
    meta,
    gene_length_map,
    group_col,
    reference,
    tested,
    split_col=None,
    min_mean_cpm=1,
    sensitivity_thresholds=(0.0, 0.05, 0.10, 0.25),
):
    """
    Run differential expression + the gene-length logistic model.

    Parameters
    ----------
    counts : DataFrame, genes x samples (raw counts or count-like abundances).
    meta : DataFrame, one row per sample, indexed by sample name
           (matching counts' columns). Extra samples in counts are ignored.
    gene_length_map : {gene_id: length}, see make_gene_length_map().
                      Version suffixes (.5) are ignored when matching IDs.
    group_col : meta column holding the conditions (e.g. "Condition").
    reference, tested : condition labels. logFC > 0 means higher in `tested`.
    split_col : optional meta column (e.g. "tissue"). If given, the analysis is
                run separately for each value; otherwise all samples are used.
    min_mean_cpm : genes need mean CPM above this in both groups.
    sensitivity_thresholds : |logFC| cutoffs for the sensitivity analysis
                (None to skip).

    Returns
    -------
    dict with
        "summary"     one row per subset: odds ratio, 95% CI, p-value, FDR
        "sensitivity" same, repeated for each |logFC| cutoff
        "de"          {subset: edgePython results table}
        "genes"       {subset: per-gene table used by the model}
        "settings"    labels used by the plotting functions
    Pass this dict to the plot_* functions.
    """
    reference, tested = str(reference), str(tested)
    counts, lengths = check_inputs(
        counts, meta, gene_length_map, group_col, reference, tested, split_col, min_mean_cpm
    )

    if split_col is None:
        subsets = {"all_data": meta}
    else:
        subsets = {str(v): meta[meta[split_col].astype(str) == str(v)]
                   for v in meta[split_col].astype(str).unique()}

    de, genes, summary_rows, sens_rows = {}, {}, [], []

    for name, sub_meta in subsets.items():
        g = sub_meta[group_col].astype(str)
        if (g == reference).sum() < 2 or (g == tested).sum() < 2:
            continue  # already reported by check_inputs

        print(f"[{name}]")
        de[name] = _run_de(counts, sub_meta, group_col, reference, tested, lengths, min_mean_cpm)
        genes[name] = _prepare_genes(de[name])
        summary_rows.append({"subset": name, **_fit_length_model(genes[name])})

        for thr in sensitivity_thresholds or ():
            fit = _fit_length_model(_prepare_genes(de[name], thr))
            sens_rows.append({"subset": name, "min_abs_logfc": thr, **fit})

    if not summary_rows:
        raise ValueError("No subsets could be analysed.")

    summary = _add_fdr(pd.DataFrame(summary_rows))
    sensitivity = _add_fdr(pd.DataFrame(sens_rows), by="min_abs_logfc") if sens_rows else None

    return {
        "summary": summary,
        "sensitivity": sensitivity,
        "de": de,
        "genes": genes,
        "settings": {"reference": reference, "tested": tested},
    }


# ============================================================
# 2. Plots (each takes the results dict from run_gene_length_analysis)
# ============================================================

def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plain_number(value, _position):
    """Format numeric axis values without scientific notation."""
    text = f"{value:,.6f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def plot_gene_length_bins(
    results,
    subset=None,
    bin_size=500,
    figsize=(6, 4),
    ax=None,
    statistical_test="odds_ratio",
):
    """
    % of genes with positive logFC across gene-length bins (descriptive only;
    optional statistics can be added with ``statistical_test``.

    subset : which subset (e.g. tissue) to plot; optional if there is only one.
    bin_size : genes per bin (genes are sorted by length first).
    statistical_test : one of ``None``, ``"odds_ratio"``, or
        ``"linear_regression"``. ``"odds_ratio"`` reports the odds ratio,
        confidence interval, and p-value from the gene-level model.
        ``"linear_regression"`` fits ``pct_up ~ log10(length)`` across
        bins and reports the slope p-value and correlation R.
    """
    valid_tests = {None, "odds_ratio", "linear_regression"}
    if statistical_test not in valid_tests:
        raise ValueError(
            "statistical_test must be None, 'odds_ratio', or "
            "'linear_regression'."
        )

    if subset is None:
        if len(results["genes"]) > 1:
            raise ValueError(f"Choose a subset: {list(results['genes'])}")
        subset = next(iter(results["genes"]))

    genes = results["genes"][subset].sort_values("gene_length").reset_index(drop=True)
    bins = genes.groupby(np.arange(len(genes)) // bin_size).agg(
        length=("gene_length", "mean"), pct_up=("up", lambda x: 100 * x.mean())
    )

    tested, reference = results["settings"]["tested"], results["settings"]["reference"]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.axhline(50, linestyle=":", linewidth=1, color="gray")
    ax.plot(bins["length"], bins["pct_up"], marker="o", markersize=4, linewidth=1)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Gene length (bp)")
    ax.set_ylabel(f"% genes up in {tested}")
    ax.set_title(f"{tested} vs {reference}" + ("" if subset == "all_data" else f" ({subset})"))
    if statistical_test == "odds_ratio":
        s = results["summary"].set_index("subset").loc[subset]
        text = (
            f"OR = {s['odds_ratio']:.3f} "
            f"({s['ci_low']:.3f}-{s['ci_high']:.3f})\n"
            f"p = {s['pvalue']:.2g}"
        )
        ax.text(
            0.98, 0.96, text,
            transform=ax.transAxes, ha="right", va="top",
        )
    elif statistical_test == "linear_regression":
        x = np.log10(bins["length"].to_numpy())
        y = bins["pct_up"].to_numpy()
        regression = sm.OLS(y, sm.add_constant(x)).fit()
        ax.text(
            0.98, 0.96,
            f"p = {regression.pvalues[1]:.2g}\n"
            f"R = {np.corrcoef(x, y)[0, 1]:.3f}",
            transform=ax.transAxes, ha="right", va="top",
        )
    _clean_axes(ax)
    fig.tight_layout()
    return fig, ax


def plot_gene_length_forest(results, figsize=None):
    """
    Odds ratio (with 95% CI) per subset, per 10-fold increase in gene length.
    OR > 1: longer genes tend to go up in the tested group. OR < 1: down.
    """
    d = results["summary"].sort_values("odds_ratio").reset_index(drop=True)
    tested = results["settings"]["tested"]

    figsize = figsize or (6, max(2.5, 0.4 * len(d) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)

    ax.errorbar(
        d["odds_ratio"], d.index,
        xerr=[d["odds_ratio"] - d["ci_low"], d["ci_high"] - d["odds_ratio"]],
        fmt="o", capsize=3,
    )
    ax.axvline(1, linestyle=":", linewidth=1, color="gray")
    ax.set_yticks(d.index)
    ax.set_yticklabels(d["subset"])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_plain_number))
    ax.set_xlabel(f"Odds of being up in {tested}\n(per 10-fold increase in gene length)")
    _clean_axes(ax)
    fig.tight_layout()
    return fig, ax


def plot_gene_length_sensitivity(results, figsize=(6, 4)):
    """Odds ratio as genes with small |logFC| are progressively removed."""
    sens = results["sensitivity"]
    if sens is None:
        raise ValueError("Sensitivity was skipped. Re-run with sensitivity_thresholds set.")

    fig, ax = plt.subplots(figsize=figsize)
    n = sens["subset"].nunique()

    for i, (name, d) in enumerate(sens.groupby("subset")):
        x = np.arange(len(d)) + (i - (n - 1) / 2) * 0.08  # small offset so subsets don't overlap
        ax.errorbar(
            x, d["odds_ratio"],
            yerr=[d["odds_ratio"] - d["ci_low"], d["ci_high"] - d["odds_ratio"]],
            fmt="o-", capsize=3, linewidth=1, markersize=4, label=name,
        )

    ax.axhline(1, linestyle=":", linewidth=1, color="gray")
    ax.set_xticks(np.arange(sens["min_abs_logfc"].nunique()))
    ax.set_xticklabels(sorted(sens["min_abs_logfc"].unique()))
    ax.set_xlabel("Minimum |logFC| to include gene")
    ax.set_ylabel("Odds ratio (per 10-fold length)")
    if n > 1:
        ax.legend(frameon=False)
    _clean_axes(ax)
    fig.tight_layout()
    return fig, ax