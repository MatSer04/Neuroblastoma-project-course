# How to import : exec(open(f"/home/felix/projects/translation_control/tools/isoform_tools.py").read())

import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from typing import Optional, Union
import warnings

def get_isoforms_gene_list(gene_list, adata):
    # Fetches all isoforms from a list of genes
    isoforms_list = []
    for gene in gene_list:
        isoforms = adata.var[adata.var.gene==gene].index.tolist()
        isoforms_list = isoforms_list + isoforms
    return isoforms_list


def fetch_isoform_varince(adata, 
                  group_by: str, 
                  split_by: Optional[str] = None,
                  layer: Optional[str] = None,
                  extended: bool = False,
                  gex_threshold: int = None,
                  gene_name_col: str = 'gene_name',
                  transcript_name_col: str = 'transcript_name',
                  chunk_size: int = 1000) -> pd.DataFrame:
    """
    # Adjusted from the find_isoform_function. Essentially truncated version to get an intermediate metric.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing single-cell expression data
    group_by : str
        Column name in adata.obs for grouping cells
    split_by : str, optional
        Column name in adata.obs for splitting analysis into subsets
    layer : str, optional
        Layer to use for counts (if None, uses adata.X)
    extended : bool
        Whether to return extended results including filtered genes
    gex_threshold : int
        Whether to threshold and if so what is the summed GEX to threshold for.
    gene_name_col : str
        Column name in adata.var containing gene names
    transcript_name_col : str
        Column name in adata.var containing transcript/isoform names
    chunk_size : int
        Number of transcripts to process at once (default 1000)
        
    Returns:
    --------
    pd.DataFrame
        Variance of each isoform based on grouping.
    """
    
    # Validation
    if group_by not in adata.obs.columns:
        raise ValueError(f"Error: {group_by} not found in AnnData object!")
    
    if gene_name_col not in adata.var.columns:
        raise ValueError(f"Error: {gene_name_col} not found in adata.var!")
        
    if transcript_name_col not in adata.var.columns:
        raise ValueError(f"Error: {transcript_name_col} not found in adata.var!")
    
    if gex_threshold is not None:
        print(f"Threshold for GEX set to: {gex_threshold}")
        adata = adata[:,adata.var.sum_gex > gex_threshold]

    
    if split_by is None:
        print("Processing AnnData object...")
        return _fetch_variance_process_simple(adata, group_by, layer, extended,
                              gene_name_col, transcript_name_col, chunk_size)
    
    # Have not implemented this yet
    #else:
    #    if split_by not in adata.obs.columns:
    #        raise ValueError(f"Error: {split_by} not found in AnnData object!")
    #    
    #    print("Splitting AnnData object and processing each subset...")
    #    return _process_split(adata, group_by, split_by, layer, extended,
    #                         gene_name_col, transcript_name_col, chunk_size)


def _fetch_variance_process_simple(adata, group_by: str, layer: Optional[str],
                                   extended: bool, gene_name_col: str, 
                                   transcript_name_col: str, 
                                   chunk_size: int = 1000) -> pd.DataFrame:
    """
    Process AnnData object without splitting and return variances.
    """
    # Get aggregated counts table
    counts_tab = _extract_agg_tab(adata, group_by, layer, gene_name_col, 
                                 transcript_name_col, chunk_size)
    
    
    genes = counts_tab['gene_name'].unique()
    
    # Test for DUIs. 
    # Process genes in batches to manage memory
    batch_size = 100
    for i in range(0, len(genes), batch_size):
        batch_genes = genes[i:i + batch_size]
        
        for gene in batch_genes:
            gene_tab = counts_tab[counts_tab['gene_name'] == gene].copy()
            
            if len(gene_tab['transcript_name'].unique()) > 1:
                # Multiple isoforms 
                gene_results = calculate_variance(gene_tab)
            else:
                continue

            if 'combined' in locals():
                combined = pd.concat([combined, gene_results])
            else:
                combined = gene_results

        # Progress indicator and memory cleanup
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {min(i + batch_size, len(genes))}/{len(genes)} genes...")
    return combined


def calculate_variance(gene_tab: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate variance for each isoform for a gene.
    """
    # Create contingency table
    contingency = gene_tab.pivot(index='transcript_name', 
                                columns=gene_tab.columns[0],  # group_by column
                                values='count').fillna(0)
    
    # Calculate proportions for filtering
    total_counts = contingency.sum().sum()
    transcript_props = contingency.sum(axis=1) / total_counts # Perhaps this can be returned as well. 
    
    # Calculate variance of proportions across groups
    prop_by_group = contingency.div(contingency.sum(axis=0), axis=1)
    transcript_vars = prop_by_group.var(axis=1) # Variance across groups for each isoform.
    
    return transcript_vars


def find_isoforms(adata, 
                  group_by: str, 
                  split_by: Optional[str] = None,
                  layer: Optional[str] = None,
                  threshold_pval: float = 0.05,
                  threshold_abund: float = 0.1, 
                  threshold_var: float = 0.05,
                  extended: bool = False,
                  gex_threshold: int = None,
                  threshold_abund_absolute: int = None,
                  gene_name_col: str = 'gene_name',
                  transcript_name_col: str = 'transcript_name',
                  chunk_size: int = 1000) -> pd.DataFrame:
    """
    # Made using Claude Sonnet 4 from SCALPEL pipeline. Some modifications were made manually by me as well and fixes.

    Find differentially expressed isoforms across cell groups in single-cell data.
    Memory-efficient version for large datasets.
    After reviewing and testing the code there where was a major issue with p-adjustment but it should
    be fixed now. Otherwise I've checked the entire workflow and I feel that it is good and does what I intend it to do. 
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing single-cell expression data
    group_by : str
        Column name in adata.obs for grouping cells
    split_by : str, optional
        Column name in adata.obs for splitting analysis into subsets
    layer : str, optional
        Layer to use for counts (if None, uses adata.X)
    threshold_pval : float
        P-value threshold for significance (default 0.05)
    threshold_abund : float
        Abundance threshold for filtering (default 0.1)
    threshold_var : float
        Variance threshold for filtering (default 0.05)
    extended : bool
        Whether to return extended results including filtered genes
    gex_threshold : int
        Whether to threshold and if so what is the summed GEX to threshold for.
    threshold_abund_absolute : int
        Set a absolute threshold for isoform expression instead of a relative one.
    gene_name_col : str
        Column name in adata.var containing gene names
    transcript_name_col : str
        Column name in adata.var containing transcript/isoform names
    chunk_size : int
        Number of transcripts to process at once (default 1000)
        
    Returns:
    --------
    pd.DataFrame
        Results table with differential isoform analysis
    """
    
    # Validation
    if group_by not in adata.obs.columns:
        raise ValueError(f"Error: {group_by} not found in AnnData object!")
    
    if gene_name_col not in adata.var.columns:
        raise ValueError(f"Error: {gene_name_col} not found in adata.var!")
        
    if transcript_name_col not in adata.var.columns:
        raise ValueError(f"Error: {transcript_name_col} not found in adata.var!")
    
    if gex_threshold is not None:
        print(f"Threshold for GEX set to: {gex_threshold}")
        adata = adata[:,adata.var.sum_gex > gex_threshold]
    
    if split_by is None:
        print("Processing AnnData object...")
        return _process_simple(adata, group_by, layer, threshold_pval, 
                              threshold_abund, threshold_var, extended,
                              gene_name_col, transcript_name_col, chunk_size, threshold_abund_absolute)
    else:
        if split_by not in adata.obs.columns:
            raise ValueError(f"Error: {split_by} not found in AnnData object!")
        
        print("Splitting AnnData object and processing each subset...")
        return _process_split(adata, group_by, split_by, layer, threshold_pval,
                             threshold_abund, threshold_var, extended,
                             gene_name_col, transcript_name_col, chunk_size, threshold_abund_absolute)


def _extract_agg_tab(adata, group_by: str, layer: Optional[str] = None,
                     gene_name_col: str = 'gene_name', 
                     transcript_name_col: str = 'transcript_name',
                     chunk_size: int = 1000) -> pd.DataFrame:
    """
    Extract aggregated counts table grouped by specified variable.
    Memory-efficient version that processes genes in chunks.
    After reviewing code I am pretty sure this works well. 
    """
    print("Aggregating counts in conditions...")
    
    # Get expression data
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    
    # Get grouping information
    groups = adata.obs[group_by].values
    unique_groups = np.unique(groups)
    
    # Prepare gene and transcript mapping
    var_info = adata.var[[gene_name_col, transcript_name_col]].copy()
    var_info['transcript_id'] = var_info.index
    
    # Process in chunks to avoid memory issues
    all_results = []
    n_genes = X.shape[1]
    
    print(f"Processing {n_genes} transcripts in chunks of {chunk_size}...")
    
    for start_idx in range(0, n_genes, chunk_size):
        end_idx = min(start_idx + chunk_size, n_genes)
        
        # Get chunk of expression data
        if hasattr(X, 'toarray'):
            X_chunk = X[:, start_idx:end_idx].toarray()
        else:
            X_chunk = X[:, start_idx:end_idx]
        
        # Get transcript info for this chunk
        chunk_var_info = var_info.iloc[start_idx:end_idx].copy()
        
        # Aggregate counts for each transcript and group combination
        chunk_results = []
        
        # Iterate over transcripts in this chunk
        for i, (transcript_id, row) in enumerate(chunk_var_info.iterrows()):
            transcript_counts = X_chunk[:, i]
            gene_name = row[gene_name_col]
            transcript_name = row[transcript_name_col]
            
            # Skip if all counts are zero
            if transcript_counts.sum() == 0:
                continue
            
            # Aggregate by group
            # This gets the sum of counts for a particular group for this transcript.
            for group in unique_groups:
                group_mask = groups == group
                group_count = transcript_counts[group_mask].sum()
                
                if group_count > 0:  # Only keep non-zero counts
                    chunk_results.append({
                        group_by: group,
                        'gene_name': gene_name,
                        'transcript_name': transcript_name,
                        'count': group_count
                    })
        
        if chunk_results:
            all_results.extend(chunk_results)
        
        # Progress indicator
        if (start_idx // chunk_size + 1) % 10 == 0:
            print(f"Processed {end_idx}/{n_genes} transcripts...")
    
    print("Combining results...")
    agg_counts = pd.DataFrame(all_results)
    
    # Clean up memory
    del all_results
    
    return agg_counts


def _perform_chi2_test(gene_tab: pd.DataFrame, 
                      threshold_abund: float, 
                      threshold_var: float,
                      threshold_abund_absolute: int = None) -> pd.DataFrame:
    """
    Perform chi-square test for isoform usage differences and apply filtering.
    After reviewing code I am pretty sure this works well.
    """
    # Create contingency table
    # Essentially the matrix of transcript sum by group
    contingency = gene_tab.pivot(index='transcript_name', 
                                columns=gene_tab.columns[0],  # group_by column
                                values='count').fillna(0)
    
    # Calculate proportions for filtering
    total_counts = contingency.sum().sum()
    transcript_props = contingency.sum(axis=1) / total_counts # Proportions of each isoform across groups.
    
    # Calculate variance of proportions across groups
    prop_by_group = contingency.div(contingency.sum(axis=0), axis=1)
    transcript_vars = prop_by_group.var(axis=1) # Variance across groups for each isoform.
    
    # Apply thresholds
    if threshold_abund_absolute != None: # If you use absolute abundance change the threshold. 
        threshold_abund = threshold_abund_absolute/total_counts # Convert the absolute  abundance to a proportional threshold. 

    thr_abund = transcript_props >= threshold_abund
    thr_var = transcript_vars >= threshold_var
    
    # Perform chi-square test if we have multiple groups and transcripts
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        try:
            # chi2_contingency will test if the distribution of transcript isopforms is the same 
            # across all clusters or if the transcription isoform usage differs significantly between clusters.
            # If it passes then it would suggest differential isoform usage.
            chi2, p_value, dof, expected = chi2_contingency(contingency) 
            statistic = chi2
        except:
            p_value = np.nan
            statistic = np.nan
    else:
        p_value = np.nan
        statistic = np.nan

    # Check if expression is dominated by one single isoform.
    # If 0 or 1> then the gene will either be filtered away or remain with
    # the solo_isoform = False attribute in the resulting df. 
    if len(thr_abund[thr_abund]) == 1:
        solo_isoform = True
    else:
        solo_isoform = False
    
    # Prepare results
    results = []
    for transcript in contingency.index:
        for group in contingency.columns:
            results.append({
                'gene_name': gene_tab['gene_name'].iloc[0],
                'transcript_name': transcript,
                gene_tab.columns[0]: group,  # group_by column
                'count': contingency.loc[transcript, group],
                'thr_abund': thr_abund[transcript],
                'thr_var': thr_var[transcript],
                'p_value': p_value,
                'statistic': statistic,
                'solo_isoform' : solo_isoform
            })
    
    return pd.DataFrame(results)


def _process_simple(adata, group_by: str, layer: Optional[str],
                   threshold_pval: float, threshold_abund: float, 
                   threshold_var: float, extended: bool, 
                   gene_name_col: str, transcript_name_col: str,
                   chunk_size: int = 1000, threshold_abund_absolute : int = None) -> pd.DataFrame:
    """
    Process AnnData object without splitting.
    After correction with p-values I am pretty sure it works as intended now.
    """
    # Get aggregated counts table
    counts_tab = _extract_agg_tab(adata, group_by, layer, gene_name_col, 
                                 transcript_name_col, chunk_size)
    
    # Process filtering and comparison test for each gene
    print("Performing filtering and comparison test...")
    
    all_results = []
    genes = counts_tab['gene_name'].unique()
    
    # Test for DUIs. 
    # Process genes in batches to manage memory
    batch_size = 100
    for i in range(0, len(genes), batch_size):
        batch_genes = genes[i:i + batch_size]
        batch_results = []
        
        for gene in batch_genes:
            gene_tab = counts_tab[counts_tab['gene_name'] == gene].copy()
            
            if len(gene_tab['transcript_name'].unique()) > 1:
                # Multiple isoforms - perform test
                gene_results = _perform_chi2_test(gene_tab, threshold_abund, threshold_var, threshold_abund_absolute)
            else:
                # Single isoform - mark as passed thresholds but no test
                gene_results = gene_tab.copy()
                gene_results['thr_abund'] = True
                gene_results['thr_var'] = True
                gene_results['p_value'] = np.nan
                gene_results['statistic'] = np.nan
            
            batch_results.append(gene_results)
        
        if batch_results:
            all_results.extend(batch_results)
        
        # Progress indicator and memory cleanup
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {min(i + batch_size, len(genes))}/{len(genes)} genes...")
    
    print("Combining results...")
    # Combine results
    all_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate adjusted p-values
    print("Adjusting p-values...")
    
    # Initialize adjusted p-values
    all_results['p_value_adjusted'] = np.nan
    
    # Identify genes that PASS both thresholds and have valid p-values
    mask_pass = (all_results['thr_abund']) & (all_results['thr_var']) & (~all_results['p_value'].isna())
    
    # Only adjust p-values for genes that pass quality thresholds
    if mask_pass.sum() > 0:
        valid_pvals = all_results.loc[mask_pass, 'p_value']
        _, adjusted_pvals, _, _ = multipletests(valid_pvals, method='fdr_bh')
        all_results.loc[mask_pass, 'p_value_adjusted'] = adjusted_pvals
    
    # Set adjusted p-values to 1.0 for genes that fail thresholds (indicating non-significance)
    mask_fail = (~all_results['thr_abund']) | (~all_results['thr_var'])
    all_results.loc[mask_fail, 'p_value_adjusted'] = 1.0
    
    # Sort results
    all_results = all_results.sort_values(['p_value_adjusted', 'gene_name', 'transcript_name'])
    all_results = all_results.drop_duplicates().reset_index(drop=True)
    
    # Apply p-value threshold if not extended
    if not extended:
        all_results = all_results[all_results['p_value_adjusted'] < threshold_pval]
        all_results = all_results.drop(columns=['thr_abund', 'thr_var'])
    
    return all_results


def _process_split(adata, group_by: str, split_by: str, layer: Optional[str],
                  threshold_pval: float, threshold_abund: float, 
                  threshold_var: float, extended: bool,
                  gene_name_col: str, transcript_name_col: str,
                  chunk_size: int = 1000, threshold_abund_absolute : int = None) -> pd.DataFrame:
    """
    Process AnnData object with splitting by another variable.
    Looks ok. 
    """
    split_values = adata.obs[split_by].unique()
    all_split_results = []
    
    print("Generation of aggregate counts...")
    
    for split_val in split_values:
        print(f"Processing {split_val}...")
        
        # Subset data
        mask = adata.obs[split_by] == split_val
        adata_subset = adata[mask].copy()
        
        # Process subset
        subset_results = _process_simple(adata_subset, group_by, layer,
                                       threshold_pval, threshold_abund, 
                                       threshold_var, True,  # Always extended for splits
                                       threshold_abund_absolute, 
                                       gene_name_col, transcript_name_col, chunk_size)
        
        # Add split information
        subset_results[split_by] = split_val
        all_split_results.append(subset_results)
        
        # Clean up memory
        del adata_subset
    
    # Combine results from all splits
    combined_results = pd.concat(all_split_results, ignore_index=True)
    
    # Apply final filtering if not extended
    if not extended:
        combined_results = combined_results[combined_results['p_value_adjusted'] < threshold_pval]
        combined_results = combined_results.drop(columns=['thr_abund', 'thr_var'])
    
    return combined_results


# Example usage:
"""
# Assuming you have an AnnData object with isoform-level data
# and appropriate gene_name and transcript_name columns in adata.var

# Basic usage
results = find_isoforms(adata, 
                       group_by='cell_type',
                       gene_name_col='gene_name',
                       transcript_name_col='transcript_id')

# With splitting
results = find_isoforms(adata, 
                       group_by='cell_type',
                       split_by='condition', 
                       gene_name_col='gene_name',
                       transcript_name_col='transcript_id')

# Extended results (include all genes, not just significant)
results = find_isoforms(adata, 
                       group_by='cell_type',
                       extended=True,
                       gene_name_col='gene_name',
                       transcript_name_col='transcript_id')
"""