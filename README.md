# curvature
Scalar Curvature Estimation

# Biological Datasets
* Contain datasets from several scRNAseq experiments (listed below):
    * 1ks3c
        * https://www.10xgenomics.com/resources/datasets/1-k-human-pbm-cs-stained-with-a-panel-of-total-seq-b-antibodies-single-indexed-3-1-standard-4-0-0
    * 2kglioblastoma
        * https://www.10xgenomics.com/resources/datasets/2-k-sorted-cells-from-human-glioblastoma-multiforme-3-v-3-1-3-1-standard-6-0-0
    * 5k-pbmc-antibody
        * https://www.10xgenomics.com/resources/datasets/5-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-with-cell-surface-proteins-v-3-chemistry-3-1-standard-3-1-0
    * 5k-pbmc-nextgem
        * https://www.10xgenomics.com/resources/datasets/5-k-peripheral-blood-mononuclear-cells-pbm-cs-from-a-healthy-donor-with-cell-surface-proteins-next-gem-3-1-standard-3-1-0
    * 200-braintumor
        * https://www.10xgenomics.com/resources/datasets/200-sorted-cells-from-human-glioblastoma-multiforme-3-lt-v-3-1-3-1-low-6-0-0
* There are also datasets with matrix counts that were not used, from this paper: https://www.pnas.org/doi/10.1073/pnas.2100473118
    * gastrulation
    * PBMC-PNAS
* Protocol for analyzing biological data:
    * Get a matrix of all the data points (if sparse, unpack into a dense format)
    * Use the estimator to get scalar curvatures at all of the points
    * Get UMAP and PCA coordinates for the data
    * Plot the scalar curvature estimates using a color bar on the UMAP coordinates

# File Descriptions
* biodata.ipynb
    * functions for extracting coordinates and data from files, computing curvature, and plotting results
    * sample plots for a few different data sets (described above)
    * functions for extracting manifold dimension from computing residuals from Isomap algorithm
* curvature_estimation.ipynb
    * (untouched) older tests and plots for the estimator
* curvature.py
    * compilation of all functions used to compute and analyze the curvature estimates
* example.ipynb
    * notebook with examples of calling the estimator and plotting various aspects of the results
* manifold.py
    * classes of manifolds
    * edited with functions generalized to not need manifold dimension (when possible)
    * added mse functions for analysis
* plots.ipynb   
    * generation of plots to see mse on manifolds during experiments
    * comparison of all density computation versions
    * attempts to correct errors on boundary for Euclidean disk
* test.ipynb
    * tests run on different manifolds with mse functions
    * commentary on each stage of variations
* testpy.py
    * compilation of all functions for automated grid search of parameters

# Experimental Logs
* https://docs.google.com/spreadsheets/d/1B6o-5LCz1H0vSC61H4UluSPhINkV0UQXc9NjnJvIAqc/edit?usp=sharing
    * General Results: records of each run of the estimator grid search on parameters
    * Rm Graphs: plots of grid search trends and relationships
    * Density Improvements: effect of density computation variants on curvature estimates for S2 and Euclidean unit disk
    * Bio Data: application of algorithm to biological data sets with manifold dimension determined by Isomap residuals
* https://docs.google.com/presentation/d/17rGH_oQrGp9azQbMOZ12b35ial5Fgzpk-slXxaF8WCs/edit?usp=sharing
    * overview of summer work
