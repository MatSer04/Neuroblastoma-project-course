# Use scFates conda enviornment. 
import scanpy as sc
import numpy as np
import pandas as pd
import tqdm
import os

# Import isoform_tools
exec(open("/home/matei/Neuroblastoma-project-course/isoform_tools.py").read())


adata = sc.read("/home/matei/data/adata_neuroblastoma_assigned.h5ad")

# Removing unwanted clusters
adata=adata[~adata.obs.leiden.isin(["7","10","11","6","11","8"])]

######### Filter based on total isoform expression ##########
# Calculate expr per isoform
expr = np.asarray(adata.X.sum(axis=0)).ravel()
df = pd.DataFrame({
    "transcript": adata.var["transcript"].values,
    "expr": expr
})
sum_iex = df.groupby("transcript")["expr"].sum().to_dict()

# Filter iso <30 expr
iso_to_keep = df.transcript[df["expr"]>30].tolist()
adata = adata[:, adata.var["transcript"].isin(iso_to_keep)]

######### Filter based on isoform number (<1) ##########
gene_list = adata.var["gene"].unique().tolist()
multi_iso = []

for gene in tqdm.tqdm(gene_list):
    n_iso = len(adata.var[adata.var["gene"]==gene])
    #print(f"n_iso:{n_iso}")
    if n_iso > 1:
        multi_iso.append(gene)
adata = adata[:, adata.var["gene"].isin(multi_iso)]


dui = find_isoforms(adata,
                    group_by="leiden",
                    threshold_var = 0.003, # adjust according to variance distribution notebook
                    gene_name_col="gene",
                    transcript_name_col="transcript",
                    threshold_abund_absolute = 30 # Lower incrementaly after inspection of DUI solo isoforms
                   )

save_directory="dui"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save
dui.to_csv("/home/matei/data/dui_leiden_filter30.csv")