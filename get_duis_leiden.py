# Use scFates conda enviornment. 
import scanpy as sc
import numpy as np
import pandas as pd
import tqdm
import os

# Import isoform_tools
exec(open(f"C:/Users/Felix/Desktop/vs_code/translational_control/tools/isoform_tools.py").read())

DATA_PATH="C:/Users/Felix/Desktop/vs_code/translational_control/data"

adata=sc.read(f"{DATA_PATH}/adata_thresholded.h5ad")


# Filter on isoform count.
# Read keep isoforms
with open("C:/Users/Felix/Desktop/vs_code/translational_control/information_content/keep_isoforms.txt", "r") as f:
    list_ = f.readlines()

keep=[]
for x in list_:
    keep.append(x[:-1])

# Need to redo keep because it was calculated on adata_preprocessed
keep_new = []
for x in tqdm.tqdm(keep):
    if x in adata.var_names.tolist():
        keep_new.append(x)
adata = adata[:,keep_new]




dui = find_isoforms(adata,
                    group_by="leiden",
                    threshold_var = 0.004, # adjust according to variance distribution notebook
                    gene_name_col="gene",
                    transcript_name_col="transcript",
                    threshold_abund_absolute = 100 # Lower incrementaly after inspection of DUI solo isoforms
                   )

save_directory="dui"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save
dui.to_csv(f"{save_directory}/dui_leiden.csv")