# %%

#investigate reactome genes overlapped with resveratrol target genes 
 
import gseapy as gp
import pandas as pd
import os

# ===============================
# INPUT
# ===============================
resveratrol_genes = [
    'CFTR','CD4','ERCC1','NFE2L3','TNS1','TFAP2C','SIRT4','ESR1','SIRT1','MAPK1',
    'XBP1','HMOX1','APEX1','PCK2','NFKBIA','EEF1A2','CSNK2A1','MMP2','GSR',
    'TNFRSF10A','CYP4F2','TGFB1','GSK3A','PON1','NAMPT','SERPINE1','NRF1','C5',
    'RPS6KB1','CCL2','SULT1E1','NFKB1','CCND1','GLI1','LTA4H','IFNG','NANOG',
    'MAPK14','APOB','ODC1','CTSD','EGR1','IAPP','CAT','TNFSF10','AHR','NPY',
    'ADGRE5','PLCG1','CDKN1A','VASP','KLF2','HSPB1','SIRT2','INS','ALOX12',
    'BST2','GDF15','JUND','CYP2E1','TRIM28','SESN2','KRAS','EIF2S1','CDK4',
    'IL6','TLR2','MMP13','CYP2C9','LMNB1','HEXB','CDH1','ATP5F1B','ATF1',
    'SMAD2','TRPA1','VMP1','MAPK3','TYR','IL1B','ELANE','KDR','ATF2','STAT3',
    'POMC','TP63','ICAM1','PPARGC1A','ANXA3','SDHA','CASP6','EGF','VDAC1',
    'NBN','CDK2','RB1','CYP11A1','TP53','ERBB2','AKT1','AGTR1','EGFR',
    'TNFRSF10B','ADM','PAK1','ATM','PDCD4','IL18','ATP5F1A','FLT1','YAP1',
    'TMPRSS11D','APP','CASP10','AIFM1','PPARG','HMGCR','ACE','S100B','BAX',
    'VCAM1','ALPG','MITF','IL15','NOS3','ALDH1A1','SCNN1G','PELP1','FADD',
    'CCL11','MAP2K1','BCL2L1','IGF1','HSPA4','HSPB3','UGT1A1','IL13','CEBPB',
    'CXCL10','SMAD1','FOS','CXCL8','EIF2AK3','RPTOR','SULT1B1','OLR1',
    'CASP3','CFLAR','LEP','CASP2','SPHK1','KLK3','CD209','BID','AGTRAP',
    'NQO1','PCK1','PKM','ADIPOQ','SLC2A4','CDC25C','ETV4','SULT1A1','SMC1A',
    'SOX2','GSK3B','STK11','PGR','MAT2B','NOS2','DNMT3B','SP1','SIRT7',
    'BCL2','CASP9','SOCS3','PDGFB','CD86','SMAD3','PRKD1','BSG','ABCC5',
    'NR1I2','SIRT6','NOS1','PNPLA2','NQO2','CYP3A4','HTR7','HIF1A','EIF4G1',
    'SULT1A2','CAV1','FOXO3','EIF4EBP1','PTPN11','ACACB','PTK2','CYP1A2',
    'CS','PTGES','SULT1A3','UGT1A10','ESR2','ATF3','CTNNB1','PFKM','PRKAA1',
    'UGT1A9','PDE5A','FAS','SREBF1','ATP5F1C','DAPK1','SRC','CASP8','ARNT',
    'MAP3K5','PIK3CG','DNMT1','TFRC','SCNN1A','MAPK8','MME','SREBF2','MTOR',
    'PTGS1','NFE2L1','ABCG1','BECN1','TGM2','AGT','PARP1','SOD2','PTGS2',
    'FASLG','NR1I3','BGLAP','LMNA','AMD1','MCL1','CASP7','CYP17A1','JUN',
    'PRKAA2','CYP2C19','PTPN1','PTEN','NCOA3','VEGFA','CEL','MMP9','SPTAN1',
    'PLAU','DNM1','UGT1A7','UGT1A8','TLR4','FOXO4','AR','ALPL','ECE1','H2AX',
    'SYK','MYC','TP73','ATAD3A','NPHS1','SIRT5','EDN1','PRNP','FOXO1',
    'CYP1A1','SMAD9','RPS6','JAK2','CHEK2','SIRT3','GPX3','STAB2','ALPP',
    'CUL5','BCL2L11','CDK1','CXCL1','NFE2L2','KHSRP','ABCC1','RELA',
    'ELAVL1','ACAN','CREB1','CASP4','MAPK9','IGF2','TBXA2R','TNF','MYD88',
    'WEE1','GPX1','PRKCA','TOP2A','IL10','GABRR1','BDNF','SERPINE2','CDK5',
    'HDAC2','HSF1','TXNRD1','RPS6KA1','SMAD5','DIO2','LDLR','HSPB2-C11orf52'
]
# %%
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests
import gseapy as gp
import os

# ===============================
# INPUT GENE SET
# ===============================
resveratrol_genes = set(resveratrol_genes)

# ===============================
# TARGET REACTOME KEYWORDS
# ===============================
target_keywords = {
    #"Skin_Tone": "Melanin",
    "Hyaluronan_biosynthesis": "Hyaluronan Metabolism",
    #"Aquaporin_transport": "Aquaporin",
    "Keratinization": "Keratinization",
    #"Cornified_envelope": "cornified",
    "Sphingolipid_metabolism": "Sphingolipid Metabolism",
    "Negative_control_neuronal": "Neurotransmitter release"
}

# ===============================
# MANUAL MITF PATHWAY
# ===============================
MITF_PATHWAY_NAME = (
    "Regulation of MITF-M-dependent genes involved in pigmentation R-HSA-9824585"
)

MITF_GENES = {
    "ACTB","ACTL6A","AKT2","ARID1A","ARID1B","BCL7A","BCL7B","BCL7C",
    "CREB1","CTNNB1","DCT","DPF1","DPF2","DPF3","GPR143","IRF4",
    "LEF1","MAPK14","MITF","MLANA","MLPH","MYO5A","MYRIP","PMEL",
    "RAB27A","SMARCA2","SMARCA4","SMARCB1","SMARCC1","SMARCC2",
    "SMARCD1","SMARCD2","SMARCD3","SMARCE1","SOX10","SS18",
    "SS18L1","SYTL2","TFAP2A","TYR","TYRP1","USF1"
}

# ===============================
# SKIN AGING PATHWAYS (MANUAL)
# ===============================
SKIN_AGING_PATHWAYS = {
    "Extracellular Matrix Organization R-HSA-1474244": "R-HSA-1474244"
    #"Cellular Senescence R-HSA-2559583": "R-HSA-2559583",
    #"Senescence-Associated Secretory Phenotype (SASP) R-HSA-2559582": "R-HSA-2559582",
    #"Integrin Signaling R-HSA-354192": "R-HSA-354192"
}

# ===============================
# LOAD REACTOME
# ===============================
reactome_sets = gp.get_library(name="Reactome_2022", organism="Human")

# Add manual MITF pathway
reactome_sets[MITF_PATHWAY_NAME] = list(MITF_GENES)

# ===============================
# BACKGROUND
# ===============================
background_genes = set().union(*reactome_sets.values())
M = len(background_genes)
N = len(resveratrol_genes)

results = []

print("\n🔍 Matched Reactome pathways:")

# ===============================
# KEYWORD-BASED TARGETED TEST
# ===============================
for label, keyword in target_keywords.items():
    matched = [p for p in reactome_sets if keyword.lower() in p.lower()]

    if not matched:
        print(f"❌ No match for {label}")
        continue

    for pathway in matched:
        pathway_genes = set(reactome_sets[pathway])
        overlap = resveratrol_genes & pathway_genes

        k = len(overlap)
        n = len(pathway_genes)

        pval = hypergeom.sf(k - 1, M, n, N)

        results.append({
            "Target_category": label,
            "Reactome_pathway": pathway,
            "Pathway_size": n,
            "Overlap_size": k,
            "Overlap_genes": ", ".join(sorted(overlap)),
            "p_value": pval
        })

        print(f"  ✔ {pathway}")

# ===============================
# MANUAL MITF TEST
# ===============================
overlap = resveratrol_genes & MITF_GENES
k = len(overlap)
n = len(MITF_GENES)

if k > 0:
    pval = hypergeom.sf(k - 1, M, n, N)

    results.append({
        "Target_category": "Skin_Tone",
        "Reactome_pathway": MITF_PATHWAY_NAME,
        "Pathway_size": n,
        "Overlap_size": k,
        "Overlap_genes": ", ".join(sorted(overlap)),
        "p_value": pval
    })

    print(f"  ✔ {MITF_PATHWAY_NAME}")

# ===============================
# SKIN AGING PATHWAY TESTS (FORCED)
# ===============================
for pathway_name, pathway_id in SKIN_AGING_PATHWAYS.items():

    matched = [p for p in reactome_sets if pathway_id in p]

    if not matched:
        print(f"❌ {pathway_name} not found in Reactome")
        continue

    pathway = matched[0]
    pathway_genes = set(reactome_sets[pathway])
    overlap = resveratrol_genes & pathway_genes

    k = len(overlap)
    n = len(pathway_genes)

    pval = hypergeom.sf(k - 1, M, n, N)

    results.append({
        "Target_category": "Skin_Aging",
        "Reactome_pathway": pathway,
        "Pathway_size": n,
        "Overlap_size": k,
        "Overlap_genes": ", ".join(sorted(overlap)),
        "p_value": pval
    })

    print(f"  ✔ {pathway}")

# ===============================
# FINALIZE
# ===============================
if not results:
    raise RuntimeError("❌ No pathways enriched.")

df = pd.DataFrame(results)

df["FDR"] = multipletests(df["p_value"], method="fdr_bh")[1]
df = df.sort_values("p_value")

# ===============================
# SAVE
# ===============================
out_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/skin_feature_gsea"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(
    out_dir,
    "Final_Resveratrol_targeted_Reactome_skin_pathways_with_MITF_and_SkinAging.xlsx"
)

df.to_excel(out_path, index=False)

print(f"\n✅ Targeted Reactome enrichment saved:\n{out_path}")
# %%

#Fig.4C: Draw barplot of enrichment test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# INPUT / OUTPUT
# ===============================
input_file = (
    "/home/parksb0518/SBI_Lab/COSMAX/Result/skin_feature_gsea/"
    "Resveratrol_targeted_Reactome_skin_pathways_with_MITF_and_SkinAging.xlsx"
)
out_dir = "/home/parksb0518/SBI_Lab/COSMAX/Result/skin_feature_gsea"
os.makedirs(out_dir, exist_ok=True)

# ===============================
# TARGET PATHWAYS (fixed order)
# ===============================
target_pathways = [
    "Regulation of MITF-M-dependent genes involved in pigmentation R-HSA-9824585",
    "Hyaluronan Metabolism R-HSA-2142845",
    "Sphingolipid Metabolism R-HSA-428157",
    "Formation Of Cornified Envelope R-HSA-6809371",
    "Extracellular Matrix Organization R-HSA-1474244"
]

mitf_pathway = "Regulation of MITF-M-dependent genes involved in pigmentation R-HSA-9824585"

# ===============================
# LOAD DATA
# ===============================
df = pd.read_excel(input_file)

df_plot = df[df["Reactome_pathway"].isin(target_pathways)].copy()
if df_plot.empty:
    raise ValueError("❌ Target pathways not found in the input file.")

# ===============================
# CALCULATE -log10(p-value)
# ===============================
df_plot["minus_log10_p"] = -np.log10(df_plot["p_value"])

# Preserve desired order
df_plot["Reactome_pathway"] = pd.Categorical(df_plot["Reactome_pathway"], categories=target_pathways, ordered=True)
df_plot = df_plot.sort_values("Reactome_pathway", ascending=False)

# ===============================
# COLORS
# ===============================
colors = [
    "#0000E6" if p == mitf_pathway else "#B0B0B0"
    for p in df_plot["Reactome_pathway"]
]

# ===============================
# BAR PLOT (HORIZONTAL)
# ===============================
plt.figure(figsize=(8, 4))

plt.barh(
    df_plot["Reactome_pathway"],
    df_plot["minus_log10_p"],
    color=colors,
    edgecolor="black"
)

# Add vertical dashed red line at x=2
plt.axvline(x=2, color="red", linestyle="--", linewidth=1.5) #, label="p = 0.01 (-log10 = 2)"

plt.xlabel("-log10(p-value)", fontsize=12)
plt.ylabel("")
plt.title("Targeted Reactome Enrichment of Resveratrol Target Genes", fontsize=13)
plt.legend(loc="lower right", fontsize=9, frameon=False)

plt.tight_layout()

# ===============================
# SAVE
# ===============================
png_path = os.path.join(out_dir, "Resveratrol_targeted_pathways_with_skin_aging_negative_control.png")
pdf_path = os.path.join(out_dir, "Resveratrol_targeted_pathways_with_skin_aging_negative_control.pdf")

plt.savefig(png_path, dpi=300)
plt.savefig(pdf_path)
plt.close()

print("✅ Bar plot saved:")
print(png_path)
print(pdf_path)
