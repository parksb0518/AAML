if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c(
  "ALDEx2", "DESeq2", "edgeR", "limma", "Maaslin2", "metagenomeSeq", 
  "SummarizedExperiment", "phyloseq", "biomformat", "lefser"
))

install.packages("ggpicrust2")
install.packages("ggprism")
install.packages("GGally")
install.packages("patchwork")

library(readr)
library(ggpicrust2)
library(tibble)
library(tidyverse)
library(ggprism)
library(patchwork)
library(GGally)
library(KEGGREST)
library(readxl)
library(dplyr)

#Reading data

raw_data <- read_excel("/home/lgy/picrust2/cosmax_analysis/data/Streptococcus_gordonii/debug_OM_TE_score_Streptococcus_gordonii_20_79.xlsx")


n_det <- raw_data %>%
  filter(SkinGroup == "Deteriorated", MicrobePresence == 1) %>%
  nrow()

n_imp <- raw_data %>%
  filter(SkinGroup == "Improved", MicrobePresence == 0) %>%
  nrow()

cat("Deteriorated & MicrobePresence == 1:", n_det, "\n")
cat("Improved     & MicrobePresence == 0:", n_imp, "\n")

# Data processing
metadata <- raw_data %>%
  # change the column name
  rename(SampleID   = `metadata-id`,
         enrichment = SkinGroup) %>%
  # Data type alteration
  mutate(SampleID = as.character(SampleID)) %>%
  # filtered the condition
  filter(
    (enrichment == "Deteriorated" & MicrobePresence == 1) |
    (enrichment == "Improved"     & MicrobePresence == 0)
  ) %>%
  # selecting the column
  select(SampleID, enrichment)

metadata

kegg_abundance <-
  ko2kegg_abundance(
    "/home/lgy/picrust2/cosmax_analysis/picrust2_out_pipeline_backup/KO_metagenome_out/pred_metagenome_unstrat.tsv"
  )
kegg_abundance

kegg_abundance2 <- kegg_abundance[, metadata$SampleID]


metadata$body.site <- factor(metadata$enrichment, levels = c("Deteriorated", "Improved"))

# Aldex2 analysis
ko_AlDEx <-
  pathway_daa(
    abundance = kegg_abundance2,
    metadata = metadata,
    group = "enrichment", 
    daa_method = "ALDEx2",
    select = NULL,
    reference = NULL
  )


ko_AlDEx %>% head()
#   feature                method group1 group2   p_values adj_method   p_adjust
# 1 ko05340 ALDEx2_Welch's t test    gut tongue 0.00402396         BH 0.01078215
# 2 ko00564 ALDEx2_Welch's t test    gut tongue 0.17365998         BH 0.26492653
# 3 ko00680 ALDEx2_Welch's t test    gut tongue 0.03217803         BH 0.06656202
# 4 ko00562 ALDEx2_Welch's t test    gut tongue 0.67601305         BH 0.75152515
# 5 ko03030 ALDEx2_Welch's t test    gut tongue 0.06168772         BH 0.10879943
# 6 ko00561 ALDEx2_Welch's t test    gut tongue 0.02946089         BH 0.06329062

ko_AlDEx %>% tail() 

ko_AlDEx_df <-   ko_AlDEx[ko_AlDEx$method == "ALDEx2_Wilcoxon rank test", ]

# enrichment calculation
enrich_samples <- metadata[metadata$enrichment == "Improved", ]$SampleID
depleted_samples <- metadata[metadata$enrichment == "Deteriorated", ]$SampleID

enrich_means <- rowMeans(kegg_abundance2[, enrich_samples])
depleted_means <- rowMeans(kegg_abundance2[, depleted_samples])

# Log2 fold change calculation
log2_fc <- log2((enrich_means + 1) / (depleted_means + 1))


ko_AlDEx_df$log2_fold_change <- log2_fc[ko_AlDEx_df$feature]
ko_AlDEx_df$regulation <- ifelse(ko_AlDEx_df$p_adjust < 0.05,
                               ifelse(ko_AlDEx_df$log2_fold_change > 0, 
                                     "Upregulated_in_ Improved", 
                                     "Upregulated_in_Deteriorated"),
                               "Not_significant")

ko_annotation <-pathway_annotation(pathway = "KO",
                                   daa_results_df = ko_AlDEx_df, 
                                   ko_to_kegg = TRUE)

write.table(ko_annotation, file = "/home/lgy/picrust2/cosmax_analysis/data/Streptococcus_gordonii/Streptococcus_gordonii_SkinGroup_ko_annotation.tsv",
            sep = "\t", row.names = FALSE, quote = FALSE)