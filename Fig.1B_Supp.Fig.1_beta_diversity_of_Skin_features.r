library(readxl)
library(dplyr)
library(ggside)

# Set the working directory
setwd("/home/parksb0518/SBI_Lab/COSMAX")

# Specify the file names and sheet names
file1 <- "피부 임상 측정.xlsx"
file2 <- "마이크로바이옴 분석 데이터.xlsx"
sheet1 <- "clinical_KSC_994ea"
sheet2 <- "Beta1.braycurtis_pcoa"
sheet3 <- "alpha-diversity.normalization"

# Read data from the first Excel file and sheet
clinical_KSC_994ea <- read_excel(file1, sheet = sheet1)
clinical_KSC_994ea_OMTE <- read_excel("/home/parksb0518/SBI_Lab/COSMAX/Data/clinical_data_OM_TE.xlsx")
head(clinical_KSC_994ea_OMTE)

colnames(clinical_KSC_994ea_OMTE)

# Read data from the second Excel file and sheet
Beta1.braycurtis_pcoa <- read_excel(file2, sheet = sheet2)
alpha_diversity.normalization <- read_excel(file2, sheet = sheet3)

# Assuming data2 is a data frame with column names in the first row
#new_column_names <- clinical_KSC_994ea_OMTE[1, ]

# Set the column names of data2 to the values in the first row
#colnames(clinical_KSC_994ea_OMTE) <- new_column_names

# Remove the first row, as it is now used as column names
#clinical_KSC_994ea_OMTE <- clinical_KSC_994ea_OMTE[-1, ]

# Assuming data2 is your data frame
clinical_KSC_994ea_OMTE <- clinical_KSC_994ea_OMTE[!is.na(clinical_KSC_994ea_OMTE$'mb.selected_sample-id'), ]

clinical_KSC_994ea_OMTE <- clinical_KSC_994ea_OMTE[!is.na(clinical_KSC_994ea_OMTE$'Age'), ]
clinical_KSC_994ea_OMTE <- clinical_KSC_994ea_OMTE[!is.na(clinical_KSC_994ea_OMTE$'Wrinkle.solabial.Max_depth_biggest_wrinkle'), ]
clinical_KSC_994ea_OMTE <- clinical_KSC_994ea_OMTE[!is.na(clinical_KSC_994ea_OMTE$'baumann'), ]

clinical_KSC_994ea_OMTE$TE.ss <- as.numeric(clinical_KSC_994ea_OMTE$TE.ss)
clinical_KSC_994ea_OMTE$OM.ss <- as.numeric(clinical_KSC_994ea_OMTE$OM.ss)

#make numeric value!
numeric_cols <- sapply(clinical_KSC_994ea_OMTE, function(x) !any(grepl("[^0-9.-]", as.character(x))))
clinical_KSC_994ea_OMTE[numeric_cols] <- lapply(clinical_KSC_994ea_OMTE[numeric_cols], as.numeric)

clinical_KSC_994ea_OMTE$Pore.Cheek.Left <- as.numeric(clinical_KSC_994ea_OMTE$Pore.Cheek.Left)
clinical_KSC_994ea_OMTE$Pore.Cheek.Right <- as.numeric(clinical_KSC_994ea_OMTE$Pore.Cheek.Right)
clinical_KSC_994ea_OMTE$Pore.Cheek.avg <- as.numeric(clinical_KSC_994ea_OMTE$Pore.Cheek.avg)


# Print the first few rows of each data frame
head(clinical_KSC_994ea_OMTE)
head(Beta1.braycurtis_pcoa)

# Assuming you have the required data frames: alpha_diversity, Beta1.braycurtis_pcoa, and clinical_KSC_994ea_OMTE

# Load the vegan package
library(vegan)
library(veganUtils)
library(writexl)

# Read the CSV file
#merged_data <- read.csv("/home/parksb0518/SBI_Lab/COSMAX/Data/merged_data.csv")

# View the first few rows
#head(merged_data)

# Merge clinical features with beta diversity data
#merged_data <- merge(clinical_KSC_994ea_OMTE, Beta1.braycurtis_pcoa, by = "sample_id", all.x = TRUE)
merged_data <- merge(clinical_KSC_994ea_OMTE, Beta1.braycurtis_pcoa, 
                     by.x = "mb.selected_sample-id", by.y = "sample-id", all.x = TRUE)

# Set a specific column as row names (metadata-id)
rownames(merged_data) <- merged_data$'mb.selected_sample-id'  # Remove the column from the data frame
rownames(merged_data) <- NULL

# Identify and remove columns containing character values
char_cols <- sapply(merged_data, is.character)
merged_data_val <- merged_data[, !char_cols]

char_cols <- sapply(clinical_KSC_994ea_OMTE, is.character)
clinical_val <- clinical_KSC_994ea_OMTE[, !char_cols]
#merged_data <- na.omit(merged_data)
# Extract PCoA coordinates
beta_coordinates <- merged_data[, c("PC1", "PC2", "PC3")]  # Adjust based on your PCoA dimensions

#clinical_val <- merged_data[, c("Age", "Hydration.Forehead.1st", "Hydration.Forehead.3rd", "Hydration.Forehead.2nd", "TE.ss", "OM.ss")] 
# Calculate Bray-Curtis dissimilarity matrix
#bray_curtis_matrix <- vegdist(merged_data_val[, -c(1, 2)], method = "manhattan", na.rm = TRUE)
#bray_curtis_matrix <- vegdist(beta_coordinates, method = "manhattan", na.rm = TRUE)
bray_curtis_matrix <- vegdist(merged_data_val, method = "bray", na.rm = TRUE)

envfit_result <- envfit(bray_curtis_matrix, beta_coordinates, na.rm = TRUE)
envfit_result2 <- envfit(bray_curtis_matrix, clinical_val, na.rm = TRUE)

print(envfit_result)
print(envfit_result2)

write_xlsx(envfit_result2, "/home/parksb0518/SBI_Lab/COSMAX/Data/beta_diversity_clinical_value_OMTE.xlsx")

# Convert envfit result to a data frame
# Combine vectors and factors into one data frame
# Extract vectors (for numeric variables)
vectors <- envfit_result2$vectors

# Create data frame with p-values and r-squared
envfit_summary <- data.frame(
  Variable = rownames(vectors$arrows),
  R2 = vectors$r,
  P_value = vectors$pvals
)

# Save to Excel
library(writexl)
write_xlsx(envfit_summary, "/home/parksb0518/SBI_Lab/COSMAX/Data/beta_diversity_clinical_value_OMTE.xlsx")

#---------------

envfit_summary <- read_excel("/home/parksb0518/SBI_Lab/COSMAX/plots/alpha_beta_cli_cor/Beta_cli/envfit_data_cli_with_beta.xlsx")

library(ggplot2)
library(dplyr)

# === Extract relevant summary data ===
selected_vars <- c("Age", "OM.ss", "TE.ss", "Oil.ss", "Moist.ss", 
                   "R7.ss", "ITA.ss", "TEWL.Cheek.avg", "Hydration.Cheek.avg", "Density.Cheek") #, "OM_TE_score"

# Filter to selected variables
plot_df <- envfit_summary %>%
  filter(Variable %in% selected_vars) %>%
  mutate(
    color = ifelse(P_value < 0.05, "#3677AD", "gray"), #blue
        asterisk = case_when(
      p_value <= 0.001 ~ "***",
      p_value <= 0.01 ~ "**",
      p_value <= 0.05 ~ "*",
      TRUE ~ ""
    )
  )

# Filter to selected variables and assign colors/asterisks by p-value
plot_df <- envfit_summary %>%
  filter(variable %in% selected_vars) %>%
  mutate(
    color = ifelse(p_value < 0.05, "#3677AD", "gray"),
    asterisk = case_when(
      p_value <= 0.001 ~ "***",
      p_value <= 0.01 ~ "**",
      p_value <= 0.05 ~ "*",
      TRUE ~ ""
    )
  )

# === Create barplot ===
p <- ggplot(plot_df, aes(x = reorder(variable, R_square), y = R_square, fill = color)) +
  geom_bar(stat = "identity", color = "black") +
  coord_flip() +
  geom_text(aes(label = asterisk), hjust = -0.3, size = 6) +
  scale_fill_identity() +
  labs(
    #title = "R² of Clinical Variables (envfit)",
    x = "Variable",
    y = expression(R^2)
  ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major.y = element_blank(),
    axis.title.y = element_blank()
  ) +
  ylim(0, max(plot_df$R_square) + 0.02)



  # === Save plot ===
ggsave(
  filename = "/home/parksb0518/SBI_Lab/COSMAX/plots/alpha_beta_cli_cor/Beta_cli/cli_variable_beta_no_OMTE.pdf",
  plot = p,
  width = 8,
  height = 5,
  dpi = 300
)

#-------
