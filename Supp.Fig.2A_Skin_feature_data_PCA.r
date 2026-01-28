
library(readxl)
library(dplyr)
library(ggside)

options (stringsAsFactors = FALSE)

library(factoextra)
library(FactoMineR)
library(corrplot)
library(ggplot2)
library(cowplot)

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

# Read data from the second Excel file and sheet
Beta1.braycurtis_pcoa <- read_excel(file2, sheet = sheet2)
alpha_diversity.normalization <- read_excel(file2, sheet = sheet3)

# Assuming data2 is a data frame with column names in the first row
new_column_names <- clinical_KSC_994ea[1, ]

# Set the column names of data2 to the values in the first row
colnames(clinical_KSC_994ea) <- new_column_names

# Remove the first row, as it is now used as column names
clinical_KSC_994ea <- clinical_KSC_994ea[-1, ]

# Assuming data2 is your data frame
clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$'mb.selected_sample-id'), ]

clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$'Age'), ]
clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$'Wrinkle.solabial.Max_depth_biggest_wrinkle'), ]
clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$'baumann'), ]

clinical_KSC_994ea$TE.ss <- as.numeric(clinical_KSC_994ea$TE.ss)
clinical_KSC_994ea$OM.ss <- as.numeric(clinical_KSC_994ea$OM.ss)

#make numeric value!
numeric_cols <- sapply(clinical_KSC_994ea, function(x) !any(grepl("[^0-9.-]", as.character(x))))
clinical_KSC_994ea[numeric_cols] <- lapply(clinical_KSC_994ea[numeric_cols], as.numeric)

clinical_KSC_994ea$Pore.Cheek.Left <- as.numeric(clinical_KSC_994ea$Pore.Cheek.Left)
clinical_KSC_994ea$Pore.Cheek.Right <- as.numeric(clinical_KSC_994ea$Pore.Cheek.Right)
clinical_KSC_994ea$Pore.Cheek.avg <- as.numeric(clinical_KSC_994ea$Pore.Cheek.avg)


# Print the first few rows of each data frame
head(clinical_KSC_994ea)
head(Beta1.braycurtis_pcoa)



# Ensure Age is numeric
clinical_KSC_994ea$Age <- as.numeric(clinical_KSC_994ea$Age)

# Create Age_group column
clinical_KSC_994ea$Age_group <- cut(
  clinical_KSC_994ea$Age,
  breaks = c(20, 44, 60, 79),
  labels = c("20-44 years", "45-60 years", "61-79 years"),
  right = TRUE,
  include.lowest = TRUE
)

clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$Age_group), ]

# Drop rows with NA in mb.selected_sample-id (in case any appeared again)
clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$'mb.selected_sample-id'), ]
clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$Wrinkle.Eye.Avg_wrinkle_depth), ]
clinical_KSC_994ea <- clinical_KSC_994ea[!is.na(clinical_KSC_994ea$ITA.ss), ]

# View the result
head(clinical_KSC_994ea[, c("Age", "Age_group", "mb.selected_sample-id")])
write.csv(clinical_KSC_994ea, "clinical_KSC_994ea.csv", row.names = FALSE)


# 0. plotting theme
age_group_color <- c("#D5221E","#4EA74A","#EF7B19")
theme <- theme(
  plot.title=element_text(size=18, face="bold", colour="black"),
  axis.title.x=element_text(size=15, colour="black"),
  axis.title.y=element_text(size=15, angle=90, colour="black"),
  axis.text.x=element_text(size=8,angle=45,colour="black",vjust=0.5),
  axis.text.y=element_text(size=8,colour="black"),
  axis.ticks=element_line(colour="black",size=0.5),
  panel.grid.major=element_blank(),
  panel.grid.minor=element_blank(),
  panel.background=element_blank(),
  axis.line=element_line(size=0.5))


# 1. Set mb.selected_sample-id as row names
clinical_KSC_994ea_meta <- clinical_KSC_994ea[, 1:7]
clinical_KSC_994ea_meta$Age_group <- clinical_KSC_994ea$Age_group
rownames(clinical_KSC_994ea) <- clinical_KSC_994ea$`mb.selected_sample-id`
rownames(clinical_KSC_994ea_meta) <- rownames(clinical_KSC_994ea)

# Keep only numeric columns from the original data excluding metadata
#clinical_KSC_994ea_data <- clinical_KSC_994ea[, sapply(clinical_KSC_994ea_meta, is.numeric)]


# 3. Save the meta information separately (you can choose which columns to keep)
# Here, saving everything except the numeric-only columns
numeric_cols <- sapply(clinical_KSC_994ea, function(x) is.numeric(x))
clinical_KSC_994ea_data <- clinical_KSC_994ea[, numeric_cols]
clinical_KSC_994ea_data <- clinical_KSC_994ea_data[, !(names(clinical_KSC_994ea_data) %in% "Age")]

# 4. Keep only numeric columns
#clinical_KSC_994ea_data <- clinical_KSC_994ea_data[, sapply(clinical_KSC_994ea_data, is.numeric)]

# 3. PCA analysis
PCA_data <- clinical_KSC_994ea_data
PCA_data <- scale(PCA_data)
rownames(PCA_data) <- rownames(clinical_KSC_994ea_meta)
res.pca <- PCA(PCA_data, graph = FALSE,ncp=10)
eig.val <- get_eigenvalue(res.pca)
fviz_eig(res.pca, addlabels = TRUE)
var <- get_pca_var(res.pca)

# 3.1 Plotting Figure 1B part1
p1 = fviz_pca_biplot(res.pca, 
                col.ind = clinical_KSC_994ea_meta[rownames(PCA_data),]$Age_group,
                palette = c("#D5221E","#4EA74A","#EF7B19"), 
                addEllipses = FALSE, label = "var",
                col.var = "black", repel = TRUE,
                legend.title = "Age group",select.var = list(name = NULL, cos2 = 10, contrib = NULL))+
  stat_ellipse(type = "t",level=0.80,alpha=0.5,aes(x=res.pca$ind$coord[,"Dim.1"],y=res.pca$ind$coord[,"Dim.2"], 
                                                                    color=clinical_KSC_994ea_meta[rownames(PCA_data),]$Age_group))+
  scale_fill_manual(values = age_group_color)+
  scale_shape_manual(values=c(1,1,1,1,1))

# 3.2 Plotting Figure 1B part2
group_comparisons <- list(c("20-44 years","45-60 years"),c("45-60 years","61-79 years"))


# 3.3 wilcox test to compare each age group against PC1

pca_dims <- as.data.frame(res.pca$ind$coord)
pca_dims$Age_group <- clinical_KSC_994ea_meta$Age_group #[rownames(pca_dims),]
#pca_dims$Gender <- skin_1000_meta[rownames(pca_dims),]$Gender
#pca_dims$Product <- skin_1000_meta[rownames(pca_dims),]$Product
wilcox.test(pca_dims[pca_dims$Age_group=="20-44 years",]$Dim.1,pca_dims[pca_dims$Age_group=="45-60 years",]$Dim.1)
#W = 38405, p-value < 2.2e-16
wilcox.test(pca_dims[pca_dims$Age_group=="45-60 years",]$Dim.1,pca_dims[pca_dims$Age_group=="61-79 years",]$Dim.1)
#W = 28212, p-value < 2.2e-16

library(ggpubr)
p2 <- ggplot(pca_dims, aes(y = Age_group, x = Dim.1)) +
  geom_boxplot(aes(fill = Age_group)) +
  scale_fill_manual(values = age_group_color) +
  stat_compare_means(comparisons = group_comparisons, method = "wilcox.test") +
  #xlim(-10, 10) +
  theme(
    axis.text.x = element_blank(),
    axis.title.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    legend.position = "none"
  )

# store the plots of p1 p2

figure1b = plot_grid(p1,p2,ncol=1, rel_heights = c(1, 0.3))
print(figure1b)


ggsave("/home/parksb0518/SBI_Lab/COSMAX/plots/Clin_PCA_age_new.png",figure1b,width=6,height=4)
ggsave("/home/parksb0518/SBI_Lab/COSMAX/plots/Clin_PCA_age_new.pdf",figure1b,width=6,height=4)




# Print sample counts per age group
table(clinical_KSC_994ea$Age_group)

# Print sample IDs per age group
age_groups <- unique(clinical_KSC_994ea$Age_group)

for (g in age_groups) {
  cat("\n=============================\n")
  cat("Age Group:", g, "\n")
  cat("Sample Count:", sum(clinical_KSC_994ea$Age_group == g), "\n")
  cat("Sample IDs:\n")
  print(clinical_KSC_994ea$`mb.selected_sample-id`[clinical_KSC_994ea$Age_group == g])
  cat("=============================\n")
}


# Create decade groups: 20s, 30s, 40s, ..., 70s
clinical_KSC_994ea$Age_decade <- cut(
  clinical_KSC_994ea$Age,
  breaks = c(19, 29, 39, 49, 59, 69, 79),
  labels = c("20s", "30s", "40s", "50s", "60s", "70s"),
  right = TRUE
)

age_counts <- table(clinical_KSC_994ea$Age_decade)
age_counts

library(ggplot2)

df_age <- as.data.frame(age_counts)
colnames(df_age) <- c("Age_decade", "Count")

p_age <- ggplot(df_age, aes(x = Age_decade, y = Count, fill = Age_decade)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = Count), 
            vjust = -0.5, size = 3) + 
  scale_fill_manual(values = c("#D5221E", "#EF7B19", "#F2BE22",
                               "#4EA74A", "#2A78B3", "#7B4EA7")) +
  theme_minimal(base_size = 14) +
  labs(
       x = "Age Group",
       y = "Number of Samples") +
  theme(legend.position = "none")

# 4. Save as PDF
ggsave(
  filename = "/home/parksb0518/SBI_Lab/COSMAX/Result/plots/Age_decade_barplot.pdf",
  plot = p_age,
  width = 6,
  height = 4
)

# Optional: also save PNG
ggsave(
  filename = "/home/parksb0518/SBI_Lab/COSMAX/Result/plots/Age_decade_barplot.png",
  plot = p_age,
  width = 6,
  height = 4,
  dpi = 300
)
