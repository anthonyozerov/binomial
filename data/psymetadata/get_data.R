if(!require(psymetadata)) install.packages('psymetadata', repos="http://cran.us.r-project.org")
# write each to csv
require(psymetadata)

dataset_info <- data(package = 'psymetadata')
print(dataset_info)
datasets <- dataset_info$results[, 'Item']
print(dataset_info$results)

# for (dataset in datasets) {
#     df <- get(dataset)
#     # rename column 'yi' to 'value'
#     if (dataset == 'manylabs2018')
    
#     colnames(df)[colnames(df) == 'yi'] <- 'value'
#     # rename column 'vi' to 'uncertainty'
#     if (!('vi' %in% colnames(df))) {
#         colnames(df)[colnames(df) == 'vi_r'] <- 'vi'
#     }
#     colnames(df)[colnames(df) == 'vi'] <- 'uncertainty'

#     write.csv(df, paste0(dataset, ".csv"))
# }

df <- get('manylabs2018')
colnames(df)[colnames(df) == 'yi_r'] <- 'value'
colnames(df)[colnames(df) == 'vi_r'] <- 'uncertainty'
df$uncertainty <- sqrt(df$uncertainty)
# group by 'analysis' and save each to csv
for (analysis in unique(df$analysis)) {
    df_analysis <- df[df$analysis == analysis, ]
    # keep only columns 'lab', 'value', 'uncertainty'
    df_analysis <- df_analysis[, c('lab', 'value', 'uncertainty')]
    write.csv(df_analysis, paste0(analysis, ".csv"), row.names=FALSE)
}