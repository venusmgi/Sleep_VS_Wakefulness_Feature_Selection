
# This script runs the lasso regularization  with lambda.1se 
# in 10 fold of cross validation to identify which features are used 
# and to compute the accuracy of each fold based on 
# the selected features. The results, including accuracy and selected features, 
# are saved to result_df.


# Load necessary libraries
library(epiDisplay)
library(ggeffects)
library(patchwork)
library(ROCR)
library(ggplot2)
library(dplyr)
library(glmnet)
library(gganimate)


# Set working directory (adjust path as needed)
setwd("D:/Venus/Lab projects/NIMBIS/Feature Selection/Sleep_VS_Wakefulness_Feature_Selection")


##############################FUNCTIONS##################################


Robust_Scale <- function(data){
  
  (data-median(data))/IQR(data)
}

Z_Score <- function(data){
  (data - mean(data)) / sd(data)
}

Range_Scale <- function(data){
  (data - min(data, na.rm = TRUE))/ (max(data, na.rm = TRUE)-min(data, na.rm = TRUE))
  
}


# Function to run cross-validation and retrieve accuracy and selected features
 Run_cross_validation <- function (x_test, y_test, cvfit,lambda_method) {
   coefs <- as.data.frame(as.matrix(coef(cvfit, s =lambda_method )))
   coefs$features <- rownames(coefs) #adding another column called feature and put name of features in it
   colnames(coefs)[1] <- "coefficient"  # Rename the first column(S1) for better readability
   
   # Filter to select significant features
   coefs <- coefs %>%
     filter(coefficient != 0 & features %in% featuresToUse)
   
   selected_features <- coefs[order(-abs(coefs$coefficient) ),]

   # Predict the results and calculate accuracy
   predict_results <- predict(cvfit, newx = x_test, type = 'class', s= lambda_method)
   tp <- sum(predict_results== 'sleep' & y_test == 'sleep')
   tn <- sum(predict_results=='wake' & y_test == 'wake')
   accuracy <- (tp + tn)/length(y_test)

   return(list(accuracy = accuracy,
               selected_features = selected_features$features))
 }
 
# 



 ##############################MAIN SCRIPT##################################

 # Read data from CSV file
raw_data <- read.csv('allPatinetsMetrics_Cz.csv', sep = ",",header = TRUE)
 # Extract computational metric names excluding specific columns
feature_metrics <- setdiff(names(raw_data),c('patient_ID', 'file_name', 'sleep_or_awake', 'pre_or_post', 
                                             'case_or_control', 'epoch_start_index', 'epoch_stop_index',
                                             'electrod_channel'))


# If there is sleep1/sleep2 or wake1/wake2 just replacing them sleep and wake for uniformity
raw_data <- raw_data%>%
  mutate(across(sleep_or_awake, ~ gsub('Sleep1|Sleep2','sleep',.)))%>%
  mutate(across(sleep_or_awake, ~ gsub('Wake1|Wake2','wake',.)))



# Prepare data by selecting one entry per patient and scaling features
set.seed(76456687) # Set seed for reproducibility

# Function for scaling data (choose appropriate scaling function)
scale = Range_Scale

# Randomly pick 1 value for each metric to create the sleep dataset
sleep_data <- raw_data  %>%
  filter(sleep_or_awake== 'sleep',pre_or_post == 'pre') %>%
  group_by(patient_ID ) %>%
  slice_sample(n = 1)

# Scale each metric in the sleep dataset
sleep_data$scaledPermEntD = scale(sleep_data$permutation_entropy_delta)
sleep_data$scaledShanEntD = scale(sleep_data$shannon_entropy_delta)
sleep_data$scaledPermEntT = scale(sleep_data$permutation_entropy_theta)
sleep_data$scaledShanEntT = scale(sleep_data$shannon_entropy_theta)
sleep_data$scaledPermEntA = scale(sleep_data$permutation_entropy_alpha)
sleep_data$scaledShanEntA = scale(sleep_data$shannon_entropy_alpha)
sleep_data$scaledPermEntB = scale(sleep_data$permutation_entropy_beta)
sleep_data$scaledShanEntB = scale(sleep_data$shannon_entropy_beta)
sleep_data$scaledPSDD = scale(sleep_data$power_spectral_density_delta)
sleep_data$scaledPSDT = scale(sleep_data$power_spectral_density_theta)
sleep_data$scaledPSDA = scale(sleep_data$power_spectral_density_alpha)
sleep_data$scaledPSDB = scale(sleep_data$power_spectral_density_beta)
sleep_data$scaledPSDBB = scale(sleep_data$power_spectral_density_broadband)
sleep_data$scaledAmp = scale(sleep_data$amplitude_range)
sleep_data$scaledSEF = scale(sleep_data$spectral_edge_frequency)

# Randomly pick 1 value for each metric to create the wakefulness dataset
wake_data <- raw_data %>%
  filter (sleep_or_awake== 'wake', pre_or_post == 'pre') %>%
  group_by(patient_ID) %>%
  slice_sample(n = 1)


# Scale each metric in the wakefulness dataset
wake_data$scaledPermEntD = scale(wake_data$permutation_entropy_delta)
wake_data$scaledShanEntD = scale(wake_data$shannon_entropy_delta)
wake_data$scaledPermEntT = scale(wake_data$permutation_entropy_theta)
wake_data$scaledShanEntT = scale(wake_data$shannon_entropy_theta)
wake_data$scaledPermEntA = scale(wake_data$permutation_entropy_alpha)
wake_data$scaledShanEntA = scale(wake_data$shannon_entropy_alpha)
wake_data$scaledPermEntB = scale(wake_data$permutation_entropy_beta)
wake_data$scaledShanEntB = scale(wake_data$shannon_entropy_beta)
wake_data$scaledPSDD = scale(wake_data$power_spectral_density_delta)
wake_data$scaledPSDT = scale(wake_data$power_spectral_density_theta)
wake_data$scaledPSDA = scale(wake_data$power_spectral_density_alpha)
wake_data$scaledPSDB = scale(wake_data$power_spectral_density_beta)
wake_data$scaledPSDBB = scale(wake_data$power_spectral_density_broadband)
wake_data$scaledAmp = scale(wake_data$amplitude_range)
wake_data$scaledSEF = scale(wake_data$spectral_edge_frequency)


# Merge sleep and wake datasets into one dataset
my_data <- rbind(sleep_data,wake_data)

# Ensure no grouping is active
my_data <- my_data %>% ungroup()

# Define features used in regularization
featuresToUse = c('scaledPermEntD','scaledShanEntD','scaledPermEntT','scaledShanEntT',
                  'scaledPermEntA','scaledShanEntA','scaledPermEntB','scaledShanEntB',
                  'scaledPSDD','scaledPSDT','scaledPSDA','scaledPSDB','scaledPSDBB','scaledAmp','scaledSEF')
# Set up cross-validation parameters
num_outer_folds = 10
num_patients <- length(unique(my_data$patient_ID))
fold_size <- ceiling(num_patients/num_outer_folds)

results_list <- vector ("list",num_outer_folds)


# Outer cross-validation loop
for (i in 1:num_outer_folds){

  # Define test and train datasets for the current outer fold
  start_idx = (i-1)*fold_size+1
  end_idx = min(i*fold_size,num_patients)

  
  test_patients <- unique( my_data$patient_ID[start_idx:end_idx])
  train_patients <- unique(setdiff(my_data$patient_ID,test_patients))
  
  # Prepare test data matrices
  test_data <- my_data %>%
    filter(patient_ID %in% test_patients)
  y_test <- as.matrix(test_data$sleep_or_awake)
  x_test <- as.matrix(test_data %>%
                        select(all_of(featuresToUse)))


  # Prepare train data matrices
  train_data <- my_data %>%
    filter(patient_ID %in% train_patients)
  y_train <- as.matrix(train_data$sleep_or_awake)
  x_train<- as.matrix(train_data %>%
                        select(all_of(featuresToUse)))


  # Assigning each patient with a foldid for inner cross-validation
  # , so I make sure that in each fold, the same patient is
  # getting selected for sleep and wakefulness
  num_inner_folds <-9
  fold_assignment <-rep(1:num_inner_folds,each = ceiling(length(train_patients)/num_inner_folds))[1:length(train_patients)]

  foldid <- train_data %>%
    mutate(fold = fold_assignment[match(patient_ID , train_patients )] ) %>%
    pull(fold)

  # Fit lasso model and run cross-validation
   epsilone = 0
   cvfit_lasso <- cv.glmnet(x_train, y_train,intercept=FALSE, alpha = 1,type.measure = "deviance",foldid = foldid,  family = "binomial",standardize = TRUE)
   lambda_method = "lambda.1se"
   lasso_1se_lambda_results <- Run_cross_validation(x_test, y_test, cvfit_lasso,lambda_method)
   
   # Store accuracy and selected features
   results_list[[i]] <- list(
     Accuracy = lasso_1se_lambda_results$accuracy,
     Selected_features = lasso_1se_lambda_results$selected_features
     
   )


}


# Process results into a data frame
max_length <- max(sapply(results_list, function(x){length(x$Selected_features)}))
result_df <- data.frame(matrix(nrow = num_outer_folds, ncol = max_length+2))
colnames(result_df)<- c("Accuracy","num_features",paste0("feature_",1:max_length))

for (i in 1:10){
  result_df[i,"Accuracy"] <- results_list[[i]]$Accuracy
  current_features <- results_list[[i]]$Selected_features
  current_length <- length(current_features)
  result_df[i,"num_features"] <- as.numeric(current_length)
  result_df[i,3:(current_length+2)] <- current_features
}

# The script ends here with the result_df containing accuracy and feature information for each fold


