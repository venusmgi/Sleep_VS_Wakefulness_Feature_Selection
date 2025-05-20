# Load necessary libraries
library(epiDisplay)
library(ggeffects)
library(patchwork)
library(ROCR)
library(ggplot2)
library(dplyr)
library(glmnet)
library(gganimate)

# This code compares different regularization methods (lasso, ridge, and elastic net)
# with different lambda selections. the output stats has the info of this comparison.
# It's updated to process the output from ConvertMatToCSV.m.
# For creating a video from generated plots, use Make_Video_From_R_plots.m."

#Note: change the name of Feature_selection_results1.csv and deviance_run1_fold 
# and histogram_run1_fold for each different run of this function so it will not overwrite the previous images


######## Functions #################

Robust_Scale <- function(data){
  
  (data-median(data))/IQR(data)
}

Z_Score <- function(data){
  (data - mean(data)) / sd(data)
}

Range_Scale <- function(data){
  (data - min(data, na.rm = TRUE))/ (max(data, na.rm = TRUE)-min(data, na.rm = TRUE))
  
}


# Function to run cross-validation and return accuracy and selected features
Run_cross_validation <- function (x_test, y_test, cvfit,lambda_method) {
  coefs <- as.data.frame(as.matrix(coef(cvfit, s =lambda_method )))
  coefs$features <- rownames(coefs) # Add a 'features' column with feature names
  colnames(coefs)[1] <- "coefficient"  # Rename the first column(S1) for better readability
  
  # Filter to get selected features
  selected_features <- coefs %>%
    filter(coefficient != 0 & features %in% featuresToUse)
  
  # Predict results using the model and calculate accuracy
  predict_results <- predict(cvfit, newx = x_test, type = 'class', s= lambda_method)
  tp <- sum(predict_results== 'sleep' & y_test == 'sleep')
  tn <- sum(predict_results=='wake' & y_test == 'wake')
  accuracy <- (tp + tn)/length(y_test)
  
  return(list(accuracy = accuracy,
              selected_features = selected_features$features))
  
  
}

# Function to store results in a structured format
# storing_results 
Store_results <- function( results,regression_method,lambda_selection){
  list(
    selected_features = results$selected_features,
    accuracy = results$accuracy,
    regression_method = regression_method,
    lambda_selection = lambda_selection)
  
}

##############################MAIN SCRIPT##################################

# Set working directory (change the path as necessary)
setwd("D:/Venus/Lab projects/NIMBIS/Feature Selection/Sleep_VS_Wakefulness_Feature_Selection")


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
  
#num_epochs_wake <-raw_data %>%
#  filter(sleep_or_awake== 'Wake1' | sleep_or_awake== 'Wake2') %>%
#  group_by(patient_ID) %>%
#  summarise(num_epochs = n())





# Function for scaling data (choose appropriate scaling function)
scale = Range_Scale

# Prepare data by selecting one entry per patient and scaling features

# (change the seed point to get different results)
set.seed(76456687) # Set seed for reproducibility

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


# 
# scaled_sleep_data <- sleep_data %>%mutate( 
#   scaledPermEntD = scale(permutation_entropy_delta),
#           scaledShanEntD = scale(shannon_entropy_delta),
#           scaledPermEntT = scale(permutation_entropy_theta),
#           scaledShanEntT = scale(shannon_entropy_theta),
#           scaledPermEntA = scale(permutation_entropy_alpha),
#           scaledShanEntA = scale(shannon_entropy_alpha),
#           scaledPermEntB = scale(permutation_entropy_beta),
#           scaledShanEntB = scale(shannon_entropy_beta),
#           scaledPSDD = scale(power_spectral_density_delta),
#           scaledPSDT = scale(power_spectral_density_theta),
#           scaledPSDA = scale(power_spectral_density_alpha),
#           scaledPSDB = scale(power_spectral_density_beta),
#           scaledPSDBB = scale(power_spectral_density_broadband),
#           scaledAmp = scale(amplitude_range),
#           scaledSEF = scale(spectral_edge_frequency))
# 

# Randomly pick 1 value for each metric to create the wakefulness dataset
wake_data <- raw_data %>%
  filter (sleep_or_awake== 'wake', pre_or_post == 'pre') %>%
  group_by(patient_ID) %>%
  slice_sample(n = 1)

# wake_data <- wake_data %>%
#   mutate( scaledPermEntD = scale(permutation_entropy_delta),
#           scaledShanEntD = scale(shannon_entropy_delta),
#           scaledPermEntT = scale(permutation_entropy_theta),
#           scaledShanEntT = scale(shannon_entropy_theta),
#           scaledPermEntA = scale(permutation_entropy_alpha),
#           scaledShanEntA = scale(shannon_entropy_alpha),
#           scaledPermEntB = scale(permutation_entropy_beta),
#           scaledShanEntB = scale(shannon_entropy_beta),
#           scaledPSDD = scale(power_spectral_density_delta),
#           scaledPSDT = scale(power_spectral_density_theta),
#           scaledPSDA = scale(power_spectral_density_alpha),
#           scaledPSDB = scale(power_spectral_density_beta),
#           scaledPSDBB = scale(power_spectral_density_broadband),
#           scaledAmp = scale(amplitude_range),
#           scaledSEF = scale(spectral_edge_frequency))


# scaling the wake dataset
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


# Concatenate sleep and wake data into one dataset
my_data <- rbind(sleep_data,wake_data)

# Ensure no grouping is active
my_data <- my_data %>% ungroup()

# Define features to use in analysis
featuresToUse = c('scaledPermEntD','scaledShanEntD','scaledPermEntT','scaledShanEntT',
                  'scaledPermEntA','scaledShanEntA','scaledPermEntB','scaledShanEntB',
                  'scaledPSDD','scaledPSDT','scaledPSDA','scaledPSDB','scaledPSDBB','scaledAmp','scaledSEF')

# Set up cross-validation parameters
num_outer_folds = 10
num_patients <- length(unique(my_data$patient_ID))
fold_size <- ceiling(num_patients/num_outer_folds)

results_list <- vector ("list",num_outer_folds)


# outer crtoss-validaton loop
for (i in 1:num_outer_folds){
  
  start_idx = (i-1)*fold_size+1
  end_idx = min(i*fold_size,num_patients)
  
  # Determine test and train patients for outer cross validation
  test_patients <- unique( my_data$patient_ID[start_idx:end_idx])
  train_patients <- unique(setdiff(my_data$patient_ID,test_patients))
  
  # Prepare test data
  test_data <- my_data %>%
    filter(patient_ID %in% test_patients)
  y_test <- as.matrix(test_data$sleep_or_awake)
  x_test <- as.matrix(test_data %>%
    select(all_of(featuresToUse)))

  
  # Prepare train data
  train_data <- my_data %>%
    filter(patient_ID %in% train_patients)
  y_train <- as.matrix(train_data$sleep_or_awake)
  x_train<- as.matrix(train_data %>%
    select(all_of(featuresToUse)))
  
  
  #assiging each patient with a foldid, so I make sure that in each fold, 
  # the same patient is getting selected for sleep and wakefulness
  num_inner_folds <-9
  fold_assignment <-rep(1:num_inner_folds,each = ceiling(length(train_patients)/num_inner_folds))[1:length(train_patients)]
  
  foldid <- train_data %>%
    mutate(fold = fold_assignment[match(patient_ID , train_patients )] ) %>%
    pull(fold)
  
  
  # Fit models using different regularization methods
  epsilone = 0
  cvfit_ridge <- cv.glmnet(x_train, y_train, alpha = 0,type.measure = "deviance",foldid = foldid,  family = "binomial",standardize = TRUE)
  cvfit_elastic <- cv.glmnet(x_train, y_train, alpha = 0.5,type.measure = "deviance",foldid = foldid,  family = "binomial",standardize = TRUE)
  cvfit_lasso <- cv.glmnet(x_train, y_train, alpha = 1,type.measure = "deviance",foldid = foldid,  family = "binomial",standardize = TRUE)
  
  # Run cross-validation and store results for lambda.min
  lambda_method = "lambda.min"
  lasso_min_lambda_results <- Run_cross_validation(x_test, y_test, cvfit_lasso,lambda_method)
  elastic_min_lambda_results <- Run_cross_validation(x_test, y_test, cvfit_elastic,lambda_method)
  ridge_min_lambda_results <- Run_cross_validation(x_test, y_test, cvfit_ridge,lambda_method)
  
  # Run cross-validation and store results for lambda.1se
  lambda_method = "lambda.1se"
  lasso_1se_lambda_results <- Run_cross_validation(x_test, y_test, cvfit_lasso,lambda_method)
  elastic_1se_lambda_results <- Run_cross_validation(x_test, y_test, cvfit_elastic,lambda_method)
  ridge_1se_lambda_results <- Run_cross_validation(x_test, y_test, cvfit_ridge,lambda_method)
  

  # Save results for each method and lambda selection

  results_list [[i]] <- list(
    Store_results(lasso_min_lambda_results,'lasso','lambda_min'),
    Store_results(lasso_1se_lambda_results,'lasso','lambda_1se'),
    Store_results(elastic_min_lambda_results,'elastic','lambda_min'),
    Store_results(elastic_1se_lambda_results,'elastic','lambda_1se'),
    Store_results(ridge_min_lambda_results,'ridge','lambda_min'),
    Store_results(ridge_1se_lambda_results,'ridge','lambda_1se'))
  
  
  
  # Plotting histograms of each variable in each outer fold
  plot_lits <- list()
  for (metric in colnames(x_train)){
    metric_median <- format(round(median (train_data[[metric]]),2), nsmall = 2) #only keeping 2 digits after decimal
   p <- ggplot(x_train, aes_string(x = metric)) +
     geom_histogram(bins =30)+
     ggtitle(paste('median =', metric_median ))+
     theme(plot.title = element_text ( size = 12))
   plot_lits[[metric]] <- p
  }
  
  # Combine all plots using patchwork
  hist_plots <- wrap_plots(plot_lits,ncol = 5)+
    plot_annotation(title = paste0('fold number = ',i ))
  
  # Save plot as PNG
  ggsave(filename = paste0("histogram_run2_fold", i, ".png"), plot =hist_plots, width = 15, height = 10)
  
  
  # Plotting the log of lambda based on the mean deviance (lambda.1se which is mean square error) for each cv
  # Set up the file to save the plots
  # Set up the file to save the plots
  png(filename = paste0("deviance_run2_fold", i, ".png"), width = 600, height = 400)
  
  # Set up the plotting area
  p1 <-par(mfrow = c(2,2))
  plot(cvfit_lasso)
  plot(cvfit_elastic)
  plot(cvfit_ridge)
  
  # cvm is mean cross-validated error which here in binomial is mean deviance
  #plotting the behavior of the model
  plot(log(cvfit_lasso$lambda),cvfit_lasso$cvm, pch = 19, col = 'red',
       xlab = "log(Lambda)", ylab = cvfit_lasso$name)
  points(log(cvfit_elastic$lambda),cvfit_elastic$cvm, pch = 19, col = 'blue')
  points(log(cvfit_ridge$lambda), cvfit_ridge$cvm, pch = 19, col = 'green')
  legend('topright', legend = c('alpha = 1 ','alpha = 0.5','alpha = 0'), pch = 19, col = c('red','blue','green') )
  title(paste0('Fold number = ',i))
  
  dev.off()
  


}

# Flatten the list if needed
results_list2 <- do.call(c, results_list)  #do.call applies function, here "c" to all elements in results_list2 to flatent it

# Create a data frame from results
max_length <- max(sapply(results_list2, function(x){length(x$selected_features)}))
result_df <- data.frame(matrix(nrow = 6*num_outer_folds, ncol = max_length+4))
colnames(result_df)<- c("Accuracy", "regression_method","lambda_selection","num_features",paste0("feature_",1:max_length))

# Populate the data frame with results
for (i in 1:60){
  result_df[i,"Accuracy"] <- results_list2[[i]]$accuracy
  result_df[i,"regression_method"] <- results_list2[[i]]$regression_method
  result_df[i,"lambda_selection"] <- results_list2[[i]]$lambda_selection
  current_features <- results_list2[[i]]$selected_features
  current_length <- length(current_features)
  result_df[i,"num_features"] <- as.numeric(current_length)
  result_df[i,5:(current_length+4)] <- current_features
}

result_df$num_features <- as.numeric(result_df$num_features)
# Save to a file
write.csv(result_df, "Feature_selection_results2.csv", row.names = FALSE)

# Calculate summary statistics (this is the output you want to look at)
stats <- result_df %>%
  group_by(regression_method,lambda_selection) %>%
  summarise(
    average_accuracy = mean(Accuracy , na.rm = TRUE),
    mean_num_features = mean(num_features, na.rm = TRUE))



