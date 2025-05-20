#Reading data from short UCB dataset

library(epiDisplay)
library(ggeffects)
library(patchwork)

library(ROCR)
library(ggplot2)


setwd("D:/Venus/Lab projects/NIMBIS/Feature Selection/Sleep_VS_Wakefulness_Feature_Selection")


##############################FUNCTIONS##################################

Range_Scale <- function(data){
  
  (data - min(data, na.rm = TRUE))/(max(data, na.rm = TRUE)-min(data, na.rm = TRUE))
}

Get_model_accuracy <- function(data, variables,fold_size,inner_validation){
  outer_accuracies <- numeric(0)
  inner_accuracies <- numeric(0)
  average_inner_accuracies <- numeric(0)
  my_variables <- character(0)
  
  
  num_patients <- length(data$PatientID)
  patientIDs <- unique(data$PatientID)
  num_patients <- length(patientIDs)
  num_folds <- floor(num_patients/fold_size)
  
  #if (is.character(variables)){
  for (i in 1:length(variables)){
    
    formula_vars <- paste(variables[[i]], collapse  = " + ")
    model_formula <- as.formula(paste("sleep_or_wake ~",formula_vars))
    
    for (j in 1:num_folds){
      start_idx = (j-1)*fold_size+1
      end_idx = min(j*fold_size, num_patients)
      my_variables <- c(my_variables,formula_vars)
      
      testing_patients <- patientIDs[start_idx:end_idx]
      training_patients <- setdiff(patientIDs,testing_patients)
      
      train_data <- data %>% filter(PatientID %in% training_patients )
      test_data <- data %>% filter (PatientID %in% testing_patients)
      
      training_model <- glm(model_formula,data = train_data, family = binomial(link = "logit"))
      
      test_data$predicted_prob <-predict(training_model,newdata = test_data , type = "response") #outer fold accuracy
      
      test_data$predicted_class <- ifelse(test_data$predicted_prob>0.5, 1,0)
      outer_accuracy <- sum(test_data$predicted_class == test_data$sleep_or_wake)/nrow(test_data) #inner accuracy
      outer_accuracies <-c(outer_accuracies,outer_accuracy)
      
      if (inner_validation){
        inner_results <-Get_model_accuracy(train_data,variables[[i]],fold_size,inner_validation = FALSE) #6 to 50
        inner_accuracies = c(inner_accuracies,inner_results$accuracies)
        average_inner_accuracy = mean(inner_results$accuracies) #average over all 9 inner accuracies for 1 metric
        average_inner_accuracies = c(average_inner_accuracies,average_inner_accuracy)
      }
    }
  }
  if (inner_validation){
    return (list(inner_accuracies = inner_accuracies, #150 (10 folds and I have 150)
                 outer_accuracies = outer_accuracies,
                 average_inner_accuracy = average_inner_accuracies,
                 variable = my_variables ))
  }else{
    return(list(accuracies = outer_accuracies, variable = my_variables))
    
    
  }
  #} else {
  stop("The 'variables' argument must be a character vector.")
  #}
}



Run_LGM_Model_nTimes <- function(whole_sleep_data,whole_wake_data,variable_list, fold_size,nTimes) {
  
  
  library(dplyr)
  selected_outer_vars <- character(0)
  selected_inner_vars <- character(0)
  outer_accuracy <- numeric(0)
  inner_accuracy <- numeric (0)
  
  set.seed(853)
  
  for (i in 1:nTimes){
    
    
    sleep_data_random <- whole_sleep_data %>%
      group_by(PatientID) %>%
      slice_sample(n = 1) #Randomly pick one row for each patient
    
    wake_data_random <- whole_wake_data %>%
      group_by(PatientID) %>%
      slice_sample(n = 1) %>%
      ungroup()
    
    random_data = rbind(wake_data_random,sleep_data_random)
    
    
    results <- Get_model_accuracy(random_data,variables_list,fold_size,inner_validation = TRUE)
    df <- data.frame( "variables" = results$variable,
                      "average_inner_accuracy" =  results$average_inner_accuracy,
                      "outer_accuracies" =  results$outer_accuracies)
    df_sumery <- df %>%
      group_by(variables) %>%
      summarise(across(c("average_inner_accuracy","outer_accuracies"), mean, na.rm = TRUE))
    
    best_inner_var <- df_sumery$variables[which.max(df_sumery$average_inner_accuracy)]
    
    best_outer_var <- df_sumery$variables[which.max(df_sumery$outer_accuracies)]
    
    selected_outer_vars <- c(selected_outer_vars,best_outer_var)
    outer_accuracy <- c(outer_accuracy,max(df_sumery$outer_accuracies))
    selected_inner_vars <- c(selected_inner_vars,best_inner_var)
    inner_accuracy <- c(inner_accuracy,max(df_sumery$average_inner_accuracy))
  }
  return(list(selected_outer_vars = selected_outer_vars,
              outer_accuracy = outer_accuracy,
              selected_inner_vars = selected_inner_vars,
              inner_accuracy = inner_accuracy))
}


################################################



#Reading the data
rawData <-read.csv('allPatinetsMetrics.csv', sep=",", header=TRUE)

rawData <- rawData %>%
  mutate(across(sleep_or_awake, ~gsub('Sleep1|Sleep2','sleep'),.))%>%
  mutate(acorss(sleep_pr_awake, ~gsub('Wake1|Wake2','wake'),.))


all_sleep_data = myRawData[myRawData$sleep_or_wake=='sleep',]
all_wake_data = myRawData[myRawData$sleep_or_wake=='wake',]



allColumns <- names(myRawData)
feature_mtercis <-setdiff(names(myRawData),c("PatientID","sleep_or_wake","Clean_30secStart"))


  



##########




variables_list <- c("ShanEntD","ShanEntT","ShanEntA","ShanEntB",
                    "PermEntD","PermEntT","PermEntA","PermEntB",
                    "PowerSpecD","PowerSpecT","PowerSpecA","PowerSpecB","PowerSpecBroad",
                    "SEF",  "Amplitude")

fold_size = 5
nTimes = 1000;
overall_results1 <- Run_LGM_Model_nTimes (all_sleep_data,all_wake_data,
                                         variable_list, fold_size,nTimes)


df_results1 <- data.frame(selected_outer_vars = overall_results1$selected_outer_vars,
                          outer_accuracy = overall_results1$outer_accuracy,
                          selected_inner_vars =overall_results1 $ selected_inner_vars,
                          inner_accuracy =overall_results1 $inner_accuracy)
mean_outer_acc1 <- df_results1 %>%
  group_by(selected_outer_vars) %>%
  summarise(mean_outer_acc = mean(outer_accuracy, na.rm = TRUE))

mean_inner_acc1 <- df_results1 %>%
  group_by(selected_outer_vars) %>%
summarise(mean_inner_acc1 = mean(inner_accuracy, na.rm = TRUE))

best_overal_inner_var1<- names(sort(table(overall_results1$selected_inner_vars),decreasing = TRUE))[1]
best_overal_outer_var1 <- names(sort(table(overall_results1$selected_outer_vars),decreasing = TRUE))[1]




  
# 
#   
# 
#   
# fold_size = 5
# variables_list <- list(c("PowerSpecB", "ShanEntD"),
#                        c("PowerSpecB", "ShanEntT"), 
#                        c("PowerSpecB", "ShanEntA"), 
#                        c("PowerSpecB", "ShanEntB"),
#                        c("PowerSpecB", "PermEntD"),
#                        c("PowerSpecB", "PermEntT"),
#                        c("PowerSpecB", "PermEntA"),
#                        c("PowerSpecB", "PermEntB"),
#                        c("PowerSpecB", "PowerSpecD"),
#                        c("PowerSpecB", "PowerSpecT"),
#                        c("PowerSpecB", "PowerSpecA"),
#                        c("PowerSpecB", "PowerSpecBroad"),
#                        c("PowerSpecB", "SEF"),
#                        c("PowerSpecB", "Amplitude"))
# 
#   
# fold_size = 5
# nTimes = 100
# overall_results2 <- Run_LGM_Model_nTimes (all_sleep_data,all_wake_data,
#                                          variable_list, fold_size,nTimes)
# 
# best_overal_inner_var2<- names(sort(table(overall_results2$selected_inner_vars),decreasing = TRUE))[1]
# best_overal_outer_var2 <- names(sort(table(overall_results2$selected_outer_vars),decreasing = TRUE))[1]
# 
# 
# 

  




