
#### Libraries ####
library(tidymodels)
library(modeldata)
library(themis)
tidymodels_prefer()

set.seed(1231)

## Load the data 
heart_failure <- readRDS("C:/Users/Huawei/OneDrive/Heart_Failure_Prediction/Heart-Failure-Prediction/Data/Data_cleaned/heart_failure_clean.rds")

## Split the data 
heart_failure_split <- initial_split(heart_failure,prop = 0.8,strata = death)

# Create a training data 
heart_failure_train <- training(heart_failure_split)

# Create a testing data
heart_failure_test <- testing(heart_failure_split)

## Feature Engineering 

# Make a recipe
heart_failure_recipe <- recipe(death~.,data = heart_failure_train) %>%
  
  # Remove near zero var features
  step_nzv(all_nominal_predictors()) %>%
  
  # Normalize all num features 
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Center and Scale all num features
  step_normalize(all_numeric_predictors()) %>%
  
  # Dummy encoding
  step_dummy(all_nominal(), -all_outcomes())
  
## SVM ##

# Define a SVM specifications 
svm_spec <- svm_rbf(cost = tune(),rbf_sigma = tune()) %>%
  set_engine("kernlab")%>%
  set_mode("classification")

# Create a SVM workflow 
svm_workflow <- 
  workflow() %>%
  add_recipe(heart_failure_recipe)%>%
  add_model(svm_spec)

# Resample fuction 10 k cross validation 
cell_folds <- vfold_cv(heart_failure_train,v = 10)

# Metric 
roc_res <- metric_set(roc_auc)

# Set parameters
svm_param <- svm_workflow %>%
  extract_parameter_set_dials() %>%
  update(rbf_sigma = rbf_sigma(c(-6,-1)))

# Start grid with Latin Hypercube
start_grid <- svm_param %>%
  update(
    cost = cost(c(-6,1)),
    rbf_sigma = rbf_sigma(c(-6,-1))
  ) %>%
  grid_latin_hypercube(size = 10)

# Initial search 
svm_initial <- svm_workflow %>%
  tune_grid(resamples = cell_folds,
            grid = start_grid,
            metrics = roc_res
            )
  
# Define a BO (Bayesian optimization)
control_bo <- control_bayes(verbose = TRUE)

# Execute a BO
svm_bo <- 
  svm_workflow %>%
  tune_bayes(
    resamples = cell_folds,
    iter = 25,
    param_info = svm_param,
    metrics = roc_res,
    initial = svm_initial,
    control = control_bo
    )

show_best(svm_bo)
autoplot(svm_bo,type = "performance")









