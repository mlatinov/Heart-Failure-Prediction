
#### Libraries ####
library(tidymodels)
library(modeldata)
library(themis)
library(finetune)

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
  
#### Define Models ####

# Define a SVM specifications 
svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune())%>%
  set_engine("kernlab")%>%
  set_mode("classification")

# Define a  RF specifications
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 120)%>%
  set_engine("ranger") %>%
  set_mode("classification")

# Define a KNN specifications
knn_spec <- nearest_neighbor(
  neighbors = tune())%>%
  set_engine("kknn")%>%
  set_mode("classification")

# Define a XGBoost specifications
xgb_spec <- boost_tree(
  mtry = tune(),        # N of predictors that will be randomly sampled at each split 
  min_n = tune(),       # Minimum number of data points in a node that is required for the node to be split further
  learn_rate = tune(),  # Rate at which the boosting algorithm adapts from iteration-to-iteration
  sample_size = tune(), # Proportion of data that is exposed to the fitting routine.
  trees = tune(),      # Number of trees contained in the ensemble
  loss_reduction = tune()) %>%
  set_engine("xgboost")%>%
  set_mode("classification")

# Define a Log Regression specifications
logr_spec <- logistic_reg(
  penalty = tune(),      #   L2 regularization
  mixture = tune()) %>%  #   L1 regularization 
  set_engine("glmnet")%>%
  set_mode("classification")

#### Models Workflow ####

# Create a SVM workflow 
svm_workflow <- 
  workflow() %>%
  add_recipe(heart_failure_recipe)%>%
  add_model(svm_spec)

# Create a RF workflow 
rf_workflow <-
  workflow() %>%
  add_recipe(heart_failure_recipe)%>%
  add_model(rf_spec)

# Create a KNN workflow
knn_workflow <-
  workflow() %>%
  add_recipe(heart_failure_recipe) %>%
  add_model(knn_spec)

# Create a XGB workflow 
xgb_workflow <-
  workflow() %>%
  add_recipe(heart_failure_recipe) %>%
  add_model(xgb_spec)

# Create a Logistic Regression workflow
logr_workflow <-
  workflow()%>%
  add_recipe(heart_failure_recipe)%>%
  add_model(logr_spec)

#### Set parameters ####

# Set parameters SVM
svm_param <- 
  svm_workflow %>%
  extract_parameter_set_dials() %>%
  update(rbf_sigma = rbf_sigma(c(-6,-1)))

# Set parameters RF
rf_param <- 
  rf_workflow %>%
  extract_parameter_set_dials() %>%
  finalize(heart_failure_train) %>%
  update(
    mtry = mtry(c(1L,12L)),
    min_n = min_n(c(2L,20L))
  )

# Set parameters KNN
knn_param <- 
  knn_workflow %>%
  extract_parameter_set_dials()%>%
  update(neighbors = neighbors(c(1L,20L)))

# Set parameters XGB

xgb_param <- 
  xgb_workflow %>%
  extract_parameter_set_dials()%>%
  finalize(heart_failure_train) %>%
  update(
    mtry = mtry(c(1,12)),
    min_n = min_n(c(2,20)),
    learn_rate = learn_rate(c(-10,-2)),
    sample_size = sample_size(c(0,1)),
    trees = trees(c(500,2000)),
    loss_reduction = loss_reduction(c(-10,1))
  )

# Set parameters Log Regression
logr_param <- 
  logr_workflow %>%
  extract_parameter_set_dials()%>%
  update(
    penalty = penalty(c(-10,0)),
    mixture = mixture(c(0,1))
  )

#### Start Grid with LHC ####

# Start Grid with SVM 
start_grid_svm <- svm_param %>%
  update(
    cost = cost(c(-6,1)),
    rbf_sigma = rbf_sigma(c(-6,-1))) %>%
  grid_latin_hypercube(size = 10)

# Start Grid with FR 
start_grid_rf <- rf_param %>%
  update(
    mtry = mtry(c(1L,12L)),
    min_n = min_n(c(2L,20L)))%>%
  grid_latin_hypercube(size = 10)
  
# Start Grid with KNN
start_grid_knn <- knn_param %>%
  update(
    neighbors = neighbors(c(1,20))) %>%
  grid_latin_hypercube(size = 10)
  
# Start Grid with XGB
start_grid_xgb <- xgb_param %>%
  update(
    mtry = mtry(c(1,12)),
    min_n = min_n(c(2,20)),
    learn_rate = learn_rate(c(-10,-2)),
    sample_size = sample_size(c(0,1)),
    trees = trees(c(500,2000)),
    loss_reduction = loss_reduction(c(-10,1))) %>%
  grid_latin_hypercube(size = 10)

# Start Grid with Log Regression
start_grid_logr <- logr_param %>%
  update(
    penalty = penalty(c(-10,0)),
    mixture = mixture(c(0,1))) %>%
  grid_latin_hypercube(size = 10)

#### Resample and Eval Metric ####

# Cross Validation k = 10
cell_folds <- vfold_cv(heart_failure_train,v = 10)

# Metric AUC
roc_res <- metric_set(roc_auc)

#### Initial Search ####

# SVM Initial search 
svm_initial <- svm_workflow %>%
  tune_grid(
    resamples = cell_folds,
    grid = start_grid_svm,
    metrics = roc_res
    )

# Collect the results 
model_metric <- svm_initial %>%
  collect_metrics() %>%
  select(mean) %>%
  mutate(svm_max_auc = max(mean))
  

# RF Initial search
rf_initial <- rf_workflow %>%
  tune_grid(
    resamples = cell_folds,
    grid = start_grid_rf,
    metrics = roc_res
  )

# Collect the results 
model_metric$rf_max_auc <- rf_initial %>%
  collect_metrics() %>%
  select(mean)%>%
  summarise(
    rf_max_auc = max(mean)
  )

# KNN Initial search
knn_initial <- knn_workflow %>%
  tune_grid(
    resamples = cell_folds,
    grid = start_grid_knn,
    metrics = roc_res
  )

# Collect the results 
model_metric$knn_max_auc <- knn_initial %>%
  collect_metrics() %>%
  select(mean)%>%
  summarise(
    knn_max_auc = max(mean)
  )

# XGB Initial search
xgb_initial <- xgb_workflow %>%
  tune_grid(
    resamples = cell_folds,
    grid = start_grid_xgb,
    metrics = roc_res
  )

# Collect the results 
model_metric$xgb_max_auc <- xgb_initial %>%
  collect_metrics() %>%
  select(mean)%>%
  summarise(
    xgb_max_auc = max(mean)
  )

# Logr Initial search
logr_initial <- logr_workflow %>%
  tune_grid(
    resamples = cell_folds,
    grid = start_grid_logr,
    metrics = roc_res
  )

# Collect the results 
model_metric$logr_max_auc <- logr_initial %>%
  collect_metrics() %>%
  select(mean)%>%
  summarise(
    logr_max_auc = max(mean)
  )

# Clean the results 
# Unnest all nested tibbles 
model_metric_clean <- model_metric %>%
  mutate(across(where(is.list), ~ map_dbl(.x, mean)))


## Plot the models 
model_metric_clean %>%
  select(-mean) %>%
  pivot_longer(cols = everything(), names_to = "Models", values_to = "Value") %>%
  ggplot(aes(x = reorder(Models, Value), y = Value, fill = Models)) +  
  geom_col() +
  scale_fill_viridis_d() +  
  coord_flip() + 
  labs(
    title = "Model Performance Comparison",
    x = "Models",
    y = "AUC Value",
    fill = "Models") +
  theme_minimal()+
  theme(
    strip.text = element_text(size = 12),
    title = element_text(size = 15),
    axis.title = element_text(size = 12)
  )


#### Bayesian optimization ####

# Define a BO control
control_bo <- control_bayes(verbose = TRUE)

# Execute a BO on the Random Forest Model
models_bo <- 
  rf_workflow %>%
  tune_bayes(
    resamples = cell_folds,
    initial = rf_initial,
    param_info = rf_param,
    metrics = roc_res,
    iter = 20,
    control = control_bo
  )

#### Simulated Annealing ####

# Define a SA control
control_sa <- control_sim_anneal(verbose = TRUE)

# Execute SA 
models_sa <- 
  rf_workflow %>%
  tune_sim_anneal(
    resamples = cell_folds,
    initial = rf_initial,
    param_info = rf_param,
    metrics = roc_res,
    iter = 20,
    control = control_sa
  )






