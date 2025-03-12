
#### Libraries ####
library(tidymodels)
library(modeldata)
library(themis)
library(finetune)
library(DALEXtra)
library(ROCR)

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

# Extract the best model
model_best_rf <- select_best(models_bo,metric = "roc_auc")

# Finalize the workflow with the best params
final_bo_rf_workflow <- finalize_workflow(rf_workflow,model_best_rf)

# Fit the model 
final_rf_model <- fit(final_bo_rf_workflow,heart_failure_train)

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

#### Model Evaluation ####

## Predict on the testing set 
prediction <- predict(final_rf_model,new_data = heart_failure_test,type = "prob")

## Take the prob of the pos class
prediction_pos <- prediction$.pred_0

# Combine the actual prediction with the prob of the pos class
data <- tibble(
  truth = heart_failure_test$death, 
  .pred = prediction_pos  
)

# Compute the ROC curve
roc_result <- roc_curve(data, truth ,.pred)

## Plot the ROC
ggplot(roc_result, aes(x = 1 - specificity, y = sensitivity)) + 
  geom_line(color = "blue") + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +  # Diagonal line
  labs(x = "False Positive Rate", y = "True Positive Rate", title = "ROC Curve") +
  theme_minimal()

#### Feature Importance ####

# Create a vector of features 
vip_features <- c("age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure",
                  "platelets","serum_creatinine","serum_sodium","sex","smoking","time")

# Select of the features from the training set
vip_train <- heart_failure_train %>%
  select(all_of(vip_features))

# Create explainer 
explainer_rf <-
  explain_tidymodels(
    model = final_rf_model,
    data = vip_train,
    y = as.numeric(as.character(heart_failure_train$death)),
    verbose = FALSE
    )

## Local Explanations 

# Get 1000 obs from the vip_train
observations <- vip_train[1000,]

# Break down the obs importance 
rf_break_down <- predict_parts(
  explainer = explainer_rf,
  new_observation = observations,
  type = "shap",
  B = 20
  )

# Plot the Importance 
rf_break_down %>%
  group_by(variable) %>%
  mutate(mean_val = mean(contribution)) %>%
  ungroup() %>%
  mutate(variable = fct_reorder(variable, abs(mean_val))) %>%
  ggplot(aes(contribution, variable, fill = mean_val > 0)) +
  geom_boxplot(width = 0.5) +
  theme_minimal()+
  scale_fill_viridis_d(option = "magma") +
  labs(
    y = "Feature Importance",
    x = "Contribution",
    title = "SHAP Local Interpretations")+
  theme(legend.position = "none")

## Global Explanations 
vip_global_rf <- model_parts(explainer = explainer_rf)

plot(vip_global_rf)






