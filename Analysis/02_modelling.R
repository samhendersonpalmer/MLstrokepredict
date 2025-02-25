# Info --------------------------------------------------------------------
# This is the ML script for stroke prediction as a binary classification problem.
# ML methods used include multilayer perceptron (MLP),support vector machine (SVM), 
# gradient boosted decision trees (XGBoost) and random forest (RF). There is also
# an attempt at model stacking (ensemble approach)

# Use the document outline to navigate the various modelling sections of the 
# script.


# Load packages and data --------------------------------------------------
library(tidyverse)
library(tidymodels)
tidymodels_prefer()

stroke_complete <- readRDS("Data/stroke_complete.RDS")
str(stroke_complete)


# Data splitting ----------------------------------------------------------
set.seed(123)
stroke_split <- initial_split(stroke_complete, prop = 0.8, strata = stroke)
stroke_train <- training(stroke_split)
stroke_test <- testing(stroke_split)


## Cross validation --------------------------------------------------------
set.seed(123)
stroke_folds <- vfold_cv(stroke_train, v = 10, strata = stroke)


# Recipes -----------------------------------------------------------------
# Load the themis library required for step_smote to oversample unbalanced data
library(themis)


## Normalised --------------------------------------------------------------

# Used for MLP and SVM
normalised_rec <-
  recipe(stroke ~., data = stroke_train) %>% 
  step_BoxCox(bmi, avg_glucose_level, age) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  # Unbalanced data so use step_smote
  step_smote(stroke, seed = 123)

# We can check the recipe steps have worked
prep(normalised_rec)

# We can also bake this recipe to see it as a dataframe
bake(normalised_rec %>% prep(), new_data = NULL)


## XGBoost -----------------------------------------------------------------

# Used for XGBoost as encoding required
xgb_rec <-
  recipe(stroke ~., data = stroke_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_smote(stroke, seed = 123)


## RF ----------------------------------------------------------------------

# Used for RF so no feature engineering required
simple_rec <-
  recipe(stroke ~., data = stroke_train) %>% 
  # Have had to add step_dummy for step_smote as requires integers
  # RF doesn't typically need any steps
  step_dummy(all_nominal_predictors()) %>%
  step_smote(stroke, seed = 123)



# Metrics -----------------------------------------------------------------

# Set metrics with pr_auc as primary
stroke_metrics <- metric_set(yardstick::pr_auc, yardstick::recall, yardstick::precision, yardstick::f_meas)


# Modelling ---------------------------------------------------------------


## MLP ---------------------------------------------------------------------

### Workflow ----------------------------------------------------------------

# Specify the model
mlp_spec <- mlp(hidden_units = tune(), 
                #penalty      = tune(),
                dropout      = tune(),
                 epochs      = tune(), 
                learn_rate   = tune(),
              activation     = "sigmoid") %>% 
  set_engine("brulee") %>% 
  set_mode("classification")

# Build model workflow
mlp_wflow <-
  workflow() %>% 
  add_model(mlp_spec) %>% 
  add_recipe(normalised_rec)

### Tuning ------------------------------------------------------------------

#### Hyperparameters ---------------------------------------------------------

# We can view which parameters are being tuned:
mlp_spec %>% 
  extract_parameter_set_dials()

# Default parameter ranges can be found in help file
# View default ranges:
hidden_units()
#penalty()
dropout()
epochs()
activation()
learn_rate()

# Update parameter ranges if required
# Create a parameter set that includes updated ranges to pass to tuning
mlp_param <-
  mlp_spec %>% 
  extract_parameter_set_dials() %>% 
  update(
    hidden_units = hidden_units(c(1, 10))
  )

#### Creating grid  ---------------------------------------------------------
set.seed(123)
mlp_grid_tune <-
  mlp_param %>% 
  grid_latin_hypercube(size = 50)

# We can check the parameter combinations
mlp_grid_tune %>% glimpse(width = 50)

#### Racing ------------------------------------------------------------------
library(finetune)
set.seed(123)

mlp_race <- 
  mlp_wflow %>% 
  tune_race_anova(
    stroke_folds,
    grid = mlp_grid_tune,
    metrics = stroke_metrics,
    param_info = mlp_param,
    control = control_race(verbose_elim = TRUE,
                           save_pred = TRUE,
                           save_workflow = TRUE))
Sys.time()
#### Visualise ---------------------------------------------------------------
# Points are too thick on plot in publication so making my own
mlp_tuning_results_blank <- autoplot(mlp_race) +
  scale_colour_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  theme_light()

# Remove geom points
mlp_tuning_results_blank$layers[[1]] <- NULL

# Then add my own labels to highlight those models that made it to the end
mlp_tuning_results <- mlp_tuning_results_blank +
  geom_point(aes(alpha = `# resamples`), stroke=NA, size = 2) +
  theme(legend.position = "top")

# Save these results for figure in project 
saveRDS(mlp_tuning_results, file = "Analysis/Plot objects/mlp_tuning_results.RDS")

# We can also visualise the dropping of poorer parameter configurations:
mlp_race_plot <- plot_race(mlp_race)

# Save these results for figure in project
saveRDS(mlp_race_plot, file = "Analysis/Plot objects/mlp_race_plot.RDS")


#### Show best ---------------------------------------------------------------

# See the top 5 parameter configurations
mlp_show_best <- show_best(mlp_race, metric = "pr_auc") %>% 
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         `AUPRC (95% CI)` = paste0(sprintf("%.3f", round(mean, 3)), 
                                   " (", 
                                   sprintf("%.3f", round(lower95, 3)), 
                                   ", ", 
                                   sprintf("%.3f", round(upper95, 3)), ")"))

# Save these results for table in project
saveRDS(mlp_show_best, file = "Analysis/Plot objects/mlp_show_best.RDS")

### Finalise workflow -------------------------------------------------------
# Select the best model parameter configuration from tuning
# All models relatively similar so will go with one with highest mean AUPRC from 
# resampling which happens to also be simplest with 1 hidden node
mlp_best_pr <- select_best(mlp_race, metric = "pr_auc")

# Finalise by updating workflow with optimal parameters from best model
final_mlp_wflow <-
  mlp_wflow %>% 
  finalize_workflow(mlp_best_pr)

# We can check the performance of this model on resamples to compare to other methods before model selection
set.seed(123)

final_mlp_fit <-
  final_mlp_wflow %>% 
  fit_resamples(stroke_folds, 
                metrics = stroke_metrics, 
                control = control_resamples(save_pred = TRUE))

# Final metrics for tuned model
mlp_metrics_table <- final_mlp_fit %>% 
  collect_metrics()

# With 95% CIs
mlp_PRAUC_table_interval <- mlp_metrics_table %>% 
  filter(.metric == "pr_auc") %>%
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         .estimate = paste0( sprintf("%.3f", round(mean, 3)), 
                             " (",  
                             sprintf("%.3f", round(lower95, 3)),
                             ", ", 
                             sprintf("%.3f",round(upper95, 3)),
                             ")"))


### Decision threshold ------------------------------------------------------

#### Find optimal threshold that maximises F1 as point estimate ------------
library(probably)
mlp_resample_pred <- collect_predictions(final_mlp_fit)

checking_mlp_thresh <- threshold_perf(mlp_resample_pred,
                                        truth = stroke,
                                        .pred_Stroke,
                                        metrics = metric_set(f_meas,recall, precision),
                                        threshold = seq(0, 1, 0.01))

mlp_optimal_f1 <- max(checking_mlp_thresh[checking_mlp_thresh$.metric == "f_meas", ".estimate" ], na.rm = TRUE)
mlp_optimal_f1_thresh <- checking_mlp_thresh %>%
  filter(.metric == "f_meas") %>% 
  slice(which.max(.estimate) )%>% 
  pull(.threshold)
# 0.65 threshold to get optimal F1 of 0.2553191

#### Visualise optimal threshold ------------
checking_mlp_thresh %>% 
  mutate(.metric = case_match(
    .metric,
    "f_meas" ~ "F1 Score",
    "precision" ~ "Precision",
    "recall" ~ "Recall")) %>% 
  ggplot(aes(x = .threshold, y = .estimate, colour = .metric))+
  geom_line() +
  labs(x = "Decision threshold", y = "Metric estimate") +
  theme_minimal() +
  theme(legend.title = element_blank())

#### Use new threshold to generate metrics and confusion matrix ------------
threshold_metrics <- metric_set(f_meas, recall, precision)

mlp_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = mlp_optimal_f1_thresh )) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  threshold_metrics(truth = stroke, estimate = thresh_pred) %>% 
  mutate(.estimate = as.character( sprintf("%.3f", round(.estimate, 3)))) -> mlp_threshold_metrics

mlp_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = mlp_optimal_f1_thresh)) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  conf_mat(stroke, thresh_pred) -> mlp_conf_mat

saveRDS(mlp_conf_mat, file = "Analysis/Plot objects/mlp_conf_mat.RDS")


## Final metrics table -----------------------------------------------------
bind_rows(mlp_PRAUC_table_interval, mlp_threshold_metrics) %>% 
  select(.metric, .estimate) -> mlp_metrics_table_interval


# Save these results for table in project
saveRDS(mlp_metrics_table_interval, file = "Analysis/Plot objects/mlp_resample_metrics.RDS")


## SVM ---------------------------------------------------------------------


### Workflow ----------------------------------------------------------------

# Specify the model
svm_spec <- 
  svm_rbf(cost = tune(), 
          rbf_sigma = tune()
          ) %>% 
  set_engine("kernlab") %>% 
  set_mode("classification")

# Build model workflow
svm_wflow <-
  workflow() %>% 
  add_model(svm_spec) %>% 
  add_recipe(normalised_rec)


### Tuning ------------------------------------------------------------------

#### Hyperparameters ---------------------------------------------------------


# We can view which parameters are being tuned:
svm_spec %>% 
  extract_parameter_set_dials()

# Default parameter ranges can be found in help file
# View default ranges:
cost()
rbf_sigma()

# Update parameter ranges if required
# Create a parameter set that includes updated ranges to pass to tuning
svm_param <-
  svm_spec %>% 
  extract_parameter_set_dials() %>% 
  update(
    cost = cost(c(-10, 15)),
    rbf_sigma = rbf_sigma(c(-10, 1))
  )

#### Creating grid  ---------------------------------------------------------
set.seed(123)
svm_grid_tune <-
  svm_param %>% 
  grid_latin_hypercube(size = 50)

# We can check the parameter combinations
svm_grid_tune %>% glimpse(width = 50)


#### Racing ------------------------------------------------------------------
set.seed(123)

svm_race <- 
  svm_wflow %>% 
  tune_race_anova(
    stroke_folds,
    grid = svm_grid_tune,
    metrics = stroke_metrics,
    param_info = svm_param,
    control = control_race(verbose_elim = TRUE, 
                           save_pred = TRUE, 
                           save_workflow = TRUE))
Sys.time()
#### Visualise ---------------------------------------------------------------
# Points are too thick on plot in publication so making my own
svm_tuning_results_blank <- autoplot(svm_race) +
  scale_colour_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  theme_light()

# Remove geom points
svm_tuning_results_blank$layers[[1]] <- NULL

# Then add my own labels to highlight those models that made it to the end
svm_tuning_results <- svm_tuning_results_blank +
  geom_point(aes(alpha = `# resamples`), stroke=NA, size = 2) +
  theme(legend.position = "top")

# We can also visualse the dropping of poorer parameter configurations:
svm_race_plot <- plot_race(svm_race)

# Save these results for figure in project
saveRDS(svm_race_plot, file = "Analysis/Plot objects/svm_race_plot.RDS")

# Save these results for figure in project
saveRDS(svm_tuning_results, file = "Analysis/Plot objects/svm_tuning_results.RDS")


#### Show best ---------------------------------------------------------------

# See the top 5 parameter configurations
svm_show_best <- show_best(svm_race, metric = "pr_auc")  %>% 
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         `AUPRC (95% CI)` = paste0(sprintf("%.3f", round(mean, 3)), 
                                   " (", 
                                   sprintf("%.3f", round(lower95, 3)), 
                                   ", ", 
                                   sprintf("%.3f", round(upper95, 3)), ")"))

# Save these results for table in project
saveRDS(svm_show_best, file = "Analysis/Plot objects/svm_show_best.RDS")


### Finalise workflow -------------------------------------------------------
# Select the best model parameter configuration from tuning
# All models relatively similar so will go with one with highest mean AUPRC from 
# resampling
svm_best_pr <- select_best(svm_race, metric = "pr_auc")

# Finalise by updating workflow with optimal parameters from best model
final_svm_wflow <-
  svm_wflow %>% 
  finalize_workflow(svm_best_pr)

# We can check the performance of this model on resamples to compare to other methods before model selection
set.seed(123)

final_svm_fit <-
  final_svm_wflow %>% 
  fit_resamples(stroke_folds, 
                metrics = stroke_metrics, 
                control = control_resamples(save_pred = TRUE))

# Final metrics for tuned model
svm_metrics_table <- final_svm_fit %>% 
  collect_metrics()

# With 95% CIs
svm_PRAUC_table_interval <- svm_metrics_table %>% 
  filter(.metric == "pr_auc") %>%
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         .estimate = paste0( sprintf("%.3f", round(mean, 3)), 
                             " (",  
                             sprintf("%.3f", round(lower95, 3)),
                             ", ", 
                             sprintf("%.3f",round(upper95, 3)),
                             ")"))




### Decision threshold ------------------------------------------------------

#### Find optimal threshold that maximises F1 as point estimate ------------
library(probably)
svm_resample_pred <- collect_predictions(final_svm_fit)

checking_svm_thresh <- threshold_perf(svm_resample_pred,
                                      truth = stroke,
                                      .pred_Stroke,
                                      metrics = metric_set(f_meas,recall, precision),
                                      threshold = seq(0, 1, 0.01))

svm_optimal_f1 <- max(checking_svm_thresh[checking_svm_thresh$.metric == "f_meas", ".estimate" ], na.rm = TRUE)
svm_optimal_f1_thresh <- checking_svm_thresh %>%
  filter(.metric == "f_meas") %>% 
  slice(which.max(.estimate) )%>% 
  pull(.threshold)
# 0.77 threshold to get optimal F1 of 0.2764706


#### Visualise optimal threshold ------------
checking_svm_thresh %>% 
  mutate(.metric = case_match(
    .metric,
    "f_meas" ~ "F1 Score",
    "precision" ~ "Precision",
    "recall" ~ "Recall")) %>% 
  ggplot(aes(x = .threshold, y = .estimate, colour = .metric))+
  geom_line() +
  labs(x = "Decision threshold", y = "Metric estimate") +
  theme_minimal() +
  theme(legend.title = element_blank())

#### Use new threshold to generate metrics and confusion matrix ------------
threshold_metrics <- metric_set(f_meas, recall, precision)

svm_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = svm_optimal_f1_thresh )) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  threshold_metrics(truth = stroke, estimate = thresh_pred) %>% 
  mutate(.estimate = as.character(sprintf("%.3f", round(.estimate, 3)))) -> svm_threshold_metrics

svm_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = svm_optimal_f1_thresh)) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  conf_mat(stroke, thresh_pred) -> svm_conf_mat

saveRDS(svm_conf_mat, file = "Analysis/Plot objects/svm_conf_mat.RDS")



## Final metrics table -----------------------------------------------------
bind_rows(svm_PRAUC_table_interval, svm_threshold_metrics) %>% 
  select(.metric, .estimate) -> svm_metrics_table_interval


# Save these results for table in project
saveRDS(svm_metrics_table_interval, file = "Analysis/Plot objects/svm_resample_metrics.RDS")


## XGBoost -----------------------------------------------------------------

### Workflow ----------------------------------------------------------------

# Specify the model
xgb_spec <- 
  boost_tree(mtry = tune(),
             trees = tune(),
             min_n = tune(),
             tree_depth = tune(),
             learn_rate = tune(),
             loss_reduction = tune(),
             sample_size = tune(),
             stop_iter = tune()) %>%  
  set_engine("xgboost") %>% 
  set_mode("classification")

# Build model workflow
xgb_wflow <-
  workflow() %>% 
  add_model(xgb_spec) %>% 
  add_recipe(xgb_rec)

### Tuning ------------------------------------------------------------------

#### Hyperparameters ---------------------------------------------------------


# We can view which parameters are being tuned:
xgb_spec %>% 
  extract_parameter_set_dials()

# Default parameter ranges can be found in help file
# View default ranges:
mtry()
trees()
min_n()
trees()
tree_depth()
learn_rate()
loss_reduction()
sample_size()
stop_iter()

# Update parameter ranges if required
# Create a parameter set that includes updated ranges to pass to tuning

# Create upper bound of randomly selected features for mtry
N_feature <- ncol(stroke_train) - 1

xgb_param <-
  xgb_spec %>% 
  extract_parameter_set_dials() %>% 
  update(
    mtry = mtry(c(1, N_feature)),
    learn_rate = learn_rate(c(-4, -0.5)),
    sample_size = sample_prop(c(0.5, 0.8))
  )

#### Creating grid  ---------------------------------------------------------
set.seed(123)
xgb_grid_tune <-
  xgb_param %>% 
  grid_latin_hypercube(size = 50)

# We can check the parameter combinations
xgb_grid_tune %>% glimpse(width = 50)


#### Racing ------------------------------------------------------------------
set.seed(123)

xgb_race <- 
  xgb_wflow %>% 
  tune_race_anova(
    stroke_folds,
    grid = xgb_grid_tune,
    metrics = stroke_metrics,
    param_info = xgb_param,
    control = control_race(save_pred = TRUE,
                           save_workflow = TRUE))
Sys.time()
#### Visualise ---------------------------------------------------------------
# Points are too thick on plot in publication so making my own
xgb_tuning_results_blank <- autoplot(xgb_race) +
  scale_colour_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  theme_light()

# Remove geom points
xgb_tuning_results_blank$layers[[1]] <- NULL

# Then add my own labels to highlight those models that made it to the end
xgb_tuning_results <- xgb_tuning_results_blank +
  geom_point(aes(alpha = `# resamples`), stroke=NA, size = 2) +
  theme(legend.position = "top")

# We can also visualse the dropping of poorer parameter configurations:
xgb_race_plot <- plot_race(xgb_race)

# Save these results for figure in project
saveRDS(xgb_race_plot, file = "Analysis/Plot objects/xgb_race_plot.RDS")

# Save these results for figure in project
saveRDS(xgb_tuning_results, file = "Analysis/Plot objects/xgb_tuning_results.RDS")


#### Show best ---------------------------------------------------------------

# See the top 5 parameter configurations
xgb_show_best <- show_best(xgb_race, metric = "pr_auc") %>% 
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         `AUPRC (95% CI)` = paste0(sprintf("%.3f", round(mean, 3)), 
                                   " (", 
                                   sprintf("%.3f", round(lower95, 3)), 
                                   ", ", 
                                   sprintf("%.3f", round(upper95, 3)), ")"))


# Save these results for table in project
saveRDS(xgb_show_best, file = "Analysis/Plot objects/xgb_show_best.RDS")


### Finalise workflow -------------------------------------------------------
# Select the best model parameter configuration from tuning
# All models relatively similar so will go with one with the fewer number of trees 
# as less complexity which is model 4 out of the 5 with 322 trees
xgb_best_pr <- xgb_show_best %>% 
  slice(4) %>% 
  select(1:8, .config)

# Finalise by updating workflow with optimal parameters from best model
final_xgb_wflow <-
  xgb_wflow %>% 
  finalize_workflow(xgb_best_pr)

# We can check the performance of this model on resamples to compare to other methods before model selection
set.seed(123)

final_xgb_fit <-
  final_xgb_wflow %>% 
  fit_resamples(stroke_folds, 
                metrics = stroke_metrics, 
                control = control_resamples(save_pred = TRUE))

# Final metrics for tuned model
xgb_metrics_table <- final_xgb_fit %>% 
  collect_metrics()

# With 95% CIs
xgb_PRAUC_table_interval <- xgb_metrics_table %>% 
  filter(.metric == "pr_auc") %>%
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         .estimate = paste0( sprintf("%.3f", round(mean, 3)), 
                             " (",  
                             sprintf("%.3f", round(lower95, 3)),
                             ", ", 
                             sprintf("%.3f",round(upper95, 3)),
                             ")"))




### Decision threshold ------------------------------------------------------

#### Find optimal threshold that maximises F1 as point estimate ------------
library(probably)
xgb_resample_pred <- collect_predictions(final_xgb_fit)

checking_xgb_thresh <- threshold_perf(xgb_resample_pred,
                                      truth = stroke,
                                      .pred_Stroke,
                                      metrics = metric_set(f_meas,recall, precision),
                                      threshold = seq(0, 1, 0.01))

xgb_optimal_f1 <- max(checking_xgb_thresh[checking_xgb_thresh$.metric == "f_meas", ".estimate" ], na.rm = TRUE)
xgb_optimal_f1_thresh <- checking_xgb_thresh %>%
  filter(.metric == "f_meas") %>% 
  slice(which.max(.estimate) )%>% 
  pull(.threshold)
# 0.18 threshold to get optimal F1 of 0.2403561


#### Visualise optimal threshold ------------
checking_xgb_thresh %>% 
  mutate(.metric = case_match(
    .metric,
    "f_meas" ~ "F1 Score",
    "precision" ~ "Precision",
    "recall" ~ "Recall")) %>% 
  ggplot(aes(x = .threshold, y = .estimate, colour = .metric))+
  geom_line() +
  labs(x = "Decision threshold", y = "Metric estimate") +
  theme_minimal() +
  theme(legend.title = element_blank())

#### Use new threshold to generate metrics and confusion matrix ------------
threshold_metrics <- metric_set(f_meas, recall, precision)

xgb_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = xgb_optimal_f1_thresh )) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  threshold_metrics(truth = stroke, estimate = thresh_pred) %>% 
  mutate(.estimate = as.character(sprintf("%.3f",round(.estimate, 3)))) -> xgb_threshold_metrics

xgb_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = xgb_optimal_f1_thresh)) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  conf_mat(stroke, thresh_pred) -> xgb_conf_mat

saveRDS(xgb_conf_mat, file = "Analysis/Plot objects/xgb_conf_mat.RDS")


## Final metrics table -----------------------------------------------------
bind_rows(xgb_PRAUC_table_interval, xgb_threshold_metrics) %>% 
  select(.metric, .estimate) -> xgb_metrics_table_interval


# Save these results for table in project
saveRDS(xgb_metrics_table_interval, file = "Analysis/Plot objects/xgb_resample_metrics.RDS")


## RF ----------------------------------------------------------------------

### Workflow ----------------------------------------------------------------

# Specify the model
rf_spec <- 
  rand_forest(mtry = tune(),
              trees = tune(),
              min_n = tune()) %>%  
  set_engine("ranger") %>% 
  set_mode("classification")

# Build model workflow
rf_wflow <-
  workflow() %>% 
  add_model(rf_spec) %>% 
  add_recipe(simple_rec)


### Tuning ------------------------------------------------------------------

#### Hyperparameters ---------------------------------------------------------


# We can view which parameters are being tuned:
rf_spec %>% 
  extract_parameter_set_dials()

# Default parameter ranges can be found in help file
# View default ranges:
mtry()
trees()
min_n()

# Update parameter ranges if required
# Create a parameter set that includes updated ranges to pass to tuning

# Create upper bound of randomly selected features for mtry
N_feature <- ncol(stroke_train) - 1

rf_param <-
  rf_spec %>% 
  extract_parameter_set_dials() %>% 
  update(
    mtry = mtry(c(1, N_feature)),
    min_n = min_n(c(1,40)))

#### Creating grid  ---------------------------------------------------------
set.seed(123)
rf_grid_tune <-
  rf_param %>% 
  grid_latin_hypercube(size = 50)

# We can check the parameter combinations
rf_grid_tune %>% glimpse(width = 50)


#### Racing ------------------------------------------------------------------
set.seed(123)

rf_race <- 
  rf_wflow %>% 
  tune_race_anova(
    stroke_folds,
    grid = rf_grid_tune,
    metrics = stroke_metrics,
    param_info = rf_param,
    control = control_race(verbose_elim = TRUE,
                           save_pred = TRUE,
                           save_workflow = TRUE))
Sys.time()
#### Visualise ---------------------------------------------------------------
# Had issue not visualising parameter plot properly as only 1 final observation
# so will plot without any label for resample stage first
rf_tuning_results_blank <- autoplot(rf_race %>% select(-.predictions)) +
  scale_colour_viridis_d(direction = -1) +
  theme(legend.position = "top") +
  theme_light()

# Remove geom points
rf_tuning_results_blank$layers[[1]] <- NULL

# Then add my own labels to highlight only model in grid that made it
rf_tuning_results <- rf_tuning_results_blank  +
  geom_point(aes(alpha = `# resamples`), stroke=NA, size = 2) +
  theme(legend.position = "top")

# We can also visualse the dropping of poorer parameter configurations:
rf_race_plot <- plot_race(rf_race)

# Save these results for figure in project
saveRDS(rf_race_plot, file = "Analysis/Plot objects/rf_race_plot.RDS")

# Save these results for figure in project
saveRDS(rf_tuning_results, file = "Analysis/Plot objects/rf_tuning_results.RDS")


#### Show best ---------------------------------------------------------------

# See the top 5 parameter configurations
rf_show_best <- show_best(rf_race, metric = "pr_auc") %>% 
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         `AUPRC (95% CI)` = paste0(sprintf("%.3f", round(mean, 3)), 
                                   " (", 
                                   sprintf("%.3f", round(lower95, 3)), 
                                   ", ", 
                                   sprintf("%.3f", round(upper95, 3)), ")"))

# Save these results for table in project
saveRDS(rf_show_best, file = "Analysis/Plot objects/rf_show_best.RDS")


### Finalise workflow -------------------------------------------------------
# Select the best model parameter configuration from tuning
# Only one model returned from tuning
rf_best_pr <- select_best(rf_race, metric = "pr_auc")

# Finalise by updating workflow with optimal parameters from best model
final_rf_wflow <-
  rf_wflow %>% 
  finalize_workflow(rf_best_pr)

# We can check the performance of this model on resamples to compare to other methods before model selection
set.seed(123)

final_rf_fit <-
  final_rf_wflow %>% 
  fit_resamples(stroke_folds, 
                metrics = stroke_metrics, 
                control = control_resamples(save_pred = TRUE))

# Final metrics for tuned model
rf_metrics_table <- final_rf_fit %>% 
  collect_metrics()

# With 95% CIs
rf_PRAUC_table_interval <- rf_metrics_table %>% 
  filter(.metric == "pr_auc") %>%
  mutate(lower95 = mean - (1.96*std_err),
         upper95 = mean  + (1.96*std_err),
         .estimate = paste0( sprintf("%.3f", round(mean, 3)), 
                             " (",  
                             sprintf("%.3f", round(lower95, 3)),
                             ", ", 
                             sprintf("%.3f",round(upper95, 3)),
                             ")"))




### Decision threshold ------------------------------------------------------

#### Find optimal threshold that maximises F1 as point estimate ------------
library(probably)
rf_resample_pred <- collect_predictions(final_rf_fit)

checking_rf_thresh <- threshold_perf(rf_resample_pred,
                                      truth = stroke,
                                      .pred_Stroke,
                                      metrics = metric_set(f_meas,recall, precision),
                                      threshold = seq(0, 1, 0.01))

rf_optimal_f1 <- max(checking_rf_thresh[checking_rf_thresh$.metric == "f_meas", ".estimate" ], na.rm = TRUE)
rf_optimal_f1_thresh <- checking_rf_thresh %>%
  filter(.metric == "f_meas") %>% 
  slice(which.max(.estimate) )%>% 
  pull(.threshold)
# 0.37 threshold to get optimal F1 of 0.2476636


#### Visualise optimal threshold ------------
checking_rf_thresh %>% 
  mutate(.metric = case_match(
    .metric,
    "f_meas" ~ "F1 Score",
    "precision" ~ "Precision",
    "recall" ~ "Recall")) %>% 
  ggplot(aes(x = .threshold, y = .estimate, colour = .metric))+
  geom_line() +
  labs(x = "Decision threshold", y = "Metric estimate") +
  theme_minimal() +
  theme(legend.title = element_blank())

#### Use new threshold to generate metrics and confusion matrix ------------
threshold_metrics <- metric_set(f_meas, recall, precision)

rf_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = rf_optimal_f1_thresh )) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  threshold_metrics(truth = stroke, estimate = thresh_pred) %>% 
  mutate(.estimate = as.character(sprintf("%.3f",round(.estimate, 3)))) -> rf_threshold_metrics

rf_resample_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = rf_optimal_f1_thresh)) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  conf_mat(stroke, thresh_pred) -> rf_conf_mat

saveRDS(rf_conf_mat, file = "Analysis/Plot objects/rf_conf_mat.RDS")


## Final metrics table -----------------------------------------------------
bind_rows(rf_PRAUC_table_interval, rf_threshold_metrics) %>% 
  select(.metric, .estimate) -> rf_metrics_table_interval


# Save these results for table in project
saveRDS(rf_metrics_table_interval, file = "Analysis/Plot objects/rf_resample_metrics.RDS")


## Stacking ----------------------------------------------------------------

# After tuning each algorithm we can use the results for stacking
library(stacks)


### Assembling --------------------------------------------------------------
# coef is a function in brulee and kernlab so need to specify
# base R coef function to avoid conflicts
conflicted::conflicts_prefer(stats::coef)

# Initialise stack and add candidate members
stroke_stack_data <- 
  stacks() %>% 
  add_candidates(mlp_race, name = "mlp") %>% 
  add_candidates(svm_race, name = "svm") %>% 
  add_candidates(xgb_race, name = "xgb") %>% 
  add_candidates(rf_race, name = "rf")

# To see how many candidate members in each model definition:
stroke_stack_data
# A data stack with 4 model definitions and 47 candidate members:
  #   mlp: 16 model configurations
  #   svm: 10 model configurations
  #   xgb: 20 model configurations
  #   rf: 1 model configuration

# To see the predictions made by each candidate member based on
# out-of-bag cv samples:
as_tibble(stroke_stack_data)

# Create the stacking model by combining predictions
# This creates coefficients for each candidate
set.seed(123)

stroke_stack_model <- 
  stroke_stack_data %>% 
  blend_predictions(metric = stroke_metrics, penalty = 0.01)

# We can view the coefficients (weights) given to each
stroke_stack_model

# Out of 47 possible candidate members, the ensemble retained 4.
# Penalty: 0.01.
# Mixture: 1.
# 
# The 4 highest weighted member classes are:
#   member                   type       weight
# 1 .pred_No.stroke_svm_1_41 svm_rbf     1.57 
# 2 .pred_No.stroke_svm_1_34 svm_rbf     1.26 
# 3 .pred_No.stroke_xgb_1_40 boost_tree  1.13 
# 4 .pred_No.stroke_mlp_1_39 mlp         0.286


### Tuning ------------------------------------------------------------------
# We can check stack has been optimised in minimising 
# number of members by visualising:
autoplot(stroke_stack_model) 

# We can visualise coefficients for each member 
stackplot_object <- autoplot(stroke_stack_model, type = "weights")


# Save this for figure in project
saveRDS(stackplot_object, file = "Analysis/Plot objects/stackplot_weights.RDS")

### Fit members -------------------------------------------------------------
# We can now fit candidates with non-zero stacking coefficients on
# full training set
stroke_stack_fit <-
  stroke_stack_model %>% 
  fit_members()

# We can also see coefficients for each hyperparameter configuration
collect_parameters(stroke_stack_fit, candidates = "svm")

### Metrics on training set --------------------------------------------------------------------
# Now we're ready to see what predictions would have been using out of sample predictions on training set
# First we need to create the predictions the stack makes
stroke_stack_train_pred <- as_tibble(stroke_stack_data) %>% 
  select(stroke,
         .pred_No.stroke_svm_1_41,
         .pred_No.stroke_svm_1_34,
         .pred_No.stroke_xgb_1_40,
         .pred_No.stroke_mlp_1_39) %>% 
  # Add predicted column based on equation
  mutate(.pred_Stroke = stats::binomial()$linkinv(-(0.280334205969429 + (.pred_No.stroke_mlp_1_39 * 
                                                                           0.28643687676548) + (.pred_No.stroke_svm_1_34 * 1.25686636073911) + 
                                                      (.pred_No.stroke_svm_1_41 * 1.57292050112202) + (.pred_No.stroke_xgb_1_40 * 
                                                                                                         1.12963982072838))))

# Calculate PR AUC for the model
yardstick::pr_auc(stroke_stack_train_pred,
                  truth = stroke,
                  .pred_Stroke)

# Make table
stack_PRAUC_table <- yardstick::pr_auc(stroke_stack_train_pred,
                                       truth = stroke,
                                       .pred_Stroke)
  
# pr_auc was 0.261


### Decision threshold ------------------------------------------------------
# The confidence matrix demonstrated poor performance at the default 0.5 decision threshold


#### Find optimal threshold that maximises F1 as point estimate ------------
library(probably)

checking_stack_thresh <- threshold_perf(stroke_stack_train_pred,
                                        truth = stroke,
                                        .pred_Stroke,
                                        metrics = metric_set(f_meas,recall, precision),
                                        threshold = seq(0, 1, 0.01))

stack_optimal_f1 <- max(checking_stack_thresh[checking_stack_thresh$.metric == "f_meas", ".estimate" ], na.rm = TRUE)
stack_optimal_f1_thresh <- checking_stack_thresh %>%
  filter(.metric == "f_meas") %>% 
  slice(which.max(.estimate) )%>% 
  pull(.threshold)
# 0.1 threshold to get optimal F1 of 0.2820809


#### Visualise optimal threshold ------------
checking_stack_thresh %>% 
  mutate(.metric = case_match(
    .metric,
    "f_meas" ~ "F1 Score",
    "precision" ~ "Precision",
    "recall" ~ "Recall")) %>% 
ggplot(aes(x = .threshold, y = .estimate, colour = .metric))+
  geom_line() +
  labs(x = "Decision threshold", y = "Metric estimate") +
  theme_minimal() +
  theme(legend.title = element_blank())

#### Use new threshold to generate metrics and confusion matrix ------------
threshold_metrics <- metric_set(f_meas, recall, precision)

stroke_stack_train_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = stack_optimal_f1_thresh )) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  threshold_metrics(truth = stroke, estimate = thresh_pred) %>% 
  mutate(.estimate =round(.estimate, 3)) -> stack_threshold_metrics

stroke_stack_train_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = stack_optimal_f1_thresh)) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  conf_mat(stroke, thresh_pred) -> stack_conf_mat

saveRDS(stack_conf_mat, file = "Analysis/Plot objects/stack_conf_mat.RDS")


## Final metrics table -----------------------------------------------------
bind_rows(stack_PRAUC_table, stack_threshold_metrics) %>% 
  select(.metric, .estimate) -> stack_metrics_table


# Save these results for table in project
saveRDS(stack_metrics_table, file = "Analysis/Plot objects/stack_train_metrics.RDS")



# Model selection ---------------------------------------------------------


## Comparing metrics -------------------------------------------------------


# Now we can compare the metrics for all algorithms optimal model from the tuning results:
# MLP
mlp_final_metrics <- mlp_metrics_table_interval %>%
  select(metric = .metric, .estimate) %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  mutate(Model = "MLP") %>% 
  select(Model, `AUPRC (95% CI)` = pr_auc, F1 = f_meas, Precision = precision, Recall = recall)

# SVM
svm_final_metrics <- svm_metrics_table_interval %>%
  select(metric = .metric, .estimate) %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  mutate(Model = "SVM") %>% 
  select(Model, `AUPRC (95% CI)` = pr_auc, F1 = f_meas, Precision = precision, Recall = recall)

# XGB
xgb_final_metrics <- xgb_metrics_table_interval %>%
  select(metric = .metric, .estimate) %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  mutate(Model = "XGBoost") %>% 
  select(Model, `AUPRC (95% CI)` = pr_auc, F1 = f_meas, Precision = precision, Recall = recall)

# RF
rf_final_metrics <- rf_metrics_table_interval %>%
  select(metric = .metric, .estimate) %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  mutate(Model = "RF") %>% 
  select(Model, `AUPRC (95% CI)` = pr_auc, F1 = f_meas, Precision = precision, Recall = recall)

# Stacked model
stack_final_metrics <- stack_metrics_table %>%
  mutate(.estimate = as.character(round(.estimate, 3))) %>% 
  select(metric = .metric, .estimate) %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  mutate(Model = "Stacked") %>% 
  select(Model, `AUPRC (95% CI)` = pr_auc, F1 = f_meas, Precision = precision, Recall = recall)

# Now combine into dataframe object
final_metric_df <- bind_rows(mlp_final_metrics,
                             svm_final_metrics,
                             xgb_final_metrics,
                             rf_final_metrics,
                             stack_final_metrics)

# Save this for table in project
saveRDS(final_metric_df, file = "Analysis/Plot objects/final_metric_df.RDS")


## Individual last fit ----------------------------------------------------------------

# Performance of individual models very similar although SVM demonstrated
# highest mean AUPRC of 0.195 however overlap in performance across models so likely
# similar performance could be obtained using other ML methods 

# Therefore we can now use last_fit to train model on full training set and test on test set
svm_last_fit <- 
  final_svm_wflow %>% 
  last_fit(split = stroke_split,
           metrics = stroke_metrics)

# View metrics for final fit
svm_last_fit_metrics_all <- collect_metrics(svm_last_fit)

svm_PRAUC_table_lastfit <- svm_last_fit_metrics_all %>% 
  filter(.metric == "pr_auc") 

# collect predictions
svm_test_pred <- collect_predictions(svm_last_fit)

### Decision threshold ------------------------------------------------------

#### Find optimal threshold that maximises F1 as point estimate ------------
library(probably)

checking_svm_last_thresh <- threshold_perf(svm_test_pred,
                                        truth = stroke,
                                        .pred_Stroke,
                                        metrics = metric_set(f_meas,recall, precision),
                                        threshold = seq(0, 1, 0.01))

svm_last_optimal_f1 <- max(checking_svm_last_thresh[checking_svm_last_thresh$.metric == "f_meas", ".estimate" ], na.rm = TRUE)
svm_last_optimal_f1_thresh <- checking_svm_last_thresh %>%
  filter(.metric == "f_meas") %>% 
  slice(which.max(.estimate) )%>% 
  pull(.threshold)
# 0.74 threshold to get optimal F1 of 0.2974359

#### Visualise optimal threshold ------------
checking_svm_last_thresh %>% 
  mutate(.metric = case_match(
    .metric,
    "f_meas" ~ "F1 Score",
    "precision" ~ "Precision",
    "recall" ~ "Recall")) %>% 
  ggplot(aes(x = .threshold, y = .estimate, colour = .metric))+
  geom_line() +
  labs(x = "Decision threshold", y = "Metric estimate") +
  theme_minimal() +
  theme(legend.title = element_blank())

#### Use new threshold to generate metrics and confusion matrix ------------
threshold_metrics <- metric_set(f_meas, recall, precision)

svm_test_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = svm_last_optimal_f1_thresh )) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  threshold_metrics(truth = stroke, estimate = thresh_pred) %>% 
  mutate(.estimate =round(.estimate, 3)) -> svm_last_threshold_metrics


### Final metrics table -----------------------------------------------------
bind_rows(svm_PRAUC_table_lastfit, svm_last_threshold_metrics) %>% 
  select(.metric, .estimate) -> svm_last_fit_metrics


# Save these results for table in project
saveRDS(svm_last_fit_metrics, file = "Analysis/Plot objects/svm_last_fit_metrics.RDS")

### PRC ---------------------------------------------------------------------

# Final AUPRC
svm_pr_curve_final <- collect_predictions(svm_last_fit) %>% 
  pr_curve(stroke, .pred_Stroke) 

svm_pr_curve_final %>% 
  ggplot(aes(recall, precision)) +
  geom_line() +
  labs(x = "Recall", y = "Precision")

# Save these results for figure in project
saveRDS(svm_pr_curve_final , file = "Analysis/Plot objects/svm_pr_curve_final.RDS")

### Confusion matrix --------------------------------------------------------

# Confusion matrix for final fit using threshold of 0.74
svm_last_fit_conf_mat <- svm_test_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = svm_last_optimal_f1_thresh )) %>% 
  conf_mat(stroke, thresh_pred)

# Save this for table in project
saveRDS(svm_last_fit_conf_mat, file = "Analysis/Plot objects/svm_last_fit_conf_mat.RDS")


## Stacking last fit --------------------------------------------------------------------
# Now we're ready to predict on test dataset
stroke_stack_pred <-
  stroke_test %>% 
  bind_cols(predict(stroke_stack_fit, ., type = "class"),
            predict(stroke_stack_fit, ., type = "prob"))



# Calculate PR AUC for the model
yardstick::pr_auc(stroke_stack_pred,
                  truth = stroke,
                  .pred_Stroke)
# pr_auc was 0.176

stack_final_PRAUC_table <- yardstick::pr_auc(stroke_stack_pred,
                                       truth = stroke,
                                       .pred_Stroke)

### Decision threshold ------------------------------------------------------

#### Find optimal threshold that maximises F1 as point estimate ------------
library(probably)

checking_stack_last_thresh <- threshold_perf(stroke_stack_pred,
                                        truth = stroke,
                                        .pred_Stroke,
                                        metrics = metric_set(f_meas,recall, precision),
                                        threshold = seq(0, 1, 0.01))

stack_last_optimal_f1 <- max(checking_stack_last_thresh[checking_stack_last_thresh$.metric == "f_meas", ".estimate" ], na.rm = TRUE)
stack_last_optimal_f1_thresh <- checking_stack_last_thresh %>%
  filter(.metric == "f_meas") %>% 
  slice(which.max(.estimate) )%>% 
  pull(.threshold)
# 0.13 threshold to get optimal F1 of 0.3040936


#### Visualise optimal threshold ------------
checking_stack_last_thresh %>% 
  mutate(.metric = case_match(
    .metric,
    "f_meas" ~ "F1 Score",
    "precision" ~ "Precision",
    "recall" ~ "Recall")) %>% 
  ggplot(aes(x = .threshold, y = .estimate, colour = .metric))+
  geom_line() +
  labs(x = "Decision threshold", y = "Metric estimate") +
  theme_minimal() +
  theme(legend.title = element_blank())

#### Use new threshold to generate metrics and confusion matrix ------------
threshold_metrics <- metric_set(f_meas, recall, precision)

stroke_stack_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = stack_last_optimal_f1_thresh )) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  threshold_metrics(truth = stroke, estimate = thresh_pred) %>% 
  mutate(.estimate =round(.estimate, 3)) -> stack_last_threshold_metrics

stroke_stack_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = stack_last_optimal_f1_thresh)) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  conf_mat(stroke, thresh_pred)


### Final metrics table -----------------------------------------------------
bind_rows(stack_final_PRAUC_table, stack_last_threshold_metrics) %>% 
  select(.metric, .estimate) -> stack_last_metrics_table


# Save these results for table in project
saveRDS(stack_last_metrics_table, file = "Analysis/Plot objects/stack_last_metrics_table.RDS")

### PRC ---------------------------------------------------------------------

stacked_pr_curve_final <- stroke_stack_pred %>% 
  pr_curve(stroke, .pred_Stroke) 

stacked_pr_curve_final %>% 
  ggplot(aes(recall, precision)) +
  geom_line() +
  labs(x = "Recall", y = "Precision")


# Save these results for figure in project
saveRDS(stacked_pr_curve_final , file = "Analysis/Plot objects/stacked_pr_curve_final.RDS")

### Confusion matrix --------------------------------------------------------


# Confidence matrix to check classification
stack_last_conf_mat <- 
  stroke_stack_pred  %>% 
  mutate(thresh_pred = make_two_class_pred(
    estimate = .pred_Stroke,
    levels = levels(stroke),
    threshold = stack_last_optimal_f1_thresh)) %>% 
  select(stroke, thresh_pred, .pred_Stroke) %>% 
  conf_mat(stroke, thresh_pred)

# Save this for table in project
saveRDS(stack_last_conf_mat, file = "Analysis/Plot objects/stack_last_conf_mat.RDS")


# Last fit metrics table --------------------------------------------------

# SVM
svm_last_table_metrics <- svm_last_fit_metrics %>%
  select(metric = .metric, .estimate) %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  mutate(Model = "SVM") %>% 
  select(Model, `AUPRC` = pr_auc, `F1 score` = f_meas, Precision = precision, Recall = recall)


# Stacked model
stack_last_table_metrics <- stack_last_metrics_table %>%
  select(metric = .metric, .estimate) %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  mutate(Model = "Stacked") %>% 
  select(Model, `AUPRC` = pr_auc, `F1 score` = f_meas, Precision = precision, Recall = recall)


# Now combine into dataframe object
last_metric_df <- bind_rows(svm_last_table_metrics,
                            stack_last_table_metrics)

# Save this for table in project
saveRDS(last_metric_df, file = "Analysis/Plot objects/last_metric_df.RDS")


# PRC combined ------------------------------------------------------------


p1 <- stacked_pr_curve_final %>% mutate(Model = "Stacked")
p2 <- svm_pr_curve_final %>% mutate(Model = "SVM")
p3 <- bind_rows(p1, p2)

p3 %>% 
  filter(.threshold < Inf) %>% 
  ggplot(aes(recall, precision, colour = Model)) +
  geom_line() +
  labs(x = "Recall", y = "Precision") +
  scale_color_brewer(palette="Dark2") +
  ylim(c(0,1)) +
  theme_minimal()

# Single SVM one for project

p2 %>% 
  filter(.threshold < Inf) %>% 
  ggplot(aes(recall, precision)) +
  geom_line(colour = "#F49509", size = 1) +
  labs(x = "Recall", y = "Precision") +
  ylim(c(0,1)) +
  theme_classic() +
  theme(axis.text.x = element_text(colour = "black", size = 20),
        axis.text.y = element_text(colour = "black", size = 20))

# Save these results for figure in project
saveRDS(p3, file = "Analysis/Plot objects/pr_curve_combined.RDS")


# Model interpretation ----------------------------------------------------


## Feature importance ------------------------------------------------------

# Using the final model we can derive feature importance
vip_test <- stroke_test %>% 
  select(-stroke)

# The model is then fitted to the full training data to pass to explainer
svm_model_fit <-
  final_svm_wflow %>% 
  fit(stroke_train)

library(DALEXtra)
explainer_svm <-
  explain_tidymodels(
    svm_model_fit,
    data = vip_test,
    y = stroke_test$stroke,
    predict_function_target_column = 1,
    label = "SVM",
    verbose = TRUE,
    type = "classification")

# Changed loss metric to pr_auc as that's how the model has been evaluated
# So normally 1-AUC loss is used because permutation usually makes AUC worse for important variables so 
# AUC goes down therefore 1-AUC goes up so the plot shows the loss
set.seed(123)
stroke_svm_featureimp <- model_parts(explainer_svm, loss_function = loss_yardstick(pr_auc, reverse = TRUE), B = 100)

# Save this for figure or table in project
saveRDS(stroke_svm_featureimp, file = "Analysis/Plot objects/feature_importance_testdata.RDS")

# Feature importance plot:
plot(stroke_svm_featureimp)


plot_fi <- stroke_svm_featureimp %>% mutate(variable = case_match(variable,
                                                          "work_type" ~ "Work type",
                                                          "Residence_type" ~ "Residence type",
                                                          "gender" ~ "Gender",
                                                          "bmi" ~ "BMI",
                                                          "ever_married" ~ "Ever married",
                                                          "avg_glucose_level" ~ "Average glucose level",
                                                          "hypertension"~ "Hypertension",
                                                          "smoking_status"~"Smoking status",
                                                          "heart_disease"~ "Heart disease",
                                                          "age"~"Age",
                                                          .default =variable)) %>%  plot()+ labs(y = "1 - AUPRC loss after 100 permutations")
# Change colour for presentation
plot_fi +
  theme(axis.text=element_text(colour="black", size = 15))



# Had to remove legacy title from incorrect "Multilayer perceptron" label as now SVM
plot_fi$labels$subtitle <- NULL
plot_fi$data <- plot_fi$data %>% 
  mutate(label = case_match(label,
                            "SVM" ~ "Multilayer Perceptron"))
plot_fi <- plot_fi+ theme(
  strip.background = element_blank(),
  strip.text.x = element_blank()
)



# Save this for figure or table in project
saveRDS(plot_fi, file = "Analysis/Plot objects/plot_fi.RDS")


## Partial effects ----------------------------------------------

# Partial dependence plots for general trend:
pdp_svm <- model_profile(explainer_svm, N = 100,
                        variables = c("gender",
                                      "age",
                                      "hypertension",
                                      "heart_disease",
                                      "ever_married",
                                      "work_type",
                                      "Residence_type",
                                      "avg_glucose_level",
                                      "bmi",
                                      "smoking_status"))
plot(pdp_svm, geom = "profiles")

## Local effects ------------------------------------------------

# Local dependence profile:
ldp_svm <- model_profile(explainer_svm, N =100,
                        type = "conditional",
                        variables = c("gender",
                                      "age",
                                      "hypertension",
                                      "heart_disease",
                                      "ever_married",
                                      "work_type",
                                      "Residence_type",
                                      "avg_glucose_level",
                                      "bmi",
                                      "smoking_status"))

# Save this for figure or table in project
saveRDS(ldp_svm$agr_profiles, file = "Analysis/Plot objects/ldp_testdata.RDS")


plot(ldp_svm)

## Accumulated local effects ---------------------------------------
alp_svm <- model_profile(explainer_svm, N = 500,
                        type = "accumulated", 
                        variables = c("age",
                                      "avg_glucose_level",
                                      "bmi"))

# Save this for figure or table in project
saveRDS(alp_svm$agr_profiles, file = "Analysis/Plot objects/alp_testdata.RDS")

plot(alp_svm)


# For presentation
as_tibble(alp_svm$agr_profiles) %>%
  select(-`_label_` ) %>%
  rename(variable = "_vname_") %>% 
  filter(variable == "age") %>% 
  mutate(variable = case_match(variable,
                               "age" ~ "Age (Years)")) %>% 
  ggplot(aes(`_x_`, `_yhat_`)) +
  geom_line(size = 2, alpha = 0.8, colour = "#F49509") +
  ylim(c(0,1)) +
  #facet_wrap(~variable, scales = "free_x") +
  labs(
    x = NULL,
    y = NULL,
    title = NULL,
    subtitle = NULL) +
  theme_classic()+
  theme(text = element_text(colour = "black"),
        axis.text = element_text(colour = "black", size = 15))

# For categorical features
alp_svm_cat <- model_profile(explainer_svm, N = 500,
                             type = "accumulated", 
                             variables = c("gender",
                                           "hypertension",
                                           "heart_disease",
                                           "ever_married",
                                           "work_type",
                                           "Residence_type",
                                           "smoking_status"),
                         variable_type = "categorical")

# Save this for figure or table in project
saveRDS(alp_svm_cat$agr_profiles, file = "Analysis/Plot objects/alp_testdata_cat.RDS")


plot(alp_svm_cat)