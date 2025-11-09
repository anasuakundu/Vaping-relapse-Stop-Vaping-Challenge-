library(dplyr)
library(tidyr)
library(devtools)
library(forcats)
library(tableone)
library(naniar)
library(mice)
library(xgboost)
library(ggplot2)
library(caret)
library(Hmisc)
library(pROC)
library(ROCR)
library(gbm)
library(survival)
library(SHAPforxgboost)
library(data.table)
library(shapviz)
library(reshape2)
library(lubridate)
library(Boruta)
library(survAUC)
library(survXgboost)
library(survex)
library(patchwork)
library(kernelshap)
library(gmodels)
library(shapr)
library(randomForestSRC)
library(survminer)
library(timeROC)
library(progress)
library(fastshap)
library(gridExtra)
library(ggpubr)
library(grid)
library(corrplot)
library(rms)
library(glmnet)

setwd("C:/Anya/IMS PhD/Thesis/Stop vaping challenge/Data 290425")

######1st dataset#####
#load data
dat <- read.csv("imputed_dataset_final.csv")
str(dat)
dat$X<- NULL

###reinserting factor variables as factor
dat[,c(1:5, 7:11, 13:15, 19:29, 31, 32,35:37, 39)] <- lapply(dat[,c(1:5, 7:11, 13:15, 19:29, 31, 32,35:37, 39)], as.factor)

#scaling numeric variables
dat[,c(6,12,16,17,18,30,33, 34)] <- lapply(dat[,c(6,12,16,17,18,30,33, 34)], function(x) c(scale(x)))
str(dat)
#n=387, p=39

###checking collinearity in dataset
df_numeric <- dat[sapply(dat, is.numeric)]
cor_numeric <- cor(df_numeric, method = "pearson")
corrplot(cor_numeric, method = "circle", title = "Pearson Correlation (Numeric)", mar = c(0,0,2,0))
cor_df <- as.data.frame(as.table(cor_numeric))
cor_df <- subset(cor_df, abs(Freq) >= 0.6 & Var1 != Var2)
#none of the numerical variable pairs had pearson correlation more than or equal 0.6

df_cat <- dat[sapply(dat, is.factor)]
cat_cor <- DescTools::PairApply(df_cat, DescTools::CramerV)
corrplot(cat_cor, method = "circle", title = "Cramér's V (Categorical)", mar = c(0,0,2,0))
cat_cor_df <- as.data.frame(as.table(cat_cor))
cat_cor_df <- subset(cat_cor_df, abs(Freq) >= 0.6 & Var1 != Var2)
#none of the categorical variable pairs had Cramer's V more than or equal 0.6

###Random survival forest#######
###############################
#nested cross-validation parameters
outerfold <- 10
innerfold <- 5

# Define the tuning grid
mtry <- round(sqrt(ncol(dat) - 2))
rf.grid <- expand.grid(mtry = c((mtry/2):mtry), ntree = c(200, 500, 700))

# Folds for outer cross-validation
set.seed(123)
folds_list <- createFolds(dat$status, k = outerfold, list = TRUE)

nested.cv.rf<- list(c_index = numeric(), mtry = numeric(), ntree = numeric(), train_data = list(),test_data = list())

# Building RSF model
set.seed(123)
for (i in 1:outerfold) {
  cat("Processing outer fold:", i, "\n")
  fold <- folds_list[[i]]
  train_data <- dat[-fold, ]
  test_data <- dat[fold, ]
  innerfolds <- createFolds(train_data$status, k = innerfold, list = TRUE)
  best_c_index <- 0
  best_mtry <- NULL
  best_ntree <- NULL
  for (grid in 1:nrow(rf.grid)) {
    mtry_val <- rf.grid$mtry[grid]
    ntree_val <- rf.grid$ntree[grid]
    inner_c_indices <- numeric()
    for (inner_fold in innerfolds) {
      inner_train_data <- train_data[-inner_fold, ]
      inner_valid_data <- train_data[inner_fold, ]
      rf_model<- rfsrc(Surv(Time.elapsed, status) ~ ., data = inner_train_data,
                        ntree = ntree_val, mtry = mtry_val)
      rf_pred <- predict(rf_model, newdata = inner_valid_data)
      cif_event2 <- rf_pred$predicted[, "event.2"]
      rc <- rcorr.cens(-cif_event2, Surv(inner_valid_data$Time.elapsed, inner_valid_data$status == 1))
      inner_c_indices <- c(inner_c_indices, rc[1])
    }
    mean_c_index <- mean(inner_c_indices)
    if (mean_c_index > best_c_index) {
      best_c_index <- mean_c_index
      best_mtry <- mtry_val
      best_ntree <- ntree_val
    }
  }
  best_rf_model<- rfsrc(Surv(Time.elapsed, status) ~ ., data = train_data,
                         ntree = best_ntree, mtry = best_mtry)
  rf_pred <- predict(best_rf_model, newdata = test_data)
  cif_event2 <- rf_pred$predicted[, "event.2"]
  rc <- rcorr.cens(-cif_event2, Surv(test_data$Time.elapsed, test_data$status == 1))
  c_index <- rc[1]
  nested.cv.rf$c_index <- c(nested.cv.rf$c_index, c_index)
  nested.cv.rf$mtry <- c(nested.cv.rf$mtry, best_mtry)
  nested.cv.rf$ntree <- c(nested.cv.rf$ntree, best_ntree)
  nested.cv.rf$train_data[[i]] <- train_data
  nested.cv.rf$test_data[[i]] <- test_data
} 

# best parameters and C-index for each outer fold
for (i in 1:outerfold) {
  cat("  Fold:", i, "  C-index:", nested.cv.rf$c_index[i], "\n\n", "mtry:", nested.cv.rf$mtry[i], "ntree:", nested.cv.rf$ntree[i], "\n\n"
      )
}
# Fold: 1   C-index: 0.5671642 mtry: 6 ntree: 500 
#  Fold: 2   C-index: 0.469352   mtry: 4 ntree: 200 
#  Fold: 3   C-index: 0.6383764 mtry: 6 ntree: 700 
#  Fold: 4   C-index: 0.7258383 mtry: 3 ntree: 700 
#  Fold: 5   C-index: 0.6753247 mtry: 6 ntree: 200 
#  Fold: 6   C-index: 0.6187943 mtry: 5 ntree: 700 
#  Fold: 7   C-index: 0.6428571 mtry: 3 ntree: 700 
#  Fold: 8   C-index: 0.6603774 mtry: 6 ntree: 200 
#  Fold: 9   C-index: 0.6772824 mtry: 3 ntree: 200 
#  Fold: 10 C-index: 0.5641476 mtry: 4 ntree: 700 

#train data: n=348, test data:39


# Summary of the C-index
cat("Mean C-index across all folds:", mean(nested.cv.rf$c_index), "\n")
#Mean C-index across all folds: 0.6239514    
cat("Standard Deviation of C-index across all folds:", sd(nested.cv.rf$c_index), "\n")
#Standard Deviation of C-index across all folds: 0.07339811 

##############GBM survival model####
###################################
# Define the tuning grid for GBM
tuning_grid <- expand.grid(
  interaction.depth = c(1, 3, 5), # tree depth
  n.trees = c(100, 200, 500),     # number of trees
  shrinkage = c(0.01, 0.1),       # learning rate
  n.minobsinnode = 10             # minimum number of observations in the terminal nodes
)

# Number of outer and inner folds
outerfold <- 10
innerfold <- 5

# Create folds for outer cross-validation
set.seed(123)
folds <- createFolds(dat$status, k = outerfold, list = TRUE)

nested_cv_gbm <- list(c_index = numeric(), interaction.depth = numeric(), n.trees = numeric(), 
                      shrinkage = numeric(), n.minobsinnode = numeric(), train_data = list(),test_data = list())

set.seed(123)
for (i in 1:outerfold) {
  fold <- folds[[i]]
  train_data <- dat[-fold, ]
  test_data <- dat[fold, ]
  best_c_index <- 0
  best_params <- NULL
  innerfolds <- createFolds(train_data$status, k = innerfold, list = TRUE)
  for (j in 1:nrow(tuning_grid)) {
    interaction.depth <- tuning_grid$interaction.depth[j]
    n.trees <- tuning_grid$n.trees[j]
    shrinkage <- tuning_grid$shrinkage[j]
    n.minobsinnode <- tuning_grid$n.minobsinnode[j]
    inner_c_indices <- numeric()
    for (inner_fold in innerfolds) {
      inner_train_data <- train_data[-inner_fold, ]
      inner_valid_data <- train_data[inner_fold, ]
      gbm_model <- gbm(
        formula = Surv(Time.elapsed, status == 1) ~ .,
        data = inner_train_data,
        distribution = "coxph",
        n.trees = n.trees,
        interaction.depth = interaction.depth,
        shrinkage = shrinkage,
        n.minobsinnode = n.minobsinnode
      )
      best_iter <- gbm.perf(gbm_model, method = "test", plot.it = FALSE)
      inner_pred <- predict(gbm_model, newdata = inner_valid_data, n.trees = best_iter)
      inner_c_index <- rcorr.cens(-inner_pred, Surv(inner_valid_data$Time.elapsed, inner_valid_data$status == 1))["C Index"]
      inner_c_indices <- c(inner_c_indices, inner_c_index)
    }
    mean_c_index <- mean(inner_c_indices)
    if (mean_c_index > best_c_index) {
      best_c_index <- mean_c_index
      best_params <- list(interaction.depth = interaction.depth, n.trees = n.trees, shrinkage = shrinkage, n.minobsinnode = n.minobsinnode)
    }
  }
  best_gbm_model <- gbm(
    formula = Surv(Time.elapsed, status == 1) ~ .,
    data = train_data,
    distribution = "coxph",
    n.trees = best_params$n.trees,
    interaction.depth = best_params$interaction.depth,
    shrinkage = best_params$shrinkage,
    n.minobsinnode = best_params$n.minobsinnode
  )
  best_iter <- gbm.perf(best_gbm_model, method = "test", plot.it = FALSE)
  test_pred <- predict(best_gbm_model, newdata = test_data, n.trees = best_iter)
  test_c_index <- rcorr.cens(-test_pred, Surv(test_data$Time.elapsed, test_data$status == 1))["C Index"]
  
  nested_cv_gbm$c_index <- c(nested_cv_gbm$c_index, test_c_index)
  nested_cv_gbm$interaction.depth <- c(nested_cv_gbm$interaction.depth, best_params$interaction.depth)
  nested_cv_gbm$n.trees <- c(nested_cv_gbm$n.trees, best_params$n.trees)
  nested_cv_gbm$shrinkage <- c(nested_cv_gbm$shrinkage, best_params$shrinkage)
  nested_cv_gbm$n.minobsinnode <- c(nested_cv_gbm$n.minobsinnode, best_params$n.minobsinnode)
  nested_cv_gbm$train_data[[i]] <- train_data
  nested_cv_gbm$test_data[[i]] <- test_data
}

###results
# best parameters and C-index for each outer fold
for (i in 1:outerfold) {
  cat("  Fold:", i, "C-index:", nested_cv_gbm$c_index[i], "\n\n", 
      "interaction depth:", nested_cv_gbm$interaction.depth[i], "\n\n",
      "ntrees:", nested_cv_gbm$n.trees[i], "\n\n",
      "learning rate", nested_cv_gbm$shrinkage[i], "\n\n",
      "n.minobsinnode", nested_cv_gbm$n.minobsinnode[i], "\n\n"
      )
}
#  Fold: 1 C-index: 0.5783582 interaction depth: 3 ntrees: 100 learning rate 0.01 n.minobsinnode 10 
#  Fold: 2 C-index: 0.4903678 interaction depth: 3 ntrees: 200 learning rate 0.01 n.minobsinnode 10 
#  Fold: 3 C-index: 0.6051661 interaction depth: 3 ntrees: 500 learning rate 0.01 n.minobsinnode 10 
#  Fold: 4 C-index: 0.6804734 interaction depth: 1 ntrees: 100 learning rate 0.01 n.minobsinnode 10 
#  Fold: 5 C-index: 0.6477273 interaction depth: 3 ntrees: 100 learning rate 0.01 n.minobsinnode 10 
#  Fold: 6 C-index: 0.6489362 interaction depth: 5 ntrees: 100 learning rate 0.01 n.minobsinnode 10 
#  Fold: 7 C-index: 0.6884921 interaction depth: 3 ntrees: 100 learning rate 0.01 n.minobsinnode 10 
#  Fold: 8 C-index: 0.6943396 interaction depth: 1 ntrees: 200 learning rate 0.01 n.minobsinnode 10 
#  Fold: 9 C-index: 0.6539278 interaction depth: 1 ntrees: 200 learning rate 0.01 n.minobsinnode 10 
# Fold:10 C-index: 0.5975395 interaction depth: 3 ntrees: 100 learning rate 0.01 n.minobsinnode 10 

# Summary of the C-index
cat("Mean C-index across all folds:", mean(nested_cv_gbm$c_index), "\n")
#Mean C-index across all folds: 0.6285328  
cat("Standard Deviation of C-index across all folds:", sd(nested_cv_gbm$c_index), "\n")
#Standard Deviation of C-index across all folds:0.06250934  

########XGBoost survival model###############
############################################
dat1<- dat

#One hot encoding
non_encoded_vars <- dat1[, c("status", "Time.elapsed")]
encoded_vars <- dat1[, !names(dat1) %in% c("status", "Time.elapsed")]
dummy_model <- dummyVars(~ ., data = encoded_vars, fullRank = TRUE) 
dat1 <- predict(dummy_model, newdata = encoded_vars)
dat1 <- as.data.frame(dat1)
dat1 <- cbind(dat1, non_encoded_vars)
dat1$status <- as.numeric(dat1$status) - 1
str(dat1)
#N=387, p=51

#folds of nested cross-validation
outerfold <- 10
innerfold <- 5
nested_cv_xgb <- list(c_index = numeric(), 
                       max_depth = numeric(), 
                       nrounds = numeric(), 
                       eta = numeric(), 
                       min_child_weight = numeric(), 
                       subsample = numeric(), 
                       colsample_bytree = numeric(),
                      train_data = list(),
                      test_data = list())

# Define the tuning grid for XGBoost
tuning_grid <- expand.grid(
  max_depth = c(3, 5, 7),
  nrounds = c(100, 200, 300),
  eta = c(0.01, 0.1),
  min_child_weight = c(1, 5),
  subsample = c(0.8),
  colsample_bytree = c(0.8)
)

# Folds for outer cross-validation
set.seed(123)
folds <- createFolds(dat1$status, k = outerfold, list = TRUE)

# Nested cross-validation process
for (i in 1:outerfold) {
  cat("Outer fold", i, "of", outerfold, "\n")
  train_indices <- unlist(folds[-i])
  test_indices <- folds[[i]]
  train_data <- dat1[train_indices, ]
  test_data <- dat1[test_indices, ]
  innerfolds <- createFolds(train_data$status, k = innerfold, list = TRUE)
  best_c_index <- 0
  best_params <- list()
  for (params in 1:nrow(tuning_grid)) {
    param_values <- tuning_grid[params, ]
    inner_c_indices <- numeric()
    for (inner_fold in innerfolds) {
      inner_train_data <- train_data[-inner_fold, ]
      inner_valid_data <- train_data[inner_fold, ]
      dtrain <- xgb.DMatrix(data = as.matrix(inner_train_data[, !names(inner_train_data) %in% c("Time.elapsed", "status")]), label = inner_train_data$Time.elapsed)
      dvalid <- xgb.DMatrix(data = as.matrix(inner_valid_data[, !names(inner_valid_data) %in% c("Time.elapsed", "status")]), label = inner_valid_data$Time.elapsed)
      xgb_model <- xgb.train(
        params = list(
          objective = "survival:cox",
          eval_metric = "cox-nloglik",
          max_depth = param_values$max_depth,
          eta = param_values$eta,
          min_child_weight = param_values$min_child_weight,
          subsample = param_values$subsample,
          colsample_bytree = param_values$colsample_bytree
        ),
        data = dtrain,
        nrounds = param_values$nrounds,
        watchlist = list(train = dtrain, eval = dvalid),
        verbose = 1  
      )
      inner_predictions <- predict(xgb_model, newdata = dvalid)
      inner_c_index <- concordance(Surv(inner_valid_data$Time.elapsed, inner_valid_data$status == 1) ~ inner_predictions)$concordance
      inner_c_indices <- c(inner_c_indices, inner_c_index)
    }
    mean_c_index <- mean(inner_c_indices)
    
    if (mean_c_index > best_c_index) {
      best_c_index <- mean_c_index
      best_params <- param_values
    }
  }
  dtrain <- xgb.DMatrix(data = as.matrix(train_data[, !names(train_data) %in% c("Time.elapsed", "status")]), label = train_data$Time.elapsed)
  dtest <- xgb.DMatrix(data = as.matrix(test_data[, !names(test_data) %in% c("Time.elapsed", "status")]), label = test_data$Time.elapsed)
  final_xgb_model<- xgb.train(
    params = list(
      objective = "survival:cox",
      eval_metric = "cox-nloglik",
      max_depth = best_params$max_depth,
      eta = best_params$eta,
      min_child_weight = best_params$min_child_weight,
      subsample = best_params$subsample,
      colsample_bytree = best_params$colsample_bytree
    ),
    data = dtrain,
    nrounds = best_params$nrounds,
    watchlist = list(train = dtrain, eval = dtest),
    verbose = 1  
  )
  test_predictions <- predict(final_xgb_model, newdata = dtest)
  test_c_index <- concordance(Surv(test_data$Time.elapsed, test_data$status == 1) ~ test_predictions)$concordance
  nested_cv_xgb$c_index <- c(nested_cv_xgb$c_index, test_c_index)
  nested_cv_xgb$max_depth <- c(nested_cv_xgb$max_depth, best_params$max_depth)
  nested_cv_xgb$nrounds <- c(nested_cv_xgb$nrounds, best_params$nrounds)
  nested_cv_xgb$eta <- c(nested_cv_xgb$eta, best_params$eta)
  nested_cv_xgb$min_child_weight <- c(nested_cv_xgb$min_child_weight, best_params$min_child_weight)
  nested_cv_xgb$subsample <- c(nested_cv_xgb$subsample, best_params$subsample)
  nested_cv_xgb$colsample_bytree <- c(nested_cv_xgb$colsample_bytree, best_params$colsample_bytree)
  nested_cv_xgb$train_data[[i]] <- train_data
  nested_cv_xgb$test_data[[i]] <- test_data
}

###results
# best parameters and C-index for each outer fold
for (i in 1:outerfold) {
  cat("  Fold:", i, "C-index:", nested_cv_xgb$c_index[i], "\n\n", 
      "maximum depth:", nested_cv_xgb$max_depth[i], "\n\n",
      "nrounds:", nested_cv_xgb$nrounds[i], "\n\n",
      "learning rate", nested_cv_xgb$eta[i], "\n\n",
      "Minimum child weight", nested_cv_xgb$min_child_weight[i], "\n\n",
      "Subsample", nested_cv_xgb$subsample[i], "\n\n",
      "colsample_bytree", nested_cv_xgb$colsample_bytree[i], "\n\n"
  )
}

# Fold: 1 C-index: 0.4251701 maximum depth: 5 nrounds: 100 learning rate 0.1 Minimum child weight 5 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 2 C-index: 0.4864865 maximum depth: 3 nrounds: 200 learning rate 0.1 Minimum child weight 1 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 3 C-index: 0.4800693 maximum depth: 3 nrounds: 300 learning rate 0.1 Minimum child weight 5 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 4 C-index: 0.3643725 maximum depth: 3 nrounds: 200 learning rate 0.1 Minimum child weight 1 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 5 C-index: 0.4592902 maximum depth: 3 nrounds: 100 learning rate 0.1 Minimum child weight 1 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 6 C-index: 0.4203152 maximum depth: 7 nrounds: 300 learning rate 0.1 Minimum child weight 5 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 7 C-index: 0.402214   maximum depth: 7 nrounds: 300 learning rate 0.1 Minimum child weight 5 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 8 C-index: 0.4770017 maximum depth: 3 nrounds: 300 learning rate 0.1 Minimum child weight 1 Subsample 0.8 colsample_bytree 0.8 
#  Fold: 9 C-index: 0.3858407 maximum depth: 5 nrounds: 200 learning rate 0.1 Minimum child weight 5 Subsample 0.8 colsample_bytree 0.8 
# Fold:10 C-index: 0.4870259 maximum depth: 5 nrounds: 300 learning rate 0.1 Minimum child weight 5 Subsample 0.8 colsample_bytree 0.8 

# Summary of the C-index
mean_c_index <- mean(nested_cv_xgb$c_index)
cat("Average C-index from cross-validation:", mean_c_index, "\n")
#Mean C-index across all folds:0.4387786 
cat("Standard Deviation of C-index across all folds:", sd(nested_cv_xgb$c_index), "\n")
#Standard Deviation of C-index across all folds: 0.04520601 

###Plotting c-index
cindex_df <- tibble(
  Fold = rep(1:10, 3),
  Model = rep(c("RSF", "GBM survival", "XGBoost survival"), each = 10),
  C_Index = c(nested.cv.rf$c_index, nested_cv_gbm$c_index, nested_cv_xgb$c_index)
)

ggplot(cindex_df, aes(x = Fold, y = C_Index, color = Model)) +
  geom_line() +
  geom_point() +
  labs(title = "C-index Comparison Across Outer Folds",
       y = "C-index", x = "Outer Fold") +
  scale_x_continuous(breaks = 1:10)+
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
    panel.background = element_blank(), axis.line = element_line(colour = "black")
  )



###Calibration curve for 7 days time point
#RSF model
all_rsf_pred <- numeric()
all_test_data_rsf <- do.call(rbind, nested.cv.rf$test_data)

time_point <- 7
for (i in seq_along(nested.cv.rf$train_data)) {
  rf_model <- rfsrc(Surv(Time.elapsed, status) ~ ., 
                    data = nested.cv.rf$train_data[[i]], 
                    ntree = nested.cv.rf$ntree[i], 
                    mtry = nested.cv.rf$mtry[i])
  rf_pred <- predict(rf_model, newdata = nested.cv.rf$test_data[[i]])
  rf_survival <- exp(-rf_pred$chf[, , "condCHF.1"])
  time_index <- which.min(abs(rf_model$time.interest - time_point))
  all_rsf_pred <- c(all_rsf_pred, rf_survival[, time_index])
}

all_test_data_rsf$RSF_pred <- all_rsf_pred
all_test_data_rsf$status <- as.numeric(as.character(all_test_data_rsf$status))

#booststarpping
dd <- datadist(all_test_data_rsf)
options(datadist = "dd")

cal_model <- cph(Surv(Time.elapsed, status) ~ RSF_pred, 
                 data = all_test_data_rsf, 
                 x = TRUE, y = TRUE, surv = TRUE, time.inc = time_point)
cal_rsf <- calibrate(cal_model, method = "boot", u = time_point, B = 200)

#GBM model
all_gbm_pred <- numeric()
all_test_data_gbm <- do.call(rbind, nested_cv_gbm$test_data)

time_point <- 7 
for (i in seq_along(nested_cv_gbm$train_data)) {
  gbm_model <- gbm(
    formula = Surv(Time.elapsed, status == 1) ~ .,
    data = nested_cv_gbm$train_data[[i]],
    distribution = "coxph",
    n.trees = nested_cv_gbm$n.trees[i],
    interaction.depth = nested_cv_gbm$interaction.depth[i],
    shrinkage = nested_cv_gbm$shrinkage[i],
    n.minobsinnode = nested_cv_gbm$n.minobsinnode[i]
  )
  gbm_pred_train <- predict(gbm_model, 
                          newdata = nested_cv_gbm$train_data[[i]], 
                          n.trees = nested_cv_gbm$n.trees[i], 
                          type = "link")
  cox_fit <- coxph(Surv(Time.elapsed, status == 1) ~ gbm_pred_train, 
                   data = nested_cv_gbm$train_data[[i]])
  base_surv <- survfit(cox_fit, newdata = data.frame(gbm_pred_train = 0))
  baseline_survival <- summary(base_surv, times = time_point, extend = TRUE)$surv
  gbm_pred_test <- predict(gbm_model, 
                         newdata = nested_cv_gbm$test_data[[i]], 
                         n.trees = nested_cv_gbm$n.trees[i], 
                         type = "link")
  predicted_survival <- baseline_survival ^ exp(gbm_pred_test)
  predicted_event_prob <- 1 - predicted_survival
  all_gbm_pred <- c(all_gbm_pred, predicted_event_prob)
}

all_test_data_gbm$GBM_pred <- all_gbm_pred
all_test_data_gbm$status <- as.numeric(as.character(all_test_data_gbm$status))

# Bootstrapping Calibration
dd <- datadist(all_test_data_gbm)
options(datadist = "dd")

cal_model_gbm <- cph(Surv(Time.elapsed, status) ~ GBM_pred,
                     data = all_test_data_gbm,
                     x = TRUE, y = TRUE, surv = TRUE, time.inc = time_point)

cal_gbm <- calibrate(cal_model_gbm, method = "boot", u = time_point, B = 200)

##Combining plots
par(mfrow = c(1, 2))

plot(cal_rsf, 
     xlim = c(0, 0.4), ylim = c(0, 0.45),
     xlab = "Prediction", ylab = "Observation",
     main = "Calibration Curve for RSF at 7 days",
     lwd = 2)
lines(cal_rsf[, "pred"], cal_rsf[, "calibrated.corrected"], 
      col = "blue", lwd = 2)
abline(0, 1, lty = 2, col = "red")

plot(cal_gbm,
     xlim = c(0, 0.4), ylim = c(0, 0.45),
     xlab = "Prediction", ylab = "Observation",
     main = "Calibration Curve for GBM at 7 days",
     lwd = 2)
lines(cal_gbm[, "pred"], cal_gbm[, "calibrated.corrected"], 
      col = "blue", lwd = 2)
abline(0, 1, lty = 2, col = "red")

par(mfrow = c(1,1))

########Selecting GBM survival model as final model based on performance (C-index 0.694 for the best model)
####getting data from the outerfolds with best performance
best_fold_gbm <- which.max(nested_cv_gbm$c_index)
best_index_gbm <- max(nested_cv_gbm$c_index)
best_interaction.depth_gbm <- nested_cv_gbm$interaction.depth[best_fold_gbm]
best_ntree_gbm <- nested_cv_gbm$n.trees[best_fold_gbm]
best_shrinkage_gbm <- nested_cv_gbm$shrinkage[best_fold_gbm]
best_n.minobsinnode_gbm <- nested_cv_gbm$n.minobsinnode[best_fold_gbm]

best_train_data_gbm <- nested_cv_gbm$train_data[[best_fold_gbm]]
best_test_data_gbm <- nested_cv_gbm$test_data[[best_fold_gbm]]
X_test_best_gbm <- best_test_data_gbm[, setdiff(names(best_test_data_gbm), c("Time.elapsed", "status"))]
y_test_best_gbm <- best_test_data_gbm[, c("Time.elapsed", "status")]

# Fit the best model using the training data from the best fold
set.seed(123)
best_gbm <- gbm(
  formula = Surv(Time.elapsed, status == 1) ~ .,
  data = best_train_data_gbm,
  distribution = "coxph",
  n.trees = best_ntree_gbm,
  interaction.depth = best_interaction.depth_gbm,
  shrinkage = best_shrinkage_gbm,
  n.minobsinnode = best_n.minobsinnode_gbm
)

#####SHAP analysis
# Define the prediction function
predict_function <- function(model, newdata) {
  predict(model, newdata = newdata, n.trees = best_ntree_gbm)
}

# Compute SHAP values
set.seed(123)
shap_values_gbm <- fastshap::explain(
  object = best_gbm,
  X = X_test_best_gbm,
  pred_wrapper = predict_function,
  nsim = 100
)
print(shap_values_gbm)
saveRDS(shap_values_gbm, 'shap_gbm_v1.rds')

shp.gbm <- shapviz(shap_values_gbm, X= X_test_best_gbm)
str(shp.gbm$X)

#renaming variables
rename_map <- c(
  ecusv_1___2 = 'Reasons for quitting-addiction concern',
  ecusv_1___3 = 'Reasons for quitting-cost',
  ecusv_1___4 = 'Reasons for quitting-health concern',
  ecusv_1___7 = 'Reasons for quitting-family or peers demand',
  ecusv_1___13 = 'Reasons for quitting-others',
  ecu8_1 = 'Past month frequency of vaping',
  ecu11b_1 = 'Time to first vape',
  ecu12a_1 = 'Puffs per session',
  ecu16_1 = 'Self-perceived addiction',
  ecu17_1 = 'Intention to quit',
  ecusv_2 = 'Past year quit attempts',
  ecusv_3 = 'Self-confidence in quitting',
  epp1_1 = 'Device type',
  epp7_1 = 'Flavor used',
  epp10_1 = 'Nicotine strength',
  epp13_1 = 'Monthly vaping expense',
  epp14_1 = 'Average e-liquid per week',
  epp15_1 = 'Single pod lasting',
  cur_csmk_1 = 'Past 30-day cigarette smoking',
  cur_can_1 = 'Past 30-day cannabis use',
  cur_alc_1 = 'Past 30-day alcohol drinking',
  cur_otob_1 = 'Past 30-day other tobacco products use',
  osu16_ecig_1 = 'Peer vaping',
  kab3_1 = 'Believing vaping is safer than smoking',
  ghealth_1 = 'Perceived general health',
  mhealth1_1 = 'Perceived mental health',
  svc_mood = 'Overall mood at baseline',
  svc_cravings = 'Overall cravings at baseline',
  pstress_1 = 'Perceived stress',
  age = "Age",
  gender = 'Gender',
  sexorient = 'Sexual orientation',
  first_mood = 'Initial mood during challenge',
  first_craving = 'Initial craving during challenge',
  mood_trend = 'Mood trend during challenge',
  craving_trend = 'Craving trend during challenge',
  race = 'Race'
)

colnames(shp.gbm$X)[colnames(shp.gbm$X) %in% names(rename_map)] <- rename_map[colnames(shp.gbm$X) %in% names(rename_map)]
str(shp.gbm$X)
colnames(shp.gbm[["S"]])[colnames(shp.gbm[["S"]]) %in% names(rename_map)] <- rename_map[colnames(shp.gbm[["S"]]) %in% names(rename_map)]
colnames(shp.gbm[["S"]])

#feature importance
shp.gbm.imp<- sv_importance(shp.gbm)
gbm.imp<- data.frame(shp.gbm.imp$data)
gbm.imp$value <- round(gbm.imp$value,2)
G1_gbm<- ggplot(data=gbm.imp,aes(x=reorder(feature,value),y=value))+
  geom_bar(fill= "navyblue", stat="identity")+
  geom_text(aes(label= value), y=0.01,color="White",size=3)+
  coord_flip()+
  labs(y="mean SHAP value",
       x= "")+
  theme(text = element_text(size= 12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
print(G1_gbm)

shp.gbm.summary<- sv_importance(shp.gbm, kind = "beeswarm", show_numbers = TRUE)
print(shp.gbm.summary)

###SHAP analysis on RSF model as the 2nd best model
best_fold <- which.max(nested.cv.rf$c_index)
best_index <- max(nested.cv.rf$c_index)
best_mtry <- nested.cv.rf$mtry[best_fold]
best_ntree <- nested.cv.rf$ntree[best_fold]

best_train_data <- nested.cv.rf$train_data[[best_fold]]
best_test_data <- nested.cv.rf$test_data[[best_fold]]
X_test_best <- best_test_data[, setdiff(names(best_test_data), c("Time.elapsed", "status"))]
y_test_best <- best_test_data[, c("Time.elapsed", "status")]

# Fit the best model using the training data from the best fold
set.seed(123)
best_rsf <- rfsrc(Surv(Time.elapsed, status) ~ ., data = best_train_data,
                  ntree = best_ntree, mtry = best_mtry, probability = TRUE)

#####SHAP analysis
# Define the prediction function
predict_function <- function(model, newdata) {
  preds <- predict(model, newdata = newdata)$predicted
  preds[, 2] 
}

# Compute SHAP values
set.seed(123)
shap_values <- fastshap::explain(
  object = best_rsf,
  X = X_test_best,
  pred_wrapper = predict_function,
  nsim = 100
)
print(shap_values)
saveRDS(shap_values, 'shap_rsf.rds')

shp.rsf <- shapviz(shap_values, X= X_test_best)
str(shp.rsf$X)

#renaming variables
rename_map <- c(
  ecusv_1___2 = 'Reasons for quitting-addiction concern',
  ecusv_1___3 = 'Reasons for quitting-cost',
  ecusv_1___4 = 'Reasons for quitting-health concern',
  ecusv_1___7 = 'Reasons for quitting-family or peers demand',
  ecusv_1___13 = 'Reasons for quitting-others',
  ecu8_1 = 'Past month frequency of vaping',
  ecu11b_1 = 'Time to first vape',
  ecu12a_1 = 'Puffs per session',
  ecu16_1 = 'Self-perceived addiction',
  ecu17_1 = 'Intention to quit',
  ecusv_2 = 'Past year quit attempts',
  ecusv_3 = 'Self-confidence in quitting',
  epp1_1 = 'Device type',
  epp7_1 = 'Flavor used',
  epp10_1 = 'Nicotine strength',
  epp13_1 = 'Monthly vaping expense',
  epp14_1 = 'Average e-liquid per week',
  epp15_1 = 'Single pod lasting',
  cur_csmk_1 = 'Past 30-day cigarette smoking',
  cur_can_1 = 'Past 30-day cannabis use',
  cur_alc_1 = 'Past 30-day alcohol drinking',
  cur_otob_1 = 'Past 30-day other tobacco products use',
  osu16_ecig_1 = 'Peer vaping',
  kab3_1 = 'Believing vaping is safer than smoking',
  ghealth_1 = 'Perceived general health',
  mhealth1_1 = 'Perceived mental health',
  svc_mood = 'Overall mood at baseline',
  svc_cravings = 'Overall cravings at baseline',
  pstress_1 = 'Perceived stress',
  age = "Age",
  gender = 'Gender',
  sexorient = 'Sexual orientation',
  first_mood = 'Initial mood during challenge',
  first_craving = 'Initial craving during challenge',
  mood_trend = 'Mood trend during challenge',
  craving_trend = 'Craving trend during challenge',
  race = 'Race'
)

colnames(shp.rsf$X)[colnames(shp.rsf$X) %in% names(rename_map)] <- rename_map[colnames(shp.rsf$X) %in% names(rename_map)]
str(shp.rsf$X)
colnames(shp.rsf[["S"]])[colnames(shp.rsf[["S"]]) %in% names(rename_map)] <- rename_map[colnames(shp.rsf[["S"]]) %in% names(rename_map)]
colnames(shp.rsf[["S"]])

###Feature importance
shp.rsf.imp<- sv_importance(shp.rsf)
rsf.imp<- data.frame(shp.rsf.imp$data)
rsf.imp$value <- round(rsf.imp$value,2)
G1_rsf<- ggplot(data=rsf.imp,aes(x=reorder(feature,value),y=value))+
  geom_bar(fill= "navyblue", stat="identity")+
  geom_text(aes(label= value), y=0.45,color="White",size=3)+
  coord_flip()+
  labs(y="mean SHAP value",
       x= "")+
  theme(text = element_text(size= 12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
print(G1_rsf)

shp.rsf.summary<- sv_importance(shp.rsf, kind = "beeswarm", show_numbers = TRUE)
print(shp.rsf.summary)


###Sensitivity analysis
###LASSO Cox regression
set.seed(123)
x <- model.matrix(Surv(Time.elapsed, status == 1) ~ ., data = dat)[, -1]
y <- Surv(dat$Time.elapsed, dat$status == 1)
#10-fold cross-validation
folds <- createFolds(dat$status, k = 10, list = TRUE)
c_index_results <- numeric(length(folds))
coefficients_list <- list()

for (i in seq_along(folds)) {
  cat("Processing fold:", i, "\n")
  test_idx <- folds[[i]]
  x_train <- x[-test_idx, ]
  y_train <- y[-test_idx]
  x_test <- x[test_idx, ]
  y_test <- y[test_idx]
  cv_fit <- cv.glmnet(x_train, y_train, family = "cox", alpha = 1, nfolds = 5) 
  lp_test <- predict(cv_fit, newx = x_test, s = "lambda.min", type = "link")
  concordance <- survConcordance(y_test ~ lp_test)
  c_index <- concordance$concordance
  c_index_results[i] <- c_index
  cat("Fold", i, "- C-index:", round(c_index, 3), "\n")
  coef_fold <- coef(cv_fit, s = "lambda.min")
  coef_fold <- as.matrix(coef_fold)
  coef_fold <- coef_fold[coef_fold != 0, , drop = FALSE]
  coefficients_list[[i]] <- coef_fold
}
# Fold 1 - C-index: 0.589 
# Fold 2 - C-index: 0.515 
# Fold 3 - C-index: 0.585 
# Fold 4 - C-index: 0.662 
# Fold 5 - C-index: 0.642 
# Fold 6 - C-index: 0.649 
# Fold 7 - C-index: 0.641 
# Fold 8 - C-index: 0.65 
# Fold 9 - C-index: 0.639 
# Fold 10 - C-index: 0.594 

# Average C-index across folds
mean_c_index <- mean(c_index_results)
cat("Average C-index from cross-validation:", mean_c_index, "\n")
#Average C-index from cross-validation: 0.6165057

# Standard deviation of C-index across folds
sd_c_index <- sd(c_index_results)
cat("Standard Deviation of C-index from cross-validation:", sd_c_index, "\n")
#Standard Deviation of C-index from cross-validation:0.0454419  

###Feature importance
# Combine nonzero coefficients across folds
combined_coefs <- do.call(rbind, lapply(coefficients_list, function(x) {
  data.frame(Variable = rownames(x), Coefficient = as.vector(x))
}))

#using both mean absolute co-efficient and number of folds selected to get feature importance
var_importance <- combined_coefs %>%
  group_by(Variable) %>%
  summarise(
    n_selected = n(),
    mean_abs_coef = mean(abs(Coefficient)),
    .groups = "drop"
  ) %>%
  arrange(desc(n_selected), desc(mean_abs_coef))

#filter to those selected in at least 2 folds 
robust_vars <- var_importance %>%
  filter(n_selected >= 5) %>%                  
  arrange(desc(mean_abs_coef)) %>%             
  slice_max(order_by = mean_abs_coef, n = 15)

#HRs
robust_vars <- robust_vars %>%
  mutate(order_id = row_number())
filtered_coefs <- combined_coefs %>%
  inner_join(robust_vars %>% select(Variable, order_id), by = "Variable")

hr_summary <- filtered_coefs %>%
  group_by(Variable, order_id) %>%
  summarise(
    n_selected = n(),
    mean_HR = exp(mean(Coefficient)),
    sd_logHR = sd(Coefficient),
    lower_logHR = mean(Coefficient) - 1.96 * sd_logHR / sqrt(n_selected),
    upper_logHR = mean(Coefficient) + 1.96 * sd_logHR / sqrt(n_selected),
    lower_HR = exp(lower_logHR),
    upper_HR = exp(upper_logHR),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_HR))

orig_order <- robust_vars$Variable 
hr_summary <- hr_summary %>%
  mutate(Variable = factor(Variable, levels = orig_order)) %>%
  arrange(Variable)
write.csv(hr_summary, "lasso_HR.csv")

##Change names
rename_map <- c(
  ecu17_11 = 'Intention to quit within next month',
  mood_trend2 = 'Mood trend during challenge-elevated',
  ecusv_3 = 'Self-confidence in quitting',
  ecusv_1___131 = 'Reasons for quitting-others',
  ecu11b_11 = 'Time to first vape 6-30 mins',
  ecu16_11 = 'Self-perceived addiction-very addicted',
  cur_alc_11 = 'Past 30-day alcohol drinking',
  mood_trend3 = 'Mood trend during challenge-inconsistent',
  epp15_1 = 'Single pod lasting',
  ecu8_1 = 'Past month frequency of vaping',
  ghealth_11 = 'Perceived general health-good'
)

robust_vars <- robust_vars %>%
  mutate(Variable = recode(Variable, !!!rename_map))
robust_vars$MeanAbsCoeff <- round(robust_vars$mean_abs_coef,2)


G1_lasso<- ggplot(robust_vars, aes(x = reorder(Variable, MeanAbsCoeff), y = MeanAbsCoeff)) +
  geom_bar(fill= "navyblue", stat="identity")+
  geom_text(aes(label= round(MeanAbsCoeff, 3)), y=0.05,color="White",size=3)+
  coord_flip()+
  labs(y="mean absolute coefficient",
       x= "")+
  theme(text = element_text(size= 12), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black"))
print(G1_lasso)


##Predictor comparisons
G1_gbm_with_title <- arrangeGrob(G1_gbm, top = textGrob("GBM Survival", gp = gpar(fontsize = 12, font = 2)))
G1_rsf_with_title <- arrangeGrob(G1_rsf, top = textGrob("RSF", gp = gpar(fontsize = 12, font = 2)))
G1_lasso_with_title <- arrangeGrob(G1_lasso, top = textGrob("Lasso-Cox regression", gp = gpar(fontsize = 12, font = 2)))

grid.arrange(G1_gbm_with_title, G1_rsf_with_title, G1_lasso_with_title, ncol = 3)

#SHAP dependence plots from mainly GBM and also from RSF model
p1<- sv_dependence(shp.gbm, v = "Self-confidence in quitting", color_var = NULL)
p1<- p1 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, linewidth = 1)


p2<- sv_dependence(shp.gbm, v = "Intention to quit", color_var = NULL)
p2$data$"Intention to quit" <- factor(p2$data$"Intention to quit", 
                                      levels= c(0,1), 
                                      labels= c('Beyond next month', "Within next month"))
p2<- p2 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")

p3<- sv_dependence(shp.gbm, v = "Monthly vaping expense", color_var = NULL)
p3<- p3 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, linewidth = 1)


p4<- sv_dependence(shp.gbm, v = "Single pod lasting", color_var = NULL)
p4<- p4 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, linewidth = 1)


p5<- sv_dependence(shp.gbm, v = "Time to first vape", color_var = NULL)
p5$data$"Time to first vape" <- factor(p5$data$"Time to first vape", levels= c(0,1,2), 
                                       labels= c('0-5 mins', '6-30 mins', ">30 mins"))
p5<- p5 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")


p6<- sv_dependence(shp.gbm, v = "Past 30-day alcohol drinking", color_var = NULL)
p6$data$"Past 30-day alcohol drinking" <- factor(p6$data$"Past 30-day alcohol drinking", 
                                                 levels= c(0,1), labels= c('No', 'Yes'))
p6<- p6 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")


p7<- sv_dependence(shp.gbm, v = "Past month frequency of vaping", color_var = NULL)
p7<- p7 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, linewidth = 1)


p8<- sv_dependence(shp.gbm, v = "Initial craving during challenge", color_var = NULL)
p8<- p8 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, linewidth = 1)


p9<- sv_dependence(shp.gbm, v = "Initial mood during challenge", color_var = NULL)
p9<- p9 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, linewidth = 1)


p10<- sv_dependence(shp.gbm, v = "Craving trend during challenge", color_var = NULL)
p10$data$"Craving trend during challenge" <- factor(p10$data$"Craving trend during challenge", levels= c(0,1,2,3), 
                                                   labels= c("Decreased", 'Stable', 'Increased', "Inconsistent"))
p10<- p10 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")


p11<- sv_dependence(shp.gbm, v = "Mood trend during challenge", color_var = NULL)
p11$data$"Mood trend during challenge" <- factor(p11$data$"Mood trend during challenge", levels= c(0,1,2,3), 
                                                labels= c("Depressed", 'Stable', 'Elevated', "Inconsistent"))
p11<- p11 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")


p12<- sv_dependence(shp.gbm, v = "Reasons for quitting-others", color_var = NULL)
p12$data$'Reasons for quitting-others' <- factor(p12$data$'Reasons for quitting-others', 
                                                levels= c(0,1), labels= c('No', 'Yes'))
p12<- p12 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")


p13<- sv_dependence(shp.gbm, v = "Average e-liquid per week", color_var = NULL)
p13<- p13 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, linewidth = 1)


p14<- sv_dependence(shp.gbm, v = "Self-perceived addiction", color_var = NULL)
p14$data$'Self-perceived addiction' <- factor(p14$data$'Self-perceived addiction', 
                                                 levels= c(0,1), labels= c('Low', 'High'))
p14<- p14 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")


grid1<- gridExtra::grid.arrange(p1,p2,p6, p12, ncol=2)
grid2<- gridExtra::grid.arrange(p3,p4,p7,p13, ncol=2)
grid3<- gridExtra::grid.arrange(p5,p14,p8, p9, p10, p11, ncol=2)


####Socio-demographics
p15<- sv_dependence(shp.gbm, v = "Age", color_var = NULL)
p15 <- p15 + geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs", k=4),color = "black", se = TRUE, size = 1)

p16<- sv_dependence(shp.gbm, v = "Race", color_var = NULL)
p16$data$"Race" <- factor(p16$data$"Race", levels= c(0,1), 
                          labels= c('Non-White', 'White'))
p16<- p16 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")

p17<- sv_dependence(shp.gbm, v = "Gender", color_var = NULL)
p17$data$"Gender" <- factor(p17$data$"Gender", levels= c(0,1,2), 
                            labels= c('Men', 'Women', 'Others'))
p17<- p17 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")

p18<- sv_dependence(shp.gbm, v = "Sexual orientation", color_var = NULL)
p18$data$"Sexual orientation" <- factor(p18$data$"Sexual orientation", levels= c(0,1,2), 
                                        labels= c('Heterosexual', '2SLGBTQ+', 'Undisclosed'))
p18<- p18 + stat_summary(fun = mean, geom = "point",  size = 3, color = "black") +
  stat_summary(fun = mean,geom = "line",aes(group = 1),color = "black",linewidth = 0.8) +
  stat_summary(fun.data = mean_cl_normal, geom = "errorbar", width = 0.2, color = "black")

gridExtra::grid.arrange(p15,p16,p17,p18, ncol=2)

####Interactions
###SHAP interactions
#gender and age
g1<- sv_dependence2D(shp.gbm, x = "Gender", y = "Age", alpha = 0.5)
g1[["data"]][["Gender"]]<- factor(g1[["data"]][["Gender"]], 
                                  levels= c(0,1,2), 
                                  labels= c('Men', 'Women', 'Others'))

#sexual orientation and age
g2<- sv_dependence2D(shp.gbm, x = "Sexual orientation", y = "Age", alpha = 0.5)
g2[["data"]][["Sexual orientation"]]<- factor(g2[["data"]][["Sexual orientation"]], 
                                              levels= c(0,1,2), 
                                              labels= c('Heterosexual', '2SLGBTQ+', 'Undisclosed'))

#race and ace
g3<- sv_dependence2D(shp.gbm, x = "Race", y = "Age", alpha = 0.5)
g3[["data"]][["Race"]]<- factor(g3[["data"]][["Race"]], levels= c(0,1), 
                                labels= c('Non-White', 'White'))

#gender and sexual orientation
g4<- sv_dependence2D(shp.gbm, x = "Gender", y = "Sexual orientation", alpha = 0.5)
g4[["data"]][["Gender"]]<- factor(g4[["data"]][["Gender"]], 
                                  levels= c(0,1,2), 
                                  labels= c('Men', 'Women', 'Others'))
g4[["data"]][["Sexual orientation"]]<- factor(g4[["data"]][["Sexual orientation"]], 
                                              levels= c(0,1,2), 
                                              labels= c('Heterosexual', '2SLGBTQ+', 'Undisclosed'))

#gender and race
g5<- sv_dependence2D(shp.gbm, x = "Gender", y = "Race", alpha = 0.5)
g5[["data"]][["Race"]]<- factor(g5[["data"]][["Race"]], levels= c(0,1), 
                                labels= c('Non-White', 'White'))
g5[["data"]][["Gender"]]<- factor(g5[["data"]][["Gender"]], 
                                  levels= c(0,1,2), 
                                  labels= c('Men', 'Women', 'Others'))

#sexual orientation and race
g6<- sv_dependence2D(shp.gbm, x = "Sexual orientation", y = "Race", alpha = 0.5)
g6[["data"]][["Race"]]<- factor(g6[["data"]][["Race"]], levels= c(0,1), 
                                labels= c('Non-White', 'White'))
g6[["data"]][["Sexual orientation"]]<- factor(g6[["data"]][["Sexual orientation"]], 
                                              levels= c(0,1,2), 
                                              labels= c('Heterosexual', '2SLGBTQ+', 'Undisclosed'))

gridExtra::grid.arrange(g1,g2,g3, g4, g5, g6, ncol=3)


#SHAP force plot for single individual
sv_force(shp.gbm, row_id = 11)
sv_waterfall(shp.gbm, row_id = 11)


###Using Kaplan Meier to examine some predictors
#reinstating original dataset wothout standardization
df<- read.csv("imputed_dataset_final.csv")
str(df)
df$X<- NULL
###reinserting factor variables as factor
df[,c(1:5, 7:11, 13:15, 19:29, 31, 32,35:37, 39)] <- lapply(df[,c(1:5, 7:11, 13:15, 19:29, 31, 32,35:37, 39)], as.factor)
#n=387, p=39

#kaplan_Meier for alcohol
df <- df %>%
  mutate(cur_alc_1 = factor(cur_alc_1, levels = c(0, 1), labels = c("No", "Yes")))
fit <- survfit(Surv(Time.elapsed, status == 1) ~ cur_alc_1, data = df)
km_plot <- ggsurvplot(fit, data = df,
                      pval = TRUE,
                      conf.int = TRUE, 
                      risk.table = TRUE,
                      xlab = "Time (days)",
                      ylab = "Survival Probability",
                      legend.title = "Past 30-day alcohol drinking",
                      legend.labs = c("No", "Yes"),
                      palette = "Set1",
                      xlim = c(0, 30))
km_plot$plot <- km_plot$plot +
  scale_x_continuous(breaks = seq(0, 30, by = 5))
print(km_plot)


#kaplan_Meier for initial craving
summary(df$first_craving)
df$craving_group <- ifelse(df$first_craving >= 8 , "High", "Not high")
fit1 <- survfit(Surv(Time.elapsed, status == 1) ~ craving_group, data = df)
km_plot1<- ggsurvplot(fit1, data = df,
           pval = TRUE,
           conf.int = TRUE, 
           risk.table = TRUE,
           xlab = "Time (days)",
           ylab = "Survival Probability",
           legend.title = "Initial craving during challenge",
           legend.labs = c("High (≥8)", "Not high (<8)"),
           xlim = c(0, 30))
km_plot1$plot <- km_plot1$plot +
  scale_x_continuous(breaks = seq(0, 30, by = 5))
print(km_plot1)


#Checking time to first vape
df$ecu11b_1 <- factor(df$ecu11b_1,
                      levels = c(0, 1, 2),
                      labels = c("0-5 mins", "6-30 mins", "More than 30 mins"))
fit2 <- survfit(Surv(Time.elapsed, status == 1) ~ ecu11b_1, data = df)
km_plot2 <- ggsurvplot(fit2, data = df,
                       pval = TRUE,
                       conf.int = TRUE,
                       risk.table = TRUE,
                       xlab = "Time (days)",
                       ylab = "Survival Probability",
                       legend.title = "Time to first vape",
                       legend.labs = c("0-5 mins", "6-30 mins", "More than 30 mins"),
                       palette = "Set1",
                       xlim = c(0, 30))
km_plot2$plot <- km_plot2$plot +
  scale_x_continuous(breaks = seq(0, 30, by = 5)) +
  geom_vline(xintercept = 15, linetype = "dashed", color = "black") +
  annotate("text", x = 15, y = 0.7, label = "Crossover ≈ 15 days", angle = 90, vjust = 1.5, size = 3)
km_plot2$plot <- km_plot2$plot +
  scale_x_continuous(breaks = seq(0, 30, by = 5)) +
  geom_vline(xintercept = 23, linetype = "dashed", color = "black") +
  annotate("text", x = 23, y = 0.7, label = "Crossover ≈ 23 days", angle = 90, vjust = 1.5, size = 3)
print(km_plot2)

#kaplan_Meier for initial mood
summary(df$first_mood)
df$mood_group <- cut(df$first_mood,
                     breaks = c(-Inf, 3.99, 6.99, Inf),
                     labels = c("Low (<4)", "Moderate (4–6)", "High (≥7)"),
                     right = TRUE)
fit3 <- survfit(Surv(Time.elapsed, status == 1) ~ mood_group, data = df)
km_plot3<- ggsurvplot(fit3, data = df,
           pval = TRUE,
           conf.int = TRUE, 
           risk.table = TRUE,
           xlab = "Time (days)",
           ylab = "Survival Probability",
           legend.title = "Initial mood during challenge",
           legend.labs = c("Low (<4)", "Moderate (4–6)", "High (≥7)"),
           xlim = c(0, 30))

km_plot3$plot <- km_plot3$plot +
  scale_x_continuous(breaks = seq(0, 30, by = 5))
print(km_plot3)