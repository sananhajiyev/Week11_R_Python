# 1. Import dataset.
churn_data <- read.csv("Churn_Modelling.csv")
#names(churn_data)

# 2. Remove unneeded columns
columns_to_remove <- c("RowNumber", "CustomerId", "Surname")
churn_data <- churn_data[, !(names(churn_data) %in% columns_to_remove)]

# 3. Build Churn model
library(caret)
library(randomForest)
library(pROC)

set.seed(123)

# Split the data into training and test sets
train_index <- createDataPartition(churn_data$Exited, p = 0.7, list = FALSE)
train_data <- churn_data[train_index, ]
test_data <- churn_data[-train_index, ]

outcome_var <- "Exited"

model <- randomForest(formula = as.formula(paste(outcome_var, "~ .")), data = train_data, ntree = 100)

# 4. Compare model results for training and test sets
train_predictions <- predict(model, train_data, type = "response")
test_predictions <- predict(model, test_data, type = "response")

train_accuracy <- mean(train_predictions == train_data$Exited)
test_accuracy <- mean(test_predictions == test_data$Exited)

#cat("Training Predictions:", train_predictions, "\n")
#cat("Test Predictions:", test_predictions, "\n")
cat("Training Accuracy:", train_accuracy, "\n")
cat("Test Accuracy:", test_accuracy, "\n")

# 5. Evaluate and explain model results using ROC & AUC curves
roc_obj <- roc(response = test_data$Exited, predictor = test_predictions)
auc_value <- auc(roc_obj)

plot(roc_obj, main = "ROC Curve for Churn Model")
#abline(a = 0, b = 1, lty = 2, col = "red")

cat("AUC:", auc_value, "\n")
