library(car)
library(Information)
library(pROC)
library(caret)

data <- read.csv("bank_full.csv", sep = ";")


sum(is.na(data$y))
sum(is.na(data[, -1]))
unique(data$y)
data$y <- ifelse(data$y == "yes", 1, 0)


# 1. Find multicollinearity by applying VIF; 
vif_values <- vif(lm(y ~ ., data = data))
data <- data[, -which(vif_values > 5)]


# 2. Standardize features;
data[, c("age", "balance", "day", "duration", "campaign", "pdays", "previous")] <- 
       scale(data[, c("age", "balance", "day", "duration", "campaign", "pdays", "previous")])


# 3. Split data into train and test sets using seed=123;
set.seed(123)
split_index <- createDataPartition(data$y, p = 0.7, list = FALSE)
train_data <- data[split_index, ]
test_data <- data[-split_index, ]


# 4. Exclude unimportant variables (information value should be > 0.02);
iv_values <- create_infotables(data, y = "y")
important_vars <- iv_values$Summary[iv_values$Summary$IV > 0.02, "VARNAME"]
data <- data[, c("y", important_vars)]


# 5. Apply binning according to Weight of Evidence principle;
install.packages("scorecard")
library(scorecard)

variable_to_bin <- "balance"
bins <- woebin(train_data, y = "y", x = variable_to_bin)
print(bins)

variables_to_bin <- c(
  "balance", "age", "day", "duration", "campaign", "pdays", "previous")

all_bins <- list()

for (variable in variables_to_bin) {
  bins <- woebin(train_data, y = "y", x = variable)
  all_bins[[variable]] <- bins
}


for (variable in names(all_bins)) {
  print(all_bins[[variable]])
}


# 6. Build a logistic regression model. p-value variables should be max 0.05;
model <- glm(y ~ ., data = train_data, family = binomial(link = "logit"))
summary(model)

# 7. Find threshold by max f1 score;
probabilities <- predict(model, newdata = test_data, type = "response")
roc_curve <- roc(test_data$y, probabilities)
optimal_threshold <- coords(roc_curve, "best", best.method = "closest.topleft")

# 8. Calculate AUC score both for train and test sets;
train_probabilities <- predict(model, newdata = train_data, type = "response")
train_roc_curve <- roc(train_data$y, train_probabilities)
test_roc_curve <- roc(test_data$y, probabilities)
train_auc <- auc(train_roc_curve)
test_auc <- auc(test_roc_curve)

# 9. Check for overfitting;
if (train_auc > test_auc) {
  print("Overfitting detected.")
} else {
  print("No overfitting detected.")
}

cat("Train AUC:", train_auc, "\n")
cat("Test AUC:", test_auc, "\n")

plot(test_roc_curve, col = "blue", main = "ROC AUC Curve")
lines(train_roc_curve, col = "red")
legend("bottomright", legend = c("Test Set", "Train Set"), col = c("blue", "red"), lty = 1)