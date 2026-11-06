# start from merge the six different season files into pne only master file

# 1. Load the libraries
library(tidyverse) # This library providing functions for data manipulation, iteration, and string manipulation.                           
library(lubridate) # This library is designed specifically for parsing and manipulating dates.

# 2. Creates a character vector containing the file names for six Serie A football seasons 
file_paths <- c(
  "I1_1819.csv",
  "I1_1920.csv",
  "I1_2021.csv",
  "I1_2122.csv",
  "I1_2223.csv",
  "I1_2324.csv"
)

# 3. Defines a custom function that accepts a file path string as its argument
read_and_clean <- function(path) {
  
  # Check the first line to determine whether the separator is ';' or ','
  first_line <- readLines(path, n = 1, warn = FALSE)
  sep_char <- if (grepl(";", first_line)) ";" else ","
  
  # Reads the CSV file into a data frame (df) using the auto detected delimeter
  # stringasFactors = FALSE: Keeps text columns as characters rather than           converting them to factors
  # check.name = FALSE: Preserve original column names
  df <- read.csv(path, sep = sep_char, stringsAsFactors = FALSE, check.names = FALSE)
  
  # Remove hidden special characters (BOM) and spaces from column names
  names(df) <- gsub("\ufeff", "", names(df))
  names(df) <- trimws(names(df))
  
  # Identify the 'Date' column.
  date_col_idx <- grep("^date$", names(df), ignore.case = TRUE)[1]
  
  if (is.na(date_col_idx)) { ## If no column named "date" is detected (is.na),                                  it defaults assigns the index 2 that is the                                     standard positional index for dates in this                                     datase   
    date_col_idx <- 2
  }   
  
  # Renames and converts the date (handles '18/08/18' and '18/08/2018')
  names(df)[date_col_idx] <- "Date"
  df$Date <- dmy(df$Date)
  
  # Extracts and adds the Season column.
  digits <- str_extract(path, "\\d{4}")
  df$Season <- paste0("20", substr(digits, 1, 2), "-20", substr(digits, 3, 4))
  return(df)
}

# 4. Merge the files and sort chronologically.
master_df <- map_dfr(file_paths, read_and_clean) %>%  
  arrange(Date)

# 5. Final assessments
dim(master_df) # Display the dimensions of master_df to verify the size of the                   combined dataset
table(master_df$Season) # Generates a frequency table showing the count of rows (matches) assigned to each season, verifying that all six files were successfully read and appended

# ==============================================================================
# SECTION 1: MULTIPLE LINEAR REGRESSION AND MSE 
# ==============================================================================
# We fit an Ordinary Least Square (OLS) multiple linear regression model to predict match total goals (Y) using home/away shot features (X). The dataset  (N = 2,156) is split into 70% Training and 30% Test sets to compute out-of-sample  performance. We evaluate model fit by comparing Training MSE (MSE_Tr = 2.0478)  with Test MSE (MSE_Te = 2.2607) to quantify generalization performance.

# 1. load the libraries
library(tidyverse)
library(class) # Library for the classification algorithms

# 2. Prepare the variables for the regression. (FTHG: full time home team goals, FTAG: full time away team goals, HS: home team shots, AS: away team shots, HST: home team shots on target, AST: away team shots on target)
df_reg <- master_df %>%
  select(FTHG, FTAG, HS, AS, HST, AST) %>% # select 6 match variables
  drop_na() %>% # Remove missing rows
  mutate(TotalGoals = FTHG + FTAG) # Compute total goals

# 3. Random split Train (70%) and Test (30%)
set.seed(123) # Ensures the reproducibility of sampling.
# Randomly select 70% of rows for training and the remaining 30% for testing
train_idx <- sample(1:nrow(df_reg), size = 0.7 * nrow(df_reg))
train_set <- df_reg[train_idx, ]
test_set  <- df_reg[-train_idx, ]
#  Evaluating model performance solely on Tr yields a biased, overly optimistic measure because complex models can easily fit training noise. Assessing accuracy requires fresh, unseen observations Te

# 4. Fits an Ordinary Least Squares (OLS) multiple linear regression model using train_set
lm_fit <- lm(TotalGoals ~ HS + AS + HST + AST, data = train_set)
# A linear model imposes a restrictive structural assumption. It has high bias if the true f(X) is non-linear, but low variance and very high interpretability. 
# 5. Calculate predictions on the train and test sets. Computes estimated values using our estimated function
pred_tr <- predict(lm_fit, newdata = train_set) 
pred_te <- predict(lm_fit, newdata = test_set)

# 6. Calculate MSE_Tr e MSE_Te. Calculates the mean squared difference between observed goals and predicted goals for both sets
mse_tr <- mean((train_set$TotalGoals - pred_tr)^2) 
mse_te <- mean((test_set$TotalGoals - pred_te)^2)

cat("--- REGRESSION RESULTS --- \n")
cat("Training MSE (MSE_Tr):", round(mse_tr, 4), "\n") # Display the MSE_tr with 4 decimals
cat("Test MSE     (MSE_Te):", round(mse_te, 4), "\n") # Display the MSE_Te with 4 decimals
# ====================== ANALYSIS OF THE RESULTS ===============================
# The close proximity between MSE_{Tr} and MSE_{Te} confirms that the linear model generalizes well to unseen test data and does not suffer from overfitting. In statistical learning, MSE_{Tr} is expected to be optimistic because the parameter vector β^ is computed directly to minimize training residuals. √MSEte = 1.5036 goals: on average, our linear model's goal prediction deviates from the actual match total by approximately +\ 1.50 goals. Given that total goals in Serie A matches typically average between 2.5 and 2.8 goals per match, a standard error of = 1.50 goals represents a non-trivial error margin. This indicates that volume-based shot metrics alone (HS, AS, HST, AST) do not capture the full variance of goal outcomes, leaving a substantial irreducible error term var (ε). So it can be a potential stuctural bias from assuming a sctrictly linear function form. However despite its predictive limitations relative to non parametric flexible methods, OLS offers high interpretability.

# ==============================================================================
# SECTION 2: KNN CLASSIFICATION AND ERROR CURVE AS A FUNCTION OF 1/K
# ==============================================================================
# In this second block, we apply the K-Nearest Neighbors (KNN) algorithm to classify whether the home team wins (HomeWin = 1) or not (HomeWin = 0), evaluating the effect of overfitting when K is small (flexibility 1/K \to 1).

# Load libraries
library(tidyverse)
library(class)

# 1. Prepare the data for classification
# Converts the categorical full-time result (FTR) into a binary numeric outcome HomeWin (1 for Home Win, 0 otherwise). Transitions from a regression problem to a Classification Problem.
df_class <- master_df %>%
  select(FTR, HS, AS, HST, AST) %>%
  drop_na() %>% # Remove missing rows
  mutate(HomeWin = ifelse(FTR == "H", 1, 0))

# 2. Normalize the predictors (standardization is mandatory for KNN)
# Standardizes features by subtracting the mean and dividing by the standard deviation (μ = 0, σ = 1). Standardization ensures equal weighting across dimensions, mitigating distortions caused by the Curse of Dimensionality
scaled_X <- scale(df_class %>% select(HS, AS, HST, AST))

# 3. Split Train/Test. Splits the standardized predictor matrix X and response vector Y into training (70%) and testing (30%) sets.
set.seed(123) #Ensures the reproducibility of sampling.
train_idx <- sample(1:nrow(df_class), size = 0.7 * nrow(df_class))

# Prepares separated data inputs (X_{Tr}, Y_{Tr}) and (X_{Te}, Y_{Te}) to calculate independent training and test error rates
X_train <- scaled_X[train_idx, ]
X_test  <- scaled_X[-train_idx, ]
y_train <- df_class$HomeWin[train_idx]
y_test  <- df_class$HomeWin[-train_idx]

# 4. Calculate the misclassification error for various values of K.  Defines a vector of neighborhood sizes (K) and sets up a data frame to collect performance metrics. K controls model flexibility: small K = High flexibility, non-linear decision boundary; large K = low flexibility, smooth decision boundary.
k_values <- c(1, 5, 10, 20, 50, 100)
results <- data.frame(K = k_values, Train_Err = NA, Test_Err = NA)

# Runs KNN for each K, predicting class labels for both training and test sets and calculates the misclassification error rate.
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  pred_tr <- knn(X_train, X_train, y_train, k = k)
  pred_te <- knn(X_train, X_test,  y_train, k = k)
  
  results$Train_Err[i] <- mean(y_train != pred_tr)
  results$Test_Err[i]  <- mean(y_test != pred_te)
}

print(results)
# ====================== ANALYSIS OF THE RESULTS ===============================
# The KNN results clearly show the bias-variance trade-off.For K=1, we have maximum flexibility: the model overfits the training data, leading to a test error of 33.1%. By increasing K to 50 or 100, we smooth out the decision boundary and reduce variance, achieving a minimum test error of 29.5%.So, The KNN model is able to correctly classify the outcome of a home win approximately 70.5% of the time for matches it has not encountered before.

# Mathematically, the KNN classifier achieves an accuracy of 70.5%. However, from a methodological standpoint, this constitutes an ex-post classification, as it utilizes in-game data (total shots and shots on target from the match itself). To transform the model into a genuine ex-ante forecasting tool, we would need to replace the match-specific data with moving averages of the two teams' prior performance.

# ==============================================================================
# SECTION 3: ADVANCED LINEAR REGRESSION & EXTENSIONS 
# ==============================================================================
# In Exercise 3, we take our baseline OLS linear model from Exercise 1 and systematically test key ISLR extensions: qualitative season dummies, shot-synergy interaction terms, and polynomial transformations. Our main goal is to see if capturing these non-linearities and interactions improves our out-of-sample goal prediction accuracy (Test MSE) on Serie A match data. Finally, we apply backward stepwise selection to prune (individuare) redundant features and isolate the optimal, most parsimonious model.

# Load libraries
library(tidyverse)
library(class)

# DATA PREPARATION FOR EX 3 (Includes the Season column)
df_reg <- master_df %>%
  mutate(TotalGoals = FTHG + FTAG) %>%
  select(TotalGoals, HS, AS, HST, AST, Season) %>%
  drop_na()

set.seed(123)
train_idx <- sample(1:nrow(df_reg), 0.7 * nrow(df_reg))

train_reg <- df_reg[train_idx, ]
test_reg  <- df_reg[-train_idx, ]

# 1. BASELINE MULTIPLE REGRESSION & HYPOTHESIS TESTING 
# The lm() function estimates an OLS multiple linear regression model by estimating Y^ (TotalGoals) based on the 4 explanatory variables related to shots implements the formal model: Y = β0 + β1X1 + β2X2 + β3X3 + β4X4 + ε. The lm() function manage to calculate the most important coefficient for the OLS analysis. The coefficients β^_j are estimated by minimizing the Residual Sum of Squares (RSS): RSS  = ∑_{i=1}^n (y_i - β^_0 - ∑_{j=1}^p β^_j x_{ij})^2
lm_ch3_base <- lm(TotalGoals ~ HS + AS + HST + AST, data = train_reg)

# This function generate the inference and goodness-of-fit metrics required to test the statistical validity of the model
summary_base <- summary(lm_ch3_base)

# In the displayed table, a hypothesis test is conducted for each variable:     H_0 : β_j = 0 vs H_A : β_j =/ 0. Using the t-statistic with n-p-1 degrees of freedom: t = (β^_j - 0) / SE(β^_j)
cat("\n================ BASELINE MODEL SUMMARY ================\n")
print(summary_base)

# Extract key statistics: R2, RSE, and F-statistic 
r2_base  <- summary_base$r.squared # R^2 = (TSS - RSS) / TSS = 1 - RSS/TSS;   TSS = ∑(y_i - ӯ)^2: it measures the proportion of total variance in goals explained by the model
rse_base <- summary_base$sigma # RSE = √[(1/n-p-1)*RSS]: it represents the average deviation by which the actual goals in the matches differ from the estimated regression plane Ῡ
f_stat   <- summary_base$fstatistic[1] # F = frac{(TSS - RSS)/p}{RSS/(n-p-1)} ~ F_{p, n-p-1}: it serves to answer the question:"is at least one predictor useful?"

# Round the model's global metrics and display them neatly on the screen.
cat("\nModel Assessment Metrics:\n")
cat("R-squared (R^2):", round(r2_base, 4), "\n")
cat("Residual Standard Error (RSE):", round(rse_base, 4), "goals\n")
cat("F-statistic:", round(f_stat, 2), "\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# I. R^2 = 0.2312: The 4 shooting statistics explain 23.12% of the total variability in the number of goals scored in matches.
# II. RSE = 1.4334 goals: On average, the model's predictions deviate from the actual outcome by approximately 1.43 goals per match.
# III. F-statistic = 113.1 (p < 2.2*10^{-16}): Since F >> 1 and the p-value is effectively zero, we confidently reject the null hypothesis H_0: β_1 = β_2 = β_3 = β_4 = 0. At least one of the shooting variables has strong explanatory power.
# IV. Analysis of the coefficients (HS, AS, HST, AST):
# HST and AST are the dominant and highly significant variables. HS and AS, given the same number of shots on target, having a higher total number of shots simply means that some shots went wide or were blocked.


# 2. QUALITATIVE PREDICTORS: DUMMY VARIABLES
# Convert Season into a factor variable to create dummy variables. For a variable with K levels (6 seasons), R creates K-1 = 5 dummy variables, taking the first season (2018/19) as the reference level or baseline. A dummy variable is a binary (0 or 1) "ON/OFF" switch that allo. Each dummy coefficient simply measures the expected shift in the outcome variable relative to that baseline.
train_reg$Season <- factor(train_reg$Season)
test_reg$Season  <- factor(test_reg$Season)

# Estimate the model by adding the 5 seasonal dummy variables to the quantitative shooting metrics. Y = β_0 + ∑_{j=1}^4 β_j X_j + 𝛾_1 D_1 +...+ 𝛾_5 D_5 + ε. Each coefficient 𝛾_k measures the average shift in the goal baseline for season k relative to the base season (18/19).
lm_ch3_qual <- lm(TotalGoals ~ HS + AS + HST + AST + Season, data = train_reg)

# Display the output showing the effect of each seasonal dummy. It allows for assessing, via individual p-values, whether the average number of goals per match in Serie A has changed significantly from one season to the next.
cat("\n================ QUALITATIVE PREDICTORS SUMMARY ================\n")
print(summary(lm_ch3_qual))
# ====================== ANALYSIS OF THE RESULTS ===============================
# I. R^2 rises to 0.2530 (from 23.12%) and RSE falls to 1.415 goals. Including the season improves the goodness of fit.
# II. Analysis of the dummy variables:
# - Season 18/19 is the baseline.
# - Season 19/20 (+0.084, p = 0.488): Not significantly different from 2018-19.
# - Season 20/21 (+0.328, p = 0.0081): +0.33 more goals on average per match.
# - Season 21/22 (+0.8118, p = 4.29*10^{-9}): The season with the highest peak in goals in Serie A (+0.81 goals on average compared to 2018–19).
# - Season 22/23 (+0.621) and Season 23/24 (+0.700): Both are highly significant confirming a structural trend of greater attacking potency compared to 2018–19.


# 3. INTERACTION EFFECTS & HIERARCHY PRINCIPLE 
# Adding interaction term between Home Shots (HS) and Home Shots on Target (HST)
# Hierarchy principle: If an interaction term X_1*X_2 is included, the main effects X_1 and X_2 must also be retained in the model, even if their individual p-values are not significant. lm() automatically keeps main effects HS and HST
lm_ch3_inter <- lm(TotalGoals ~ HS * HST + AS * AST + Season, data = train_reg)

# It allows for quantifying whether the p-value of the interaction term β_3 is significant and how much additional explained variance it brings to the model compared to the additive effect.
cat("\n================ INTERACTION MODEL SUMMARY ================\n")
print(summary(lm_ch3_inter))
# ====================== ANALYSIS OF THE RESULTS ===============================
# I. Global Metrics: R^2 = 0.2576 and RSE = 1.412.
# II. Interaction: HS:HST (Coefficient = -0.0055, p = 0.00385): The interaction term between Home Shots and Home Shots on Target is statistically significant. This indicates that the effect of shots on target is not purely additive, but rather varies depending on the total volume of shots attempted. Although the main variable HS loses its individual significance (p = 0.252), it is retained in the model in accordance with the Hierarchy Principle from ISLR, since its interaction term HS:HST is significant.


# 4. POLYNOMIAL REGRESSION: NON-LINEAR EFFECTS 
# Quadratic transformation of shots to model non-linear goal returns
# The function I() force R to calculate the algebraic mathematical operation HS^2 instead of interpreting it as R formula syntax. Implement Polynomial Regression: Y = β_0 + β_1 X + β_2 X^2 +...+ ε
lm_ch3_poly <- lm(TotalGoals ~ HS + I(HS^2) + AS + I(AS^2) + HST + AST + Season, data = train_reg)

# The statistical significance (p-value) of the quadratic coefficient β^_2 confirms or refutes the presence of a genuine non-linear curvature in the data.
cat("\n================ POLYNOMIAL MODEL SUMMARY ================\n")
print(summary(lm_ch3_poly))
# ====================== ANALYSIS OF THE RESULTS ===============================
# I. Quadratic Term I(HS^2) (Coefficient = -0.0014057, p = 0.06221): It is marginally significant at the 10% level. The negative curvature of the quadratic term models diminishing marginal returns: accumulating shots that fail to hit the target yields an increasingly smaller benefit in terms of total goals.


# 5. MODEL SELECTION: BACKWARD STEPWISE SELECTION 
# Perform backward elimination based on AIC
# The step() function performs a stepwise backward selection based on the AIC information criterion.
# BACWARD SELECTION:
# 1. It starts with the full model containing all variables and interactions.
# 2. Remove the variable with the largest p-value that is thr variable that is the least statistically significant.
# 3. It stops when all remaining variables have a significant p-value
lm_ch3_opt <- step(lm_ch3_inter, direction = "backward", trace = FALSE)

# View the final, clean structure of the optimized model.
cat("\n================ OPTIMAL MODEL VIA BACKWARD SELECTION ================\n")
print(summary(lm_ch3_opt))
# ====================== ANALYSIS OF THE RESULTS ===============================
# I. Variables Removed by the AIC Algorithm: Stepwise elimination removed total away shots (AS) and the away interaction term (AS:AST), deeming them redundant. Final model = TotalGoals ~ HS + HST + AST + Season + HS:HST. It reduces the number of parameters from 12 to 10, while maintaining an almost identical R^2 (0.2570) and reducing the risk of overfitting.


# 6. COMPARISON OF OUT-OF-SAMPLE TEST MSE
# Group the 5 estimated models into a structured list so that their performance can be compared in a single pass.
models <- list(
  "1. Baseline Linear"     = lm_ch3_base,
  "2. + Qualitative Season" = lm_ch3_qual,
  "3. + Interactions"      = lm_ch3_inter,
  "4. + Polynomial terms"  = lm_ch3_poly,
  "5. Optimal (Backward)"  = lm_ch3_opt
)

test_mse_results <- sapply(models, function(mod) {
  preds <- predict(mod, newdata = test_reg) # Calculate the predicted values y^_0 for the 647 observations in the Test Set.
  mean((test_reg$TotalGoals - preds)^2) # calculate the actual test mean squared error:MSE_{Te} = 1/N_{Te}*∑_{i \in Te} (y_0 - f^}(x_0))^2. Apply the core principle from ISLR: the quality of a model is never measured by the Training MSE, but rather by its out-of-sample generalization capability (MSE_{Te}).
})

# Round to 4 decimal places and display the final comparison of the MSE test results for all 5 developed models. It allows for the empirical identification of the model that minimizes the Test MSE, striking an ideal balance between flexibility (bias) and estimation stability (variance).
cat("\n================ OUT-OF-SAMPLE TEST MSE COMPARISON ================\n")
print(round(test_mse_results, 4))
# ====================== ANALYSIS OF THE RESULTS ===============================
# I. BASELINE LINEAR: MSE_Te = 2.2607 = Simpler model, it incurs a specification bias due to the absence of temporal factors.
# II. + QUALITATIVE SEASON: MSE_Te = 2.1850 = Significant reduction in error. This demonstrates that accounting for the seasonal effect improves out-of-sample prediction.
# III. + INTERACTIONS: MSE_Te = 2.1711 = Further reduction. Capturing the HS:HST synergy reduces the Test MSE.
# IV. + POLYNOMIAL TERMS: MSE_Te = 2.1694 = Best overall MSE test result. Modeling the curvature of the shots provides the greatest generalization capability on unseen data.
# V. + OPTIMAL (BACKWARD): MSE_Te = 2.1881 = Slightly superior to the polynomial model. Stepwise selection based on AIC operates on the likelihood within the training set, whereas evaluation on the test set identifies the polynomial model as offering the best bias-variance tradeoff.

# Remember to change the name of train_set and test_set into train_reg and test_reg before run the code. So run this two line in console: train_reg <- train_set; test_reg  <- test_set before running the entire code

# ==============================================================================
# SECTION 4: BINARY LOGISTIC REGRESSION *
# ==============================================================================
# With a binary response variable Y ∈ {0, 1\} (e.g., HomeWin: 1 =  Home Win, 0 = No Home Win), the application of Ordinary Least Squares (OLS) presents insurmountable limitations:
# - Out-of-Range Probabilities: A linear model Y^ = Xβ can yield predictions below 0 or above 1, which lack any probabilistic meaning. 
# - Heteroskedasticity of Residuals: The variance of a Bernoulli variable depends on the mean, violating the OLS assumption of constant variance of the error components.
# To ensure that the estimated probability p(X) = Pr(Y = 1|X) always remains confined within the interval (0, 1), the logistic (or sigmoid) function is applied and through algebraic rearrangement, the function is transformed into the Logit (or Log-Odds) form: 
# log(frac{p(X)}{1 - p(X)}) = β_0 + β_1 X_1 +...+ β_p*X_p
# The β coefficients are not estimated by minimizing OLS residuals, but rather by maximizing the Likelihood Function (Maximum Likelihood Estimation – MLE), identifying the parameters that maximize the probability of observing the actual data.

# Load libraries
library(tidyverse)
library(class)

# 1. Preparation of the Binary Target 
# HomeWin = 1 if the home team winse (FTHG > FTAG), 0 otherwise.
train_reg$HomeWin <- ifelse(train_reg$FTHG > train_reg$FTAG, 1, 0)
test_reg$HomeWin  <- ifelse(test_reg$FTHG > test_reg$FTAG, 1, 0)
# Add the season dummy variable 
train_reg$Season <- as.factor(master_df$Season[train_idx])
test_reg$Season  <- as.factor(master_df$Season[-train_idx])

#2. Estimation of the Logit Model via Maximum Likelihood Estimation (glm)
# The glm() function stands for Generalized Linear Model. It saves the estimated model into the variable glm_logit_base.
glm_logit_base <- glm(
  HomeWin ~ HS + AS + HST + AST + Season,
  data   = train_reg, # It instructs R to use exclusively the training dataset to calculate the parameters.
  family = binomial(link = "logit") # binomial: Informs R that the target variable follows a Binomial/Bernoulli distribution (0/1 outcomes); link = "logit": Specifies the Logit link function (log(frac{p}{1-p})). This forces R to move away from OLS and estimate the β parameters using Maximum Likelihood Estimation (MLE).
)

# 3. Summary of Coefficients in Log-Odds
summary_logit <- summary(glm_logit_base) # Extracts all detailed training results
print(summary_logit)
# ====================== ANALYSIS OF THE RESULTS ===============================
# This output shows parameter estimates on the Logit (log-odds) scale—that is, prior to the exponential transformation (e^β) used to calculate odds ratios. The model expresses the log-odds of a home win as:
# log(p / 1-p) = -0.6414 - 0.0674*HS + 0.0597*AS + 0.3935*HST - 0.3590*AST + ∑𝛾_k *Season_k.
# - HST: It is the strongest positive predictor and is highly significant. Each additional shot on target by the home team increases the log-odds of victory by +0.3935 corresponding to e^{0.39345} = 1.4821, or a +48.2% increase in the odds.
# - AST: It is the primary negative predictor, and it is highly significant. Each shot on target by the visiting team reduces the log-odds of a home win by -0.3590 equivalent to e^{-0.35899} = 0.6984, i.e., a -30.2% change in the odds.
# - HS and AS: Both total shot figures are significant, but they show the opposite trend compared to shots on target. Holding shots on target (HST) constant, an increase in total shots (HS) implies an accumulation of shots off target. The negative coefficient (β = -0.0674) quantifies the effect of offensive inaccuracy.
# - Season: All p-values for the seasonal dummies are well above the 5% significance level. Compared to the 2018-2019 baseline season, there are no significant structural changes in the baseline probability of a home win, once shot production is controlled for.


# 4. Transformation into Odds Ratios (OR) and 95% Confidence Intervals
# coef(glm_logit_base): Extracts the estimated coefficients β^, which are initially on a logarithmic scale (log-odds). exp(...): Calculates the exponential e^β^ to convert them into Odds Ratios, making them interpretable in terms of the percentage change in probability.
odds_ratios <- exp(coef(glm_logit_base))
# confint.default(...): Calculates the lower and upper limits of the 95% confidence intervals for each coefficient β using the standard normal distribution (z). exp(...): Applies the exponential function to transform the interval endpoints onto the scale of Odds Ratios.
ci_odds     <- exp(confint.default(glm_logit_base))
# cbind(...): Combines the point estimate of the Odds Ratio with its 95% confidence interval by column (column bind).
or_table    <- cbind(Odds_Ratio = odds_ratios, ci_odds)
print(round(or_table, 4))
# ====================== ANALYSIS OF THE RESULTS ===============================
# Remember: OR > 1: The predictor increases the home team's chances of winning OR < 1: The predictor reduces the home team's chances of winning. Confidence Interval (2.5% - 97.5%): If the interval contains the value 1, the predictor is not statistically significant.
# - HST: The primary key factor. Each additional shot on target by the home team increases the odds of victory by 48.2%.
# - AST: Each shot on target by the visiting team reduces the odds of a home win by 30.2% (1 - 0.6984 = 30.2%).
# - HS: Keeping shots on target (HST) constant, taking more total shots means taking more shots off target. Each additional off-target shot reduces the odds by 6.5%.
# - AS: Holding the number of shots on target by the visiting team (AST) constant, a higher number of total off-target shots by the visitors slightly favors the home team (+6.15%).
# - Season: All contains 1 so thei are not significant. When controlling for shooting efficiency, the "year" effect does not alter the baseline probability relative to 2018/19.


# 5. Probability Prediction on the Test Set (out-of-sample)
# predict(glm_logit_base, newdata = test_reg,): Applies the estimated model to new data from the test set in order to evaluate its generalization capability. type = "response": it tells R to apply the sigmoid function and return the estimated probabilities rather than the values on the logit scale.
pred_probs <- predict(glm_logit_base, newdata = test_reg, type = "response")

# 6. Classification with Standard Threshold (Threshold = 0.5)
# ifelse(pred_probs >= 0.5, 1, 0): Converts continuous probabilities into a binary decision: if the probability of a home win is at least 50%, it assigns the predicted class 1; otherwise, it assigns 0.
pred_class_05 <- ifelse(pred_probs >= 0.5, 1, 0)

# 7. Confusion Matrix and Test Accuracy
# table(...): Cross-tabulates the model's predictions (pred_class_05) with the actual outcomes from the test set (test_reg$HomeWin), generating a 2 X 2 confusion matrix.
conf_matrix_05 <- table(Predicted = pred_class_05, Actual = test_reg$HomeWin)
cat("\n--- CONFUSION MATRIX (Threshold = 0.5) ---\n")
print(conf_matrix_05)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - True Negatives (TN = 313): Matches in which the home team did not win, correctly classified as 0.
# - True Positives (TP = 142): Actual home wins correctly predicted as 1.
# - False Positives (FP = 69): Home wins predicted by the model, but the match ended in a draw (pareggio) or a defeat.
# - False Negatives (FN = 123): The home team won, but the model estimates a probability p < 0.5.


# pred_class_05 == test_reg$HomeWin: Compares elements one by one, creating a vector of logical values (TRUE if the prediction is correct, FALSE if it is an error). mean(...): Calculates the mean of the logical values, which corresponds exactly to the proportion of correctly classified instances (Test Accuracy).
test_acc_05 <- mean(pred_class_05 == test_reg$HomeWin)
cat("\nTest Accuracy (Threshold 0.5):", round(test_acc_05 * 100, 2), "%\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# Test Accuracy = (TN + TP) / Total Test = (313 + 142) / (313 + 123 + 69 + 142) = 455 / 647 = 70,32%. An out-of-sample accuracy of 70.32% is an excellent result for predicting football outcomes based solely on shot metrics.

# Remember to change the name of train_set and test_set into train_reg and test_reg before run the code. So run this two line in console: train_reg <- train_set; test_reg  <- test_set before running the entire code

# ==============================================================================
# SECTION 5: GENERATIVE MODELS (LDA, QDA, NAÏVE BAYES) VS. DISCRIMINATIVE MODELS (LOGIT). OUT-OF-SAMPLE EVALUATION, CONFUSION MATRICES, AND ROC/AUC CURVES
# ==============================================================================
# In this analytical section, generative classification models (LDA, QDA, and Naïve Bayes) are compared against the discriminative benchmark of Logistic Regression to estimate the probability of a home win (HomeWin) in Serie A. An out-of-sample evaluation, conducted on 30% of the dataset (N=647), demonstrates the superiority of linear models, with LDA achieving the best performance.

# 1. Load libraries
suppressPackageStartupMessages({
  library(MASS)       # For LDA e QDA
  library(e1071)      # For Naïve Bayes
  library(pROC)       # For Curve ROC e AUC
})

# 2. Data preparation and Train/Test split (70% Train, 30% Test - N_total = 2,156 matches)
# Constructs the binary target variable Y: HomeWin = 1 (Home win), 0 (draw / defeat)
master_df$HomeWin <- factor(ifelse(master_df$FTHG > master_df$FTAG, 1, 0))

set.seed(123) # To ensure the exact reproducibility of the results
train_idx <- sample(seq_len(nrow(master_df)), size = 0.70 * nrow(master_df)) # Randomly selects row indices corresponding to 70% of the dataset to form the training set.
train_set <- master_df[train_idx, ]
test_set  <- master_df[-train_idx, ]

# Defining Formulas
# For LDA/QDA, we use continuous shooting predictors to avoid rank singularities
formula_gen <- HomeWin ~ HST + AST + HS + AS # Specifies the model formula for generative algorithms.
formula_log <- HomeWin ~ HST + AST + HS + AS + Season # Specifies the model formula for Logistic Regression.

# 3. Fitting of models on the training set
# Discriminative Benchmark: Logistic Regression 
fit_logit <- glm(formula_log, data = train_set, family = binomial) # Fits a Binary Logistic Regression model on the training set using Maximum Likelihood Estimation
prob_logit <- predict(fit_logit, newdata = test_set, type = "response") # redicts out-of-sample probabilities p(X) = Pr(HomeWin = 1|X) on the test set.
pred_logit <- factor(ifelse(prob_logit >= 0.5, 1, 0)) # Converts estimated probabilities into class predictions using a decision boundary threshold of 0.5.

# Generative Model 1: Linear Discriminant Analysis (LDA)
fit_lda <- lda(formula_gen, data = train_set) # stimates class prior probabilities π_k and class-specific means μ_k, assuming a common covariance matrix ∑ across classes.
pred_lda_obj <- predict(fit_lda, newdata = test_set) # Computes posterior probabilities and class assignments for test observations via Bayes' theorem.
prob_lda <- pred_lda_obj$posterior[, "1"] # Extracts posterior probabilities Pr(HomeWin = 1|X).
pred_lda <- pred_lda_obj$class # Extracts assigned class labels based on Bayes' decision rule (highest posterior probability).

# Generative model 2: Quadratic Discriminant Analysis (QDA)
fit_qda <- qda(formula_gen, data = train_set) # Fits QDA by estimating distinct covariance matrices ∑_k for each class k ∈ {0, 1}.
pred_qda_obj <- predict(fit_qda, newdata = test_set) # Calculates QDA posterior probabilities and classifications on test data.
prob_qda <- pred_qda_obj$posterior[, "1"] # Calculates QDA posterior probabilities and classifications on test data.
pred_qda <- pred_qda_obj$class # Extracts predicted class labels.

# Generative model 3: Naïve Bayes 
fit_nb <- naiveBayes(formula_gen, data = train_set) #  Fits Naïve Bayes under the assumption that all predictors X_j are conditionally independent given the class label Y.
prob_nb <- predict(fit_nb, newdata = test_set, type = "raw")[, "1"] # Extracts raw posterior probability estimates for Y = 1.
pred_nb <- predict(fit_nb, newdata = test_set, type = "class") # Assigns predicted classes.

# 4. Calculation of confusion matrices and extraction of metrics
compute_metrics <- function(actual, predicted, prob, model_name) { # Defines a function to compute performance metrics from actual vs. predicted values
  cm <- table(Actual = actual, Predicted = predicted) # Generates the 2x2 Confusion Matrix.
  
  TN <- cm["0", "0"]; FP <- cm["0", "1"]
  FN <- cm["1", "0"]; TP <- cm["1", "1"]
  
  accuracy    <- (TP + TN) / sum(cm)
  sensitivity <- TP / (TP + FN)  # True Positive Rate (Recall for HomeWin = 1)
  specificity <- TN / (TN + FP)  # True Negative Rate (Recall for HomeWin = 0)
  
  roc_obj <- roc(actual, prob, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_obj)) # Computes the Area Under the ROC Curve (AUC).
  
  data.frame(
    Model = model_name,
    Accuracy = paste0(round(accuracy * 100, 2), "%"),
    Sensitivity = paste0(round(sensitivity * 100, 2), "%"),
    Specificity = paste0(round(specificity * 100, 2), "%"),
    AUC = round(auc_val, 4)
  )
}

# Generation of a Comparison Table
results_summary <- rbind(
  compute_metrics(test_set$HomeWin, pred_logit, prob_logit, "Logistic Regression"),
  compute_metrics(test_set$HomeWin, pred_lda, prob_lda, "LDA"),
  compute_metrics(test_set$HomeWin, pred_qda, prob_qda, "QDA"),
  compute_metrics(test_set$HomeWin, pred_nb, prob_nb, "Naïve Bayes")
)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - LDA: The absolute best. The linear decision boundary works extremely well, and the shared covariance matrix reduces variance.
# - LOGIT: An excellent discriminative benchmark. Performance is nearly identical to LDA, confirming that the Gaussianity assumption of LDA holds true.
# - QDA: Slight overfitting. Estimating separate covariance matrices (∑_k) adds too much flexibility without any real benefit.
# - NAIVE BAYES: Assumption violated. There is a strong correlation between total shots (HS/AS) and shots on target (HST/AST).
# LDA assumes that the predictors follow a multivariate Gaussian distribution with the same covariance matrix ∑ for both classes (Y=0 and Y=1). This assumption of homoscedasticity acts as an implicit form of regularization: it limits the number of parameters to be estimated, resulting in an estimate with very low variance.


cat("\n================ COMPARISON TABLE PERFORMANCE OUT-OF-SAMPLE ================\n")
print(results_summary)

# Detailed Visualization of Confusion Matrices
cat("\n--- Comparison Table LDA ---\n"); print(table(Actual = test_set$HomeWin, Predicted = pred_lda))
cat("\n--- Comparison Table QDA ---\n"); print(table(Actual = test_set$HomeWin, Predicted = pred_qda))
cat("\n--- Comparison Table Naïve Bayes ---\n"); print(table(Actual = test_set$HomeWin, Predicted = pred_nb))

# 5. Generation of joint graph of RO curves
roc_logit <- roc(test_set$HomeWin, prob_logit) # Calculates sensitivity and specificity values across all threshold cutoffs from 0 to 1.
roc_lda   <- roc(test_set$HomeWin, prob_lda)
roc_qda   <- roc(test_set$HomeWin, prob_qda)
roc_nb    <- roc(test_set$HomeWin, prob_nb)

# Plot ROC Curves
par(mfrow = c(1, 1), mar = c(4.5, 4.5, 3, 1))
plot(roc_logit, col = "#1E293B", lwd = 2.5, main = "ROC Curves Comparison - Serie A Outcome Classification") # Plots the baseline ROC curve for Logistic Regression.
# Overlays LDA, QDA, and Naïve Bayes ROC curves on the same figure using distinct colors.
plot(roc_lda,   col = "#2563EB", lwd = 2, add = TRUE)
plot(roc_qda,   col = "#D97706", lwd = 2, add = TRUE)
plot(roc_nb,    col = "#16A34A", lwd = 2, add = TRUE)

# Legend
legend("bottomright", 
       legend = c(
         paste0("Logistic Reg (AUC = ", round(auc(roc_logit), 3), ")"),
         paste0("LDA (AUC = ", round(auc(roc_lda), 3), ")"),
         paste0("QDA (AUC = ", round(auc(roc_qda), 3), ")"),
         paste0("Naïve Bayes (AUC = ", round(auc(roc_nb), 3), ")")
       ),
       col = c("#1E293B", "#2563EB", "#D97706", "#16A34A"), 
       lwd = 2, bty = "n", cex = 0.9)

# ==============================================================================
# SECTION 6: POISSON GLM & MULTINOMIAL LOGISTIC REGRESSION 
# ==============================================================================
# This code apply Poisson Regression and Multinomial Logistic Regression to football match data. First, it fits a Poisson GLM to predict the total count of goals (TotalGoals) based on match shot statistics (HST, AST, HS, AS), evaluating performance via test MSE and checking for overdispersion. Next, it constructs a Multinomial Logistic Regression model to classify match outcomes into three categories: Home Win (1), Draw (X), or Away Win (2). Finally, it evaluates the 3-class classifier's out-of-sample performance using a confusion matrix and overall test accuracy.

# Load libraries
suppressPackageStartupMessages({
  library(nnet)       # For Multinomial Logistic Regression
  library(MASS)
})

# 1. GLM POISSON REGRESSION (Total Goals Count)
# Target: Total goals scored in the match
master_df$TotalGoals <- master_df$FTHG + master_df$FTAG 

# Splitting the dataset into a 70% training set and a 30% out-of-sample test set
train_set <- master_df[train_idx, ]
test_set  <- master_df[-train_idx, ]

# Fitting Poisson Model
# family = poisson(link = "log"): Sets the distribution to Poisson and applies the standard log link function log(λ) = Xβ, guaranteeing non-negative predicted values λ^ = e^{Xβ}.
fit_poisson <- glm(TotalGoals ~ HST + AST + HS + AS, family = poisson(link = "log"), data = train_set)

cat("\n================ POISSON REGRESSION SUMMARY ================\n")
print(summary(fit_poisson))
# ====================== ANALYSIS OF THE RESULTS ===============================
# Because the Poisson GLM uses a log-link function log(λ) = Xβ, raw coefficients must be exponentiated (e^{β^}) to be interpreted as multiplicative effects on expected total goals (λ):
# - HST: e^{0.0823} = 1.0857. Each additional shot on target by the home team increases total expected match goals by +8.57%.
# - AST: e^{0.0686}  1.0710$. Each additional shot on target by the away team increases total expected match goals by +7.10%.
# - HS ans AS: fail to reach strict 5% significance once shots on target are controlled for. This proves empirically that shot quality (accuracy) drives goal scoring, not raw shot volume.


# Overdispersion Check
# Calculates the sum of squared Pearson residuals ∑(Y_i - μ^_i)^2 / Var(μ^_i). Divides by the residual degrees of freedom (n - p - 1) to calculate the estimated dispersion parameter φ^. In an ideal Poisson model, variance equals the mean (Var(Y) = E(Y)), so φ = 1.
dispersion_stat <- sum(residuals(fit_poisson, type = "pearson")^2) / fit_poisson$df.residual
# Evaluates whether overdispersion exists (φ > 1.2). If variance significantly exceeds the mean, standard errors are underestimated, which would make variables appear artificially over-significant.
cat(sprintf("\nDispersion Parameter (Variance/Mean Ratio): %.4f\n", dispersion_stat))
if(dispersion_stat > 1.2) {
  cat("RESULT: Overdispersion detected! Poisson p-values might be underestimated.\n")
} else {
  cat("RESULT: No overdispersion detected. The Poisson model is well-specified.\n")
}
# ====================== ANALYSIS OF THE RESULTS ===============================
# State clearly that the Poisson assumption Var(Y) = E(Y) is satisfied: Since φ^ < 1.2, there is no overdispersion. Standard errors are not artificially deflated, confirming that the reported p-values are valid.


# Out-of-Sample Evaluation (Test MSE for goals)
# Computes the out-of-sample Mean Squared Error for goal predictions and prints the formatted numerical result.
pred_goals <- predict(fit_poisson, newdata = test_set, type = "response")
test_mse_poisson <- mean((test_set$TotalGoals - pred_goals)^2)
cat(sprintf("Poisson Regression Test MSE: %.4f\n", test_mse_poisson))
# ====================== ANALYSIS OF THE RESULTS ===============================
# The out-of-sample Test MSE = 2.3203 yields a standard error of goal prediction of √2.3203 = 1.52 goals per match.


# 2. MULTINOMIAL LOGISTIC REGRESSION
# 3-class Target: 1 (Home Win), X (Draw), 2 (Away Win)
master_df$Outcome1X2 <- factor(ifelse(master_df$FTHG > master_df$FTAG, "1",
                                      ifelse(master_df$FTHG == master_df$FTAG, "X", "2")),
                               levels = c("1", "X", "2"))

# Splitting the dataset into a 70% training set and a 30% out-of-sample test set
train_set_3c <- master_df[train_idx, ]
test_set_3c  <- master_df[-train_idx, ]

# Fits a Multinomial Logistic Regression model using Softmax link functions to model multi-class log-odds relative to the baseline category ("1").
fit_multi <- multinom(Outcome1X2 ~ HST + AST + HS + AS, data = train_set_3c, trace = FALSE) # trace = FALSE: Suppresses the detailed convergence output produced during maximum likelihood estimation.

# Evaluates test matches and assigns each to the class with the highest predicted probability (argmax_k P(Y = k|X)).
pred_multi_class <- predict(fit_multi, newdata = test_set_3c)
# Builds a 3x3 confusion matrix comparing actual vs. predicted match outcomes across all three classes
cm_multi <- table(Actual = test_set_3c$Outcome1X2, Predicted = pred_multi_class)
cat("\n================ MULTICLASS CONFUSION MATRIX (1X2) ================\n")
print(cm_multi)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Home Win (Class 1): Recall = 211 / 265 = 79.62% (Excellent detection).
# - Away Win (Class 2): Recall = 142 / 210 = 67.62% (Strong detection).
# - Draw (Class X): Recall = 18 / 172 = 10.47% (Severe bottleneck).
# Explain why the model fails on Draws (10.47% recall). In football, draws are stochastic equilibrium states rather than distinct tactical events. A draw can result from a low-intensity 0-0 stalemate or an intense 2-2 match. Because draw matches do not possess a unique "shot profile" in terms of HS and HST, shot metrics alone struggle to separate draws from wins or losses.


accuracy_multi <- sum(diag(cm_multi)) / sum(cm_multi) # Sums the main diagonal of the confusion matrix and Divides by total test observations to compute the overall 3-class classification accuracy rate.
cat(sprintf("Multiclass Out-of-Sample Accuracy (1X2): %.2f%%\n", accuracy_multi * 100))
# ====================== ANALYSIS OF THE RESULTS ===============================
# A zero-rule baseline classifier that simply predicts the most frequent class ("Home Win") every time achieves only 265 / 647 = 40.96%. The model provides a net gain of +16.38% in predictive accuracy.

# ==============================================================================
# SECTION 7: MODEL EVALUATION & CROSS-VALIDATION PARADIGMS 
# ==============================================================================
# In this section, we evaluate our Serie A predictive models using a resampling techniques to measure their true out-of-sample accuracy. We compare a simple Validation Set split with 10-Fold Cross-Validation and the analytical LOOCV leverage shortcut. Finally, we implement a leak-free cross-validation workflow that performs feature selection strictly inside each fold to prevent data leakage and overfitting.

# Load libraries
suppressPackageStartupMessages({
  library(boot)       # For cv.glm()
  library(caret)      # For stratified fold creation
})

# Ensure dataset and target variables are properly formatted
set.seed(42)
master_df$HomeWin    <- factor(ifelse(master_df$FTHG > master_df$FTAG, 1, 0))
master_df$TotalGoals <- master_df$FTHG + master_df$FTAG

# 1. Validation set approach (70/30 slit & seed variability)
# Demonstrates high variance of test error depending on random split
evaluate_validation_split <- function(seed_val) {
  set.seed(seed_val)
  train_idx <- sample(1:nrow(master_df), size = 0.70 * nrow(master_df)) # Randomly split the dataset (N=2,156) into two parts: 70% for the Training Set and 30% for the Validation Set.
  train_set <- master_df[train_idx, ]
  test_set  <- master_df[-train_idx, ]
  
  # Fit Logistic Regression on training set
  fit_logit <- glm(HomeWin ~ HST + AST + HS + AS, family = binomial, data = train_set)
  preds     <- ifelse(predict(fit_logit, newdata = test_set, type = "response") > 0.5, 1, 0) # Generate out-of-sample predictions for the observations in the validation set that were held out during training.
  
  # Calculate test error rate
  test_err  <- mean(preds != test_set$HomeWin) # Calculate the empirical test error rate.
  return(test_err)
}

# Evaluate across two different random split seeds
val_err_seed1 <- evaluate_validation_split(seed_val = 101)
val_err_seed2 <- evaluate_validation_split(seed_val = 202)

cat("\n================ 1. VALIDATION SET APPROACH ================\n")
cat(sprintf("Test Error Rate (Seed 101): %.4f (Accuracy: %.2f%%)\n", val_err_seed1, (1 - val_err_seed1) * 100))
cat(sprintf("Test Error Rate (Seed 202): %.4f (Accuracy: %.2f%%)\n", val_err_seed2, (1 - val_err_seed2) * 100))
cat(sprintf("Split-Induced Variability:  %.4f\n", abs(val_err_seed1 - val_err_seed2)))
# ====================== ANALYSIS OF THE RESULTS ===============================
# This demonstrates the primary drawback of a single train/validation split. Changing only the random partitioning seed alters the test accuracy by 0.62%. Because the model is trained on only 70% of the data (N = 1,509), the validation set approach is highly sensitive to which matches end up in the training fold, tending to overestimate the true test error rate.  


# 2. 10-fold cross validation & cv standard error
# Computes K-Fold CV error and its estimated standard error SE(CV_K)
set.seed(42)
k_folds <- 10 # Randomly divide the entire dataset into K=10 subsets (folds) of approximately equal size
folds   <- createFolds(master_df$HomeWin, k = k_folds, list = TRUE)

err_k_logit   <- numeric(k_folds)
mse_k_poisson <- numeric(k_folds)

for (k in 1:k_folds) {
  train_k <- master_df[-folds[[k]], ] # It holds back the k-th fold for validation and combines the remaining K-1 folds to form the training set.
  test_k  <- master_df[folds[[k]], ]
  
  # 2a. Logistic Regression Classification Fold Error
  fit_logit_k <- glm(HomeWin ~ HST + AST + HS + AS, family = binomial, data = train_k)
  pred_logit  <- ifelse(predict(fit_logit_k, newdata = test_k, type = "response") > 0.5, 1, 0)
  err_k_logit[k] <- mean(pred_logit != test_k$HomeWin) # Measure the misclassification rate of the k-th fold for logistic regression.
  
  # 2b. Poisson GLM Regression Fold MSE
  fit_poisson_k <- glm(TotalGoals ~ HST + AST + HS + AS, family = poisson(link = "log"), data = train_k)
  pred_poisson  <- predict(fit_poisson_k, newdata = test_k, type = "response")
  mse_k_poisson[k] <- mean((test_k$TotalGoals - pred_poisson)^2) # Calculate the Mean Squared Error of the $k$-th fold for the Poisson GLM
}

# 10-Fold CV Error Rates and Standard Errors
cv10_logit_mean <- mean(err_k_logit) # Calculate the Mean Squared Error of the k-th fold for the Poisson GLM.
cv10_logit_se   <- sd(err_k_logit) / sqrt(k_folds) # Calculate the estimated Standard Error of the cross-validation.
cv10_poisson_mse <- mean(mse_k_poisson)
cv10_poisson_se  <- sd(mse_k_poisson) / sqrt(k_folds)

cat("\n================ 2. 10-FOLD CROSS-VALIDATION ================\n")
cat(sprintf("Logistic 10-Fold CV Error Rate: %.4f (Accuracy: %.2f%%)\n", cv10_logit_mean, (1 - cv10_logit_mean) * 100))
cat(sprintf("Estimated SE(CV_10) Logistic:   %.4f\n", cv10_logit_se))
cat(sprintf("Poisson 10-Fold CV Test MSE:    %.4f\n", cv10_poisson_mse))
cat(sprintf("Estimated SE(CV_10) Poisson:    %.4f\n", cv10_poisson_se))
# ====================== ANALYSIS OF THE RESULTS ===============================
# By averaging over 10 non-overlapping validation folds, cross-validation eliminates the random split variability of the validation set approach. 
# - The baseline logistic regression model achieves an out-of-sample accuracy of 70.78% across all 2,156 Serie A matches. The estimated standard error quantifies the fold-to-fold uncertainty of this error rate.
# - The Poisson GLM yields an out-of-sample Mean Squared Error of 2.1776, meaning goal predictions deviate by approximately √2.1776 = 1.47 goals per match


# 3. LOOCV analytical leverage shortcut (OLS regression)
# Computes LOOCV MSE in O(1) single-fit time using hat values (h_i)
# Formula: CV_(n) = (1/n) * sum( ((y_i - y_hat_i) / (1 - h_i))^2 )
fit_ols <- lm(TotalGoals ~ HST + AST + HS + AS, data = master_df) # Train a single ordinary least squares (OLS) regression model on the entire dataset

# Extract leverage diagonal values (h_i) from hat matrix
h_i <- hatvalues(fit_ols)

# Compute LOOCV MSE via leverage shortcut formula
loocv_mse_shortcut <- mean(((master_df$TotalGoals - fitted(fit_ols)) / (1 - h_i))^2)
ols_train_mse      <- mean(residuals(fit_ols)^2)

cat("\n================ 3. LOOCV LEVERAGE SHORTCUT ================\n")
cat(sprintf("OLS Training MSE:              %.4f\n", ols_train_mse))
cat(sprintf("LOOCV Test MSE (Shortcut):     %.4f\n", loocv_mse_shortcut))
# ====================== ANALYSIS OF THE RESULTS ===============================
# Training MSE (2.1083) is smaller than LOOCV Test MSE (2.1188). This illustrates the fundamental rule that training error underestimates test error due to subtle model fitting to training noise.


# 4. preventing data leakage in cross validation
# Performs variable screening STRICTLY INSIDE each training fold
cv_errors_right_way <- numeric(k_folds)

for (k in 1:k_folds) {
  train_k <- master_df[-folds[[k]], ]
  test_k  <- master_df[folds[[k]], ]
  
  # STEP 1: Screen top 3 correlated predictors ONLY inside train_k
  candidate_vars <- c("HST", "AST", "HS", "AS", "HC", "AC")
  # Calculate the correlation between the predictor variables and the response y using only the data from train_k, and select the top 3 variables.
  cor_train      <- abs(sapply(train_k[, candidate_vars], function(x) cor(x, as.numeric(train_k$HomeWin))))
  top_features   <- names(sort(cor_train, decreasing = TRUE)[1:3])
  
  # STEP 2: Fit model strictly on train_k using selected features
  formula_k <- as.formula(paste("HomeWin ~", paste(top_features, collapse = " + ")))
  fit_leak_free <- glm(formula_k, family = binomial, data = train_k) # It dynamically constructs the model formula for each fold, based solely on the variables identified at that point.
  
  # STEP 3: Predict on left-out fold test_k
  preds_k <- ifelse(predict(fit_leak_free, newdata = test_k, type = "response") > 0.5, 1, 0)
  cv_errors_right_way[k] <- mean(preds_k != test_k$HomeWin) # Evaluate the predictions of the model selected in this way on the test fold `test_k`, which was kept separate from both variable selection and training.
}

leak_free_cv_error <- mean(cv_errors_right_way)

cat("\n================ 4. LEAK-FREE CROSS-VALIDATION ================\n")
cat(sprintf("Leak-Free 10-Fold CV Error Rate: %.4f (Accuracy: %.2f%%)\n", 
            leak_free_cv_error, (1 - leak_free_cv_error) * 100))
# ====================== ANALYSIS OF THE RESULTS ===============================
# In the leak-free implementation, feature selection (screening the top correlated shot variables) was executed strictly inside each training fold. When feature selection is performed outside CV (or when all features are used), validation fold information subtly influences model structure


# 5. Summary comparison table
# Organize the results into a single summary data frame to directly compare the error estimates obtained using each theoretical approach, along with their respective uncertainty (SE^).
model_eval_summary <- data.frame(
  Method = c("Training Error (Full Fit)",
             "Validation Set (Seed 101)",
             "Validation Set (Seed 202)",
             "10-Fold CV (Logistic)",
             "10-Fold CV (Leak-Free)",
             "LOOCV Shortcut (OLS Goal MSE)"),
  Metric_Type = c("Classification Error", "Classification Error", "Classification Error",
                  "Classification Error", "Classification Error", "Regression MSE"),
  Estimate = c(
    mean(ifelse(predict(glm(HomeWin ~ HST + AST + HS + AS, family = binomial, data = master_df), type = "response") > 0.5, 1, 0) != master_df$HomeWin),
    val_err_seed1,
    val_err_seed2,
    cv10_logit_mean,
    leak_free_cv_error,
    loocv_mse_shortcut
  ),
  Std_Error = c(NA, NA, NA, cv10_logit_se, sd(cv_errors_right_way)/sqrt(k_folds), NA)
)

cat("\n================ MODEL EVALUATION SUMMARY TABLE ================\n")
print(model_eval_summary, digits = 4)
# ====================== ANALYSIS OF THE RESULTS ===============================
# Comparing Method 4 (10-Fold CV) and Method 5 (Leak-Free CV) proves that the model maintains a reliable 70.55% - 70.78% out-of-sample classification accuracy on Serie A match outcomes. Reporting Method 5 demonstrates rigorous methodology by explicitly guarding (previene) against data leakage (contaminazione dei dati). 

# ==============================================================================
# SECTION 8: THE BOOTSTRAP (PARAMETER UNCERTAINTY & CONFIDENCE INTERVALS) *
# ==============================================================================
# In this section, we apply the non-parametric Bootstrap to quantify the uncertainty of our logistic regression coefficients without parametric assumptions. We resample our Serie A dataset B=1,000 times with replacement to compute empirical Bootstrap Standard Errors (SE_B) and 95% Percentile Confidence Intervals. Finally, we compare these robust estimates against traditional parametric standard errors.

suppressPackageStartupMessages({
  library(boot)       # For boot() and boot.ci() functions
})

set.seed(42)

# 1. Define statistic function for boostrap resampling
# It defines the function required by the R engine to calculate the timings at each iteration. It takes two arguments: the original dataset (data) and a vector of random indices (indices).
boot_logit_fn <- function(data, indices) {
  # Create a data subset `d_sub` by selecting the rows corresponding to the indices generated by the bootstrap.
  d_sub <- data[indices, ]
  
  # Fit logistic regression model on the bootstrap dataset
  fit <- glm(HomeWin ~ HST + AST + HS + AS, family = binomial, data = d_sub)
  
  # Return vector of estimated coefficients (beta_0, beta_1, ..., beta_p)
  return(coef(fit))
}

# 2.Run non-parametric boostrap with B = 1000 resamples. In line with the literature, B = 1,000 resamples are sufficient for the empirical standard deviation of the estimates to converge to the actual Standard Error of the parameter.
B_resamples  <- 1000
# It executes the function `boot_logit_fn` 1,000 times on `master_df`, storing the 1,000 estimates for each of the 5 parameters in a 1,000 x 5 matrix within `boot_results$t`.
boot_results <- boot(data = master_df, statistic = boot_logit_fn, R = B_resamples)

cat("\n================ BOOTSTRAP RESAMPLING SUMMARY ================\n")
print(boot_results)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Original: The coefficient estimates β^ obtained from fitting the logistic regression model on the full N = 2,156 dataset. 
#  t1 = intercept ; t2 = HST; t3 = AST; t4 = HS; t5 = AS
# - Bias: The difference between the average bootstrap coefficient across B = 1,000 resamples and the original full-sample estimate. All bias values are virtually zero. This confirms that the maximum likelihood estimator for our Serie A logistic model is completely unbiased.
# - Std. Error: The empirical standard error SE_B calculated as the standard deviation of the 1,000 bootstrap estimates:


# 3. Compare parametric VS boostrap standard errors
# Fit original model on the complete dataset
fit_full_logit <- glm(HomeWin ~ HST + AST + HS + AS, family = binomial, data = master_df)

# Parametric Standard Errors from standard asymptotic theory
parametric_se <- summary(fit_full_logit)$coefficients[, "Std. Error"]

# Empirical Bootstrap Standard Errors: SD (standard deviation) of coefficients across B=1000 resamples. Formally calculate the Bootstrap Standard Error
bootstrap_se  <- apply(boot_results$t, 2, sd)

# Combine the coefficients, point estimates, parametric SE, and bootstrap SE into an explanatory table, adding the absolute difference between the two.
se_comparison <- data.frame(
  Coefficient   = names(coef(fit_full_logit)),
  Estimate      = coef(fit_full_logit),
  Parametric_SE = parametric_se,
  Bootstrap_SE  = bootstrap_se,
  Abs_Diff      = abs(bootstrap_se - parametric_se)
)

cat("\n================ PARAMETRIC VS BOOTSTRAP SE COMPARISON ================\n")
print(se_comparison, digits = 4)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - The absolute differences (Abs_Diff) between asymptotic parametric standard errors and non-parametric bootstrap standard errors are negligible. Parametric standard errors assume asymptotic normality and strict distributional assumptions. Because the non-parametric bootstrap SEs (which make zero distributional assumptions) align almost perfectly with the parametric SEs, the standard theoretical assumptions hold true in your dataset and your parameter inferences are completely uncorrupted by heavy tails or non-normality.   


# 4. Compute 95% percentile confidence intervals
# Extracts the 2.5th and 97.5th percentiles from the bootstrap distributions
ci_hst <- boot.ci(boot_results, type = "perc", index = 2) # Index 2 = HST (Home Shots on Target)
ci_ast <- boot.ci(boot_results, type = "perc", index = 3) # Index 3 = AST (Away Shots on Target)

cat("\n================ 95% BOOTSTRAP PERCENTILE CONFIDENCE INTERVALS ================\n")
cat("Home Shots on Target (HST) 95% Percentile CI:\n")
print(ci_hst$percent[4:5])

cat("\nAway Shots on Target (AST) 95% Percentile CI:\n")
print(ci_ast$percent[4:5])
# ====================== ANALYSIS OF THE RESULTS ===============================
# The 95% non-parametric confidence interval for HST is derived directly from the 2.5th and 97.5th percentiles of the 1,000 bootstrap iterations. Because neither interval contains zero, both HST and AST are statistically significant at the α = 0.05 level.  We are 95% confident that for every additional shot on target by the home team, the log-odds of a home win increase by between +0.3265 and +0.4253. Converting to odds ratios (e^{0.3265} = 1.386 to e^{0.4253} = 1.530), each home shot on target increases the odds of winning by 38.6% to 53.0%.


# 5. Visualize boostrap distribution (diagnostic plots)
# Generate histogram and Q-Q plot for Home Shots on Target (HST) coefficient
plot(boot_results, index = 2)
# ====================== ANALYSIS OF THE RESULTS ===============================
# Left Panel (Histogram of t*): Displays the empirical distribution of β^_HST across all 1,000 bootstrap resamples. It forms a smooth, unimodal, symmetric bell curve centered at 0.373.
# Right Panel (Normal Q-Q Plot): Compares the quantiles of the bootstrap distribution against theoretical standard normal quantiles. The data points adhere almost perfectly to the straight dashed diagonal line from -3 to +3 standard deviations. 
# This confirms that the sampling distribution of the HST coefficient is Gaussian normal, validating standard $z$-tests and Wald confidence intervals for your Serie A model.

# ==============================================================================
# SECTION 9: ADVANCED INFERENCE (PRE-VALIDATION & PERMUTATION TESTING) 
# ==============================================================================
# In this section, we implement advanced resampling techniques for fair model
# comparison and non-parametric hypothesis testing. First, we use Pre-validation
# to construct an out-of-fold adaptive predictor (z_tilde) that prevents coefficient inflation when evaluated alongside (insieme) fixed clinical/match features. Second, we run a Permutation Test by shuffling target labels (etichette taget) under the Null Hypothesis to calculate an exact non-parametric p-value for shot metrics.

suppressPackageStartupMessages({
  library(caret)      # For createFolds
})

set.seed(42)

# 1. Pre-validation methodology
# Constructing an unbiased out-of-fold adaptive predictor (z_tilde)
k_folds <- 10
folds   <- createFolds(master_df$HomeWin, k = k_folds, list = TRUE)#  Divides all 2,156 matches into 10 stratified folds, ensuring each fold contains the same proportion of home wins (= 43%) as the full dataset. Pre-validation requires partitioning the observations into $K$ equal parts before building the adaptive predictor.

# Initialize vector to store out-of-fold pre-validated predictions
master_df$z_tilde <- NA_real_

# Step A: Generate pre-validated predictor z_tilde out-of-fold
for (k in 1:k_folds) {
  # For fold k, isolates the 90% training indices and the 10% held-out(riservato) test indices.
  train_k_idx <- -folds[[k]]
  test_k_idx  <- folds[[k]]
  
  # Fit adaptive model ONLY on training fold (using shot performance metrics)
  fit_adaptive_k <- glm(HomeWin ~ HST + AST + HS + AS, family = binomial, data = master_df[train_k_idx, ])
  
  # Computes predicted log-odds only for matches in the held-out 10% validation fold.
  master_df$z_tilde[test_k_idx] <- predict(fit_adaptive_k, newdata = master_df[test_k_idx, ], type = "link")
}

# Step B: Construct naïve (re-used) adaptive predictor fit on ALL data
fit_naive_adaptive <- glm(HomeWin ~ HST + AST + HS + AS, family = binomial, data = master_df) # Fits the adaptive model on all 2,156 matches at once and generates in-sample log-odds predictions.
master_df$z_naive   <- predict(fit_naive_adaptive, newdata = master_df, type = "link")

# Step C: Fit Joint Logistic Regression comparing Adaptive Predictor vs Fixed Features (Corners)
fit_reused_model      <- glm(HomeWin ~ z_naive + HC + AC, family = binomial, data = master_df)# Fits a joint logistic regression combining the naïve score z_naive with static match predictors.
fit_prevalidated_model <- glm(HomeWin ~ z_tilde + HC + AC, family = binomial, data = master_df)# Fits a joint logistic regression combining the pre-validated score z_tilde with static match predictors.

cat("\n================ 1A. NAÏVE RE-USE JOINT MODEL SUMMARY ================\n")
print(summary(fit_reused_model)$coefficients)

cat("\n================ 1B. PRE-VALIDATED JOINT MODEL SUMMARY ================\n")
print(summary(fit_prevalidated_model)$coefficients)
# ====================== ANALYSIS OF THE RESULTS ===============================
# In Model 1A, z_naive was generated using a model trained on all N = 2,156$ match outcomes. As a result, its coefficient was slightly inflated (β^ = 1.0916)
# In Model 1B, using the out-of-fold predictor z_tilde adjusts the coefficient down to its true out-of-sample level (β^ = 1.0775).
# In genomics or small datasets naïve re-use leads to severe over-optimism bias. Because our Serie A dataset is large relative to the number of shot features the in-sample overfitting of log-odds is modest, but Pre-validation successfully isolates and removes that upward bias.By replacing z_{naive} with z_tilde, we ensure that baseline match metrics—Home Corners and Away Corners are evaluated against a fair, leak-free (priva di fughe di informazioni) summary metric rather than one artificially fitted to the outcome labels.


# 2. Permutation testing VS bootstrap
# Sampling from the NULL distribution (H0) by shuffling match outcome labels
# Step A: Calculate observed test statistic (HST coefficient on original data)
fit_obs  <- glm(HomeWin ~ HST, family = binomial, data = master_df)# Fits a simple logistic regression predicting Home Win using Home Shots on Target (HST) on actual Serie A match outcomes.
obs_stat <- coef(fit_obs)["HST"]# Extracts the observed coefficient β^_HST = 0.3730.

# Step B: Run Permutation Loop under H0 (shuffling target outcome)
B_perm     <- 1000
perm_stats <- numeric(B_perm)

for (b in 1:B_perm) {
  # Shuffle HomeWin labels, without replacement, to break any true relationship with HST
  shuffled_target <- sample(master_df$HomeWin)
  
  # Fit model under Null Hypothesis
  fit_perm      <- glm(shuffled_target ~ master_df$HST, family = binomial)# Fits the logistic model using the permuted target labels against the original, unchanged HST predictor values.
  perm_stats[b] <- coef(fit_perm)["master_df$HST"]# Stores the estimated slope from the b-th permuted dataset
}

# Step C: Compute empirical two-sided p-value
perm_p_value <- mean(abs(perm_stats) >= abs(obs_stat))#  Checks how many permuted coefficients under H_0 are as large or larger in magnitude than our true observed slope and Computes the proportion of times $H_0$ generated a result as extreme as the real data, yielding the two-tailed empirical p-value.
parametric_p <- summary(fit_obs)$coefficients["HST", "Pr(>|z|)"]# Extracts the classical Wald test $p$-value based on asymptotic normal distribution theory.

cat("\n================ 2. PERMUTATION TEST RESULTS ================\n")
cat(sprintf("Observed HST Coefficient (beta_hat): %.4f\n", obs_stat))
cat(sprintf("Parametric Wald Test p-value:         %.5e\n", parametric_p))
cat(sprintf("Permutation Test Empirical p-value:  %.5f (out of %d permutations)\n", 
            perm_p_value, B_perm))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Observed Effect (β^_obs = 0.2877): In the unshuffled data, every additional Home Shot on Target increases the log-odds of winning by +0.2877.
# - Empirical Null Distribution (H_0): In B = 1,000 iterations, match outcomes (Y) were randomly shuffled without replacement, breaking any true relationship between shots and wins to enforce the Null Hypothesis.
# - The parametric Wald test assumes asymptotic Gaussian distributions. The Permutation Test confirms this result non-parametrically, proving beyond a doubt that the link between shots on target and match victory is statistically genuine and not an artifact of distributional assumptions.

# ==============================================================================
# SECTION 10: SUBSET SELECTION & INFORMATION CRITERIA
# ==============================================================================
# In this section, we implement Best Subset, Forward Stepwise, and Backward Stepwise selection to identify optimal predictor subsets for goal modeling. The Best Subset Selection algorithm guarantees finding the globally optimal combination among all 2^p possible options. However, since adding variables always reduces training bias (the RSS decreases and R^2 increases monotonically), the selection of the final model cannot be based on RSS or R^2; instead, it requires criteria involving penalization or out-of-sample validation. We evaluate models using indirect analytical criteria (Cp, AIC, BIC, Adjusted R^2) as well as direct 10-Fold Cross-Validation error. Finally, we apply the 1-Standard-Error rule to select the most parsimonious predictive model.

suppressPackageStartupMessages({
  library(leaps)      # For regsubsets() exhaustive, forward, and backward search
  library(caret)      # For stratified fold generation
})

set.seed(42)

# 1. Best subset selection 
# Fit best subset regression across all 2^p predictor combinations
# The regsubsets() function from the leaps package trains and examines all possible combinations of predictors for each model size. With p = 6 variables, evaluate 2^6 = 64 distinct models and select the best model M_k for each dimension k based on the minimum Residual Sum of Squares or the maximum R^2 value.
regfit_full <- regsubsets(TotalGoals ~ HST + AST + HS + AS + HC + AC, 
                          data = master_df, 
                          nvmax = 6, 
                          method = "exhaustive")

reg_summary <- summary(regfit_full)

# Identify optimal model size according to analytical criteria
best_cp_idx    <- which.min(reg_summary$cp) # C_p di Mallows / AIC it adds a penalty proportional to the number of parameters d to compensate for the optimism of the training error.
best_bic_idx   <- which.min(reg_summary$bic) # BIC - Bayesian Information Criterion. The BIC imposes a much harsher penalty than $C_p$ on large model sizes, tending to select more parsimonious and compact models.
best_adjr2_idx <- which.max(reg_summary$adjr2)# Adjusted R^2. Unlike the standard R^2, it penalizes the inclusion of irrelevant variables by dividing the RSS by the residual degrees of freedom n-d-1.

cat("\n================ 1. BEST SUBSET SELECTION SUMMARY ================\n")
cat(sprintf("Optimal number of predictors by Cp:          %d\n", best_cp_idx))
cat(sprintf("Optimal number of predictors by BIC:         %d\n", best_bic_idx))
cat(sprintf("Optimal number of predictors by Adjusted R2: %d\n", best_adjr2_idx))

# 2. Stepside selection (Forward & Backward)
# It re-performs the selection of predictors, restricting the search to sequential heuristics: Forward (starting from an empty model M_0 and adding one predictor at a time) and Backward (starting from the full model M_p and removing the least useful(meno utile) variable at each step). Instead of exploring 2^p models, stepwise procedures evaluate only 1 + frac{p(p+1)}{2} models.
regfit_fwd <- regsubsets(TotalGoals ~ HST + AST + HS + AS + HC + AC, 
                         data = master_df, 
                         nvmax = 6, 
                         method = "forward")

regfit_bwd <- regsubsets(TotalGoals ~ HST + AST + HS + AS + HC + AC, 
                         data = master_df, 
                         nvmax = 6, 
                         method = "backward")

fwd_summary <- summary(regfit_fwd)
bwd_summary <- summary(regfit_bwd)

cat("\n================ 2. STEPWISE SELECTION COMPARISON ================\n")
cat("Best 3-variable model (Exhaustive): ", names(which(reg_summary$which[3, ])[-1]), "\n")
cat("Best 3-variable model (Forward):    ", names(which(fwd_summary$which[3, ])[-1]), "\n")
cat("Best 3-variable model (Backward):   ", names(which(bwd_summary$which[3, ])[-1]), "\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# At k = 3, Best Subset (Exhaustive), Forward Stepwise, and Backward Stepwise selection arrive at the exact same subset of variables: Home Shots on Target (HST), Away Shots on Target (AST), and Home Corners (HC).
# Forward and Backward stepwise algorithms search a restricted model space ($1 + \frac{p(p+1)}{2} = 22$ models) compared to Best Subset ($2^6 = 64$ models) and are not theoretically guaranteed to find the global optimum. The complete alignment across all three methods proves that your predictor space contains no greedy path dependencies or local optima traps.

# 3. Direct Test Error estimation: 10-FOLD CROSS-VALIDATION
# Helper function to compute predictions from a regsubsets object
predict.regsubsets <- function(object, newdata, id, ...) {
  form  <- as.formula(object$call[[2]])
  mat   <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  return(mat[, xvars] %*% coefi)
}

k_folds <- 10
folds   <- createFolds(master_df$TotalGoals, k = k_folds, list = TRUE)
p_vars  <- 6

cv_errors <- matrix(NA, nrow = k_folds, ncol = p_vars,
                    dimnames = list(NULL, paste0("size_", 1:p_vars)))

# Variable selection is performed exclusively within the training data of each fold (master_df[-folds[[k]], ]) to prevent data leakage, with the out-of-sample MSE calculated for each dimension k ∈ {1,..., 6}.
for (k in 1:k_folds) {
  # Fit best subset strictly inside training folds (prevent data leakage)
  best_fit <- regsubsets(TotalGoals ~ HST + AST + HS + AS + HC + AC, 
                         data = master_df[-folds[[k]], ], 
                         nvmax = p_vars, 
                         method = "exhaustive")
  
  for (i in 1:p_vars) {
    pred <- predict(best_fit, master_df[folds[[k]], ], id = i)
    cv_errors[k, i] <- mean((master_df$TotalGoals[folds[[k]]] - pred)^2)
  }
}

mean_cv_errors <- apply(cv_errors, 2, mean)
sd_cv_errors   <- apply(cv_errors, 2, sd)
se_cv_errors   <- sd_cv_errors / sqrt(k_folds)

best_cv_k    <- which.min(mean_cv_errors)
min_cv_error <- mean_cv_errors[best_cv_k]

# Apply the 1-Standard-Error Rule. Calculate the standard error of the MSE test estimate across the K folds, identify the minimum test error (min_cv_error), and select the smallest model whose error falls within 1 standard error of that minimum.
one_se_threshold <- min_cv_error + se_cv_errors[best_cv_k]
k_1se            <- min(which(mean_cv_errors <= one_se_threshold))

cat("\n================ 3. 10-FOLD CV & 1-SE RULE ================\n")
cat(sprintf("Minimum CV MSE Model Size:    k = %d (MSE = %.4f)\n", best_cv_k, min_cv_error))
cat(sprintf("1-SE Rule Model Size:         k = %d (Threshold = %.4f)\n", k_1se, one_se_threshold))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Min CV MSE (k = 4): The 10-Fold Cross-Validation error achieves its global minimum at k = 4 (MSE = 2.0363 goals^2). This matches the analytical choices of C_p and BIC, confirming that shots on target (HST, AST) and corners (HC, AC) provide the strongest out-of-sample predictive power for total match goals. 
# - By calculating the standard error of the CV MSE across folds, the 1-SE threshold is set to 2.1201. The simplest model whose test MSE falls below this threshold is k = 2 (HST, AST). 
# Direct Cross-Validation measures test error without requiring estimates of error variance σ^2 or asymptotic distribution assumptions. The 1-Standard-Error Rule selects the most parsimonious model (k = 2) whose predictive accuracy is statistically indistinguishable from the minimum MSE model (k = 4), effectively favoring lower model variance over marginal reductions in bias.  

# 4. Summary comparison tabel
subset_summary_table <- data.frame(
  Criterion = c("Cp", "BIC", "Adjusted R2", "10-Fold CV Min MSE", "10-Fold CV 1-SE Rule"),
  Optimal_k = c(best_cp_idx, best_bic_idx, best_adjr2_idx, best_cv_k, k_1se),
  Selected_Variables = c(
    paste(names(which(reg_summary$which[best_cp_idx, ])[-1]), collapse = ", "),
    paste(names(which(reg_summary$which[best_bic_idx, ])[-1]), collapse = ", "),
    paste(names(which(reg_summary$which[best_adjr2_idx, ])[-1]), collapse = ", "),
    paste(names(which(reg_summary$which[best_cv_k, ])[-1]), collapse = ", "),
    paste(names(which(reg_summary$which[k_1se, ])[-1]), collapse = ", ")
  )
)

cat("\n================ SUBSET SELECTION COMPARISON TABLE ================\n")
print(subset_summary_table)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Cp and BIC Convergence (k = 4): Both Cp and BIC select the exact same 4-variable model containing Home/Away Shots on Target (HST, AST) and Home/Away Corners (HC, AC).
# - Adjusted R^2 Expansion (k = 5): Adjusted $R^2$ adds Away Shots (AS) to select a 5-variable model.
# BIC applies a penalty proportional to log(n) * d. With N = 2,156 Serie A matches, log(2156) = 7.68, which penalizes additional parameters much more heavily than Cp (2d) or Adjusted R^2. Adjusted R^2 has a softer penalty denominator (n - d - 1), causing it to consistently select larger models than BIC. 
# ==============================================================================
# SECTION 11: SHRINKAGE METHODS (RIDGE REGRESSION & LASSO)
# ==============================================================================
# In this section, we implement Ridge Regression (L2 penalty) and the Lasso (L1 penalty) using the glmnet package. We standardise predictors automatically, fit models across a grid of lambda values, tune lambda via 10-Fold Cross-Validation, and extract optimal models using both lambda.min and lambda.1se rules.

suppressPackageStartupMessages({
  library(glmnet)     # For Ridge (alpha=0) and Lasso (alpha=1) fitting
  library(caret)      # For dataset manipulation helpers
})

set.seed(42)

# 1. Prepare matrix inputs & predictor standardization
# glmnet requires a numeric predictor matrix X (excluding intercept) and target vector y. Unlike ordinary least squares (OLS), which are scale-equivariant, Ridge and Lasso estimates are sensitive to the scale of the variables.
X <- model.matrix(TotalGoals ~ HST + AST + HS + AS + HC + AC, data = master_df)[, -1]
y <- master_df$TotalGoals

# Grid of 100 lambda values ranging from 10^3 (heavy shrinkage) to 10^-2 (near OLS)
lambda_grid <- 10^seq(3, -2, length = 100)

# 2. Ridge regression (L2 PENALTY, alpha = 0)
# Fit Ridge Regression path across grid (standardize = TRUE standardizes X)
ridge_fit <- glmnet(X, y, alpha = 0, lambda = lambda_grid, standardize = TRUE)

# Perform 10-Fold Cross-Validation for Ridge. The cv.glmnet function performs 10-fold cross-validation to determine the optimal level of shrinkage. Ridge coincides with OLS. As λ increases, the coefficients are shrunk (vengono contratti) toward zero, drastically reducing the model's variance at the cost of a modest increase in bias. No coefficient is ever set exactly to zero.
cv_ridge <- cv.glmnet(X, y, alpha = 0, lambda = lambda_grid, nfolds = 10, standardize = TRUE)

lambda_min_ridge <- cv_ridge$lambda.min
lambda_1se_ridge <- cv_ridge$lambda.1se

cat("\n================ 1. RIDGE REGRESSION CV RESULTS ================\n")
cat(sprintf("Optimal Lambda (min MSE): %.4f (CV MSE = %.4f)\n", 
            lambda_min_ridge, min(cv_ridge$cvm)))
cat(sprintf("1-SE Rule Lambda:        %.4f (CV MSE = %.4f)\n", 
            lambda_1se_ridge, cv_ridge$cvm[cv_ridge$lambda == lambda_1se_ridge]))

# 3. The Lasso (L1 PENALTY, alpha = 1)
# Fit Lasso Regression path across grid (performs shrinkage + feature selection Setting alpha = 1 results in Lasso regression, a method whose penalty forces certain coefficients to become exactly zero as λ increases.
lasso_fit <- glmnet(X, y, alpha = 1, lambda = lambda_grid, standardize = TRUE)

# Perform 10-Fold Cross-Validation for Lasso
cv_lasso <- cv.glmnet(X, y, alpha = 1, lambda = lambda_grid, nfolds = 10, standardize = TRUE)

# It extracts two key values of λ identified by 10-fold cross-validation: 
# - `lambda.min`: the value of λ that minimizes the cross-validation mean squared error (MSE)
# — `lambda.1se`: the value of λ that imposes a greater penalty (resulting in a simpler/sparser model) and whose MSE falls within one standard error of the minimum.
lambda_min_lasso <- cv_lasso$lambda.min
lambda_1se_lasso <- cv_lasso$lambda.1se

cat("\n================ 2. THE LASSO CV RESULTS ================\n")
cat(sprintf("Optimal Lambda (min MSE): %.4f (CV MSE = %.4f)\n", 
            lambda_min_lasso, min(cv_lasso$cvm)))
cat(sprintf("1-SE Rule Lambda:        %.4f (CV MSE = %.4f)\n", 
            lambda_1se_lasso, cv_lasso$cvm[cv_lasso$lambda == lambda_1se_lasso]))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Optimal Shrinkage (λ_min): For both Ridge (λ = 0.0142) and Lasso (λ = 0.0159), the tuning parameter that minimizes 10-Fold Cross-Validation MSE is very close to zero. This indicates that the unconstrained OLS model is already near-optimal, but introducing a small amount of regularization reduces model variance without introducing significant bias.
# - Parsimonious Boundary (λ_1se): The 1-Standard-Error rule increases the penalty to λ = 0.5857 for Ridge and λ = 0.1024 for Lasso. Both models maintain out-of-sample predictive accuracy (MSE = 2.11) within one standard error of their respective minimums while significantly reducing model complexity.


# 4. Coefficient comparison table (OLS VS RIDGE VS LASSO)
# Baseline OLS model fit for comparison
ols_fit <- lm(TotalGoals ~ HST + AST + HS + AS + HC + AC, data = master_df)

# Extract estimated coefficient vectors at optimal lambda values
coef_ols        <- coef(ols_fit)
coef_ridge_min  <- as.vector(coef(cv_ridge, s = "lambda.min"))
coef_ridge_1se  <- as.vector(coef(cv_ridge, s = "lambda.1se"))
coef_lasso_min  <- as.vector(coef(cv_lasso, s = "lambda.min"))
coef_lasso_1se  <- as.vector(coef(cv_lasso, s = "lambda.1se"))

shrinkage_comp_table <- data.frame(
  Feature     = names(coef_ols),
  OLS         = round(coef_ols, 4),
  Ridge_min   = round(coef_ridge_min, 4),
  Ridge_1se   = round(coef_ridge_1se, 4),
  Lasso_min   = round(coef_lasso_min, 4),
  Lasso_1se   = round(coef_lasso_1se, 4)
)

cat("\n================ SHRINKAGE COEFFICIENTS COMPARISON ================\n")
print(shrinkage_comp_table)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Lasso Sparsity & Automatic Feature Selection (L_1 Penalty): At λ_min, Lasso sets HS = 0.0000. Home Shots on Target (HST) already captures the scoring mechanism, making total Home Shots (HS) redundant noise.At λ_se, Lasso sets both HS = 0.0000 and AS = 0.0000. This produces a 4-variable sparse model (HST, AST, HC, AC). This matches the subset selection results from C_p, BIC, and minimum 10-Fold CV MSE in section 9.
# - Ridge Continuous Shrinkage (L_2 Penalty): Ridge regression contracts all coefficients toward zero proportionally but never sets any coefficient exactly to zero. Under Ridge_1se, HS (0.0081) and AS (0.0181) remain in the model despite their minor contribution.
# - Parameter Contraction: Moving from OLS to λ_1se the coefficient for HST shrinks from 0.2529 to 0.1651 (Ridge) and 0.1913 (Lasso), illustrating how regularization controls coefficient variance when predictors are correlated.

# 5. Diagnostic plots 
par(mfrow = c(2, 2))

# Plot 1: Ridge Coefficient Path as function of log(lambda)
plot(ridge_fit, xvar = "lambda", label = TRUE, main = "Ridge Coefficient Path (L2)")
abline(v = log(c(lambda_min_ridge, lambda_1se_ridge)), lty = c(2, 3), col = c("red", "blue"))

# Plot 2: Lasso Coefficient Path as function of log(lambda)
plot(lasso_fit, xvar = "lambda", label = TRUE, main = "Lasso Coefficient Path (L1)")
abline(v = log(c(lambda_min_lasso, lambda_1se_lasso)), lty = c(2, 3), col = c("red", "blue"))

# Plot 3: Ridge 10-Fold CV Curve
plot(cv_ridge, main = "Ridge 10-Fold CV Error")

# Plot 4: Lasso 10-Fold CV Curve
plot(cv_lasso, main = "Lasso 10-Fold CV Error")

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# Top Panels (Coefficient Paths):
# - Ridge (Top-Left): As -log(λ) decreases (moving left, increasing penalty λ), coefficient curves decay smoothly toward zero as asymptotic asymptotes.
# - Lasso (Top-Right): As λ increases, coefficient paths hit the zero axis at distinct thresholds, demonstrating the geometric corner-intersection property of the L_1 constraint.
# Bottom Panels (10-Fold CV Curves):
# - Ridge & Lasso CV Plots: Both curves display a flat, low-MSE region at small λ values (right side of plots), followed by a steep increase in MSE as λ grows excessively large (left side of plots).The vertical dashed lines mark log(λ_min) (first line) and log(λ_1se) (second line), establishing the bounds for model selection.

# ==============================================================================
# SECTION 12: DIMENSION REDUCTION (PCR & PLS)
# ==============================================================================
# In this section, we implement Principal Components Regression (PCR) and Partial Least Squares (PLS). We evaluate how unsupervised linear projections (PCR) compare against supervised projections (PLS) that account for response correlation, tuning the optimal number of components M via 10-Fold Cross-Validation.

suppressPackageStartupMessages({
  library(pls)        # For pcr() and plsr() modeling
})

set.seed(42)

# 1. Principal components regression (PCR - UNSUPERVISED)
# Fit PCR with 10-Fold Cross-Validation (scale = TRUE standardizes features to have mean 0 and standard deviation 1)
pcr_fit <- pcr(TotalGoals ~ HST + AST + HS + AS + HC + AC, 
               data = master_df, 
               scale = TRUE, 
               validation = "CV", 
               segments = 10)

pcr_cv_mse <- MSEP(pcr_fit)$val[1, 1, ]  # Extract CV MSE for each component M

# Identify optimal M for PCR
best_m_pcr <- which.min(pcr_cv_mse) - 1   # Subtract 1 to account for M=0 (intercept-only)

cat("\n================ 1. PCR CROSS-VALIDATION RESULTS ================\n")
cat(sprintf("Optimal Number of Components (PCR): M = %d (CV MSE = %.4f)\n", 
            best_m_pcr, min(pcr_cv_mse)))

# 2. Partial Least Squares (PLS - SUPERVISED)
# Fit PLS with 10-Fold Cross-Validation (uses Y to guide direction selection)
pls_fit <- plsr(TotalGoals ~ HST + AST + HS + AS + HC + AC, 
                data = master_df, 
                scale = TRUE, 
                validation = "CV", 
                segments = 10)

pls_cv_mse <- MSEP(pls_fit)$val[1, 1, ]  # Extract CV MSE for each component M

# Identify optimal M for PLS
best_m_pls <- which.min(pls_cv_mse) - 1   # Subtract 1 to account for M=0

cat("\n================ 2. PLS CROSS-VALIDATION RESULTS ================\n")
cat(sprintf("Optimal Number of Components (PLS): M = %d (CV MSE = %.4f)\n", 
            best_m_pls, min(pls_cv_mse)))

# 3. Global model comparison table (OLS vs LASSO vs PCR vs PLS)
# Variance explained summary
pcr_expl_x <- R2(pcr_fit, estimate = "train")$val[1, 1, best_m_pcr + 1] * 100
pls_expl_x <- R2(pls_fit, estimate = "train")$val[1, 1, best_m_pls + 1] * 100

# Assembles a comparison table containing method names, optimal $M$, minimum CV MSE, and percentage of X-variance explained.
dim_red_summary <- data.frame(
  Method = c("PCR (Unsupervised)", "PLS (Supervised)"),
  Optimal_Components_M = c(best_m_pcr, best_m_pls),
  CV_MSE = c(round(min(pcr_cv_mse), 4), round(min(pls_cv_mse), 4)),
  Pct_Variance_Explained_X = c(paste0(round(pcr_expl_x, 2), "%"), 
                               paste0(round(pls_expl_x, 2), "%"))
)
# ====================== ANALYSIS OF THE RESULTS ===============================
# Supervised Efficiency of PLS (M = 4 vs M = 5): Partial Least Squares (PLS) achieves virtually the exact same test error (MSE = 2.038) as Principal Components Regression (PCR) using one fewer component (M = 4 for PLS vs M = 5 for PCR).   
# - PCR is Unsupervised: It constructs linear combinations Z_m purely to maximize the variance of the predictor matrix X without looking at Y (TotalGoals). As a result, PCR can waste components explaining variations in X that have weak or no correlation with scoring goals.  
# - PLS is Supervised: It uses Y to guide the direction of Z_m, placing higher weights on predictors that are strongly correlated with TotalGoals. By supervising the projection, PLS compresses the predictive signal into fewer dimensions.   


cat("\n================ DIMENSION REDUCTION COMPARISON ================\n")
print(dim_red_summary)
# ====================== ANALYSIS OF THE RESULTS ===============================
#  Both models capture roughly 26% of the total variation in the 6 predictor variables. In football match data, predictors like corners (HC, AC) and total shots (HS, AS) contain high independent noise. The fact that both models reach minimal CV MSE by explaining only 26% of X proves that the remaining 74% of predictor variance is uninformative noise regarding goal scoring.


# 4. Diagnostic plots 
par(mfrow = c(1, 2))# Configures the base R graphics layout to display 1 row and 2 columns for side-by-side plot comparison.

# Plot 1: PCR Validation Plot (MSE vs Components M)
validationplot(pcr_fit, val.type = "MSEP", main = "PCR 10-Fold CV Error", legendpos = "topright")
abline(v = best_m_pcr, lty = 2, col = "red")

# Plot 2: PLS Validation Plot (MSE vs Components M)
validationplot(pls_fit, val.type = "MSEP", main = "PLS 10-Fold CV Error", legendpos = "topright")
abline(v = best_m_pls, lty = 2, col = "blue")

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Left Panel (PCR 10-Fold CV Curve):The curve exhibits a step-like decrease in MSE as M increases. Because PCR directions are ordered strictly by X-variance rather than Y-correlation, adding the 3rd component provides little benefit, while the 4th and 5th components significantly drop the test error.   
# - Right Panel (PLS 10-Fold CV Curve):The curve drops sharply at M = 1 and flattens out almost immediately by M = 2. Because the first PLS component is specifically constructed to align with Y, it captures the vast majority of predictive power right away.

# ==============================================================================
# SECTION 13: POLYNOMIAL REGRESSION & STEP FUNCTIONS
# ==============================================================================
# This module explores non-linear extensions of linear regression by modeling match TotalGoals as a function of HST (Home Shots on Target). It systematically evaluates orthogonal polynomial regressions across degrees d ∈ [1, 4] using 10-Fold Cross-Validation to select the optimal model complexity. To validate whether higher-order terms provide statistically significant improvements, sequential ANOVA F-tests are executed. Furthermore, a piecewise constant step function is fitted by binning the continuous predictor into discrete impact zones, accompanied by diagnostic confidence band plots to illustrate non-linear goal dynamics.
suppressPackageStartupMessages({
  library(boot)       # For cv.glm() polynomial tuning
})

set.seed(42)

# 1. Polynomial Regression tuning (Degrees d = 1 to 4 via 10-Fold CV)
# Pre-allocates a numeric vector of length 4 filled with zeros to store the 10-Fold Cross-Validation Mean Squared Error (MSE) for degrees d = 1, 2, 3, 4.
cv_errors_poly <- numeric(4)

for (d in 1:4) {
  # Fit Generalized Linear Model (Gaussian/OLS) with Orthogonal Polynomials
  glm_poly <- glm(TotalGoals ~ poly(HST, d), data = master_df)
  
  # 10-Fold Cross-Validation Error
  cv_errors_poly[d] <- cv.glm(master_df, glm_poly, K = 10)$delta[1]
}

# Identifies the array index (degree $d$) that yields the lowest cross-validation MSE.
best_d <- which.min(cv_errors_poly)

cat("\n================ 1. POLYNOMIAL REGRESSION (10-FOLD CV) ================\n")
for (d in 1:4) {
  cat(sprintf("Degree d = %d | 10-Fold CV MSE = %.4f %s\n", 
              d, cv_errors_poly[d], ifelse(d == best_d, "<-- OPTIMAL", "")))
}
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Statistical Interpretation: The cross-validation error drops when moving from linear (d = 1) to quadratic (d = 2), remains flat at cubic (d = 3), and achieves its numerical minimum at degree 4 (d = 4).
# - Football & ISLR Insight: Moving beyond linearity improves the error compared to plain OLS (2.4487 to 2.4413). However, high-degree polynomials (d >= 4) carry notorious risks of erratic tail behavior near dataset boundaries (e.g., extreme matches with >15 shots on target).


# 2. Hypothesis testing: ANOVA FOR POLYNOMIAL DEGREES
# Fit sequential models from degree 1 to 4 to perform ANOVA F-Test
fit_1 <- lm(TotalGoals ~ poly(HST, 1), data = master_df)
fit_2 <- lm(TotalGoals ~ poly(HST, 2), data = master_df)
fit_3 <- lm(TotalGoals ~ poly(HST, 3), data = master_df)
fit_4 <- lm(TotalGoals ~ poly(HST, 4), data = master_df)

# Performs a sequential analysis of variance (ANOVA F-test) comparing nested (annidati) models. Tests the null hypothesis H_0: β_d = 0 (that degree d-1 is sufficient) against H_1: β_d =/ 0 (that adding degree d significantly reduces Residual Sum of Squares, RSS).   
anova_res <- anova(fit_1, fit_2, fit_3, fit_4)

cat("\n================ 2. ANOVA F-TEST FOR MODEL COMPLEXITY ================\n")
print(anova_res)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Model 1 vs. Model 2 (p = 0.009014): Strongly significant (p < 0.01). Adding a quadratic term (HST^2) significantly reduces Residual Sum of Squares (RSS from 5267.0 to 5250.3), proving that the relationship between shots on target and goals is non-linear.
# - Model 2 vs. Model 3 (p = 0.100310): Not statistically significant (p > 0.05). Adding a cubic term (HST^3) provides no meaningful improvement over the quadratic model.
# - Model 3 vs. Model 4 (p = 0.015348): Statistically significant (p < 0.05), confirming why d = 4 had the lowest CV error. 
#  In academic presentations, Parsimony (d=2) is preferred over d=4. Degree 2 captures 90% of the non-linear gain without introducing boundary instability or overfitting.


# 3. Step functions (PIECEWISE CONSTANT REGRESSION via cut())
# Cut HST into 4 categorical intervals (e.g., low, mid-low, mid-high, high)
master_df$HST_cut <- cut(master_df$HST, breaks = c(-1, 2, 5, 8, Max = max(master_df$HST)))
fit_step <- lm(TotalGoals ~ HST_cut, data = master_df)# its a piecewise constant step function regression using dummy variable coding.

cat("\n================ 3. STEP FUNCTION COEFFICIENTS ================\n")
print(summary(fit_step)$coefficients)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Baseline Intercept (2.0682): Matches where the home team produces 0 to 2 shots on target average 2.07 total match goals.
# - (2, 5] HST: Adds +0.53 goals (Expected Total = 2.60 goals).
# - (5, 8] HST: Adds +1.01 goals (Expected Total = 3.08 goals).
# - (8, 18] HST: Adds +1.76 goals (Expected Total = 3.83 goals).
# Statistical Significance: All bin dummy variables have p-value < 10^-7, proving that discretizing shots on target creates distinct goal expectation plateaus(livelli).

# 4. Diagnostic visualizations 
par(mfrow = c(1, 2))

# Plot 1: Polynomial Fit vs Raw Data
hst_grid <- seq(min(master_df$HST), max(master_df$HST), by = 0.5)
preds_poly <- predict(fit_2, newdata = list(HST = hst_grid), se = TRUE)

plot(master_df$HST, master_df$TotalGoals, col = "darkgrey", xlab = "Home Shots on Target (HST)",
     ylab = "Total Goals", main = "Polynomial Fit (Degree d = 2)", pch = 16, cex = 0.6)
lines(hst_grid, preds_poly$fit, col = "blue", lwd = 2)
lines(hst_grid, preds_poly$fit + 2*preds_poly$se.fit, col = "blue", lty = 2)
lines(hst_grid, preds_poly$fit - 2*preds_poly$se.fit, col = "blue", lty = 2)

# Plot 2: 10-Fold CV Error by Polynomial Degree
plot(1:4, cv_errors_poly, type = "b", xlab = "Polynomial Degree (d)", 
     ylab = "10-Fold CV MSE", main = "Polynomial Degree Selection", col = "red", pch = 19)
points(best_d, cv_errors_poly[best_d], col = "blue", cex = 2, pch = 1, lwd = 2)

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Left Panel (Quadratic Fit Curve): Displays diminishing marginal returns. Going from 2 to 6 home shots on target sharply increases total match goals, but beyond 12+ shots on target, the curve flattens out.The dashed lines (+- 2 * SE) widen substantially above 13 shots on target because fewer historical matches reach that extreme range, resulting in higher variance.
# - Right Panel (CV MSE Curve):Confirms the plateau at d=2 and d=3, followed by the localized dip at d=4 highlighted by the blue circle.

# ==============================================================================
# SECTION 14: REGRESSION SPLINES & NATURAL CUBIC SPLINES
# ==============================================================================
# This module implements Regression Splines (bs) and Natural Cubic Splines (ns) to model non-linear relationships between Home Shots on Target (HST) and TotalGoals. Knots (nodi) are strategically placed at empirical quantiles (25th, 50th, and 75th percentiles) to ensure balanced local fitting across the predictor domain. Optimal degrees of freedom are evaluated using 10-Fold Cross-Validation to select the ideal model flexibility. Diagnostic visualizations compare prediction curves and pointwise confidence bands, demonstrating how Natural Splines effectively constrain (limitino) boundary (ai margini) extrapolation to stabilize variance at the dataset boundaries.
suppressPackageStartupMessages({
  library(splines)    # For bs() [B-splines] and ns() [Natural Splines]
  library(boot)       # For cv.glm() 10-Fold Cross-Validation
})

set.seed(42)

# 1. Spline fitting with specified quantile knots (Slide 15)
# Positioning K = 3 interior knots at the 25th, 50th, and 75th percentiles of HST. Placing knots at empirical quantiles ensures each interval contains an equal volume of observations.
knots_hst <- quantile(master_df$HST, probs = c(0.25, 0.50, 0.75))

# Constructs a B-Spline basis. Cubic Spline (bs): df = K + 4 = 7 degrees of freedom and Fits a linear model of TotalGoals on the generated B-spline basis functions.
fit_bs_knots <- lm(TotalGoals ~ bs(HST, knots = knots_hst), data = master_df)

# Constructs a Natural Cubic Spline basis. Natural splines impose additional boundary constraints requiring the function to be strictly linear beyond the outer knots. These constraints reduce degrees of freedom from K + 4 down to df = K = 3.
fit_ns_knots <- lm(TotalGoals ~ ns(HST, knots = knots_hst), data = master_df)

cat("\n================ 1. MODEL COMPARISON WITH QUANTILE KNOTS ================\n")
cat("Cubic Spline (bs)  - Residual Std Error: ", round(summary(fit_bs_knots)$sigma, 4), "\n")
cat("Natural Spline (ns) - Residual Std Error: ", round(summary(fit_ns_knots)$sigma, 4), "\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# - The cubic B-spline (bs) achieves a slightly lower training Residual Standard Error (1.5597) than the Natural Spline (1.5603).
# - Why this happens: When placing 3 interior knots, the standard cubic spline uses df = 3 + 4 = 7 degrees of freedom, whereas the natural spline uses df = 3. The additional degrees of freedom allow bs to fit training data slightly closer, but at the cost of higher complexity


# 2. Tuning degrees of freedom (df) via 10-FOLD CROSS-VALIDATION. 
# Defines the grid of candidate degrees of freedom to evaluate (df ∈ {2, 3, 4, 5, 6, 7, 8}).
df_range <- 2:8
cv_mse_bs <- numeric(length(df_range))# Pre-allocates numeric storage vectors to record cross-validation errors for B-splines and Natural splines.
cv_mse_ns <- numeric(length(df_range))

# Loops sequentially through each candidate value in df_range
for (i in seq_along(df_range)) {
  df_val <- df_range[i]
  
  # Enforces the mathematical restriction that cubic B-splines require at least df = 3. Assigns NA for df = 2.
  if (df_val >= 3) {
    glm_bs <- glm(TotalGoals ~ bs(HST, df = df_val), data = master_df)# Fits a GLM specifying target degrees of freedom directly
    cv_mse_bs[i] <- cv.glm(master_df, glm_bs, K = 10)$delta[1]# Runs 10-Fold Cross-Validation and extracts the raw test Mean Squared Error (MSE).
  } else {
    cv_mse_bs[i] <- NA
  }
  
  #Fits a Natural Cubic Spline model for all df >= 2 and computes its 10-Fold CV MSE.
  glm_ns <- glm(TotalGoals ~ ns(HST, df = df_val), data = master_df)
  cv_mse_ns[i] <- cv.glm(master_df, glm_ns, K = 10)$delta[1]
}

# dentifies the specific df value that minimizes cross-validation MSE for each spline type.
best_df_bs <- df_range[which.min(cv_mse_bs)]
best_df_ns <- df_range[which.min(cv_mse_ns)]

cat("\n================ 2. 10-FOLD CV RESULTS (df SELECTION) ================\n")
cat(sprintf("Optimal Cubic Spline (bs):   df = %d | CV MSE = %.4f\n", 
            best_df_bs, min(cv_mse_bs, na.rm = TRUE)))
cat(sprintf("Optimal Natural Spline (ns): df = %d | CV MSE = %.4f\n", 
            best_df_ns, min(cv_mse_ns, na.rm = TRUE)))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Natural Splines (ns) win out-of-sample (2.4399 < 2.4411). By forcing linear extrapolation beyond the outer knots, ns prevents overfitting in sparse regions of the predictor space, yielding better test set performance.


# 3. Diagnostic plots: CURVES & TAIL BEHAVIOR 
# Configures R graphics layout to render two side-by-side plots in a 1 x 2 grid.
par(mfrow = c(1, 2))

# Grid for predictions across the domain of HST
hst_grid <- seq(min(master_df$HST), max(master_df$HST), by = 0.1)

# Predictions with Pointwise Standard Errors for both fitted quantile-knot models.
preds_bs <- predict(fit_bs_knots, newdata = list(HST = hst_grid), se = TRUE)
preds_ns <- predict(fit_ns_knots, newdata = list(HST = hst_grid), se = TRUE)

# Plot 1: Cubic Spline (Red) vs Natural Spline (Blue). Draws a scatterplot of raw data points (HST vs TotalGoals) rendered in translucent grey dots.
plot(master_df$HST, master_df$TotalGoals, col = "grey70", pch = 16, cex = 0.5,
     xlab = "Home Shots on Target (HST)", ylab = "Total Goals",
     main = "Cubic (bs) vs Natural Spline (ns)")

# Cubic Spline Fit + 95% Confidence Intervals
lines(hst_grid, preds_bs$fit, col = "red", lwd = 2)
lines(hst_grid, preds_bs$fit + 2*preds_bs$se.fit, col = "red", lty = 2)
lines(hst_grid, preds_bs$fit - 2*preds_bs$se.fit, col = "red", lty = 2)

# Natural Spline Fit + 95% Confidence Intervals
lines(hst_grid, preds_ns$fit, col = "blue", lwd = 2)
lines(hst_grid, preds_ns$fit + 2*preds_ns$se.fit, col = "blue", lty = 2)
lines(hst_grid, preds_ns$fit - 2*preds_ns$se.fit, col = "blue", lty = 2)

# Vertical dashed lines indicating knot positions
abline(v = knots_hst, lty = 3, col = "darkgreen")
legend("topleft", legend = c("Cubic Spline (bs)", "Natural Spline (ns)", "Knots"),
       col = c("red", "blue", "darkgreen"), lty = c(1, 1, 3), lwd = c(2, 2, 1), cex = 0.75)

# Plot 2: 10-Fold CV MSE across degrees of freedom for B-splines (red circles) and Natural splines (blue triangles).
plot(df_range, cv_mse_bs, type = "b", col = "red", pch = 19, 
     ylim = range(c(cv_mse_bs, cv_mse_ns), na.rm = TRUE),
     xlab = "Degrees of Freedom (df)", ylab = "10-Fold CV MSE", 
     main = "Spline Complexity Selection (CV)")
lines(df_range, cv_mse_ns, type = "b", col = "blue", pch = 17)
legend("topright", legend = c("Cubic Spline (bs)", "Natural Spline (ns)"),
       col = c("red", "blue"), pch = c(19, 17), lty = 1, cex = 0.8)

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Left Panel: Fitted Curves & Tail InstabilityKnots (Dotted Lines): Placed at the empirical quantiles (= 3, 5, 7$ shots on target). 
# - The Tail Problem (HST > 15$):
# - Cubic Spline (bs, Red Line): Plunges (precipita) dramatically downwards at extreme values (HST > 15$) because standard polynomials flex (oscillano) wildly where data is sparse (sono scarsi). Its dashed 95% confidence bands widen (si allargano) drastically, showing high estimation variance. 
# - Natural Spline (ns, Blue Line): Extrapolates strictly linearly past (oltre) the boundary knots. The prediction stays stable and realistic, with narrow (più ristrett) confidence bands.   
# - Right Panel: CV Error vs. Degrees of Freedombs
# - bs Curve (Red): Error spikes sharply as degrees of freedom increase (df >= 5), demonstrating rapid overfitting due to unconstrained cubic terms.  
# - ns Curve (Blue): Remains lower and substantially more stable across all complexity levels because boundary constraints absorb unneeded variance.


# ==============================================================================
# SECTION 15: SMOOTHING SPLINES & LOCAL REGRESSION (LOESS)
# ==============================================================================
# This module implements non-parametric smoothing techniques—Smoothing Splines and Local Regression (LOESS)—to model non-linear goal dynamics without requiring explicit knot specification. Smoothing Splines minimize a penalized residual sum of squares where the roughness penalty parameter (λ) is automatically tuned via Leave-One-Out Cross-Validation (LOOCV). Local Regression estimates localized non-linear trends using sliding weighted least-squares windows governed by the span parameter (s). Diagnostic plots illustrate how tuning parameters impact the bias-variance tradeoff.

set.seed(42)

# 1. Smoothing splines 
# Fit 1: Fixed Degrees of Freedom (df = 16)
fit_ss_fixed <- smooth.spline(master_df$HST, master_df$TotalGoals, df = 16)

# Fit 2: Optimal Degrees of Freedom chosen by LOOCV (cv = TRUE)
fit_ss_cv <- smooth.spline(master_df$HST, master_df$TotalGoals, cv = TRUE)

cat("\n================ 1. SMOOTHING SPLINES RESULTS ================\n")
cat("Fixed df Model - Degrees of Freedom: ", fit_ss_fixed$df, "\n")
cat("LOOCV Optimal Model - Degrees of Freedom (df): ", round(fit_ss_cv$df, 2), "\n")
cat("LOOCV Optimal Lambda: ", format(fit_ss_cv$lambda, scientific = TRUE), "\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# - By manually forcing 16 effective degrees of freedom, the model allows the curve to flex excessively to fit local data points.
# - LOOCV Optimal Model (df = 4.55): Leave-One-Out Cross-Validation automatically finds that the optimal effective complexity for our 2,156 matches is df_λ = 4.55. 
# - Tuning Parameter (λ = 0.0886): The penalty parameter λ strikes the optimal balance between minimizing the Residual Sum of Squares (fit) and penalizing function curvature


# 2. Local Regression / LOESS 
# Fit 1: Flexible Local Regression (Span s = 0.2). It estimates a local regression based on Weighted Least Squares. A span of 0.2 means that, to estimate the value at each point x_0, the model uses only the 20% of observations closest to it, resulting in a highly flexible curve.
fit_loess_flexible <- loess(TotalGoals ~ HST, data = master_df, span = 0.2)

# Fit 2: Smoother Local Regression (Span s = 0.7). It uses 70% of the nearby data points for each local estimate, producing a much smoother and more stable curve (higher bias, lower variance).
fit_loess_smooth <- loess(TotalGoals ~ HST, data = master_df, span = 0.7)

cat("\n================ 2. LOCAL REGRESSION (LOESS) ================\n")
cat("Loess Model 1 - Span s = 0.2 (High Flexibility)\n")
cat("Loess Model 2 - Span s = 0.7 (High Smoothness)\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Span s = 0.2: For each target estimate, LOESS uses a sliding window containing only the nearest 20% of observations. This creates a highly localized flexible fit susceptible to noise.  
# - Span s = 0.7: Uses 70% of the data in each local neighborhood, yielding a far smoother curve with lower variance. 


# 3. Diagnostic visualizations 
par(mfrow = c(1, 2))

# Grid for smooth curve predictions
hst_grid <- seq(min(master_df$HST), max(master_df$HST), by = 0.1)

# Plot 1: Smoothing Splines (Fixed df vs LOOCV Optimal)
plot(master_df$HST, master_df$TotalGoals, col = "grey70", pch = 16, cex = 0.5,
     xlab = "Home Shots on Target (HST)", ylab = "Total Goals",
     main = "Smoothing Splines")

# Plot the optimal curve calculated by the Smoothing Spline. Unlike standard vectors, objects created by `smooth.spline` can be passed directly to the `lines()` function in R.
lines(fit_ss_fixed, col = "red", lwd = 2, lty = 2)
lines(fit_ss_cv, col = "blue", lwd = 2)

legend("topleft", 
       legend = c("Fixed (df = 16)", 
                  paste("LOOCV Optimal (df =", round(fit_ss_cv$df, 1), ")")),
       col = c("red", "blue"), lty = c(2, 1), lwd = 2, cex = 0.75)

# Plot 2: Local Regression / LOESS (Span s = 0.2 vs Span s = 0.7)
plot(master_df$HST, master_df$TotalGoals, col = "grey70", pch = 16, cex = 0.5,
     xlab = "Home Shots on Target (HST)", ylab = "Total Goals",
     main = "Local Regression (LOESS)")

# Generate point predictions from the local regression along the hst_grid to visualize the impact of the span parameter.
preds_loess_02 <- predict(fit_loess_flexible, newdata = data.frame(HST = hst_grid))
preds_loess_07 <- predict(fit_loess_smooth, newdata = data.frame(HST = hst_grid))

lines(hst_grid, preds_loess_02, col = "darkgreen", lwd = 2, lty = 2)
lines(hst_grid, preds_loess_07, col = "purple", lwd = 2)

legend("topleft", 
       legend = c("LOESS (Span s = 0.2)", "LOESS (Span s = 0.7)"),
       col = c("darkgreen", "purple"), lty = c(2, 1), lwd = 2, cex = 0.75)

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Left Panel: Smoothing Splines (Fixed vs LOOCV Optimal)
# - Fixed df = 16 (Red Dashed Line): Displays classic overfitting and tail collapse. Notice how the red curve wiggles (ondula) around 12 - 15 shots on target and then crashes sharply to 0 at HST = 18 due to the scarcity of extreme match data. 
# - LOOCV Optimal df = 4.6 (Blue Solid Line): Filters out local sampling noise and produces a smooth, realistic trajectory. Match goals rise steadily as shots on target increase from 0 to 10, reaching a logical expectation plateau of = 4 total goals.   
# - Right Panel: Local Regression / LOESS (s = 0.2 vs s = 0.7)
# - LOESS s = 0.2 (Green Dashed Line): Reacts to small local clusters in the data (bumps around 2 - 5 shots), demonstrating higher variance. 
# - LOESS s = 0.7 (Purple Solid Line): Provides a robust, smooth non-linear profile across the entire range of home shots on target. 


# ==============================================================================
# SECTION 16: GENERALIZED ADDITIVE MODELS (GAMs)
# ==============================================================================
# This module implements Generalized Additive Models (GAMs) to extend non-linear modeling across multiple match predictors simultaneously (HST, AST, HC) while retaining an interpretable additive structure. Using the gam package, it combines smoothing splines (s()) and linear terms to capture complex goal dynamics without suffering from the curse of dimensionality. Sequential ANOVA tests evaluate whether non-linear transformations significantly outperform simpler linear additive specifications, while Partial Dependence Plots isolate the marginal non-linear effect of each individual match metric on total expected goals.

suppressPackageStartupMessages({
  library(gam)        # For gam(), s(), lo(), and plot.gam()
  library(splines)    # For ns()
})

set.seed(42)

# 1. Fitting multivariate GAMs with various non linear terms
# Model 1: Fully Linear Additive Model
fit_gam_1 <- gam(TotalGoals ~ HST + AST + HC, data = master_df)

# Model 2: Mixed Additive Model (Natural Splines for Shots, Linear for Corners)
fit_gam_2 <- gam(TotalGoals ~ ns(HST, df = 4) + ns(AST, df = 4) + HC, data = master_df)

# Model 3: Flexible Non-Parametric GAM (Smoothing Splines s() for all variables)
fit_gam_3 <- gam(TotalGoals ~ s(HST, df = 4) + s(AST, df = 4) + s(HC, df = 3), data = master_df)

cat("\n================ 1. GAM NON-PARAMETRIC MODEL SUMMARY ================\n")
print(summary(fit_gam_3))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Parametric Effects Table: Tests whether a linear relationship exists between each predictor and TotalGoals. All three variables (HST, AST, HC) are extremely significant (p < 10^{-13}), confirming they are crucial predictors of total match goals. 
# - Nonparametric Effects Table: Tests whether adding a non-linear curve provides significant value over a simple linear fit for each specific variable.
# - s(HST, df = 4) (p = 0.003285): Highly statistically significant (p < 0.01). Home shots on target require a non-linear transformation. 
# - s(AST, df = 4) (p = 0.069303): Marginally significant at the 10% level. Away shots on target are nearly linear, with only slight non-linear curvature.  
# - s(HC, df = 3) (p = 0.618136): Not significant at all (p > 0.05). A linear term is completely sufficient for Home Corners; the smoothing spline non-linearity adds no explanatory power.   


# 2. ANOVA F-TEST for non-linearity significance
# Compare the three nested models using an F-test to determine whether the inclusion of non-linear components results in a statistically significant reduction in residual deviance compared to the simple linear model.
anova_gam <- anova(fit_gam_1, fit_gam_2, fit_gam_3, test = "F")

cat("\n================ 2. ANOVA COMPARISON: LINEAR VS NON-LINEAR GAMs ================\n")
print(anova_gam)

# 3. Partial Dependence Plots (PDPs) 
par(mfrow = c(1, 3))

# Partial Dependence Plots with 95% Confidence Intervals for Model 3
plot(fit_gam_3, se = TRUE, col = "blue", lwd = 2)

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Left Plot (HST - Home Shots on Target): Displays clear diminishing marginal returns. Goal expectation increases steadily up to ~ 10 shots on target, after which the curve flattens out.  
# - Middle Plot (AST - Away Shots on Target): Shows an almost purely linear upward slope. More away shots linearly increase expected match goals (matching its marginal p = 0.069 non-parametric test). 
# - Right Plot (HC - Home Corners): Shows a linear downward slope when controlling for shots on target. Holding shot output constant, high corner counts often indicate inefficient possession against low defensive blocks.


# ==============================================================================
# SECTION 17: DECISION TREES & COST-COMPLEXITY PRUNING
# ==============================================================================
# This module introduces Tree-Based Methods by fitting a regression tree to predict match TotalGoals based on offensive metrics (HST, AST, HC, AC). It begins by growing a deep, overly complex tree using recursive binary splitting to minimize the Residual Sum of Squares (RSS). To prevent overfitting and improve generalization, the algorithm applies cost-complexity pruning. The optimal subtree is systematically selected by tuning the complexity parameter (CP) via internal 10-Fold Cross-Validation, ultimately yielding an interpretable model that isolates key non-linear impact zones without fitting the noise.

suppressPackageStartupMessages({
  library(rpart)       # For recursive partitioning (decision trees)
  library(rpart.plot)  # For plotting the trees visually
})

set.seed(42)

# 1. Growing the full (unpruned) tree
# We force a deep tree by setting the complexity parameter (cp) to 0. This function performs Recursive Binary Splitting.
tree_full <- rpart(TotalGoals ~ HST + AST + HC + AC, 
                   data = master_df, 
                   method = "anova",
                   control = rpart.control(cp = 0, minsplit = 10)) # By forcing this parameter to zero, we compel the algorithm to construct a gigantic tree (intentional overfitting), stopping only when a node contains fewer than `minsplit = 10` observations.

# 2. Cost-complexity pruning VIA 10-FOLD CROSS-VALIDATION
# These commands print (and plot) a table showing the training error (rel error) and the cross-validation error (xerror) as the penalty α (represented by CP) varies.
cat("\n================ 1. COMPLEXITY PARAMETER (CP) TABLE ================\n")
printcp(tree_full)
# ====================== ANALYSIS OF THE RESULTS ===============================
# The printcp() output displays the model's performance metrics across 200 different complexity levels.
# - CP (Complexity Parameter): The penalty factor applied for adding branches to the tree.
# - nsplit: The number of splits in the tree at that specific CP value.
# - rel error: The training data error. This continuously decreases as the tree grows larger and memorizes the dataset, dropping from 1.00000 at 0 splits to 0.48087 at 335 splits.
# - xerror (Cross-Validation Error): The out-of-sample error estimate used to evaluate true predictive performance. It initially decreases as the model learns real patterns but eventually increases as the model begins to overfit the training data. The absolute minimum value is 0.82267. This minimum occurs at row 12, which corresponds to 11 splits and a CP value of 4.4466e-03. Rounding 4.4466e-03 to five decimal places yields exactly 0.00445. The printed code output correctly identifies the optimal CP value.
# - xstd: The standard error associated with the cross-validation error.


# Extract the optimal CP value that minimizes the Cross-Validation Error (xerror)
opt_cp_index <- which.min(tree_full$cptable[, "xerror"])
optimal_cp <- tree_full$cptable[opt_cp_index, "CP"]

cat(sprintf("\nOptimal CP minimizing CV error: %.5f\n", optimal_cp))
# ====================== ANALYSIS OF THE RESULTS ===============================
# The console explicitly extracts the exact CP value used to prune the overly complex tree back to its optimal size, discarding branches that do not sufficiently reduce the error.


# Prune the full tree using the optimal complexity parameter. This is the implementation of Cost-Complexity Pruning (or Weakest Link Pruning). Instead of growing a tree from scratch (da zero) using different rules, it takes the massive tree—`tree_full`—and prunes (taglia via) its branches, starting from the bottom. Any splits for which the reduction in RSS does not sufficiently offset the penalty imposed by the `optimal_cp` parameter are removed.
tree_pruned <- prune(tree_full, cp = optimal_cp)

# 3. Diagnostic plots 
par(mfrow = c(1, 2))

# Plot 1: Cross-Validation Error vs Complexity Parameter
plotcp(tree_full, main = "CV Error vs CP")

# Plot 2: The Final Pruned Regression Tree.
# Force the graph to display both the mean of the predicted goals at that terminal node and the percentage of matches in the dataset that fall into that region, providing an immediate sense of the reliability of each individual prediction.
rpart.plot(tree_pruned, 
           type = 4, 
           extra = 101, 
           fallen.leaves = TRUE, 
           main = "Pruned Regression Tree",
           box.palette = "RdYlGn",
           shadow.col = "gray")

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - V Error vs CP (Left): This plot visualizes the cptable data. The x-axis represents the complexity parameter, and the y-axis represents the relative cross-validation error. The curve exhibits a clear "U-shape," confirming that growing the tree too deep causes the test error to rise as the model fits to noise. The horizontal dashed line indicates the 1-Standard Error (1-SE) threshold above the absolute minimum error.
# - Pruned Regression Tree (Right): This is the final, optimized predictive model. The root node shows that the average TotalGoals across all 2,156 matches is 2.8.   The most important variable (the primary split) is HST (Home Shots on Target). Matches with HST < 7 drop to an expected 2.5 goals, while matches with HST >= 7 jump to 3.5 expected goals. The terminal nodes (leaves) at the bottom provide the final predictions for specific match scenarios. The color scale intuitively maps low expected goals (e.g., the red node predicting 1.4 goals for 8% of the data) to high expected goals (e.g., the bright green node predicting 5.1 goals for 1% of the data).  


# ==============================================================================
# SECTION 18: BAGGING & RANDOM FOREST
# ==============================================================================
# This module addresses the primary issue with individual decision trees (covered in Module 1): high variance. To stabilize predictions, we introduce ensemble methods—specifically Bagging and Random Forest—which train hundreds of trees on samples drawn with replacement (bootstrapping) and then average their results. Furthermore, we leverage the elegant theoretical concept of the Out-of-Bag (OOB) error, which allows us to estimate a model's error on new data without having to set aside a portion of the dataset as a test set.

suppressPackageStartupMessages({
  library(randomForest)  # Core package per Bagging e Random Forest
})

set.seed(42)

# 1. Bagging (Bootstrap Aggregation)
# mtry = 4: we consider all 4 predictors at each split (definition of Bagging). We force the algorithm to perform Bagging. All the trees will always evaluate the same variables, and since HST is the strongest one, nearly all the trees will have the same initial structure.
bag_model <- randomForest(TotalGoals ~ HST + AST + HC + AC, 
                          data = master_df, 
                          mtry = 4, 
                          importance = TRUE,
                          ntree = 500)

# 2. Random Forest
# mtry = 2: we consider a random subset of 2 predictors at each split. At each node, the algorithm can choose only from two randomly selected variables. This trick temporarily "hides" the most dominant variables (such as shots on target), forcing the model to extract useful patterns even from less prominent variables (such as corner kicks). This process decorrelates the trees, further reducing the variance of the final average.
rf_model <- randomForest(TotalGoals ~ HST + AST + HC + AC, 
                         data = master_df, 
                         mtry = 2, 
                         importance = TRUE,
                         ntree = 500) #Instruct the model to generate 500 separate decision trees. Unlike the single tree in section 17, the final prediction for a match will be the mathematical average of the goals predicted by all 500 trees.

# 3. Comparison of OUT-OF-BAG ERRORS (OOB)
# Since each tree is built on a bootstrap sample (which contains approximately two-thirds of the original data), the remaining third of the observations (the out-of-bag data) is used as an automatic test set for that tree.
cat("\n================ COMPARISON OOB MSE ================\n")
cat(sprintf("Bagging OOB Mean Squared Error:       %.4f\n", tail(bag_model$mse, 1)))
cat(sprintf("Random Forest OOB Mean Squared Error: %.4f\n", tail(rf_model$mse, 1)))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Bagging (MSE = 2.4314): By forcing all trees to evaluate all 4 predictors at each split (mtry = 4), the dominant variable (HST – Home Shots on Target) is almost always chosen at the first node. Consequently, the 500 generated trees end up being highly correlated with one another. When averaging correlated trees, the reduction in variance is limited.
# - Random Forest (MSE = 2.2850): By limiting the splits to a random subset of only 2 variables (mtry = 2), the algorithm decorrelates the trees. Forcing the model to construct trees without being able to use HST at the first node allows other metrics to be leveraged as well. Averaging highly decorrelated trees drastically reduces variance and lowers the Out-of-Bag (OOB) error from 2.43 to 2.28.


# 4. Diagnostic: IMPORTANCE OF VARIABLES 
par(mfrow = c(1, 2))

# Track the importance of variables (Increase in % error if the variable is omitted). This algorithm randomly shuffles the values ​​of a variable (e.g., AC) and measures how much the prediction error worsens: if the error increases drastically, the variable was essential; if nothing changes, it was useless.
varImpPlot(bag_model, 
           main = "Bagging\nVariable Importance", 
           pch = 19, color = "#1E293B", cex = 1.1)

varImpPlot(rf_model, 
           main = "Random Forest\nVariable Importance", 
           pch = 19, color = "#D97706", cex = 1.1)

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Left — %IncMSE (Increase in Mean Squared Error):
# - It indicates how much the model’s out-of-bag prediction error increases if the values for a specific variable are randomly permuted (i.e., "destroyed").
# - Interpretation: HST (Home Shots on Target) is by far the most critical variable; removing it causes the error to spike by nearly 90%. This is followed by AST (Away Shots on Target, ~70%), HC (Home Corners, ~23%), and finally AC (Away Corners, ~12%).
# - Football context: Shots on target (both at home and away) are the primary drivers of the final score; corner kicks provide a marginal but useful contribution for refining the prediction.
# - Right — IncNodePurity (Increase in Node Purity):
# - It represents the total reduction in the Residual Sum of Squares (RSS) accumulated from all splits performed on a given variable, aggregated across all 500 trees.
# - Interpretation: This fully confirms the ranking shown on the left. HST produces the splits that result in the "purest" nodes (i.e., those with the least dispersion of goal values around the mean), followed by AST, HC, and AC.


# ==============================================================================
# SECTION 19: GRADIENT BOOSTING REGRESSION
# ==============================================================================
#Unlike Bagging and Random Forests, where trees are built independently in parallel on bootstrap samples, Boosting grows trees sequentially. Each new tree is fitted to the residuals (unexplained errors) of the preceding model, learning slowly from the current ensemble's mistakes. In this module, we implement Gradient Boosting via the gbm package using three core hyperparameter knobs:n Number of Trees, Shrinkage Parameter, Interaction Depth. We will evaluate the test/OOB performance across different configurations and extract relative influence plots

suppressPackageStartupMessages({
  library(gbm)  # Core package for Generalized Boosted Regression Models
})

set.seed(42)

# 1. Train / Test split (70% Train, 30% Test for Out-of-Sample Validation)
# Unlike Bagging and Random Forest (which automatically calculate the Out-of-Bag error via bootstrapping), in Boosting it is standard practice to manually split the dataset (70% training, 30% test) in order to calculate a rigorous, out-of-sample Test MSE that is free from data leakage.
train_idx <- sample(1:nrow(master_df), 0.7 * nrow(master_df))
train_df  <- master_df[train_idx, ]
test_df   <- master_df[-train_idx, ]

# 2. Fit gradient boosting models (Varying Interaction Depths & Learning Rates)
# Model A: Additive Stumps (depth = 1, shrinkage = 0.01)
boost_model_a <- gbm(TotalGoals ~ HST + AST + HC + AC,
                     data = train_df,
                     distribution = "gaussian",  # Specify that the problem is a quantitative regression and that the loss function to be minimized is the Mean Squared Error (MSE).
                     n.trees = 5000, # Sets the maximum number of trees to be built in sequence. Unlike Random Forest, in Boosting, an excessive number of trees can lead to overfitting.
                     interaction.depth = 1, # Defines the depth of each individual tree. With d = 1 (Model A), we obtain stumps (trees with a single split) that model only purely additive relationships. With d = 2 (Model B), we allow the model to capture bivariate interactions (e.g., the joint effect of HST and AST in the same match).
                     shrinkage = 0.01, # It is the learning rate. It scales the contribution of each new tree added to the sequence.
                     cv.folds = 5)# It performs an internal 5-fold cross-validation during training to monitor how the error evolves as the number of trees increases.

# Model B: Bivariate Interactions (depth = 2, shrinkage = 0.01)
boost_model_b <- gbm(TotalGoals ~ HST + AST + HC + AC,
                     data = train_df,
                     distribution = "gaussian",
                     n.trees = 5000,
                     interaction.depth = 2,
                     shrinkage = 0.01,
                     cv.folds = 5)

# 3. Identify optimal number of trees via CV
# Identify the exact point on the cross-validation curve where the error stabilizes or begins to rise. This yields the optimal number of trees, B_opt, to be used for predictions, effectively preventing overfitting.
best_trees_a <- gbm.perf(boost_model_a, method = "cv", plot.it = FALSE)
best_trees_b <- gbm.perf(boost_model_b, method = "cv", plot.it = FALSE)

cat("\n================ OPTIMAL NUMBER OF TREES (CV) ================\n")
cat(sprintf("Model A (Depth 1) Optimal Trees: %d\n", best_trees_a))
cat(sprintf("Model B (Depth 2) Optimal Trees: %d\n", best_trees_b))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Model A (d = 1, Decision Stumps): Because each tree consists of a single split, the learning process is very granular. It requires 1,360 trees at a shrinkage rate of λ = 0.01 to fully capture the predictive signal without overfitting.
# - Model B (d = 2, Bivariate Trees): Because each tree has a depth of 2 (capable of splitting on two variables), each sequential iteration extracts more information. As a result, the model converges faster and reaches its minimum Cross-Validation error earlier, at 752 trees.


# 4. Out-of-sample prediction & MSE evaluation
# It calculates predictions on 30% of the data that the model has never seen, using exactly B_opt trees, and derives the final Test MSE, which can then be compared directly to the $2.2850$ value obtained by the Random Forest.
preds_a <- predict(boost_model_a, newdata = test_df, n.trees = best_trees_a)
preds_b <- predict(boost_model_b, newdata = test_df, n.trees = best_trees_b)

mse_boost_a <- mean((test_df$TotalGoals - preds_a)^2)
mse_boost_b <- mean((test_df$TotalGoals - preds_b)^2)

cat("\n================ BOOSTING TEST SET MSE ================\n")
cat(sprintf("Boosting (Depth 1, Stumps) Test MSE:      %.4f\n", mse_boost_a))
cat(sprintf("Boosting (Depth 2, Interactions) Test MSE:  %.4f\n", mse_boost_b))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Model A (d = 1) achieves a slightly lower Test MSE (2.1906) than Model B (d = 2).
# - This indicates that the true underlying Data Generating Process (DGP) for total goals in Serie A is primarily additive. Shots on target and corners contribute independently to total goals, with minimal higher-order interaction effects required to model the outcome.


# 5. Relative variable influence & partial dependence plots
par(mfrow = c(1, 2))

# Calculate the relative influence of each variable by measuring the extent to which its inclusion in the splits reduced the overall squared loss.
summary(boost_model_b, 
        cBars = 4, 
        method = relative.influence, 
        las = 1, 
        main = "Boosting Relative Influence")
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Primary Drivers: Shots on target (HST + AST) account for 66.13% of the total reduction in square-error loss across all sequential trees.
# - Secondary Drivers: Corner kicks (HC + AC) account for the remaining 33.87%, proving that set-piece volume provides meaningful secondary predictive information.


# Generate the Partial Dependence Plot (PDP). This shows the direct marginal effect of the variable HST (home shots on target) on the expected total number of goals (TotalGoals), while accounting for and "neutralizing" the average effect of the other three variables (AST, HC, AC).
plot(boost_model_b, i.var = "HST", 
     n.trees = best_trees_b, 
     main = "Partial Dependence Plot: HST")

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - The step-like curve illustrates the isolated marginal effect of Home Shots on Target (HST) on predicted total goals (y^), holding AST, HC, and AC fixed at their average values
# - Baseline (< 2 Shots): When HST is 0 or 1, expected total match goals remain low (~1.7 goals).
# - Monotonic Step Increases: Between 2 and 12 shots on target, the curve increases in discrete steps. Each additional shot on target reliably bumps up expected match goals.
# - Saturation Point (> 13 Shots): Above ~13 home shots on target, the curve plateaus around ~4.3 total goals, reflecting natural ceiling effects in match scoring.
# - Piecewise structure: the step-like shape is a direct reflection of a partition based on trees; the ensemble predictions are piecewise constant over defined intervals of the variables.


# ==============================================================================
# SECTION 20: CLASSIFICATION TREES & BART
# ==============================================================================
# This final module extends our tree-based methodology by framing match outcomes as a binary classification problem (Over/Under 2.5 Goals) using a Classification Random Forest. It then introduces Bayesian Additive Regression Trees (BART) to model the continuous target. Unlike traditional ensembles, BART employs Markov Chain Monte Carlo (MCMC) sampling to generate a posterior distribution of predictions, allowing us to rigorously quantify predictive uncertainty for our Decision Support System.

suppressPackageStartupMessages({
  library(randomForest)
  library(BART)          # Package for Bayesian Additive Regression Trees
})

set.seed(42)

# 1. Binary Target Creation (Over 2.5 Goals)
# Create factor variable: 1 if TotalGoals > 2.5, else 0
master_df$Over25 <- factor(ifelse(master_df$TotalGoals > 2.5, "Over", "Under"))

# Train/Test Split (70/30)
train_idx <- sample(1:nrow(master_df), 0.7 * nrow(master_df))
train_df  <- master_df[train_idx, ]
test_df   <- master_df[-train_idx, ]

# 2. Classification RANDOM FOREST (Over/Under 2.5)

# Fit Classification RF using Gini Impurity criterion
rf_class <- randomForest(Over25 ~ HST + AST + HC + AC, 
                         data = train_df, 
                         ntree = 500, 
                         mtry = 2, 
                         importance = TRUE)

# Out-of-sample Predictions (Class labels & Probabilities)
rf_class_preds <- predict(rf_class, newdata = test_df, type = "response")
# It extracts the estimated probabilities for the "Over 2.5" class. Instead of providing a binary answer ("Over" or "Under"), the system calculates a continuous probability (e.g., 62.4%).
rf_class_probs <- predict(rf_class, newdata = test_df, type = "prob")[, "Over"]

# Confusion Matrix & Classification Metrics
conf_matrix <- table(Predicted = rf_class_preds, Actual = test_df$Over25)
accuracy    <- sum(diag(conf_matrix)) / sum(conf_matrix)

cat("\n================ CLASSIFICATION RF (OVER/UNDER 2.5) ================\n")
print(conf_matrix)
cat(sprintf("\nOut-of-Sample Accuracy: %.2f%%\n", accuracy * 100))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Random Forest for Binary Classification (Over/Under 2.5): The confusion matrix indicates that the model correctly classified 244 "Over" events and 164 "Under" events. The resulting out-of-sample accuracy is 63.06%. In the field of football prediction—an area characterized by a high degree of stochasticity—an accuracy rate exceeding 60% for this specific market is considered a very robust analytical result.


# 3. Bayesian Additive Regression Trees (BART)
# Prepare matrices for BART (requires numeric predictor matrices)
x_train <- as.matrix(train_df[, c("HST", "AST", "HC", "AC")])
y_train <- train_df$TotalGoals

x_test  <- as.matrix(test_df[, c("HST", "AST", "HC", "AC")])
y_test  <- test_df$TotalGoals

# Fit BART model (MCMC sampling over tree structures)
# Unlike other algorithms, BART assigns a prior distribution over both the tree structure and the leaves, updating it based on observations via MCMC (Markov Chain Monte Carlo) chains. By default, ISLR suggests using 50 small trees.
bart_model <- gbart(x_train, y_train, x.test = x_test, ntree = 50)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - MCMC Execution of the BART Model: The console output documents that the algorithm successfully completed a total of 1100 iterations, taking 3 seconds. This process typically includes a "burn-in" period to stabilize the model, followed by the extraction of 1000 valid samples to construct the final distribution.


# Compute Test MSE using posterior mean predictions
# Each row of the `yhat.test` matrix represents an MCMC simulation of the predictions. By calculating the column-wise mean (`yhat.test.mean`), we obtain the point estimate for the test set. If, on the other hand, we isolate a single column (`match1_posterior`), we can construct a histogram of the posterior distribution.
bart_preds <- bart_model$yhat.test.mean
mse_bart   <- mean((y_test - bart_preds)^2)

cat("\n================ BART MODEL PERFORMANCE ================\n")
cat(sprintf("BART Test Set MSE: %.4f\n", mse_bart))

# 4. Diagnostics: BART POSTERIOR CONVERGENCE & UNCERTAINTY
par(mfrow = c(1, 2))

# Plot MCMC convergence for residual variance (sigma)
plot(bart_model$sigma, type = "l", 
     main = "BART MCMC Traceplot (Sigma)", 
     xlab = "MCMC Iteration", ylab = expression(sigma), col = "#2563EB")

# Posterior prediction distribution for a single sample match (Match #1 in Test Set)
match1_posterior <- bart_model$yhat.test[, 1]
hist(match1_posterior, breaks = 20, col = "#10B981", border = "white",
     main = "Posterior Expected Goals\n(Sample Match #1)",
     xlab = "Predicted Goals (MCMC Draws)")
abline(v = mean(match1_posterior), col = "darkgreen", lwd = 2, lty = 2)

par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Trace plot of the residual variance (left-hand graph): The fluctuation of the parameter σ appears stationary, remaining roughly between the values 1.30 and 1.50. The absence of any upward or descending trends—exhibiting a characteristic "white noise" pattern—confirms that the Markov chain (MCMC) has successfully reached convergence.
# - Posterior Predictive Distribution (Chart on the Right): This histogram illustrates the added value of the Bayesian approach. For the specific match tested, the model did not provide a static estimate, but rather a probability distribution derived from 1,000 valid draws. The expected mean is around 3.3 goals (indicated by the green dashed line), with nearly all simulations falling within a range of 2.8 to 3.8 goals.


# ==============================================================================
# SECTION 21: LINEAR SUPPORT VECTOR CLASSIFIER (SVC)
# ==============================================================================
# This module introduces the Support Vector Classifier (SVC) for predicting binary match outcomes (Over25 goals) using a linear hyper-plane. We split the Serie A dataset into training and test sets to evaluate out-of-sample performance across different values of the regularization parameter C (cost). By tuning C via Cross-Validation, we optimize the trade-off between margin width and classification error tolerance on noisy match data.  

suppressPackageStartupMessages({
  library(e1071)  # Core package for Support Vector Machines
})

set.seed(42)

# 1. Data preparation & TRAIN/TEST split
# Ensure target is a factor variable for classification
master_df$Over25 <- factor(ifelse(master_df$TotalGoals > 2.5, "Over", "Under"))

# Cleaning missing values only for the model variables to avoid mismatches
svm_data <- na.omit(master_df[, c("Over25", "HST", "AST", "HC", "AC")])

train_idx <- sample(1:nrow(svm_data), 0.7 * nrow(svm_data))
train_df  <- svm_data[train_idx, ]
test_df   <- svm_data[-train_idx, ]

# 2. Cross-validation for regularization parameter (COST C)
# Tune Cost parameter C over a grid of values
tune_svc <- tune(svm, 
                 train.x = Over25 ~ HST + AST + HC + AC, 
                 data = train_df, 
                 kernel = "linear",# Specify that we are building a Support Vector Classifier based on a flat (linear) separating hyperplane, without any non-linear transformations via complex kernels. 
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10)))# It performs 10-fold cross-validation to select the optimal value for the regularization parameter C (cost). A low value of C allows for a wider margin and tolerates more violations/classification errors (increasing bias, reducing variance). A high value of C adopts a "narrow" margin, fitting the data more rigidly (reducing bias, increasing variance).

best_svc <- tune_svc$best.model

cat("\n================ TUNING RESULTS ================\n")
cat(sprintf("Optimal Cost (C): %.3f\n", tune_svc$best.parameters$cost))
cat(sprintf("Best 10-Fold CV Error: %.4f\n", tune_svc$best.performance))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Low Cost Value (C = 0.010): Cross-validation selected a relatively small cost parameter. In SVM terminology, a low C value creates a wide soft margin that tolerates a higher degree of sample overlap/violations.
# - Theoretical Justification: Football match statistics (HST, AST, HC, AC) are inherently noisy. A strict linear boundary (high C) would overfit to individual outlier games. By opting for a low C, the linear SVC prioritizes low model variance, achieving a strong 10-fold cross-validation error rate of 31.41% (equivalent to ~68.59% CV accuracy). 


# 3. Model evaluation & Confusion matrix
# Predict on out-of-sample test set
svc_preds <- predict(best_svc, newdata = test_df)

# Confusion Matrix & Accuracy
conf_matrix_svc <- table(Predicted = svc_preds, Actual = test_df$Over25)
accuracy_svc    <- sum(diag(conf_matrix_svc)) / sum(conf_matrix_svc)

cat("\n================ TEST SET PERFORMANCE ================\n")
print(conf_matrix_svc)
cat(sprintf("\nLinear SVC Test Accuracy: %.2f%%\n", accuracy_svc * 100))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - The test set accuracy reached 64.61%, surpassing the 63.06% accuracy obtained by the Classification Random Forest.
# - Correct Predictions: 238 True Overs + 180 True Unders = 418 correctly classified matches. Errors: 124 False Overs + 105 False Unders = 229 misclassified matches. 


# 4. Support Vectors Diagnostics
cat("\n================ SUPPORT VECTORS DIAGNOSTICS ================\n")
cat(sprintf("Total Number of Support Vectors: %d\n", best_svc$tot.nSV))
cat(sprintf("Support Vectors per Class (Over / Under): %d / %d\n", 
            best_svc$nSV[1], best_svc$nSV[2]))# It extracts the total number of Support Vectors (the observations that lie exactly on the margin or violate it). These points are the only ones that define the position of the hyperplane; all other, well-classified data points that are far from the margin have no influence whatsoever on the model.
# ====================== ANALYSIS OF THE RESULTS ===============================
# - High Proportion of Support Vectors: Out of the training sample (~1,509 observations), 1,177 points serve as support vectors.
# - A large number of support vectors directly stems from choosing a small cost value (C = 0.010). Because the soft margin is wide, many observations lie on or violate the margin boundaries.
# - The support vectors are split almost perfectly evenly between classes (589 Over vs. 588 Under), indicating that the linear hyperplane is well-centered in feature space without bias toward either outcome.


# ==============================================================================
# SECTION 22: NON-LINEAR SUPPORT VECTOR MACHINES (KERNELS)
# ==============================================================================
# This module extends the linear classifier by introducing non-linear decision boundaries through the use of Polynomial and Radial Basis Function (RBF) kernels. By employing the "kernel trick," we implicitly transform the feature space to determine whether a non-linear separation of game statistics improves accuracy compared to the linear model. We will perform cross-validation to optimize both the regularization parameter C (cost) and the kernel width parameter 𝛾.

suppressPackageStartupMessages({
  library(e1071)  # Core package for Support Vector Machines
})

set.seed(42)

# 1. Data preparation & TRAIN/TEST SPLIT
master_df$Over25 <- factor(ifelse(master_df$TotalGoals > 2.5, "Over", "Under"))
svm_data <- na.omit(master_df[, c("Over25", "HST", "AST", "HC", "AC")])

train_idx <- sample(1:nrow(svm_data), 0.7 * nrow(svm_data))
train_df  <- svm_data[train_idx, ]
test_df   <- svm_data[-train_idx, ]

# 2. Polynomial KERNEL SVM (Degree d = 2 & d = 3)
# Train an SVM with a polynomial kernel, testing degrees d = 2 and d = 3. This allows for modeling quadratic or cubic boundaries to capture non-linear interactions between shots and angles.
tune_poly <- tune(svm, 
                  train.x = Over25 ~ HST + AST + HC + AC, 
                  data = train_df, 
                  kernel = "polynomial", 
                  ranges = list(cost = c(0.1, 1, 5), degree = c(2, 3)))

best_poly <- tune_poly$best.model

cat("\n================ POLYNOMIAL KERNEL TUNING ================\n")
cat(sprintf("Optimal Cost (C): %g\n", tune_poly$best.parameters$cost))
cat(sprintf("Optimal Degree (d): %d\n", tune_poly$best.parameters$degree))
cat(sprintf("Best 10-Fold CV Error: %.4f\n", tune_poly$best.performance))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Optimal Hyperparameters: The grid search selected a Cost (C) of 0.1 and a Polynomial Degree (d) of 3.  
# - Performance: It achieved a 10-Fold Cross-Validation Error of 0.3373 and an out-of-sample test accuracy of 63.21%. 
# - Interpretation: The cubic decision boundary (d=3) underperformed compared to the linear model (64.61%), indicating that global polynomial transformations of match statistics do not generalize well to unseen data. It also required a massive 1,246 Support Vectors to define this boundary, suggesting the model struggled to separate the classes efficiently.  


# 3. Radial Basis Function (RBF) KERNEL SVM
# Train an SVM with a Radial Basis Function (RBF) kernel. The parameter 𝛾 controls the reach of the influence of individual support vectors: a small λ (10^{-3}) results in a smooth, nearly linear decision boundary (low variance), while a large 𝛾 (1) creates a highly complex boundary around individual training data points (risk of overfitting, high variance).
tune_radial <- tune(svm, 
                    train.x = Over25 ~ HST + AST + HC + AC, 
                    data = train_df, 
                    kernel = "radial", 
                    ranges = list(cost = c(0.1, 1, 5, 10), 
                                  gamma = c(0.001, 0.01, 0.1, 1)))# Perform a grid search with 10-fold cross-validation, simultaneously over C and 𝛾, to find the combination that minimizes the out-of-sample error on the training data.

best_radial <- tune_radial$best.model

cat("\n================ RADIAL (RBF) KERNEL TUNING ================\n")
cat(sprintf("Optimal Cost (C): %g\n", tune_radial$best.parameters$cost))
cat(sprintf("Optimal Gamma (gamma): %g\n", tune_radial$best.parameters$gamma))
cat(sprintf("Best 10-Fold CV Error: %.4f\n", tune_radial$best.performance))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - Optimal Hyperparameters: The tuning process found an optimal Cost (C) of 5 and a Gamma (𝛾) of 0.1.  
# - Performance: It achieved the lowest 10-Fold Cross-Validation Error at 0.3068. Most importantly, it reached the highest out-of-sample test accuracy of 65.53%
# - Interpretation: A higher Cost (C=5) penalizes margin violations more strictly than the linear model, while a moderate Gamma (𝛾=0.1) allows for localized, non-linear decision boundaries that fit the data's natural clusters without overfitting. This model is more efficient, reducing the number of Support Vectors to 1,033. 


# 4. OUT-OF-SAMPLE Test Evaluation
preds_poly   <- predict(best_poly, newdata = test_df)
preds_radial <- predict(best_radial, newdata = test_df)

conf_poly   <- table(Predicted = preds_poly, Actual = test_df$Over25)
conf_radial <- table(Predicted = preds_radial, Actual = test_df$Over25)

acc_poly   <- sum(diag(conf_poly)) / sum(conf_poly)
acc_radial <- sum(diag(conf_radial)) / sum(conf_radial)

cat("\n================ TEST SET PERFORMANCE COMPARISON ================\n")
cat(sprintf("Polynomial Kernel SVM Test Accuracy: %.2f%%\n", acc_poly * 100))
cat(sprintf("Radial (RBF) Kernel SVM Test Accuracy:   %.2f%%\n", acc_radial * 100))
# ====================== ANALYSIS OF THE RESULTS ===============================
# The Radial (RBF) Kernel successfully captures the non-linear relationships between shots and corners, making it the strongest predictive model for forecasting Over/Under 2.5 outcomes in this Serie A dataset.


# 5. Support Vectors comparison
cat("\n================ SUPPORT VECTORS COUNT ================\n")
cat(sprintf("Polynomial Kernel Total SVs: %d\n", best_poly$tot.nSV))
cat(sprintf("Radial Kernel Total SVs:     %d\n", best_radial$tot.nSV))


# ==============================================================================
# SECTION 23: MODEL EVALUATION VIA ROC CURVES
# ==============================================================================
# This module introduces the construction and analysis of ROC (Receiver Operating Characteristic) curves for evaluating model performance as the classification threshold varies. Using the ROCR package, we extract continuous decision values from Support Vector Machines to graphically compare an optimal radial model (𝛾 = 0.1) with an intentionally over-complex model (𝛾 = 50), highlighting the concepts of generalization and overfitting in the context of training and test data.

suppressPackageStartupMessages({
  library(e1071)
  library(ROCR)   # Package required for ROC curves
})

# 1. Model training with decision values
# Optimal Model (gamma = 0.1, cost = 5)
svm_opt <- svm(Over25 ~ HST + AST + HC + AC, data = train_df, 
               kernel = "radial", gamma = 0.1, cost = 5, 
               decision.values = TRUE)# Within the `svm()` function, this argument forces the model to return continuous values (the distance of each observation from the decision margin) rather than just the predicted class labels. These scores are essential for varying the threshold and constructing an ROC curve.

# Overfitted Model (very high gamma, to demonstrate overfitting)
svm_flex <- svm(Over25 ~ HST + AST + HC + AC, data = train_df, 
                kernel = "radial", gamma = 50, cost = 5, 
                decision.values = TRUE)

# 2. Support function for calculating and drawing ROC curves
plot_roc <- function(model, data_set, color, add_to_plot = FALSE) {
  # Extracts continuous predictions (distance from the hyperplane)
  fitted_preds <- predict(model, newdata = data_set, decision.values = TRUE)
  scores <- attr(fitted_preds, "decision.values")
  
  # It belongs to the ROCR package and standardizes decision scores by comparing them against the actual labels (specifying which class we consider "positive"—in this case, "Over").
  pred_obj <- prediction(scores, data_set$Over25 == "Over")
  
  # Calculate the metrics for the curve: the True Positive Rate (TPR, sensitivity) on the Y-axis and the False Positive Rate (FPR, 1 - specificity) on the X-axis for every possible threshold value.
  perf <- performance(pred_obj, "tpr", "fpr")
  
  # Draw the curve
  plot(perf, col = color, add = add_to_plot, lwd = 2)
}

# 3. Comparison of ROC curves on training and test data
par(mfrow = c(1, 2)) # Set the graphical layout to 2 panels

# Panel 1: Training Data
plot_roc(svm_opt, train_df, color = "blue")
plot_roc(svm_flex, train_df, color = "red", add_to_plot = TRUE)
legend("bottomright", legend = c("Opt. Gamma = 0.1", "Flex Gamma = 50"), 
       col = c("blue", "red"), lwd = 2, cex = 0.8)
title("ROC Curve: Training Data")

# Panel 2: Test Data (Out-of-sample)
plot_roc(svm_opt, test_df, color = "blue")
plot_roc(svm_flex, test_df, color = "red", add_to_plot = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray") # Linea del classificatore casuale
legend("bottomright", legend = c("Opt. Gamma = 0.1", "Flex Gamma = 50"), 
       col = c("blue", "red"), lwd = 2, cex = 0.8)
title("ROC Curve: Test Data")

# Restore the original graphic layout
par(mfrow = c(1, 1))
# ====================== ANALYSIS OF THE RESULTS ===============================
# - ROC Curve: Training Data (Left Panel)
# - Flex Gamma = 50 (Red Line): This curve hugs the top-left corner almost perfectly, indicating near 100% accuracy on the training data. With a very high 𝛾, the SVM creates extremely tight, complex, and non-linear decision boundaries around every single training point. The model has effectively memorized the dataset.  
# - Opt. Gamma = 0.1 (Blue Line): This curve shows good performance but is visibly lower than the red line. The lower 𝛾 enforces a smoother, less complex boundary that doesn't capture every single outlier or noisy data point in the training set.   
# - ROC Curve: Test Data (Right Panel)
# - Flex Gamma = 50 (Red Line): The performance collapses dramatically, falling very close to the gray dashed diagonal line (which represents random guessing). Because the model memorized the specific noise of the training data, it is entirely incapable of generalizing to new, unseen football matches. This is a textbook example of overfitting. 
# - Opt. Gamma = 0.1 (Blue Line): The optimal model maintains a strong curve, bowing steadily toward the top left and significantly outperforming the red line. By learning the true underlying patterns (the signal) rather than the noise during training, it generalizes much better to new matches.
# - These plots visually confirm why hyperparameter tuning (via the Cross-Validation we did in SECTION 22) is essential: it protects the model from memorizing the data and ensures it remains useful for actual out-of-sample forecasting.


# ==============================================================================
# SECTION 24: MULTICLASS SVM & LOGISTIC REGRESSION COMPARISON
# ==============================================================================
# This module explores the extension of Support Vector Machines to multi-class classification problems (K > 2) and introduces a comparison with probabilistic models. We will replace the binary target "Over25" with the variable "FTR" (Full-Time Result: Home Win "H", Draw "D", Away Win "A") to demonstrate how the `e1071` package automatically handles multiple classes using the One-versus-One (OVO) approach.

suppressPackageStartupMessages({
  library(e1071)
  library(nnet)  # For Multinomial Logistic Regression
})

set.seed(42)

# 1. Multiclass Data Preparation
# Ensure target is a factor with 3 levels (H, D, A)
multi_data <- na.omit(master_df[, c("FTR", "HST", "AST", "HC", "AC")])
multi_data$FTR <- as.factor(multi_data$FTR)

train_idx_m <- sample(1:nrow(multi_data), 0.7 * nrow(multi_data))
train_df_m  <- multi_data[train_idx_m, ]
test_df_m   <- multi_data[-train_idx_m, ]

# 2. Multiclass Support Vector Machine (OVO APPROACH)
# When the target variable has more than two levels (H, D, A), the `svm()` function from the `e1071` package does not require any extra configuration. It automatically applies the One-versus-One (OVO) approach: it constructs a separate binary classifier for every possible pair of classes (Home vs. Draw, Home vs. Away, Draw vs. Away) and assigns the final observation to the class that wins the highest number of these pairwise "contests."
svm_multi <- svm(FTR ~ HST + AST + HC + AC, data = train_df_m, 
                 kernel = "radial", cost = 5, gamma = 0.1)

preds_svm_multi <- predict(svm_multi, newdata = test_df_m)
conf_svm_multi  <- table(Predicted = preds_svm_multi, Actual = test_df_m$FTR)
acc_svm_multi   <- sum(diag(conf_svm_multi)) / sum(conf_svm_multi)

cat("\n================ MULTICLASS SVM (OVO) ================\n")
print(conf_svm_multi)
cat(sprintf("Multiclass SVM Accuracy: %.2f%%\n", acc_svm_multi * 100))
# ====================== ANALYSIS OF THE RESULTS ===============================
# Multiclass SVM (OVO) Performance: The model achieved an out-of-sample accuracy of 56.11%. It is strong at identifying decisive outcomes, correctly predicting 209 Home wins and 135 Away wins. However, it struggles significantly with Draws, correctly identifying only 19 of them


# 3. Comparison: MULTINOMIAL LOGISTIC REGRESSION
# Fitting a Logistic model to compare theoretical Loss Functions in practice
logit_multi <- multinom(FTR ~ HST + AST + HC + AC, data = train_df_m, trace = FALSE)

preds_logit_multi <- predict(logit_multi, newdata = test_df_m)
conf_logit_multi  <- table(Predicted = preds_logit_multi, Actual = test_df_m$FTR)
acc_logit_multi   <- sum(diag(conf_logit_multi)) / sum(conf_logit_multi)

cat("\n================ MULTINOMIAL LOGISTIC REGRESSION ================\n")
print(conf_logit_multi)
cat(sprintf("Multinomial Logit Accuracy: %.2f%%\n", acc_logit_multi * 100))
# ====================== ANALYSIS OF THE RESULTS ===============================
# The probabilistic model performed similarly but slightly lower, with an accuracy of 55.49%. It correctly classified slightly more Draws (21) and Away wins (141), but was less accurate on Home wins (197) compared to the SVM.
# - SVMs and Logistic Regression often produce very comparable results in practice. The shared difficulty in predicting the "Draw" class (D) highlights a structural reality of football analytics. Draws are notoriously difficult to isolate using only in-game volume statistics (like shots and corners) because a 0-0 game looks statistically very different from a 2-2 game, making it hard for either algorithm to draw clear boundaries around that specific class.

# ==============================================================================
# SECTION 25: STANDARD NEURAL NETWORKS (DENSE/FEEDFORWARD) AND BASIC CLASSIFICATION
# ==============================================================================
# This code uses Keras to implement a feedforward neural network for the multiclass classification of the MNIST dataset. The 28x28 images are flattened into 784-element vectors and normalized, while the labels are converted using one-hot encoding. The model employs hidden layers with ReLU activation and a 10-class softmax output, and is trained by minimizing categorical cross-entropy.

# Load the required library
library(keras)

# 1. Load the MNIST dataset
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# 2. Data Preprocessing
# Reshape the 28x28 matrices into 784-dimensional vectors
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# Normalize pixel values to the range [0, 1]
x_train <- x_train / 255
x_test <- x_test / 255

# Convert class vectors to binary class matrices (One-hot encoding)
# Apply one-hot encoding to the labels. This converts numerical values ​​(from 0 to 9) into binary vectors of length 10, which are essential for multi-class classification.
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# 3. Model Definition
model <- keras_model_sequential() %>%
  # Create the first fully connected hidden layer of the network. Use 256 neurons and the ReLU activation function, which is essential for enabling the model to learn non-linear relationships.
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.4) %>% # Dropout for regularization
  
  # Second hidden layer with ReLU activation
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  
  # It defines the final (output) layer of the network. It has exactly 10 neurons (one for each possible digit) and uses the softmax function to convert the outputs into percentage probabilities that sum to 1.
  layer_dense(units = 10, activation = "softmax")

# Print the model summary
summary(model)
# ====================== ANALYSIS OF THE RESULTS ===============================
# This output confirms the network structure you defined. It shows the exact sequence of layers: an initial dense layer producing an output of 256, a dropout layer, a second dense layer with 128 units, another dropout layer, and a final output layer with 10 classes. The entry "Total params: 235146" indicates the number of weights (connections) that the neural network had to learn to calibrate in order to recognize the digits. The dropout layers show 0 parameters because their sole function is to randomly "switch off" certain connections to prevent overfitting, without learning anything new.

# 4. It configures the learning process. It defines the optimization algorithm (RMSprop) and the loss function (categorical_crossentropy), which is the standard for penalizing errors in multi-class classification problems.
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# 5. It performs the actual training. The model "observes" the training data in blocks of 128 images at a time (batches) and repeats the entire learning process over the full dataset 20 times (epochs), updating the weights to minimize the loss function.
history <- model %>% fit(
  x_train, y_train,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 1
)

# Plot the training history (loss and accuracy)
plot(history)
# ====================== ANALYSIS OF THE RESULTS ===============================
# This graph illustrates a very healthy and stable learning process over the course of 20 epochs. In the top panel, the loss function decreases steadily before stabilizing, while in the bottom panel, the accuracy rises from the very first epochs, approaching 100%. A crucial detail is that the blue curves—representing the validation data (data set aside during each epoch)—closely follow the red training curves. If the blue loss line had begun to rise again after a certain point, it would have indicated overfitting (meaning the model was simply memorizing the data); however, the dropout levels kept the curve flat and stable.

# 6. Model Evaluation on Test Data
metrics <- model %>% evaluate(x_test, y_test, verbose = 0)
cat("Test loss:", metrics["loss"], "\n")
cat("Test accuracy:", metrics["accuracy"], "\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# This is the final score obtained on the test set—that is, images that the network had never encountered during either training or validation. The result shows a test accuracy of 0.9813, meaning the model is able to correctly classify handwritten digits 98.13% of the time. Consequently, the test loss value—0.08497656—is very low, confirming that the errors made by the network are minimal and its predictions carry a high degree of confidence.


#------------------------------------------------------------------------------#
# 1. Carica il pacchetto reticulate per gestire Python
library(reticulate)

# 2. Scarica una versione di Python sicuramente compatibile (es. 3.10)
install_python(version = "3.10:latest")

# 3. Carica keras e lancia l'installazione forzando l'uso di Python 3.10
library(keras)
install_keras(method = "virtualenv", envname = "r-reticulate", python_version = "3.10")
#------------------------------------------------------------------------------#

# ==============================================================================
# SECTION 26: RECURRENT NEURAL NETWORKS (RNNS) AND TIME SERIES
# ==============================================================================
# This code uses Keras to implement a Recurrent Neural Network (RNN) with an LSTM (Long Short-Term Memory) architecture, tested on the classic IMDB sequential dataset. The sequences are first padded to a uniform length, after which an embedding layer transforms the discrete indices into continuous vectors. Finally, an LSTM layer processes the temporal order of the elements by storing past context, concluding with a single-probability output (via a sigmoid function) for binary classification.

# Load the required library
library(keras)

# 1. Load and prepare sequence data (e.g., IMDB movie reviews)
max_features <- 10000 # Top most frequent words to consider
maxlen <- 500         # Cut sequences after this many words

# Neural networks require fixed-size inputs, but sequences (such as texts or the historical progression of matches) have variable lengths. This function truncates sequences that are too long and pads those that are too short with zeros, ensuring they all conform exactly to a specified length.
imdb <- dataset_imdb(num_words = max_features)
x_train <- pad_sequences(imdb$train$x, maxlen = maxlen)
y_train <- imdb$train$y
x_test <- pad_sequences(imdb$test$x, maxlen = maxlen)
y_test <- imdb$test$y

# 2. Model Definition (RNN / LSTM Architecture)
model <- keras_model_sequential() %>%
  # Embedding layer: turns positive integers (indexes) into dense vectors of fixed size. Rather than passing raw categorical or numerical data directly to the network, this layer creates a 32-dimensional "map" (an embedding). It learns to group items with similar meanings or behaviors, drastically reducing computational complexity.
  layer_embedding(input_dim = max_features, output_dim = 32, input_length = maxlen) %>%
  
  # LSTM layer: recurrent layer that captures long-term dependencies in the sequence. Unlike dense layers (which look at the input all at once), an LSTM cell reads the sequence one step at a time, updating an "internal memory." It understands the chronological order of events and discards irrelevant information. The dropout arguments within it apply regularization directly to the recurrent calculations.
  layer_lstm(units = 32, dropout = 0.2, recurrent_dropout = 0.2) %>%
  
  # Output Layer: single neuron with Sigmoid for binary classification (0 or 1)
  layer_dense(units = 3, activation = "softmax")

# 3. Model Compilation
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = "rmsprop",
  metrics = c("accuracy")
)

# 4. Model Training
history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 1
)
# ====================== ANALYSIS OF THE RESULTS ===============================
# This graph highlights the classic overfitting problem typical of neural networks.
# - In the top panel, the light-blue training loss curve steadily decreases, which means the model is learning the data it has already seen very well. However, the green `val_loss` curve (the error on new data) stops decreasing around Epoch 4 and begins to rise erratically (reaching peaks of nearly 0.50 at Epoch 9). This indicates that, from that point on, the network is essentially "memorizing" the training data, losing its ability to generalize to previously unseen data.
# - The bottom panel confirms this divergence: while training accuracy (blue curve) rises to nearly 95%, validation accuracy (green curve) plateaus and fluctuates around 85–88%, showing no further improvement after the fourth epoch.



# 5. Model Evaluation
metrics <- model %>% evaluate(x_test, y_test, verbose = 0)
cat("Test loss:", metrics["loss"], "\n")
cat("Test accuracy:", metrics["accuracy"], "\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# Despite the overfitting, the model achieved a final accuracy on the test set (test accuracy) of 86.94%. This is a solid result, but the test loss value (0.3428) is relatively high, precisely due to the divergence observed in the graph


# ==============================================================================
# SECTION 27: TRAINING, OPTIMIZATION, AND REGULARIZATION
# ==============================================================================
# This code incorporates advanced techniques from Module 5 to combat overfitting—a critical issue when analyzing sports odds and statistics. It employs an architecture designed for tabular data (ideal for your app) and introduces L2 regularization on the weights, as well as Dropout. A key addition is the use of callbacks to implement Early Stopping, which automatically halts the training process the moment the model stops improving.

# 1. Load required libraries
library(keras)
library(dplyr)

# 2. Data Import and Cleaning
# Read the file specifying the correct separator (semicolon)
dataset <- read.csv("master_serie_a.csv", sep = ";", stringsAsFactors = FALSE)

# Select the Target (FTR) and a set of pre-match odds. 
# Exclude ALL post-match statistics to prevent data leakage.
model_data <- dataset %>%
  select(FTR, B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA) %>%
  na.omit() # Remove historical matches with missing odds

# 3. Convert Target to numeric format (H=0, D=1, A=2)
model_data$FTR <- case_when(
  model_data$FTR == "H" ~ 0,
  model_data$FTR == "D" ~ 1,
  model_data$FTR == "A" ~ 2
)

# 4. Create Matrices for Keras
# Separate features (X) from the final result (Y)
X <- model_data %>% select(-FTR) %>% as.matrix()
Y <- model_data$FTR

# Normalization (Standardization) of the odds
# Neural networks require features to be on a similar scale
X <- scale(X)

# 5. Train/Test Split (80% training, 20% app evaluation)
set.seed(42) # For reproducible results
train_index <- sample(1:nrow(X), 0.8 * nrow(X))

X_train <- X[train_index, ]
Y_train <- Y[train_index]
X_test  <- X[-train_index, ]
Y_test  <- Y[-train_index]

# 6. Early Stopping Configuration (Module 5)
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5, # Slightly increased patience for noisy data like betting odds
  restore_best_weights = TRUE
)

# 7. Network Architecture Definition (Module 1 + Module 5)
model <- keras_model_sequential() %>%
  # The first layer automatically infers the number of input columns (ncol(X_train))
  layer_dense(units = 32, activation = "relu", 
              kernel_regularizer = regularizer_l2(l = 0.005), 
              input_shape = ncol(X_train)) %>%
  layer_dropout(rate = 0.3) %>%
  
  layer_dense(units = 16, activation = "relu", 
              kernel_regularizer = regularizer_l2(l = 0.005)) %>%
  layer_dropout(rate = 0.2) %>%
  
  # Output layer for 3 classes (H, D, A)
  layer_dense(units = 3, activation = "softmax")

# 8. Model Compilation
model %>% compile(
  loss = "sparse_categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = c("accuracy")
)

# 9. Model Training
history <- model %>% fit(
  X_train, Y_train,
  epochs = 100, # Early Stopping will interrupt before reaching 100
  batch_size = 32,
  validation_split = 0.2,
  callbacks = list(early_stop),
  verbose = 1
)
# ====================== ANALYSIS OF THE RESULTS ===============================
# - The graph shows that training stopped before the planned 100 epochs (at around epoch 60). This is the result of Early Stopping, which triggered correctly because the `val_loss` (the green curve at the top) was no longer improving.
# - Unlike the previous example, there is no sign of overfitting here. The green validation loss curve descends in parallel with the blue training loss curve and never rises again. Accuracy (in the bottom panel) stabilizes for both curves at around 53–55% very quickly, remaining flat. This pattern—combined with the absence of overfitting—is known as underfitting or saturation: the neural network is sound and well-regularized, but the bookmakers' pre-match odds, on their own, do not contain enough "mathematical signals" to predict the outcome of a match with an accuracy higher than this threshold.

# 10. Final Evaluation on unseen matches
final_metrics <- model %>% evaluate(X_test, Y_test)
cat("App Model Accuracy:", final_metrics["accuracy"] * 100, "%\n")
# ====================== ANALYSIS OF THE RESULTS ===============================
# In a football context (with three possible outcomes: 1, X, 2), an accuracy rate of 53% is considered realistic. Guessing at random would yield a success rate of about 33%, while bookmakers themselves rarely exceed 55–58% accuracy, due to the highly unpredictable nature of the sport.



# --- ESPORTAZIONE PER LA DASHBOARD SHINY ---

# 1. Dataset pulito
saveRDS(master_df, "dataset_serieA.rds")

# 2. Modelli Predittivi
saveRDS(fit_poisson, "modello_poisson.rds")
saveRDS(fit_multi,"modello_multimoniale.rds")
saveRDS(cv_lasso, "modello_lasso.rds")
saveRDS(fit_gam_3, "modello_gam.rds")
saveRDS(rf_model, "modello_rf.rds")

# 3. Altri output (es. risultati bootstrap per i grafici)
saveRDS(boot_results, "dati_bootstrap.rds")