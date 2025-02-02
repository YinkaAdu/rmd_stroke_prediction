---
title: "Build and deploy a stroke prediction model using R"
date: "02/02/2025"
output: html_document
author: "Olayinka Adu"
---

# About Data Analysis Report

This RMarkdown file contains the report of the data analysis done for the project on building and deploying a stroke prediction model in R. It contains analysis such as data exploration, summary statistics and building the prediction models. The final report was completed on 02/02/2025. 

**Data Description:**

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.

This data set is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient. The dataset is comprised of 5,110 observations of demographic data and health indicators. Each row in the dataset represents a patient and each column represents relevant information about patients. The information should make it possible to explore the probability of stroke incidence in the population.
I realized that some variables are more relevant to the purpose of identifying the risk of stroke. As such, I have selected the following:

 gender: (Male, Female, Other)
 age: age of participant at the time of observation.
 avg_glucose_level: average glucose level
 bmi: Body Mass Index
 hypertension: does the participant have a history of hypertension? (true (1)/false(0))
 heart_disease: does the participant have a history of heart disease? (true (1)/false(0))
 work_type: type of work (private, govt_job, self-employed, never_worked, children)
 smoking_status: history of smoking (formerly smoked, never_smoked, unknown, smokes)
 stroke: has the participant suffered a stroke (true(1)/false(0))

# Task One: Import data and data preprocessing.

## Install Packages
```{r}
install.packages("caret") 
install.packages("tidymodels")
install.packages("statsr")
install.packages("janitor")
install.packages("tidyverse")
install.packages("gridExtra")
install.packages("corrplot")
install.packages("ggExtra")
install.packages("shinydashboard")
```

## Load data and libraries
```{r}
library(workflows)
library(tune)
library(caret)
library(tidyr)
library(janitor)
library(tidyverse)
library(gridExtra)
library(corrplot)
library(pROC)

str_dataset <- read.csv ("healthcare-dataset-stroke-data.csv")
```

## Data Preprocessing

### Check how the values look like
```{r}
head(str_dataset)
```
![image](https://github.com/user-attachments/assets/9484b2ce-c271-4abd-a3f4-d63b6393a1dc)

### Select variables of interest
```{r}
selected_dataset<- select(str_dataset, -ever_married, -Residence_type, -id)
```

### Remove any possible duplicate rows
```{r}
stroke_unique <- selected_dataset %>% distinct(.keep_all=TRUE)
```

### Summary of the dataset
```{r}
summary(stroke_unique)
```
![image](https://github.com/user-attachments/assets/98612a89-5c31-49e9-9e16-f07767231b28)

### Check for misspelt characters that could increase number of categories
```{r}
unique_genders<- unique(stroke_unique$gender)
unique_hypertension <- unique(stroke_unique$hypertension)
unique_work_type <- unique(stroke_unique$work_type)
unique_heart_disease <- unique(stroke_unique$heart_disease)
unique_smoking_status <- unique(stroke_unique$smoking_status)
unique_stroke <- unique(stroke_unique$stroke)
```

### Change datatypes

Some variables appear to have been allocated the wrong data type. For instance bmi (body mass index) should be numeric but I found that the data type was character.

```{r}
stroke_unique$bmi <- as.numeric(stroke_unique$bmi) 
stroke_unique$hypertension <- as.logical(stroke_unique$hypertension != 0) 
stroke_unique$heart_disease <- as.logical(stroke_unique$heart_disease != 0) 
stroke_unique$stroke <- as.logical(stroke_unique$stroke != 0) 

# Edit nominal data
stroke_unique <- stroke_unique %>%
  mutate(work_type = recode(work_type,
                           "Self-employed" = "self_employed",
                           "Never_worked" = "never_worked",
                           "Govt_job" = "govt_job",
                            "Private" = "private")) %>%
  mutate(smoking_status = recode(smoking_status,
                                "formerly smoked" = "formerly_smoked",
                                "never smoked" = "never_smoked",
                                "Unknown" = "unknown"))


```

### Replace missing data with NA

```{r}
# Replace 0, "N/A", and existing NAs with NA in selected columns
stroke_dataset <- stroke_unique %>% 
  mutate_at(vars(avg_glucose_level, age, bmi, gender, work_type, smoking_status), 
            ~ ifelse(is.na(.) | . == 0 | . == "N/A", NA, .))

```



## Exploratory Data Analysis

Stroke is the dependent variable. Now, we explore the rate of stroke incidence by nominal variables (gender, work_type, smoking_status), logical variables (heart_disease, hypertension) and numeric variables (age, bmi, avg_glucose_level).

### 1a. Stacked Bar plots for Bivariate Distribution of Nominal variables
```{r}
p1 <- ggplot(data = stroke_dataset) +
  geom_bar(mapping = aes(x = gender, fill = factor(stroke))) + 
  ggtitle('Gender') +
  scale_fill_manual(values = c("skyblue", "maroon"), name = "Stroke Incidence")  # Custom colors

p2 <- ggplot(data = stroke_dataset) +
  geom_bar(mapping = aes(x = smoking_status, fill = factor(stroke))) + 
  ggtitle('Smoking Status') +
  scale_fill_manual(values = c("skyblue", "maroon"), name = "Stroke Incidence")

p3 <- ggplot(data = stroke_dataset) +
  geom_bar(mapping = aes(x = work_type, fill = factor(stroke))) + 
  ggtitle('Work Type') +
  scale_fill_manual(values = c("skyblue", "maroon"), name = "Stroke Incidence")

grid.arrange(p1, p2, p3, ncol=1)

# Cross-tabulation for more specific detail

t1 <- stroke_dataset %>%
  tabyl(gender, stroke) %>%  # Create a table of gender vs stroke
  adorn_totals("row") %>%    # Add row totals
  adorn_percentages("row") %>%  # Convert counts to percentages
  adorn_pct_formatting(digits = 2)  # Format as percentages

t2 <- stroke_dataset %>%
  tabyl(smoking_status, stroke) %>%  # Create a table of gender vs stroke
  adorn_totals("row") %>%    # Add row totals
  adorn_percentages("row") %>%  # Convert counts to percentages
  adorn_pct_formatting(digits = 2)  # Format as percentages

t3 <- stroke_dataset %>%
  tabyl(work_type, stroke) %>%  # Create a table of gender vs stroke
  adorn_totals("row") %>%    # Add row totals
  adorn_percentages("row") %>%  # Convert counts to percentages
  adorn_pct_formatting(digits = 2)  # Format as percentages


print(t1)
print(t2)
print(t3)
```
![image](https://github.com/user-attachments/assets/1a711ad3-4365-458d-bebb-96ab10234c93)
![image](https://github.com/user-attachments/assets/31c84cb1-417a-4b8f-bca2-7480d0650179)

### 1b. Compare work_type to smoking status 

```{r}

p4 <- ggplot(data = stroke_dataset) +
  geom_bar(mapping = aes(x = work_type, fill = factor(smoking_status))) + 
  ggtitle('Work Type') +
  scale_fill_manual(values = c("skyblue", "blue", "maroon", "pink"), name = "Smoking Status")

grid.arrange(p4, ncol=1)

t4 <- stroke_dataset %>%
  tabyl(work_type, smoking_status) %>%  # Create a table of gender vs stroke
  adorn_totals("row") %>%    # Add row totals
  adorn_percentages("row") %>%  # Convert counts to percentages
  adorn_pct_formatting(digits = 2)  # Format as percentages

print(t4)
```
![image](https://github.com/user-attachments/assets/221253d0-f9ab-4d88-8388-df32a30dd84b)
![image](https://github.com/user-attachments/assets/3e1eadf1-634f-4e64-9940-dcfb71ff40f5)

### 1c. Stacked Bar plots for Bivariate Distribution of logical variables
```{r}
p5 <- ggplot(data = stroke_dataset) +
  geom_bar(mapping = aes(x =hypertension, fill = factor(stroke))) + 
  ggtitle('Hypertension') +
  scale_fill_manual(values = c("skyblue", "maroon"), name = "Stroke Incidence")  # Custom colors


p6 <- ggplot(data = stroke_dataset) +
  geom_bar(mapping = aes(x = heart_disease, fill = factor(stroke))) + 
  ggtitle('Heart Disease') +
  scale_fill_manual(values = c("skyblue", "maroon"), name = "Stroke Incidence")  # Custom colors


grid.arrange(p5, p6, ncol=1)

t5 <- stroke_dataset %>%
  tabyl(hypertension, stroke) %>%  # Create a table of gender vs stroke
  adorn_totals("row") %>%    # Add row totals
  adorn_percentages("row") %>%  # Convert counts to percentages
  adorn_pct_formatting(digits = 2)  # Format as percentages

t6 <- stroke_dataset %>%
  tabyl(heart_disease, stroke) %>%  # Create a table of gender vs stroke
  adorn_totals("row") %>%    # Add row totals
  adorn_percentages("row") %>%  # Convert counts to percentages
  adorn_pct_formatting(digits = 2)  # Format as percentages

print(t5)

print(t6)
```
![image](https://github.com/user-attachments/assets/844ccee3-90ed-44d1-a20c-5cf202fcdf91)
![image](https://github.com/user-attachments/assets/5c959d51-4815-4022-9ef9-6365747a3185)

### 1d. Stacked Histograms for Bivariate Distribution of Numeric Variables

```{r}
p8 <- ggplot(data = stroke_dataset) +
  geom_histogram(mapping = aes(x = bmi, fill = factor(stroke)), 
                 color = "black", binwidth = 2, position = "stack") +
  ggtitle("Body Mass Index (BMI) Distribution by Stroke") +
  labs(x = "BMI", y = "Count", fill = "Stroke") +
  scale_fill_manual(values = c("skyblue", "maroon"), labels = c("False", "True")) +
  theme_minimal()

p9 <- ggplot(data = stroke_dataset) +
  geom_histogram(mapping = aes(x = avg_glucose_level, fill = factor(stroke)), 
                 color = "black", binwidth = 4, position = "stack") +
  ggtitle("Average Glucose Level Distribution by Stroke") +
  labs(x = "BMI", y = "Count", fill = "Stroke") +
  scale_fill_manual(values = c("skyblue", "maroon"), labels = c("False", "True")) +
  theme_minimal()

p10 <- ggplot(data = stroke_dataset) +
  geom_histogram(mapping = aes(x = age, fill = factor(stroke)), 
                 color = "black", binwidth = 2, position = "stack") +
  ggtitle("Age Distribution by Stroke") +
  labs(x = "BMI", y = "Count", fill = "Stroke") +
  scale_fill_manual(values = c("skyblue", "maroon"), labels = c("False", "True")) +
  theme_minimal()

grid.arrange(p8, p9, p10, ncol=1)

```
![image](https://github.com/user-attachments/assets/5f946f75-c93d-451e-a372-858805a4016e)

### Treat missing values in numeric variables
I will be filling up all missing bmi, age and avg_glucose_level with the mean of non-missing values in these variables

```{r}

stroke_treat_dataset <- stroke_dataset %>%
  mutate(age = ifelse(is.na(age), mean(age, na.rm = TRUE), age)) %>%
  mutate(bmi = ifelse(is.na(bmi), mean(bmi, na.rm = TRUE), bmi)) %>%
  mutate(avg_glucose_level = ifelse(is.na(avg_glucose_level), mean(avg_glucose_level, na.rm = TRUE), avg_glucose_level))

```

### 1e. Box Plot assessing the relationship between work_type, Stroke Incidence and avg_glucose_level

Stroke Incidence is generally higher for patients who had higher Average Glucose Level. I will assess the impact of excluding vs including the children data on model performance because children show a contrary relationship when compared with other Work Type categories.
```{r}

set.seed(123)

ggplot(stroke_treat_dataset, aes(x =work_type , y = avg_glucose_level, fill = stroke)) +
  geom_boxplot() +
  labs(title = "Box Plot of Average Glucose Level by Work Type and Stroke Incidence",
       x = "Work Type",
       y = "Average Glucose Level") +
  theme_bw()

```
![image](https://github.com/user-attachments/assets/e55787b9-b954-441e-b6ad-c5df065949af)

### 1f. Box Plot assessing the relationship between smoking_status, stroke and age
There appears to be a positive relatioship between age and stroke incidence as well. Smokers also have the least average age of smoke incidence
```{r}

set.seed(456)

ggplot(stroke_treat_dataset, aes(x = smoking_status, y = age, fill = stroke)) +
  geom_boxplot() +
  labs(title = "Box Plot of Age by Smoking Status and Stroke Incidence",
       x = "Smoking Status",
       y = "Age") +
  theme_bw()  
```
![image](https://github.com/user-attachments/assets/ad7aa07c-403b-4ea9-9491-275c62c29a04)

### 1g. Scatterplot showing the distribution of avg_glucose_level by age and stroke incidence
```{r}
set.seed(789)

# Create the scatter plot using ggplot2
library(ggplot2)

ggplot(stroke_treat_dataset, aes(x = age, y = avg_glucose_level, color = stroke)) +
  geom_point() +  # Use geom_point for scatter plot
  labs(title = "Scatter Plot of Age vs. Average Glucose Level",
       x = "Age",
       y = "Average Glucose Level",
       color = "Stroke") +  # Label the color legend
  theme_bw()  # Optional: Use a black and white theme

```
![image](https://github.com/user-attachments/assets/9d1be40d-c510-4fe3-9076-0390f7e639d9)

**Regression analysis:** age, bmi, and avg_glucose_levels all have reasonable broad distribution, therefore, they will be considered for the regression analysis. As the outcome variable, stroke, is a logical variable, a logistic regression model would be considered.

### One-hot Encode Logical Variables and Nominal Variable in the train data
```{r}

# Define the variables to be one-hot encoded
categorical_vars <- c('gender', 'hypertension', 'heart_disease', 'work_type', 'smoking_status', 'stroke')

# Subset the dataset to include only the specified categorical variables
treat_subset <- stroke_treat_dataset[, categorical_vars]

# Ensure that the variables are treated as factors
treat_subset[] <- lapply(treat_subset, as.factor)

# Define the one-hot encoding model
dummies_model <- dummyVars(~ ., data = treat_subset)

# Apply the model to the data to create the one-hot encoded variables
one_hot_encoded_data <- predict(dummies_model, newdata = treat_subset)

# Convert the result to a data frame
one_hot_encoded_df <- as.data.frame(one_hot_encoded_data)

# If you want to combine the one-hot encoded variables with the original dataset
# (excluding the original categorical variables), you can do so as follows:
stroke_dataset_cd <- cbind(stroke_treat_dataset[, !names(stroke_treat_dataset) %in% categorical_vars], one_hot_encoded_df)
```

### Recreate the factor for the outcome variable from one-hot encoded columns
```{r}

stroke_dataset_cd$stroke <- apply(stroke_dataset_cd[, c("stroke.FALSE", "stroke.TRUE")], 1, function(row) {
  which(row == 1) - 1
})

stroke_dataset_cd <- stroke_dataset_cd %>%
  mutate(stroke = as.factor(stroke))
```

### Oversample Stroke Patients to Address Class Imbalance
```{r}

# Apply MWMOTE (or your preferred oversampling method)
new_stroke_data <- mwmote(stroke_dataset_cd, classAttr = "stroke", numInstances = 4000)

# Combine with original data
balanced_data <- rbind(stroke_dataset_cd, new_stroke_data)

# Check the class distribution
table(balanced_data$stroke)

```

# Task Two and Task Three: Build prediction models and Evaluate and select prediction models 

First model including the children
Second model excluding the children 

## Splitting the data into train and test sets
```{r}
set.seed(123)

library(rsample)

index <- sample(1:nrow(balanced_data), 0.8 * nrow(balanced_data)) # 80% for training
stroke_train <- balanced_data[index, ]
stroke_test <- balanced_data[-index, ]
```
### Extract train and test for reproducibility
```{r}
summary(stroke_test)
```
![image](https://github.com/user-attachments/assets/71fe7cab-c253-4690-81e7-75ee449281ba)

```{r}
summary(stroke_train)
```
![image](https://github.com/user-attachments/assets/d8fe34de-d9c8-43b4-b7c1-422de71c4c00)

### Create Cross Validation object from training data
```{r}
stroke_cv <- vfold_cv(stroke_train)
metric <- "Accuracy"
```

## Correlation between numerical variables

```{r}

library(recipes)

# Define the variables to select
selected_vars <- names(stroke_train) %in% c("age", "bmi", "avg_glucose_level") 

# Extract the numeric variables from the recipe
selected_train <-stroke_train[selected_vars]
selected_train <- as.matrix(selected_train)

numeric_train <- selected_train[sapply(selected_train, is.numeric)]

# dplyr (if you are using the dplyr package)
mar = c(0, 0, 5, 0)
corr.matrix <- cor(selected_train, use = "complete.obs")
corrplot(corr.matrix, main="\n\nCorrelation Plot of Numeric Variables", method="number", mar=mar)
```
![image](https://github.com/user-attachments/assets/d28e29a3-d06f-4d3e-ad9b-8852e94ea6aa)

### Visualise Data Structure
```{r}
str(stroke_train)
```
![image](https://github.com/user-attachments/assets/dfa4cf49-fb65-41d8-9e53-636c33463f1d)

```{r}
print(colnames(stroke_train))
```
![image](https://github.com/user-attachments/assets/0dd796e3-2f20-4305-85ad-99b8aa250b25)

## Logistic Regression Model (logistic regression version 1)
```{r}
logit_model <- glm(formula = stroke ~ avg_glucose_level + age + bmi + gender.Male + smoking_status.unknown + smoking_status.smokes + smoking_status.formerly_smoked + work_type.private + work_type.govt_job + work_type.self_employed + work_type.children + heart_disease.TRUE + hypertension.TRUE, 
                   data = stroke_train, 
                   family = binomial())

# View model summary
summary(logit_model)

# Make predictions (probabilities) for the training data
pred_prob1 <- predict(logit_model, newdata = stroke_test, type = "response")

# Convert probabilities to binary outcomes
pred_outcome1 <- ifelse(pred_prob1 > 0.5, 1, 0)

# Display the first few predicted outcomes
head(pred_outcome1)

# View the odds ratios for each coefficient
exp(coef(logit_model))

```
![image](https://github.com/user-attachments/assets/9b6c0fba-9bc6-4b13-be7e-e232a9b6f3bb)

## Logistic Regression Model (logistic regression version 2)
Picked variables identified by significant p-values.
```{r}
logit_model_filtered <- glm(formula = stroke ~ age + avg_glucose_level + bmi + smoking_status.unknown + smoking_status.smokes + smoking_status.formerly_smoked + heart_disease.TRUE + hypertension.TRUE, 
                   data = stroke_train, 
                   family = binomial())

# View model summary
summary(logit_model_filtered)

# Make predictions (probabilities) for the training data
pred_prob2 <- predict(logit_model_filtered, newdata = stroke_test, type = "response")

# Convert probabilities to binary outcomes
pred_outcome2 <- ifelse(pred_prob2 > 0.5, 1, 0)

# Display the first few predicted outcomes
head(pred_outcome2)


confusionMatrix(factor(pred_outcome2), factor(stroke_test$stroke))

# View the odds ratios for each coefficient
exp(coef(logit_model_filtered))

# Create the ROC curve

roc_obj2 <- roc(stroke_test$stroke, pred_prob2) # Create ROC object

# Plot the ROC curve
plot(roc_obj2,
     main = "ROC Curve",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     col = "skyblue",
     lwd = 2,
     print.auc = TRUE, # Print AUC on the plot
     legacy.axes=TRUE) # Use traditional axes for better visualization

# Calculate and display AUC (Area Under the Curve)

auc2 <- auc(roc_obj2)
print(paste("AUC:", auc2))
```
![image](https://github.com/user-attachments/assets/d7f4aa3a-2abf-4c64-91e8-f1ce717e2471)

# Build Prediction model excluding children category

### Filter out rows that fall in 'children' category
```{r}

# Filter out rows where work_type is 'children'
stroke_no_kids <- stroke_treat_dataset %>%
  filter(work_type != "children")

# Check the dataset after filtering
head(stroke_no_kids)
```
![image](https://github.com/user-attachments/assets/03b9ddbc-f188-4b62-92ff-2c7307f34c73)


### 3a. Box Plot assessing the relationship between work_type, Stroke Incidence and avg_glucose_level

```{r}

set.seed(123)

ggplot(stroke_no_kids, aes(x =work_type , y = avg_glucose_level, fill = stroke)) +
  geom_boxplot() +
  labs(title = "Box Plot of Average Glucose Level by Work Type and Stroke Incidence",
       x = "Work Type",
       y = "Average Glucose Level") +
  theme_bw()

```
![image](https://github.com/user-attachments/assets/f3a73dfb-b4df-4b59-9a68-fa309a8fdd29)

### One-hot Encode Logical Variables and Nominal Variable
```{r}

# Define the variables to be one-hot encoded
categorical_vars_nk <- c('gender', 'hypertension', 'heart_disease', 'work_type', 'smoking_status', 'stroke')

# Subset the dataset to include only the specified categorical variables
train_subset_nk <- stroke_no_kids[, categorical_vars_nk]

# Ensure that the variables are treated as factors
train_subset_nk[] <- lapply(train_subset_nk, as.factor)

# Define the one-hot encoding model
dummies_model_nk <- dummyVars(~ ., data = train_subset_nk)

# Apply the model to the data to create the one-hot encoded variables
one_hot_encoded_data_nk <- predict(dummies_model_nk, newdata = train_subset_nk)

# Convert the result to a data frame
one_hot_encoded_nk <- as.data.frame(one_hot_encoded_data_nk)

# If you want to combine the one-hot encoded variables with the original dataset
# (excluding the original categorical variables), you can do so as follows:
stroke_dataset_nk <- cbind(stroke_no_kids[, !names(stroke_no_kids) %in% categorical_vars_nk], one_hot_encoded_nk)
```

### Recreate the factor for the outcome variable from one-hot encoded columns
```{r}

stroke_dataset_nk$stroke <- apply(stroke_dataset_nk[, c("stroke.FALSE", "stroke.TRUE")], 1, function(row) {
  which(row == 1) - 1
})

stroke_dataset_nk <- stroke_dataset_nk %>%
  mutate(stroke = as.factor(stroke))
```

### Oversample Stroke Patients to Address Class Imbalance
```{r}
# Apply MWMOTE (or your preferred oversampling method)
new_stroke_nk <- mwmote(stroke_dataset_nk, classAttr = "stroke", numInstances = 4000)

# Combine with original data
balanced_data_nk <- rbind(stroke_dataset_nk, new_stroke_nk)

# Check the class distribution
table(balanced_data_nk$stroke)

```
## Splitting the data into train and test sets
```{r}
set.seed(123)


index <- sample(1:nrow(balanced_data_nk), 0.8 * nrow(balanced_data_nk)) # 80% for training
stroke_train_nk <- balanced_data_nk[index, ]
stroke_test_nk <- balanced_data_nk[-index, ]
```
### Extract train and test for reproducibility
```{r}

summary(stroke_test_nk)
```
![image](https://github.com/user-attachments/assets/4f314ed0-9df3-42a7-aa1e-e92c4a726916)

```{r}
summary(stroke_train_nk)
```
![image](https://github.com/user-attachments/assets/d6062b6b-294c-40d9-b5ee-28d07a0ba8ad)

### Create Cross Validation object from training data
```{r}
stroke_nk_cv <- vfold_cv(stroke_train_nk)
```

## Correlation between numerical variables after excluding children
```{r}

library(recipes)

# Define the variables to select
selected_vars_nk <- names(stroke_train_nk) %in% c("age", "bmi", "avg_glucose_level") 

# Extract the numeric variables from the recipe
selected_train_nk <-stroke_train_nk[selected_vars_nk]
selected_train_nk <- as.matrix(selected_train_nk)

numeric_train_nk <- selected_train_nk[sapply(selected_train_nk, is.numeric)]

# dplyr (if you are using the dplyr package)
mar = c(0, 0, 5, 0)
corr.matrix <- cor(selected_train_nk, use = "complete.obs")
corrplot(corr.matrix, main="\n\nCorrelation Plot of Numeric Variables", method="number", mar=mar)

```
![image](https://github.com/user-attachments/assets/254a73a5-344f-4c18-80e3-d75179b139dd)

### Structure of Balanced Data after excluding children
```{r}
str(balanced_data_nk)
```
![image](https://github.com/user-attachments/assets/450f5d10-1d12-4579-962f-f4d2e9ba50f2)

```{r}
print(colnames(balanced_data_nk))
```
![image](https://github.com/user-attachments/assets/71fe0615-be4f-430b-b42d-ba59728d5334)

## Logistic Regression Model (version 3)
```{r}
logit_model_nk <- glm(formula = stroke ~ avg_glucose_level + age + bmi + gender.Male + smoking_status.unknown + smoking_status.smokes + smoking_status.formerly_smoked + work_type.private + work_type.govt_job + work_type.self_employed + heart_disease.TRUE + hypertension.TRUE, 
                   data = stroke_train_nk, 
                   family = binomial())

# View model summary
summary(logit_model_nk)

# Make predictions (probabilities) for the training data
pred_prob3 <- predict(logit_model_nk, new_data = stroke_test_nk, type = "response")

# Convert probabilities to binary outcomes
pred_outcome3 <- ifelse(pred_prob3 > 0.5, 1, 0)

# Display the first few predicted outcomes
head(pred_outcome3)

# View the odds ratios for each coefficient
exp(coef(logit_model_nk))
```
![image](https://github.com/user-attachments/assets/4e6a345a-86aa-4430-92ec-d6f6f9e517f2)

## Logistic Regression Model (version 4)
Picked variables identified by significant p-values.
```{r}
logit_model_filtered_nk <- glm(formula = stroke ~ age + avg_glucose_level + bmi + smoking_status.unknown + smoking_status.smokes + smoking_status.formerly_smoked + heart_disease.TRUE + hypertension.TRUE, 
                   data = stroke_train_nk, 
                   family = binomial())

# View model summary
summary(logit_model_filtered_nk)

# Make predictions (probabilities) for the training data
pred_prob4 <- predict(logit_model_filtered_nk, newdata = stroke_test_nk, type = "response")

# Convert probabilities to binary outcomes
pred_outcome4 <- ifelse(pred_prob4 > 0.5, 1, 0)

# Display the first few predicted outcomes
head(pred_outcome4)


confusionMatrix(factor(pred_outcome4), factor(stroke_test_nk$stroke))

# View the odds ratios for each coefficient
exp(coef(logit_model_filtered_nk))

# Create the ROC curve

roc_obj4 <- roc(stroke_test_nk$stroke, pred_prob4) # Create ROC object

# Plot the ROC curve
plot(roc_obj4,
     main = "ROC Curve",
     xlab = "False Positive Rate (1 - Specificity)",
     ylab = "True Positive Rate (Sensitivity)",
     col = "skyblue",
     lwd = 2,
     print.auc = TRUE, # Print AUC on the plot
     legacy.axes=TRUE) # Use traditional axes for better visualization

# Calculate and display AUC (Area Under the Curve)

auc4 <- auc(roc_obj4)
print(paste("AUC:", auc4))
```
![image](https://github.com/user-attachments/assets/5bc3fb21-278b-40a7-b43a-560d065b4af6)

I now have two logistic regression models and have chosen to keep the 'children' category as the model with this category performed better in terms of balanced accuracy, AUC Score, and sensitivity.

# Findings and Conclusions
The model above demonstrates that it is possible to predict the risk of stroke, as measured by the incidence of stroke with six predictors - Age, Body Mass Index(BMI), Average Glucose Level, Smoking Status and Disease History (Hypertension and Heart Disease). The healthcare sector can use the similar methods when advising patients. 
However, the 'stroke' variable was highly imbalanced, with under 10% of the data representing positive cases. Second, the heart disease variable was limited to a binary representation. I believe that greater detail regarding specific heart disease types (e.g., atherosclerosis, CPVT) would have enhanced the analysis.

