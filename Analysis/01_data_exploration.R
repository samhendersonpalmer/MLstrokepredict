# Load packages and data --------------------------------------------------

# Load packages
library(tidyverse)

# Read in cleaned data
stroke_complete <- readRDS("Data/stroke_complete.RDS")

# rename factor levels for reporting
levels(stroke_complete$hypertension) <- c("No", "Yes")
levels(stroke_complete$heart_disease) <- c("No", "Yes")
levels(stroke_complete$work_type) <- c("Children", "Government", "Private", "Self-employed")
levels(stroke_complete$smoking_status) <- c("Formerly smoked", "Never smoked", "Smokes", "Unknown")


# Summary -----------------------------------------------------------------

# See summary of all variables
str(stroke_complete)
summary(stroke_complete)
skimr::skim(stroke_complete)

# A lot of younger age groups with indication they have jobs/smoke etc
stroke_complete %>% 
  filter(age < 18) %>% 
  summary()

# So we have some patients under 18 that have government, private or self_employed jobs
# and smoke or formerly smoked which seems unlikely but we don't have context for which
# country this data is from so inclined to leave it (even have 1 self-employed under 10
# and youngest working for gov is 14)

# Univariate --------------------------------------------------------------

# Continuous
# Check distribution of bmi
ggplot(stroke_complete, aes(x = bmi)) +
  geom_histogram(colour = "black", fill = "cornflowerblue") +
  labs(x = expression(BMI~kg/m^{2}), y = "Observations") +
  theme_minimal()  

# Check distribution of age
ggplot(stroke_complete, aes(x = age)) +
  geom_histogram(colour = "black", fill = "cornflowerblue") +
  labs(x = "Age (Years)", y = "Observations") +
  theme_minimal()

# Check distribution of avg glucose level
ggplot(stroke_complete, aes(x = avg_glucose_level)) +
  geom_histogram(colour = "black", fill = "cornflowerblue") +
  labs(x = "Average glucose level (mg/dL)", y = "Observations") +
  theme_minimal()

# Categorical
# Check distribution of stroke outcome
ggplot(stroke_complete, aes(x = stroke)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(y = "Observations") +
  theme_minimal()

# Check distribution of gender
ggplot(stroke_complete, aes(x = gender)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(x = "Gender", y = "Observations") +
  theme_minimal()

# Check distribution of hypertension
ggplot(stroke_complete, aes(x = hypertension)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(x = "Hypertension", y = "Observations") +
  theme_minimal()

# Check distribution of heart_disease
ggplot(stroke_complete, aes(x = heart_disease)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(x = "Heart disease", y = "Observations") +
  theme_minimal()

# Check distribution of ever_married
ggplot(stroke_complete, aes(x = ever_married)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(x = "Ever married", y = "Observations") +
  theme_minimal()

# Check distribution of work_type
ggplot(stroke_complete, aes(x = work_type)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(x = "Work type", y = "Observations") +
  theme_minimal()

# Check distribution of Residence_type
ggplot(stroke_complete, aes(x = Residence_type)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(x = "Residence type", y = "Observations") +
  theme_minimal()

# Check distribution of smoking_status
ggplot(stroke_complete, aes(x = smoking_status)) +
  geom_bar(colour = "black", fill = "cornflowerblue") +
  labs(x = "Smoking status", y = "Observations") +
  theme_minimal()


# Summary table -----------------------------------------------------------
library(crosstable)
ct1 <- crosstable(stroke_complete, c(gender, age, hypertension, heart_disease, 
                                     ever_married, work_type, Residence_type,
                                     avg_glucose_level, bmi, smoking_status), by=stroke, total="row", 
                  percent_pattern="{n} ({p_col})", percent_digits=1) %>%
  select(Variable = label, Level = variable, Stroke, `No stroke`, Total) %>% 
  mutate(Variable = replace(Variable, Variable == "gender", "Gender"),
         Variable = replace(Variable, Variable == "age", "Age"),
         Variable = replace(Variable, Variable == "hypertension", "Hypertension"),
         Variable = replace(Variable, Variable == "heart_disease", "Heart disease"),
         Variable = replace(Variable, Variable == "ever_married", "Ever married"),
         Variable = replace(Variable, Variable == "work_type", "Work type"),
         Variable = replace(Variable, Variable == "Residence_type", "Residence type"),
         Variable = replace(Variable, Variable == "avg_glucose_level", "Avg. glucose level"),
         Variable = replace(Variable, Variable == "bmi", "BMI"),
         Variable = replace(Variable, Variable == "smoking_status", "Smoking status"))

ct1

# Transformation ----------------------------------------------------------
library(tidymodels)

# Demonstrate BoxCox transformation
# create recipe
rec <- recipe(stroke ~., data = stroke_complete)

# Create boxcox step
stroke_trans <- step_BoxCox(rec, all_numeric())

# Use BoxCox on data
stroke_estimates <- prep(stroke_trans, training = stroke_complete)

# Create transformed data frame for plots
transformed_data <- bake(stroke_estimates, stroke_complete)

# See values of lambda used
tidy(stroke_estimates, number = 1)

# age                0.840 
# avg_glucose_level -1.06  
# bmi                0.368

# Check transformed distribution of bmi
ggplot(transformed_data, aes(x = bmi)) +
  geom_histogram(colour = "black", fill = "cornflowerblue") +
  labs(x = expression((BMI~(kg/m^{2}))^{0.368}), y = "Observations") +
  theme_minimal()  

# Check transformed distribution of age
ggplot(transformed_data, aes(x = age)) +
  geom_histogram(colour = "black", fill = "cornflowerblue") +
  labs(x = expression((Age~(Years))^{0.840}), y = "Observations") +
  theme_minimal()

# Check transformed distribution of avg glucose level
ggplot(transformed_data, aes(x = avg_glucose_level)) +
  geom_histogram(colour = "black", fill = "cornflowerblue") +
  labs(x = expression((Average~glucose~level~(mg/dL))^{-1.06}), y = "Observations") +
  theme_minimal()

# Bivariate and multivariate ----------------------------------------------

## Stacked bar charts ------------------------------------------------------
stroke_complete %>% 
  select(stroke, gender,hypertension, heart_disease, ever_married, work_type,
         Residence_type, smoking_status) %>% 
  pivot_longer(gender:smoking_status) %>% 
  ggplot(aes(y = value, fill = stroke)) +
  geom_bar(position = "fill") +
  facet_wrap(vars(name), scales = "free") +
  labs(x = NULL, y = NULL, fill = NULL) +
  theme_minimal()+ 
  theme(
    strip.background = element_blank(),
  #  strip.text.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank()) +
  scale_x_continuous(labels = scales::percent)

# From this looks like hypertension, heart disease, ever married, work type and smoking status
# Can explain binary ones but visualise multi choice ones
stroke_complete %>% 
  select(stroke, work_type, smoking_status) %>% 
  pivot_longer(work_type:smoking_status) %>%
  mutate(name = recode(name, 
                       "work_type" = "Work type",
                       "smoking_status" = "Smoking status")) %>% 
  ggplot(aes(y = value, fill = stroke)) +
  geom_bar(position = "fill") +
  facet_wrap(vars(name), scales = "free") +
  labs(x = NULL, y = NULL, fill = "Outcome") +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank()) +
  scale_x_continuous(labels = scales::percent)

## Boxplots ----------------------------------------------------------------

# age and gender
ggplot(stroke_complete, aes(stroke, y = age, fill = gender)) +
  geom_boxplot() +
  labs(x = "Outcome", y = "Age (Years)", fill = "Gender") +
  theme_minimal()

ggplot(stroke_complete, aes(x = age, fill = stroke)) +
  geom_density(alpha = 0.5) +
  labs(x = "Age (Years)", y = "Density", fill = "Outcome") +
  theme_minimal()

# average_blood_glucose and gender
ggplot(stroke_complete, aes(stroke, y = avg_glucose_level, fill = gender)) +
  geom_boxplot() +
  labs(x = "Outcome", y = "Average glucose level (mg/dL)", fill = "Gender") +
  theme_minimal()

ggplot(stroke_complete, aes(x = avg_glucose_level, fill = stroke)) +
  geom_density(alpha = 0.5) +
  labs(y = "Density", x = "Average glucose level (mg/dL)", fill = "Outcome") +
  theme_minimal()

# bmi and gender
ggplot(stroke_complete, aes(stroke, y = bmi, fill = gender)) +
  geom_boxplot() +
  labs(x = "Outcome", y = expression(BMI~kg/m^{2}), fill = "Gender") +
  theme_minimal()

ggplot(stroke_complete, aes(x = bmi, fill = stroke)) +
  geom_density(alpha = 0.5) +
  labs(y = "Density", x = expression(BMI~kg/m^{2}), fill = "Outcome") +
  theme_minimal()


# So for report I'll likely include age and average blood glucose density plots as
# show the most change
stroke_complete %>% 
  select(stroke, age, avg_glucose_level) %>% 
  pivot_longer(age:avg_glucose_level) %>% 
  mutate(name = recode(name, 
                       "avg_glucose_level" = "Average glucose level (mg/dL)",
                       "age" = "Age (Years)")) %>% 
  ggplot(aes(x = value, fill = stroke)) +
  geom_density(alpha = 0.5) +
  facet_wrap(vars(name), scales = "free", strip.position = "bottom") +
  labs(x = NULL, y = "Density", fill = "Outcome") +
  theme_minimal()+ 
  theme(
    strip.placement = "outside",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank())

# Correlations ------------------------------------------------------------

# View numeric correlations for continuous variables
stroke_complete %>%
  select(age, avg_glucose_level, bmi) %>%
  cor()

# Visualise all
library(GGally)
stroke_complete %>%
  select(stroke, age, avg_glucose_level, bmi) %>%
  ggpairs(columns = 2:4, aes(color = stroke, alpha = 0.5))

# Plot age and bmi as strongest correlation
ggplot(stroke_complete, aes(x = age, y = bmi, colour = stroke)) +
  geom_point() +
  labs(x = "Age (Years)", y = expression(BMI~kg/m^{2}), colour = "Outcome") +
  theme_minimal()

# Then plot age and average glucose level as second strongest correlation
ggplot(stroke_complete, aes(x = age, y = avg_glucose_level, colour = stroke)) +
  geom_point() +
  labs(x = "Age (Years)", y = "Average glucose level (mg/dL)", colour = "Outcome") +
  theme_minimal()


# So for report I'll likely include age and bmi and age and average blood glucose
# scatter plots as they show the most change
stroke_complete %>% 
  select(stroke, age, bmi, avg_glucose_level) %>% 
  pivot_longer(bmi:avg_glucose_level) %>% 
  mutate(name = recode(name, 
                       "avg_glucose_level" = "Average glucose level (mg/dL)",
                       "age" = "Age (Years)",
                       "bmi" = "BMI")) %>% 
  ggplot(aes(x = age, y = value, colour = stroke)) +
  geom_point(alpha = 0.9, size = 2) +
  facet_wrap(vars(name), scales = "free", strip.position = "left") +
  labs(x = "Age (years)", y = NULL, colour = "Outcome") + 
  theme_minimal()+
  theme(
    strip.placement = "outside",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_blank())
