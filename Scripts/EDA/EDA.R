
#### Libraries ####
library(tidyverse)
library(corrplot)

## Load the data 
heart_failure_records <- read_csv("Data/Data_raw/heart_failure_clinical_records.csv")

# Change  the data types of features encoded as 0 and 1 from num to factors
heart_failure_clean <- heart_failure_records %>%
     rename(death = DEATH_EVENT) %>%
     mutate(
         anaemia = as.factor(anaemia),
         diabetes = as.factor(diabetes),
         high_blood_pressure = as.factor(high_blood_pressure),
         sex = as.factor(sex),
         smoking = as.factor(smoking),
         death = as.factor(death)
         )


## Plot the num distributions

heart_failure_clean %>%
  select_if(is.numeric) %>%  # Select only numeric columns
  pivot_longer(cols = everything(), names_to = "Features", values_to = "Count") %>%
  ggplot(aes(x = Count, fill = Features)) +  # Use fill to color the histogram
  geom_histogram() +
  scale_fill_viridis_d(option = "plasma") +  # Apply Viridis color scale
  facet_wrap(~Features, scale = "free_x") +  # Facet by feature, allowing different x-scales
  theme_minimal() +
  labs(
    title = "Numerical Features Distribution", # Give a Title
    x = "Features ",  # Label for the x-axis
    y = "Frequency") +# Label for the y-axis
  
  # Change the text size  
  theme(
    title = element_text(size = 15),          
    axis.title.x = element_text(size = 13),
    axis.text.y = element_text(size = 10),
    strip.text = element_text(size = 8)
  )

## Plot the count per Category
heart_failure_clean %>%
  select_if(is.factor) %>% 
  pivot_longer(cols = everything(),names_to = "Features",values_to = "Value")%>%
  ggplot(aes(x = Value,fill = Features))+
  geom_bar()+
  scale_fill_viridis_d(option = "inferno")+
  facet_wrap(~Features)+
  theme_minimal()+
  labs(
    title = "Categorical Distribution",
    x = "Feature",
    y = "Count") +
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 15),
    title = element_text(size = 17))


## Boxplot
heart_failure_clean %>%
  select_if(is.numeric)%>%
  pivot_longer(cols = everything(),names_to = "Features",values_to = "Value")%>%
  ggplot(aes(x = Value))+
  geom_boxplot(outlier.colour = "red")+
  facet_wrap(~Features,scale = "free_x")+
  theme_minimal()+
  labs(
    title = "Numerical Features Boxplot",
    x = "Features",
    y = "Value")+
  theme(
    strip.text = element_text(size = 10),
    title = element_text(size = 15),
    axis.title = element_text(size = 15))

## Box plot with classified outliers
heart_failure_clean %>%
  select(creatinine_phosphokinase,platelets,serum_creatinine,serum_sodium,death)%>%
  pivot_longer(cols = -death,names_to = "Features",values_to = "Value")%>%
  group_by(Features) %>%
  mutate(
    Q1 = quantile(Value, 0.25, na.rm = TRUE),
    Q3 = quantile(Value, 0.75, na.rm = TRUE),
    IQR = Q3 - Q1,
    lower_bound = Q1 - 1.5 * IQR,
    upper_bound = Q3 + 1.5 * IQR,
    is_outlier = Value < lower_bound | Value > upper_bound
  ) %>%
  ggplot(aes(x = Features, y = Value)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.5) + 
  geom_jitter(aes(color = as.factor(death), alpha = is_outlier), width = 0.2) + 
  scale_alpha_manual(values = c(0, 1)) +  
  scale_color_manual(values = c("yellow", "black")) + 
  facet_wrap(~ Features, scales = "free_y") +  
  theme_minimal() +
  labs(x = "Feature", y = "Value", color = "Death", alpha = "Outlier",title = "Outliers Classification") 

## Hypothesis testing 

# Null Hypothesis : There is no difference in the central tendency of creatinine_phosphokinase levels between death = 0 and death = 1
# Alternative Hypothesis (H1): The central tendency  of creatinine_phosphokinase  levels differs between death = 0 and death = 1.

# TEST : Mann-Whitney U test
# If it is less than significance level of  0.05,reject the null hypothesis and conclude that there is a significant difference between the two groups

wilcox.test(creatinine_phosphokinase ~ death,data = heart_failure_clean) # p-value = 0.0008999

## Viz the creatinine_phosphokinase by death
heart_failure_clean %>%
  select(death,creatinine_phosphokinase)%>%
  group_by(death)%>%
  summarize(
    median_crea = median(creatinine_phosphokinase))%>%
  ggplot(aes(x = death, y=median_crea,fill = death))+
  geom_col()+
  scale_fill_manual(values = c("0" = "#1f77b4", "1" = "#d62728")) +
  labs(x = "Death (0 = No, 1 = Yes)",
       y = "Median Creatinine Phosphokinase", 
       title = "Median CPK Levels by Death Status") +
  theme_minimal()+
  theme(
    axis.text = element_text(size = 10),
    title = element_text(size = 14)
  )+
  annotate(geom = "text",x = 1,y = 350 ,label = "Mann-Whitney U test p_value = 0.0008999")
  
## Hypothesis testing 

# Null Hypothesis : There is no difference in the central tendency of platelets levels between death = 0 and death = 1
# Alternative Hypothesis (H1): The central tendency  of platelets levels differs between death = 0 and death = 1.

# TEST : Mann-Whitney U test
# If it is less than significance level of  0.05,reject the null hypothesis and conclude that there is a significant difference between the two groups

wilcox.test(platelets~ death,data = heart_failure_clean) # , p-value = 0.01896

## Hypothesis testing 

# Null Hypothesis : There is no difference in the central tendency of serum_sodium levels between death = 0 and death = 1
# Alternative Hypothesis (H1): The central tendency  of serum_sodium levels differs between death = 0 and death = 1.

# TEST : Mann-Whitney U test
# If it is less than significance level of  0.05,reject the null hypothesis and conclude that there is a significant difference between the two groups

wilcox.test(serum_sodium ~ death,data = heart_failure_clean) # p-value < 2.2e-16

## Plot the serum_sodium by death
heart_failure_clean %>%
  select(serum_sodium,death)%>%
  ggplot(aes(x = serum_sodium,color = death))+
  stat_ecdf(size = 1)+
  scale_color_manual(values = c("0" = "green","1" = "black"))+
  theme_minimal()+
  labs(
    title = "Empirical CDF of Serum sodium by Death Status",
    x = "Serum sodium",
    y = "ECDF"
  ) +
  theme(
    title = element_text(size = 13),
    axis.title = element_text(size = 12))+
  annotate(geom = "text",x = 121,y = 1,label = "Mann-Whitney U test p-value < 2.2e-16")

## Correlation Matrix
tmwr_cols <- colorRampPalette(c("#91CBD765", "#CA225E"))

cor_matrix <- heart_failure_clean %>%
  select_if(is.numeric) %>%
  cor()

# Shorten column names
colnames(cor_matrix) <- abbreviate(colnames(cor_matrix), minlength = 5)  

# Plot with improved visibility
corrplot(cor_matrix, 
         col = tmwr_cols(200),      
         tl.col = "black",          
         tl.cex = 0.8,              
         method = "ellipse",        
         addCoef.col = "black",      
         number.cex = 0.9) 
  
saveRDS(heart_failure_clean,file = "heart_failure_clean.rds")


