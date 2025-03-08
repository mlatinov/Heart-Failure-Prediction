
#### Libraries ####
 library(tidyverse)

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





