# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(dups)
library(trackr)
# Create figs directory if it doesn't exist
dir.create("figs", showWarnings = FALSE)

# Read the data
data <- read_csv("outputs/curated_outputs/combined_summary.csv")

# Check sample size for each combo of model_doctor, case_file combination
sample_sizes <- data %>%
  group_by(model_doctor, case_file) %>%
  summarise(n = n(), .groups = 'drop')

print("Sample sizes per model_doctor and case_file combination:")
print(sample_sizes)

# Clean -1 values and convert to NA for specified columns
cols_to_clean <- c(
  "num_checklist_items_asked", "checklist_completion_rate", 
  "num_correct_treatments", "num_palliative_treatments",
  "num_unnecessary_harmful_treatments", "num_not_found_treatments",
  "num_errored_treatments_classification"
)

data <- data %>%
  mutate(across(all_of(cols_to_clean), ~na_if(., -1)))

# Add a clearer case file label
data <- data %>%
  mutate(
    case_label = case_when(
      str_detect(case_file, "case1") ~ "TB",
      str_detect(case_file, "case2") ~ "Pre-eclampsia",
      str_detect(case_file, "case3") ~ "Dysentery",
      str_detect(case_file, "case4") ~ "Angina",
      str_detect(case_file, "case5") ~ "Asthma",
      TRUE ~ as.character(case_file) # Fallback if new cases are added
    )
  )

# --- Bar graphs ---

# Theme for plots
plot_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top",
    strip.background = element_rect(fill = "lightgrey", color = "grey"),
    strip.text = element_text(face = "bold")
  )

# 1. doctor_questions_count
plot_doc_questions <- ggplot(data, aes(x = model_doctor, y = doctor_questions_count, fill = model_doctor)) +
  geom_bar(stat = "summary", fun = "mean", position = "dodge") +
  stat_summary(fun = mean, aes(label = round(after_stat(y), 2)), geom = "text", position = position_dodge(width = 0.9), vjust = 1.5, size = 3) +
  facet_wrap(~case_label) +
  labs(
    title = "# questions from doctor",
    x = "Model",
    y = "# questions",
    fill = "Model"
  ) +
  plot_theme

print(plot_doc_questions)
ggsave("figs/doctor_questions_count_plot.png", plot = plot_doc_questions, width = 10, height = 7)

# 2. checklist_completion_rate
plot_checklist_completion <- ggplot(data %>% 
                                   filter(case_label != "Pre-eclampsia" & case_label != "TB"), 
                                 aes(x = model_doctor, y = checklist_completion_rate, fill = model_doctor)) +
  geom_bar(stat = "summary", fun = "mean", position = "dodge") +
  stat_summary(fun = mean, aes(label = scales::percent(after_stat(y), accuracy = 0.1)), geom = "text", position = position_dodge(width = 0.9), vjust = 1.5, size = 3) +
  facet_wrap(~case_label) +
  labs(
    title = "Average Checklist Completion Rate",
    x = "Model Doctor",
    y = "Average Checklist Completion Rate",
    fill = "Model Doctor"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), limits = c(0, 1)) +
  plot_theme

print(plot_checklist_completion)
ggsave("figs/checklist_completion_rate_plot.png", plot = plot_checklist_completion, width = 10, height = 7)

# 3. doctor_questions_without_ids
plot_doc_questions_no_ids <- ggplot(data, aes(x = model_doctor, y = doctor_questions_without_ids, fill = model_doctor)) +
  geom_bar(stat = "summary", fun = "mean", position = "dodge") +
  stat_summary(fun = mean, aes(label = round(after_stat(y), 2)), geom = "text", position = position_dodge(width = 0.9), vjust = 1.5, size = 3) +
  facet_wrap(~case_label) +
  labs(
    title = "# questions not on checklist (off script questions)",
    x = "Model Doctor",
    y = "Average Questions Count (No IDs)",
    fill = "Model Doctor"
  ) +
  plot_theme

print(plot_doc_questions_no_ids)
ggsave("figs/doctor_questions_without_ids_plot.png", plot = plot_doc_questions_no_ids, width = 10, height = 7)


# 4. diag_classification (proportion correct)
diag_summary <- data %>%
  group_by(model_doctor, case_label, diag_classification) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(model_doctor, case_label) %>%
  mutate(proportion = count / sum(count)) %>% 
  ungroup

plot_diag_classification <- ggplot(diag_summary, 
                                   aes(x = model_doctor, y = proportion, fill = diag_classification)) +
  geom_col(position = "fill") +
  geom_text(aes(label = scales::percent(proportion, accuracy = 0.1)), position = position_fill(vjust = 0.1), size = 3) +
  facet_wrap(~case_label) +
  labs(
    title = "Proportion of Diagnosis Classifications",
    x = "Model Doctor",
    y = "Proportion",
    fill = "Diagnosis Classification"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  scale_fill_manual(values = c("correct" = "forestgreen", "incorrect" = "firebrick", "unclear" = "grey")) +
  plot_theme

print(plot_diag_classification)
ggsave("figs/diag_classification_proportions_plot.png", plot = plot_diag_classification, width = 10, height = 7)

# 5. Filled bargraph for treatment proportions
treatment_data_long <- data %>%
  select(model_doctor, case_label, 
         num_correct_treatments, num_palliative_treatments, 
         num_unnecessary_harmful_treatments, num_not_found_treatments) %>%
  pivot_longer(
    cols = starts_with("num_"),
    names_to = "treatment_type",
    values_to = "count"
  ) %>%
  mutate(
    treatment_type = factor(treatment_type, levels = c(
      "num_correct_treatments", "num_palliative_treatments",
      "num_unnecessary_harmful_treatments", "num_not_found_treatments"
    ), labels = c(
      "Correct", "Palliative", "Unnecessary/Harmful", "Not Found"
    ))
  ) %>%
  filter(!is.na(count)) # Remove NAs which were -1, if any row has all treatments as NA after cleaning

treatment_summary <- treatment_data_long %>%
  group_by(model_doctor, case_label, treatment_type) %>%
  summarise(total_count = sum(count, na.rm = TRUE), .groups = 'drop') %>%
  group_by(model_doctor, case_label) %>%
  mutate(proportion = total_count / sum(total_count, na.rm = TRUE)) %>%
  filter(!is.na(proportion)) %>%  # Ensure proportion is not NA (e.g. if sum(total_count) was 0)
  mutate(treatment_type = treatment_type  %>% fct_rev())

plot_treatment_proportions <- ggplot(treatment_summary, 
                                     aes(x = model_doctor, y = proportion, fill = treatment_type)) +
  geom_bar(stat = "identity", position = "fill") +
  geom_text(aes(label = ifelse(proportion > 0.01, scales::percent(proportion, accuracy = 0.1), "")), 
            position = position_fill(vjust = 0.1), size = 2.5) +
  facet_wrap(~case_label) +
  labs(
    title = "Proportion of Treatment Types",
    x = "Model Doctor",
    y = "Proportion of Treatments",
    fill = "Treatment Type"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  plot_theme +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = c(
    "Not Found" = "grey",
    "Unnecessary/Harmful" = "salmon", 
    "Palliative" = "#9ECA6D", # beautiful light green color
    "Correct" = "mediumseagreen"
  ))

print(plot_treatment_proportions)
ggsave("figs/treatment_proportions_plot.png", plot = plot_treatment_proportions, width = 12, height = 8)

print("Analysis script finished. Plots saved to 'figs' directory as PDFs.") 




# Improving diagnosis classification ----------------------------------------------------------------------------------------------


data %>% 
filter(case_label == "Dysentery") %>% 
select(model_doctor, case_label, extracted_diagnosis,	diag_classification, diag_classification_confidence, diag_explanation) %>% 
select(extracted_diagnosis, diag_classification) %>% 
dups_drop() %>% 
arrange(diag_classification, extracted_diagnosis) %>% 
write_csv("diagnosis_checks/dysentery_diagnosis_checks.csv")



data %>% 
count_prop(case_label) %>% 
filter(case_label == "Asthma") %>% 
select(model_doctor, case_label, extracted_diagnosis,	diag_classification, diag_classification_confidence, diag_explanation) %>% 
select(extracted_diagnosis, diag_classification) %>% 
dups_drop() %>% 
arrange(diag_classification, extracted_diagnosis) %>% 
print_all

# Improving treatment classification ----------------------------------------------------------------------------------------------

# treatments_dysentery <- data %>% 
# count_prop(case_label) %>% 
# filter(case_label == "Dysentery") %>% 
# # select(model_doctor, case_label, extracted_treatments, num_correct_treatments,	num_palliative_treatments,	num_unnecessary_harmful_treatments,	num_not_found_treatments,	num_errored_treatments_classification,	treatment_classification_explanation) %>% 
# select(extracted_treatments, num_correct_treatments,	num_palliative_treatments,	num_unnecessary_harmful_treatments,	num_not_found_treatments,	num_errored_treatments_classification) %>% 
# dups_drop() %>% 
# mutate(
#   extracted_treatments = str_remove_all(extracted_treatments, "\\[|\\]"),
#   extracted_treatments = str_remove_all(extracted_treatments, '"'),
#   extracted_treatments = str_split(extracted_treatments, ","),
#   extracted_treatments = map(extracted_treatments, ~str_trim(.x))
# ) %>% 
# unnest(extracted_treatments) %>% 
# select(extracted_treatments) %>% 
# arrange(extracted_treatments) %>% 
# mutate(extracted_treatments = str_to_lower(extracted_treatments)) %>% 
# dups_drop() %>% 
# print_all


treatments <- read_csv("outputs/curated_outputs/treatment_classifications.csv") %>% 
mutate(case_label = case_when(
  str_detect(case_file, "case1") ~ "TB",
  str_detect(case_file, "case2") ~ "Pre-eclampsia",
  str_detect(case_file, "case3") ~ "Dysentery",
  str_detect(case_file, "case4") ~ "Angina",
  str_detect(case_file, "case5") ~ "Asthma",
  TRUE ~ as.character(case_file) # Fallback if new cases are added
))

treatments %>% count_prop(case_label, category)

treatments %>% filter(case_label == "Asthma") %>% 
select(case_label, source_file, stated_treatment, category, matched_key_item) %>% 
mutate(stated_treatment = str_to_lower(stated_treatment) %>% str_trim()) %>% 
select(-source_file) %>% 
dups_drop() %>% 
arrange(stated_treatment, category) %>% 
print_all










