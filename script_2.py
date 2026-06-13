# Project Part 2: COVID-19 Neighborhood and Vaccination Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats


# ============================================================
# RESEARCH QUESTION 4
# Relationship between COVID-19 cases, population size,
# cases per 100,000, and deaths
# Dataset: neighborhood_level.csv
# ============================================================

# Load neighborhood-level COVID dataset
covid_df = pd.read_csv("neighborhood_level.csv")

# Display basic information
print("Neighborhood Dataset Preview")
print(covid_df.head())
print(covid_df.info())

# Clean column names
covid_df.columns = covid_df.columns.str.strip()

# Keep relevant columns
rq4_columns = ["cases", "deaths", "population", "cases_per100k"]
covid_df = covid_df[rq4_columns]

# Remove missing values
covid_df = covid_df.dropna()

# Descriptive statistics
print("\nDescriptive Statistics for RQ4")
print(covid_df.describe())


# ----------------------------
# Simple Linear Regression
# Predictor: cases
# Response: deaths
# ----------------------------

X_simple = covid_df[["cases"]]
y = covid_df["deaths"]

X_simple = sm.add_constant(X_simple)

simple_model = sm.OLS(y, X_simple).fit()

print("\nSimple Linear Regression Results")
print(simple_model.summary())


# ----------------------------
# Multiple Linear Regression
# Predictors: cases, population, cases_per100k
# Response: deaths
# ----------------------------

X_multiple = covid_df[["cases", "population", "cases_per100k"]]
y = covid_df["deaths"]

X_multiple = sm.add_constant(X_multiple)

multiple_model = sm.OLS(y, X_multiple).fit()

print("\nMultiple Linear Regression Results")
print(multiple_model.summary())


# ----------------------------
# Visualization: Cases vs Deaths
# ----------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(data=covid_df, x="cases", y="deaths")
plt.title("COVID-19 Cases vs Deaths")
plt.xlabel("Number of Cases")
plt.ylabel("Number of Deaths")
plt.grid(True)
plt.tight_layout()
plt.savefig("cases_vs_deaths.png")
plt.show()


# ----------------------------
# Visualization: Regression Line
# ----------------------------

plt.figure(figsize=(10, 6))
sns.regplot(data=covid_df, x="cases", y="deaths", line_kws={"color": "red"})
plt.title("Simple Linear Regression: Cases Predicting Deaths")
plt.xlabel("Number of Cases")
plt.ylabel("Number of Deaths")
plt.grid(True)
plt.tight_layout()
plt.savefig("cases_deaths_regression.png")
plt.show()



# ============================================================
# RESEARCH QUESTION 5
# Do COVID-19 cases significantly differ between incorporated
# cities and unincorporated areas?
# Dataset: Vaccination_Rates_by_Neighborhood_20241203.csv
# ============================================================

# Load vaccination dataset
vaccination_df = pd.read_csv("Vaccination_Rates_by_Neighborhood_20241203.csv")

# Display basic information
print("\nVaccination Dataset Preview")
print(vaccination_df.head())
print(vaccination_df.info())

# Clean column names
vaccination_df.columns = vaccination_df.columns.str.strip()

# Check available columns
print("\nVaccination Dataset Columns")
print(vaccination_df.columns)


# Rename columns if needed
# Adjust these names if your CSV has slightly different capitalization
vaccination_df = vaccination_df.rename(columns={
    "CITY_TYPE": "city_type",
    "Cases": "cases",
    "Percent_of_People_Fully_Vaccinated": "percent_fully_vaccinated"
})

# Keep relevant columns
rq5_columns = ["city_type", "cases", "percent_fully_vaccinated"]
vaccination_df = vaccination_df[rq5_columns]

# Remove missing values
vaccination_df = vaccination_df.dropna()

# Descriptive statistics
print("\nDescriptive Statistics for RQ5")
print(vaccination_df.describe())

print("\nCity Type Distribution")
print(vaccination_df["city_type"].value_counts())


# ----------------------------
# T-test: City vs Unincorporated
# ----------------------------

city_cases = vaccination_df[vaccination_df["city_type"] == "City"]["cases"]
unincorporated_cases = vaccination_df[vaccination_df["city_type"] == "Unincorporated"]["cases"]

t_stat, p_value = stats.ttest_ind(city_cases, unincorporated_cases, equal_var=False)

print("\nT-Test Results for COVID Cases by City Type")
print("t-statistic:", t_stat)
print("p-value:", p_value)


# ----------------------------
# Visualization: Before Outlier Handling
# ----------------------------

plt.figure(figsize=(10, 6))
sns.boxplot(data=vaccination_df, x="city_type", y="cases")
plt.title("COVID Cases Distribution by City Type")
plt.xlabel("City Type")
plt.ylabel("Number of Cases")
plt.grid(True)
plt.tight_layout()
plt.savefig("covid_cases_by_city_type_before.png")
plt.show()


# ----------------------------
# Outlier Capping using IQR
# ----------------------------

Q1 = vaccination_df["cases"].quantile(0.25)
Q3 = vaccination_df["cases"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

vaccination_df["cases_capped"] = vaccination_df["cases"].clip(lower=lower_bound, upper=upper_bound)


# ----------------------------
# T-test After Capping
# ----------------------------

city_cases_capped = vaccination_df[vaccination_df["city_type"] == "City"]["cases_capped"]
unincorporated_cases_capped = vaccination_df[vaccination_df["city_type"] == "Unincorporated"]["cases_capped"]

t_stat_capped, p_value_capped = stats.ttest_ind(
    city_cases_capped,
    unincorporated_cases_capped,
    equal_var=False
)

print("\nT-Test Results After Outlier Capping")
print("t-statistic:", t_stat_capped)
print("p-value:", p_value_capped)


# ----------------------------
# Visualization: After Outlier Handling
# ----------------------------

plt.figure(figsize=(10, 6))
sns.boxplot(data=vaccination_df, x="city_type", y="cases_capped")
plt.title("COVID Cases Distribution by City Type After Outlier Capping")
plt.xlabel("City Type")
plt.ylabel("Number of Cases")
plt.grid(True)
plt.tight_layout()
plt.savefig("covid_cases_by_city_type_after.png")
plt.show()


# ----------------------------
# Scatterplot: Vaccination Rate vs Cases
# ----------------------------

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=vaccination_df,
    x="percent_fully_vaccinated",
    y="cases",
    hue="city_type"
)
plt.title("COVID Cases vs Vaccination Rates by City Type")
plt.xlabel("Percentage of Fully Vaccinated People")
plt.ylabel("Number of Cases")
plt.grid(True)
plt.tight_layout()
plt.savefig("vaccination_rate_vs_cases.png")
plt.show()