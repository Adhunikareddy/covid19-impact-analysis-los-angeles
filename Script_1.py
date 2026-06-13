# Step 1: Load Dataset
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# # Load the dataset
df = pd.read_csv('PPP_Loans_LA.csv')
# # # Show the dataset
# # print(df.head())

# # # Step 2: Detecting missing values
# # # 2.1. Remove Rows with Missing Values
df = df.dropna(how='all')

# # # Imputation
# # # for numerical variables
# # # Impute Missing Values Using Mean (for Numerical Data)
df = df.fillna(df.mean(numeric_only=True))

# Q1 = df['InitialApprovalAmount'].quantile(0.25)
# Q3 = df['InitialApprovalAmount'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# outliers = df[(df['JobsReported'] < lower_bound) | (df['JobsReported'] > upper_bound)]
# df['InitialApprovalAmount'] = df['InitialApprovalAmount'].apply(lambda x: upper_bound if x > upper_bound else
# (lower_bound if x < lower_bound else x))

# # Visualization
# sns.boxplot(data=df, x='InitialApprovalAmount')
# plt.title(f"Boxplot for InitialApprovalAmount")
# plt.show()

# # Descriptive Statistics
# # Descriptive Statistics for all numerical variables

# print("Statistics for InitialApprovalAmount")
# print("Mean:", df['InitialApprovalAmount'].mean())
# print("Median:", df['InitialApprovalAmount'].median())
# print("Mode:", df['InitialApprovalAmount'].mode()[0])
# print("Standard Deviation:", df['InitialApprovalAmount'].std())

# # Categorical variables

print("Categorical Variables:")
print("Summary for HubzoneIndicator")
print("Counts:")
print(df['HubzoneIndicator'].value_counts())
print("\nFrequencies:")
print(df['HubzoneIndicator'].value_counts(normalize=True))
print("Summary for LoanStatus")
print("Counts:")
print(df['LoanStatus'].value_counts())
print("\nFrequencies:")
print(df['LoanStatus'].value_counts(normalize=True))
print("Summary for Industry")
print("Counts:")
print(df['Industry'].value_counts())
print("\nFrequencies:")
print(df['Industry'].value_counts(normalize=True))  

# # Hypothesis testing

# T-test
group_paidinfull = df[df['HubzoneIndicator'] == False]['InitialApprovalAmount'].dropna()
group_exemption4 = df[df['HubzoneIndicator'] == True]['InitialApprovalAmount'].dropna()
# There is a significant difference between the means of these two groups.
t_stat, p_value = stats.ttest_ind(group_paidinfull,group_exemption4)
print(f'T-Test Results: t-statistic = {t_stat}, p-value = {p_value}')

# Chi-square test
contingency_table = pd.crosstab(df['Industry'], df['LoanStatus']).dropna()
chi2_stat, chi2_p_val, dof, expected = stats.chi2_contingency(contingency_table)
print(f'Chi-Squared Test Results: chi2_stat = {chi2_stat}, p-value = {chi2_p_val}')


## Dataset 2
# # Load the dataset
# df = pd.read_csv('Stay_home_order.csv')
# # # Show the dataset
# # print(df.head())

# # # Step 2: Detecting missing values
# # # 2.1. Remove Rows with Missing Values
# df = df.dropna(how='all')

# # # Imputation
# # # for numerical variables
# # # Impute Missing Values Using Mean (for Numerical Data)
# df = df.fillna(df.mean(numeric_only=True))

# #outliers
# Q1 = df['icu_capacity_percent'].quantile(0.25)
# Q3 = df['icu_capacity_percent'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# outliers = df[(df['icu_capacity_percent'] < lower_bound) | (df['icu_capacity_percent'] > upper_bound)]
# df['icu_capacity_percent'] = df['icu_capacity_percent'].apply(lambda x: upper_bound if x > upper_bound else
# (lower_bound if x < lower_bound else x))

# # Visualization
# sns.boxplot(data=df, x='icu_capacity_percent')
# plt.title(f"Boxplot for icu_capacity_percent")
# plt.show()

# # Descriptive Statistics for icu_capacity_percent
# print("Statistics for icu_capacity_percent")
# print("Mean:", df['icu_capacity_percent'].mean())
# print("Median:", df['icu_capacity_percent'].median())
# print("Mode:", df['icu_capacity_percent'].mode()[0])
# print("Standard Deviation:", df['icu_capacity_percent'].std())

# Hypothesis Testing
group_Bay_Area = df[df['region'] == 'Bay Area']['icu_capacity_percent'].dropna()
group_Greater_Sacramento = df[df['region'] == 'Greater Sacramento']['icu_capacity_percent'].dropna()
group_Northern_California = df[df['region'] == 'Northern California']['icu_capacity_percent'].dropna()
group_San_Joaquin_Valley = df[df['region'] == 'San Joaquin Valley']['icu_capacity_percent'].dropna()
group_Southern_California = df[df['region'] == 'Southern California']['icu_capacity_percent'].dropna()
f_stat, f_p_value = stats.levene(group_Bay_Area,group_Greater_Sacramento,group_Northern_California,group_San_Joaquin_Valley,group_Southern_California)
print(f'F-Test Results: F-statistic = {f_stat}, p-value = {f_p_value}')

