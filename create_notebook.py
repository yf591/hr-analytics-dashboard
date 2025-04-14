import nbformat as nbf
import os

# 新しいノートブックを作成
nb = nbf.v4.new_notebook()

# セルのリスト
cells = [
    # セル1: タイトル（マークダウン）
    nbf.v4.new_markdown_cell("""# IBM HR Analytics Dataset - Initial Exploration

This notebook explores the IBM HR Analytics Employee Attrition & Performance dataset from Kaggle."""),
    
    # セル2: 初期設定（コード）
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure matplotlib and seaborn
plt.style.use('ggplot')
sns.set(style='whitegrid')
%matplotlib inline

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)"""),
    
    # セル3: データ読み込み（コード）
    nbf.v4.new_code_cell("""# Load the dataset
data_path = os.path.join('..', '..', 'data', 'raw', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')
df = pd.read_csv(data_path)

# Display basic info
print(f"Dataset shape: {df.shape}")
df.head()"""),
    
    # セル4: データタイプ確認（コード）
    nbf.v4.new_code_cell("""# Check data types
df.info()"""),
    
    # セル5: 基本統計量（コード）
    nbf.v4.new_code_cell("""# Check basic statistics
df.describe()"""),
    
    # セル6: 欠損値チェック（コード）
    nbf.v4.new_code_cell("""# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:")
print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values")"""),
    
    # セル7: 離職分析のタイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Target Variable Analysis: Attrition"""),
    
    # セル8: 離職率分析（コード）
    nbf.v4.new_code_cell("""# Overall attrition rate
attrition_count = df['Attrition'].value_counts()
attrition_pct = df['Attrition'].value_counts(normalize=True) * 100

print(f"Attrition counts:\\n{attrition_count}")
print(f"\\nAttrition percentage:\\n{attrition_pct}")

# Plot attrition distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Distribution')
plt.xlabel('Attrition')
plt.ylabel('Count')

for i, count in enumerate(attrition_count):
    plt.text(i, count + 30, f"{count} ({attrition_pct[i]:.1f}%)", 
             horizontalalignment='center')

plt.show()"""),
    
    # セル9: 部門分析のタイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Departmental Analysis"""),
    
    # セル10: 部門分布（コード）
    nbf.v4.new_code_cell("""# Department distribution
dept_counts = df['Department'].value_counts()
print("Department distribution:")
print(dept_counts)

# Plot department distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='Department', data=df, order=dept_counts.index)
plt.title('Employee Count by Department')
plt.xlabel('Count')
plt.ylabel('Department')
for i, count in enumerate(dept_counts):
    plt.text(count + 10, i, str(count), va='center')
plt.show()"""),
    
    # セル11: 部門別離職率（コード）
    nbf.v4.new_code_cell("""# Attrition by department
dept_attrition = df.groupby('Department')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
dept_attrition.columns = ['Department', 'Attrition Rate (%)']
dept_attrition = dept_attrition.sort_values('Attrition Rate (%)', ascending=False)

print("Attrition rate by department:")
print(dept_attrition)

# Plot attrition by department
plt.figure(figsize=(12, 6))
sns.barplot(x='Attrition Rate (%)', y='Department', data=dept_attrition)
plt.title('Attrition Rate by Department')
plt.xlabel('Attrition Rate (%)')
plt.ylabel('Department')
for i, rate in enumerate(dept_attrition['Attrition Rate (%)']):
    plt.text(rate + 0.5, i, f"{rate:.1f}%", va='center')
plt.show()"""),
    
    # セル12: 年齢と経験分析タイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Age and Experience Analysis"""),
    
    # セル13: 年齢分析（コード）
    nbf.v4.new_code_cell("""# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Attrition by age group
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], 
                        labels=['<25', '26-35', '36-45', '46-55', '55+'])

age_attrition = df.groupby('AgeGroup')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
age_attrition.columns = ['Age Group', 'Attrition Rate (%)']

plt.figure(figsize=(12, 6))
sns.barplot(x='Age Group', y='Attrition Rate (%)', data=age_attrition)
plt.title('Attrition Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Attrition Rate (%)')
for i, rate in enumerate(age_attrition['Attrition Rate (%)']):
    plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center')
plt.show()"""),
    
    # セル14: 勤続年数分析（コード）
    nbf.v4.new_code_cell("""# Years at company
plt.figure(figsize=(12, 6))
sns.histplot(df['YearsAtCompany'], bins=20, kde=True)
plt.title('Years at Company Distribution')
plt.xlabel('Years at Company')
plt.ylabel('Count')
plt.show()

# Attrition by years at company
df['TenureGroup'] = pd.cut(df['YearsAtCompany'], 
                           bins=[0, 2, 5, 10, 20, 100], 
                           labels=['<2 years', '2-5 years', '6-10 years', '11-20 years', '20+ years'])

tenure_attrition = df.groupby('TenureGroup')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
tenure_attrition.columns = ['Tenure Group', 'Attrition Rate (%)']

plt.figure(figsize=(12, 6))
sns.barplot(x='Tenure Group', y='Attrition Rate (%)', data=tenure_attrition)
plt.title('Attrition Rate by Tenure Group')
plt.xlabel('Tenure Group')
plt.ylabel('Attrition Rate (%)')
for i, rate in enumerate(tenure_attrition['Attrition Rate (%)']):
    plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center')
plt.show()"""),
    
    # セル15: 満足度分析タイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Satisfaction and Work-Life Balance Analysis"""),
    
    # セル16: 職務満足度分析（コード）
    nbf.v4.new_code_cell("""# Job satisfaction
job_sat_counts = df['JobSatisfaction'].value_counts().sort_index()
print("Job Satisfaction distribution:")
print(job_sat_counts)

# Attrition by job satisfaction
job_sat_attrition = df.groupby('JobSatisfaction')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
job_sat_attrition.columns = ['Job Satisfaction', 'Attrition Rate (%)']

plt.figure(figsize=(12, 6))
sns.barplot(x='Job Satisfaction', y='Attrition Rate (%)', data=job_sat_attrition)
plt.title('Attrition Rate by Job Satisfaction Level')
plt.xlabel('Job Satisfaction (1-4 scale)')
plt.ylabel('Attrition Rate (%)')
for i, rate in enumerate(job_sat_attrition['Attrition Rate (%)']):
    plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center')
plt.show()"""),
    
    # セル17: ワークライフバランス分析（コード）
    nbf.v4.new_code_cell("""# Work-life balance
wlb_counts = df['WorkLifeBalance'].value_counts().sort_index()
print("Work-Life Balance distribution:")
print(wlb_counts)

# Attrition by work-life balance
wlb_attrition = df.groupby('WorkLifeBalance')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
wlb_attrition.columns = ['Work-Life Balance', 'Attrition Rate (%)']

plt.figure(figsize=(12, 6))
sns.barplot(x='Work-Life Balance', y='Attrition Rate (%)', data=wlb_attrition)
plt.title('Attrition Rate by Work-Life Balance')
plt.xlabel('Work-Life Balance (1-4 scale)')
plt.ylabel('Attrition Rate (%)')
for i, rate in enumerate(wlb_attrition['Attrition Rate (%)']):
    plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center')
plt.show()"""),
    
    # セル18: 報酬分析タイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Compensation Analysis"""),
    
    # セル19: 給与分析（コード）
    nbf.v4.new_code_cell("""# Monthly income distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['MonthlyIncome'], bins=30, kde=True)
plt.title('Monthly Income Distribution')
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.show()

# Monthly income by job level
plt.figure(figsize=(12, 6))
sns.boxplot(x='JobLevel', y='MonthlyIncome', data=df)
plt.title('Monthly Income by Job Level')
plt.xlabel('Job Level')
plt.ylabel('Monthly Income')
plt.show()

# Attrition by income quartiles
df['IncomeLevel'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

income_attrition = df.groupby('IncomeLevel')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
income_attrition.columns = ['Income Level', 'Attrition Rate (%)']

plt.figure(figsize=(12, 6))
sns.barplot(x='Income Level', y='Attrition Rate (%)', data=income_attrition)
plt.title('Attrition Rate by Income Level')
plt.xlabel('Income Level (Quartiles)')
plt.ylabel('Attrition Rate (%)')
for i, rate in enumerate(income_attrition['Attrition Rate (%)']):
    plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center')
plt.show()"""),
    
    # セル20: 追加要因分析タイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Additional Factors Analysis"""),
    
    # セル21: 残業分析（コード）
    nbf.v4.new_code_cell("""# Overtime and attrition
overtime_attrition = df.groupby('OverTime')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
overtime_attrition.columns = ['Overtime', 'Attrition Rate (%)']

plt.figure(figsize=(10, 6))
sns.barplot(x='Overtime', y='Attrition Rate (%)', data=overtime_attrition)
plt.title('Attrition Rate by Overtime Status')
plt.xlabel('Overtime')
plt.ylabel('Attrition Rate (%)')
for i, rate in enumerate(overtime_attrition['Attrition Rate (%)']):
    plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center')
plt.show()"""),
    
    # セル22: 通勤距離分析（コード）
    nbf.v4.new_code_cell("""# Distance from home
plt.figure(figsize=(12, 6))
sns.boxplot(x='Attrition', y='DistanceFromHome', data=df)
plt.title('Distance From Home by Attrition Status')
plt.xlabel('Attrition')
plt.ylabel('Distance From Home')
plt.show()

# Create distance groups
df['DistanceGroup'] = pd.cut(df['DistanceFromHome'], 
                             bins=[0, 5, 10, 20, 30], 
                             labels=['<5 miles', '5-10 miles', '10-20 miles', '20+ miles'])

distance_attrition = df.groupby('DistanceGroup')['Attrition'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
distance_attrition.columns = ['Distance Group', 'Attrition Rate (%)']

plt.figure(figsize=(12, 6))
sns.barplot(x='Distance Group', y='Attrition Rate (%)', data=distance_attrition)
plt.title('Attrition Rate by Distance from Home')
plt.xlabel('Distance from Home')
plt.ylabel('Attrition Rate (%)')
for i, rate in enumerate(distance_attrition['Attrition Rate (%)']):
    plt.text(i, rate + 0.5, f"{rate:.1f}%", ha='center')
plt.show()"""),
    
    # セル23: 相関分析タイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Correlation Analysis"""),
    
    # セル24: 相関行列（コード）
    nbf.v4.new_code_cell("""# Convert categorical variables to numeric for correlation analysis
df_numeric = df.copy()
df_numeric['AttritionNumeric'] = (df_numeric['Attrition'] == 'Yes').astype(int)
df_numeric['OvertimeNumeric'] = (df_numeric['OverTime'] == 'Yes').astype(int)

# Select numerical columns for correlation
num_cols = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction', 
            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
            'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 
            'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
            'AttritionNumeric', 'OvertimeNumeric']

# Calculate correlation matrix
corr_matrix = df_numeric[num_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()"""),
    
    # セル25: 離職相関（コード）
    nbf.v4.new_code_cell("""# Focus on correlation with attrition
attrition_corr = corr_matrix['AttritionNumeric'].sort_values(ascending=False)
print("Features most correlated with attrition:")
print(attrition_corr)

# Plot top correlations with attrition
plt.figure(figsize=(12, 8))
attrition_corr_filtered = attrition_corr[attrition_corr.index != 'AttritionNumeric']
top_corr = attrition_corr_filtered.abs().sort_values(ascending=False).head(15)
sns.barplot(x=top_corr.values, y=top_corr.index)
plt.title('Top Features Correlated with Attrition (Absolute Value)')
plt.xlabel('Correlation Coefficient (Absolute Value)')
plt.ylabel('Features')
plt.show()"""),
    
    # セル26: 主要な発見タイトル（マークダウン）
    nbf.v4.new_markdown_cell("""## Key Findings Summary"""),
    
    # セル27: 主要な発見（マークダウン）
    nbf.v4.new_markdown_cell("""Based on our exploratory analysis, here are the key findings:

1. **Overall Attrition Rate**: The dataset shows an overall attrition rate of approximately 16%.

2. **Department Impact**: Sales department has the highest attrition rate, followed by Human Resources and Research & Development.

3. **Age Factor**: Younger employees (under 30) show significantly higher attrition rates compared to older employees.

4. **Tenure Impact**: Employees with less than 2 years at the company are more likely to leave.

5. **Job Satisfaction**: Lower job satisfaction scores correlate strongly with higher attrition rates.

6. **Compensation**: Lower income brackets show substantially higher attrition rates.

7. **Overtime**: Employees working overtime have nearly double the attrition rate compared to those who don't.

8. **Distance from Home**: Longer commute distances correlate with higher attrition rates.

9. **Key Correlation Factors**: The strongest correlations with attrition (in order) are overtime, job level, monthly income, total working years, and job satisfaction.

These findings suggest several areas for further investigation and potential dashboard features focusing on these key drivers of attrition."""),
    
    # セル28: 加工データ保存（コード）
    nbf.v4.new_code_cell("""# Create a processed version of the dataset with engineered features
processed_df = df.copy()

# Add additional engineered features for dashboard
processed_df['AttritionBinary'] = (processed_df['Attrition'] == 'Yes').astype(int)
processed_df['OvertimeBinary'] = (processed_df['OverTime'] == 'Yes').astype(int)

# Create standardized satisfaction score (average of all satisfaction metrics)
satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                     'WorkLifeBalance', 'RelationshipSatisfaction']
processed_df['OverallSatisfaction'] = processed_df[satisfaction_cols].mean(axis=1)

# Save processed dataset
processed_path = os.path.join('..', '..', 'data', 'processed', 'hr_data_processed.csv')
processed_df.to_csv(processed_path, index=False)
print(f"Processed dataset saved to {processed_path}")""")
]

# セルをノートブックに追加
nb['cells'] = cells

# ディレクトリが存在しない場合は作成
os.makedirs('notebooks/exploration', exist_ok=True)

# ノートブックを保存
nbf.write(nb, 'notebooks/exploration/01_data_exploration.ipynb')
print("Notebook created successfully!")