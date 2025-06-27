import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# Dropbox to import csv
url = "https://www.dropbox.com/scl/fi/xwirjv3wflfl94qckcbqw/salary_cleaned.csv?rlkey=8w9zgs8psc6g775hb2b7uvv74&dl=1"

df = pd.read_csv(url)

# -------- Plot1: Salary origin distribution --------
plt.figure()
sns.histplot(df['Salary'], bins=40, kde=True)
plt.title('Salary Distribution (Original Scale)')
plt.xlabel('Salary')
plt.tight_layout()
plt.savefig("1_salary_distribution.png")
plt.show()

# -------- Plot2: Log Salary distribution --------
plt.figure()
sns.histplot(np.log1p(df['Salary']), bins=40, kde=True, color='orange')
plt.title('Salary Distribution (Log Transformed)')
plt.xlabel('Log(Salary)')
plt.tight_layout()
plt.savefig("2_log_salary_distribution.png")
plt.show()

# -------- plot3: heatmap --------
plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("3_correlation_heatmap.png")
plt.show()

# -------- plot4: salary vs education --------
plt.figure()
sns.boxplot(data=df, x='EducationLevel', y='Salary', order=df['EducationLevel'].value_counts().index)
plt.title('Salary by Education Level')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("4_salary_by_education.png")
plt.show()

# -------- plot5: salary vs age --------
plt.figure()
sns.scatterplot(data=df, x='Age', y='Salary', edgecolor='w')
plt.title('Age vs Salary')
plt.tight_layout()
plt.savefig("5_age_vs_salary.png")
plt.show()