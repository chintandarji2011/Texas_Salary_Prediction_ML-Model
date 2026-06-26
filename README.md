---
# 📊 Texas Salary Prediction — Project Report 
---
## 1. Problem Statement & Goal Definition

The Texas state government maintains a publicly available **salary database** covering **113 state agencies**. This project aims to leverage that data to build a machine learning model capable of predicting employee annual salaries — helping the government plan payroll, detect anomalies, and ensure pay equity.

### 🎯 Task Breakdown

| Task | Description |
|---|---|
| **Task 1** | Prepare a complete **data analysis report** on the Texas Salary dataset |
| **Task 2** | Build a **predictive model** to estimate employee annual salary |
| **Task 3.1** | Identify **salary outliers** — who are they? |
| **Task 3.2** | Find departments/roles with the **biggest wage disparities** between managers and employees |
| **Task 3.3** | Analyze how **salaries, compensation, and headcount** have changed over time |

## 2. Dataset Overview

The dataset contains salary records from **113 Texas state government agencies**, capturing compensation details, employee demographics, employment status, and job classification.

### 2.1 Import Libraries & Load Data

### Libraries Used

The project was implemented using Python and several data science libraries:

* **Pandas** and **NumPy** were used for data manipulation, cleaning, and numerical computations.
* **Matplotlib** and **Seaborn** were used for data visualization and Exploratory Data Analysis (EDA).
* **Scikit-learn** was used for data preprocessing, feature scaling, train-test splitting, model building, hyperparameter tuning, and performance evaluation.
* **XGBoost** was used to implement the Extreme Gradient Boosting regression model.
* **Pickle** was used to serialize and save the final trained model for deployment.
* **Datetime** was used for date handling and feature engineering (e.g., employee experience calculation).
* **Warnings** was used to suppress unnecessary warning messages during execution.

#### Machine Learning Models Used

The following regression models were developed and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. XGBoost Regressor

#### Evaluation Metrics

Model performance was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The Random Forest Regressor achieved the best performance and was selected as the final model after hyperparameter tuning.

### Dataset Loading

The Texas Employee Salary dataset was loaded into a Pandas DataFrame using the `read_csv()` function. The dataset contains employee demographic information, job details, compensation data, and employment records used for salary analysis and prediction.

```python
df = pd.read_csv('salary.csv')
```

After loading, the dataset dimensions were examined to understand the number of observations and features available for analysis.

```python
print(df.shape)
df.head()
```

This step ensured that the dataset was successfully imported and ready for exploratory data analysis (EDA), preprocessing, and model development.

## 3. Domain Analysis
Understanding the business context is critical before any analysis.

### 3.1 Texas Employee Salary — Domain Framework

#### 🏛️ Core Business Logic
- The dataset contains **raw identifiers** (`FIRST NAME`, `LAST NAME`, `MI`, `STATE NUMBER`) used for government transparency.
- The `STATUS` column classifies workers into **Full-Time, Part-Time, Elected Official,** or **Contracted** statuses — each with distinct salary structures.

#### 🗂️ Column-Level Domain Mapping

| Column | Domain Meaning |
|---|---|
| `AGENCY`, `AGENCY NAME` | Specific state department |
| `CLASS CODE`, `CLASS TITLE` | Official state job classification index |
| `ETHNICITY`, `GENDER` | Used for equity reporting and compliance audits |
| `HRLY RATE × HRS PER WK × 52` | ≈ `ANNUAL` salary (calculation rule) |
| `summed_annual_salary` | Total fiscal impact per employee across all held jobs |
| `EMPLOY DATE` | Tracks employee tenure |
|`hide_from_search` | a privacy or visibility **flag** rather than a business attribute related to salary|

#### ⚖️ Employment Status Flags
- **`duplicated`, `multiple_full_time_jobs`, `combined_multiple_jobs`**: Texas law allows employees to work for multiple state agencies under dual-employment provisions. These flags prevent double-counting headcounts in aggregations.
- **`EMPLOY DATE`**: State pay scales often increase automatically based on *years of service* (longevity pay).

#### 💡 Key Formula
$$\text{HRLY RATE} \times \text{HRS PER WK} \times 52\ \text{weeks} \approx \text{ANNUAL}$$

## 4.  Basic Checks:

Basic data exploration was performed to understand the structure, quality, and characteristics of the Texas Employee Salary dataset before conducting detailed analysis and preprocessing.

### Checks Performed

| Function                                | Purpose                                                                                                              |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `df.head()`                             | Displayed the first five records to understand the dataset structure and feature values.                             |
| `df.shape`                              | Identified the number of rows and columns in the dataset.                                                            |
| `df.columns`                            | Listed all available features for analysis.                                                                          |
| `len(df.columns)`                       | Determined the total number of variables in the dataset.                                                             |
| `df.info()`                             | Examined data types, non-null counts, and memory usage.                                                              |
| `df.describe().T`                       | Generated statistical summaries of numerical features such as mean, standard deviation, minimum, and maximum values. |
| `df.describe(include='O').T`            | Analyzed categorical features by showing unique values, most frequent categories, and frequency counts.              |
| `df["duplicated"].unique()`             | Verified the presence of duplicate employee records.                                                                 |
| `df["combined_multiple_jobs"].unique()` | Identified whether employees held multiple jobs simultaneously.                                                      |
| `df["hide_from_search"].unique()`       | Examined records flagged for privacy restrictions.                                                                   |

**📌 Basic Checks — Insights**

| Observation | Detail |
|---|---|
|**No.of Rows**|149481|
| **Total Columns** | 21 |
| **Numerical Columns** | `AGENCY`, `HRLY RATE`, `HRS PER WK`, `MONTHLY`, `ANNUAL`, `STATE NUMBER`, `multiple_full_time_jobs`, `summed_annual_salary` |
| **Categorical Columns** | `AGENCY NAME`, `LAST NAME`, `FIRST NAME`, `MI`, `CLASS CODE`, `CLASS TITLE`, `ETHNICITY`, `GENDER`, `STATUS`, `EMPLOY DATE`, `duplicated`, `combined_multiple_jobs`, `hide_from_search` |
| **Type Mismatches** | `CLASS CODE` stored as text (should be int); `EMPLOY DATE` stored as text (should be date); boolean flags stored as object |
| **Columns with NULLs** | `duplicated`, `multiple_full_time_jobs`, `combined_multiple_jobs`, `summed_annual_salary`, `hide_from_search` |

## 5. Exploratory Data Analysis (EDA)
### 5.1 Univariate Analysis
- In Univariate Analysis, this code is used to automatically separate `Numerical Columns` and `Categorical Columns`. so that you can analyze each type appropriately.
```python
num_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

```

**Univariate Analysis – Numerical Features**:

- **`Histograms`** was used to analyze the distribution and spread in numerical variables.
    * Salary-related variables such as **ANNUAL**, **MONTHLY**, and **summed_annual_salary** exhibited positively skewed distributions, indicating that most employees earn moderate salaries while a small number receive significantly higher compensation.
    * **HRLY RATE** showed considerable variation across employees, reflecting differences in job classifications, experience levels, and responsibilities.
    * **HRS PER WK** was concentrated around standard working hours, suggesting that the majority of employees work full-time schedules.
    * Variables related to multiple job holdings were heavily concentrated at lower values, indicating that most employees hold a single position.
    * No unusual patterns or invalid values were observed in the numerical features.
**📌 Univariate Analysis — Insights**:

- `ANNUAL` and `MONTHLY` are **right-skewed** — a few employees earn very high salaries, inflating the mean.
- `HRS PER WK` shows a sharp spike around **40 hours/week**, confirming most employees are full-time.
- `multiple_full_time_jobs` has near-zero variation — likely a low-importance feature.
- `STATE NUMBER` behaves as a **categorical identifier**, not a continuous numerical value.

**Univariate Analysis – Categorical Features**

Before performing `categorical analysis`, certain columns were excluded because they were not suitable for meaningful visualization and analysis.

**Columns Removed**

| Column      | Reason                                                      |
| ----------- | ----------------------------------------------------------- |
| `LAST NAME`   | Personal identifier with high uniqueness                    |
| `FIRST NAME`  | Personal identifier with high uniqueness                    |
| `MI`          | Initials with limited analytical value                      |
| `EMPLOY DATE` | Date field, better suited for time-based analysis           |
| `CLASS TITLE` | High cardinality and redundant when CLASS CODE is available |

These columns were removed to avoid overcrowded visualizations and focus on features that provide meaningful business insights.

**Selected Categorical Features**

The following categorical variables were retained for analysis:

* `Agency Name`
* `Class Code`
* `Ethnicity`
* `Gender`
* `Employment Status`
* `Duplicated Record Indicator`
* `Combined Multiple Jobs Indicator`
* `Hide from Search Indicator`

**Insights**:
* `AGENCY NAME` and `CLASS CODE` have too many categories, making the plots crowded and hard to interpret.
* Most employees belong to a few major agencies and class codes, while many categories have very low frequency.
* `ETHNICITY` shows most employees are from `WHITE`, `HISPANIC`, and `BLACK` groups.
* `GENDER` distribution indicates female employees are slightly higher than male employees.
* `STATUS` is dominated by a single employment type, showing class imbalance in employee status categories.
* `duplicated` contains only `True`, so this feature provides no useful variation for analysis or modeling.
  
### 5.2 Bivariate Analysis

- `Bivariate analysis` was performed to understand the relationship between `employee salary` and other `categorical` and `numerical` **variables**.

**1. Categorical Features vs Salary**

`Countplots` were used to examine the distribution of key categorical variables such as `Agency Name`, `Class Code`, `Ethnicity`, `Gender`, `Employment Status`, and other employment-related indicators.

**`Average Salary` by Agency and `State Number`**

**Note:**
> When **a categorical column** has *hundreds* or *thousands* of categories (like `AGENCY`, `AGENCY NAME`, `CLASS CODE`), `sns.barplot()` or `sns.countplot()` becomes very **slow** and **unreadable**. Show Only **Top N**(=`15`) Categories.

Bar plots were created to compare the `average annual` salary across `agencies` and `state numbers`.

**Purpose**

* Identify departments with the highest average salaries.
* Compare salary differences across organizational units.
* Detect agencies with higher compensation structures.

> **Business Insight:** `Agency` may have a moderate impact on `salary`, while `State Number` appears to show **stronger salary differences** and may be an important feature for salary prediction.
> Salary is influenced more by **job classification, agency, and employment status** than by demographic features like `gender` or `ethnicity`.

-----
**2. Numerical Features vs Annual Salary**

Scatter plots were used to analyze the relationship between annual salary and numerical variables such as hourly rate, monthly salary, hours worked per week, and other compensation-related features.

**Purpose**

* Identify relationships between salary and numerical attributes.
* Detect trends, clusters, and potential outliers.
* Assess the strength of associations between variables.

**📌 Bivariate Analysis — Insights**

| Comparison | Finding |
|---|---|
| **`Annual` vs `HRLY RATE`** | Positive relationship — higher hourly rates → higher annual salary |
| **`Annual` vs `MONTHLY`** | Near-perfect linear correlation (MONTHLY is derived from ANNUAL) |
| **`Annual` vs `HRS PER WK`** | Weak positive; most employees cluster at 40 hrs regardless of salary |
| **`Annual` vs `summed_annual_salary`** | Strong positive — reliable predictor |
| **`Annual` vs `multiple_full_time_jobs`** | No meaningful pattern — low importance |
| **`Ethnicity`** | Asian employees show the highest average salary |
| **`Gender`** | Male employees have slightly higher average salaries |
| **`Status`** | Salary varies significantly across employment statuses |

**Conclusion**

 >Bivariate analysis revealed that employee salary is influenced by agency affiliation, employment characteristics, and compensation-related factors. Strong relationships were observed between annual salary and other pay-related variables, while agency-level differences highlighted variations in compensation structures across Texas state departments.

### 5.3 Multivariate Analysis — Correlation Heatmap
1. **Strong Correlation Among Salary Variables**

   * `ANNUAL`, `MONTHLY`, and `summed_annual_salary` show very high positive correlations, indicating that these features contain similar salary-related information.

2. **Potential Data Leakage Risk**

   * Features that are directly derived from salary (such as `MONTHLY` or `summed_annual_salary`) may cause data leakage if used as predictors for salary prediction and should be reviewed carefully.

3. **Weak Correlation with Demographic Features**

   * Variables such as gender and ethnicity exhibit low correlation with salary, suggesting they have limited direct influence on compensation.

4. **Moderate Relationship with Work-Related Features**

   * Employment-related variables such as hours worked and job classifications show a moderate association with salary, indicating their importance in salary determination.

5. **Low Multicollinearity Among Most Features**

   * Apart from salary-derived variables, most features do not exhibit strong correlations with each other, reducing the risk of multicollinearity in machine learning models.

6. **Need for Feature Selection**

   * Highly correlated features should be evaluated and potentially removed to improve model efficiency and avoid redundant information.

**📌 Correlation Analysis — Insights**

- **`MONTHLY` ↔ `ANNUAL`**: Perfect correlation (1.00) — `MONTHLY` is directly derived from `ANNUAL` (target leakage risk).
- **`summed_annual_salary` ↔ `ANNUAL`**: Strong positive — important predictor.
- **`HRS PER WK`**: Moderate positive correlation — employees working more hours earn more.
- **`HRLY RATE`**: Moderate positive correlation with total compensation.
- **`AGENCY`, `STATE NUMBER`**: Weak correlations with salary.

### 5.4 Time-Based Analysis — Task 3.3

> **Question:** Have salaries, total compensation, and headcount changed over time?

**📈 Average Salary Trend Over Time**

**Insights:**

* The average employee salary shows a generally increasing trend over the years.
* Salary growth suggests periodic pay revisions, promotions, and adjustments in compensation policies.
* Some years exhibit fluctuations, indicating variations in workforce composition and salary structures.

---

**👥 Employee Headcount Trend**

**Insights:**

* Employee headcount has generally increased over time, indicating workforce expansion across Texas state agencies.
* Certain periods show faster growth, reflecting increased hiring activities.
* The trend highlights the growing scale of state government operations and staffing requirements.

---

**🏢 Salary Trend by Agency**

**Insights:**

* Average salaries vary significantly across agencies, indicating differences in job roles, responsibilities, and compensation structures.
* Some agencies consistently maintain higher salary levels than others.
* Salary growth patterns differ among agencies, suggesting agency-specific budgeting and workforce policies.
* The analysis reveals that agency affiliation is an important factor influencing employee compensation.
**📌 Time-Based Analysis — Insights**

| Analysis | Finding |
|---|---|
| **Average Salary Trend** | Declining trend from late 1970s to 2020; earlier cohorts had higher average salaries (~80K+), recent hires average ~40K–50K |
| **Headcount Trend** | Dramatic increase after 2010; peak headcount in 2019 — significant state government workforce expansion |
| **Department Salary Trend** | Varies across agencies; `Texas Dept. of Criminal Justice` and `Health & Human Services` show greatest fluctuations |

> **Conclusion:** Yes — salaries, compensation patterns, and workforce size have changed significantly over time. Increased entry-level hiring and changing workforce composition explain the declining average salary trend.

### 5.5 Outlier Analysis — Task 3.1

> **Question:** Who are the salary outliers?

- A small number of employees earn significantly higher salaries than the majority of the workforce and are identified as salary outliers using the IQR method.
- Most high-salary outliers belong to senior leadership, executive, specialized, or highly skilled professional roles.
- These employees are concentrated in specific agencies where strategic, administrative, or technical expertise commands higher compensation.
- The salary distribution is positively skewed due to the presence of these high-income individuals.
- Although classified as statistical outliers, these records represent legitimate business cases and should not be removed without domain justification.

**📌 Outlier Analysis — Insights**

- Salary outliers are a **small, identifiable group** earning significantly above the workforce norm.
- These outliers are predominantly **senior leadership** and **investment management professionals** — Chief Scientific Officer, Chief Investment Officer, Director of Investments, Executive Director, Senior Managing Director.
- Most outliers belong to: **Teacher Retirement System**, **Employees Retirement System**, **Texas Education Agency**, and **Texas Dept. of Transportation**.
- These are **genuine, legitimate salaries** — not data errors — reflecting leadership and specialized professional roles.

### 5.6 Wage Disparity Analysis — Task 3.2

> **Question:** What departments/roles have the biggest salary disparities between managers and employees?

- A significant salary gap exists between managerial and non-managerial positions across many agencies.
- Employees classified under leadership roles (e.g., `Manager`, `Director`, `Chief`, `Executive`, `Supervisor`) earn substantially higher average salaries than regular employees.
- Several agencies exhibit particularly large compensation differences, indicating a more hierarchical pay structure.
- The highest salary disparities are generally observed in agencies with a greater concentration of executive and senior management positions.
- The analysis suggests that job role and organizational hierarchy are major factors influencing employee compensation.

**Wage Disparity — Insights**

- Significant **salary disparities** exist between managerial and employee-level roles across several departments.
- Departments with **executive and administrative leadership** positions show the highest wage gaps.
- Roles like **Directors, Executives, Chiefs** earn substantially more than operational/support staff.
- These disparities reflect differences in **responsibility, decision-making authority, and required expertise** — not necessarily inequities.

## 6. Feature Engineering / Data Processing :
### 6.1 Handling Missing Values:

**Strategy:**
- Drop **`multiple_full_time_jobs`**. Because **`99.99%`** null. Only **14 rows** have a value. Essentially empty.
- **Numerical NULLs** → filled with **median** (robust to outliers in salary data)
- **Categorical NULLs** → filled with **mode** (preserves existing category distribution)
- No records dropped; full dataset integrity maintained.

### 6.2 Handling Outliers (IQR Capping)
**Why capping instead of removal?**
- High salaries (executives, directors) are **genuine observations** — removing them would distort payroll analysis. Capping reduces their disproportionate influence on model training while preserving the data.

### 6.3 Encoding — Categorical to Numerical

| Column | Action | Reason |
|---|---|---|
| `FIRST NAME`, `LAST NAME`, `MI` | **Drop** | Personal identifiers — no predictive value |
| `AGENCY NAME` | **Drop** | Duplicate of numeric `AGENCY` column |
| `CLASS CODE` | **Drop** | Redundant with `CLASS TITLE` |
| `EMPLOY DATE` | **→ EXPERIENCE** | Convert to years of experience |
| `ETHNICITY`, `GENDER` | **One-Hot Encoding** | Nominal categories |
| `STATUS` | **Binary flags** | Create `Classified`, `Regular`, `Full_Time` binary columns |
| `CLASS TITLE` | **Label Encoding** | High cardinality |
| `duplicated`, `combined_multiple_jobs`, `hide_from_search` | **Drop** | **`99%`** null vales.Hence these columns are completely useless for modelling. |

### 6.4 Feature Creation — EXPERIENCE

```python
df['EXPERIENCE'] = current_year - df['EMPLOY YEAR']
df.drop('EMPLOY YEAR', axis=1, inplace=True)
```

- `Experience` is more meaningful than a raw year.Hence, Drop the `employ year` column.

## 7. Feature Selection

### 7.1 Correlation with Target

**📌 Feature Selection — Insights**

- **`MONTHLY`** dropped — perfect correlation (1.00) with target `ANNUAL` constitutes target leakage.
- **`Full_Time`** dropped — `0.94` correlation with `HRS PER WK`; `HRS PER WK` retained as it contains richer continuous information.
- Remaining features provide diverse, non-redundant information for model training.

## 8. Train-Test Split & Scaling
- Split `Input` and `Output` Features.
- Split Data into `Train` and `Test` Sets.
   - **Scale** the data for `Linear Regression` model

## 9. Model Building

Four models are trained and compared:

| Model | Type | Why Used |
|---|---|---|
| **Linear Regression** | Parametric | Baseline; captures linear relationships |
| **Decision Tree** | Non-parametric | Captures non-linear decision rules |
| **Random Forest** | Ensemble | Reduces overfitting; robust predictions |
| **XGBoost** | Gradient Boosting | State-of-the-art; handles complex interactions |

## 10. Model Evaluation

Metrics used:
- **MAE** (Mean Absolute Error) — average prediction error in dollars
- **RMSE** (Root Mean Squared Error) — penalizes large errors more heavily
- **R² Score** — proportion of variance explained by the model (higher = better)

**📌 Model Evaluation — Insights**

| Model | MAE | RMSE | R² Score | Notes |
|---|---|---|---|---|
| **Random Forest** | 2,501 | 4,576 | **0.9146** | 🏆 Best overall |
| Decision Tree | ~3,200 | ~5,400 | 0.8553 | Good but lower than RF |
| XGBoost | ~4,100 | ~6,500 | 0.7718 | Moderate performance |
| Linear Regression | ~8,000 | ~12,000 | 0.2410 | Salary is non-linear — poor fit |

> **Winner: Random Forest Regressor** — explains ~91.5% of salary variance. Selected for hyperparameter tuning.

## 11. Hyperparameter Tuning — Random Forest

**Why RandomizedSearchCV?** 
- With 140,000+ rows, `GridSearchCV` is prohibitively slow. `RandomizedSearchCV` samples from the parameter distribution efficiently.

## 12. Final Evaluation & Model Comparison
**📌 Tuning Results**

| Metric | Before Tuning | After Tuning | Improvement |
|---|---|---|---|
| **MAE** | 2,501.48 | 2,461.43 | ↓ ~40 |
| **RMSE** | 4,576.07 | 4,469.37 | ↓ ~107 |
| **R² Score** | 0.9146 | **0.9186** | ↑ +0.004 |

> The tuned model explains approximately **91.86%** of annual salary variance — a meaningful improvement over the baseline.

## 13. Model Saving (.pkl)

- Save tuned model.
- Model saved as `texas_salary_prediction.pkl`.
```python
with open('texas_salary_prediction.pkl', 'wb') as file:
    pickle.dump(best_rf, file)
```

- Load the `.pkl` File
- "Model loaded successfully.
```python

with open('texas_salary_prediction.pkl', 'rb') as file:
    model = pickle.load(file)
```
## 14. Conclusion & Project Summary

---

### 🏁 Project Outcome

This project successfully built an **end-to-end salary prediction pipeline** for Texas state government employees. Starting from raw HR data, we performed domain analysis, comprehensive EDA, feature engineering, and trained four machine learning models.

---

### 📊 Key Findings — EDA

| Finding | Detail |
|---|---|
| **Salary Distribution** | Right-skewed; most employees earn $40K–$60K; executive outliers up to $550K |
| **Outliers (Task 3.1)** | Senior executives (Chief Investment Officers, Directors) at retirement systems & transport agencies |
| **Wage Disparity (Task 3.2)** | Largest manager–employee gaps in agencies with executive leadership; reflects legitimate hierarchical compensation |
| **Time Trends (Task 3.3)** | Average salary declining (more entry-level hires); headcount peaked in 2019; department trends vary |
| **Top Salary Predictors** | `summed_annual_salary`, `HRLY RATE`, `HRS PER WK`, `CLASS TITLE` |

---

### 🤖 Model Performance Summary

| Model | R² Score | MAE | RMSE |
|---|---|---|---|
| Linear Regression | 0.2410 | ~8,000 | ~12,000 |
| Decision Tree | 0.8553 | ~3,200 | ~5,400 |
| XGBoost | 0.7718 | ~4,100 | ~6,500 |
| Random Forest (Before) | 0.9146 | 2,501 | 4,576 |
| **Random Forest (Tuned)** | **0.9186** | **2,461** | **4,469** |

---

### ✅ Final Selected Model: Tuned Random Forest Regressor

- **R² Score: 0.9186** — explains ~92% of salary variation
- **MAE: `$`2,461** — on average predictions are within `$`2,461 of actual salary
- Saved as `texas_salary_prediction.pkl` for deployment

---

### 💡 Business Recommendations

1. **Salary Planning**: Use the model to benchmark new-hire salaries against peers with similar job classifications, agencies, and experience.
2. **Equity Audits**: Address gender and ethnicity-based salary gaps identified in EDA.
3. **Workforce Planning**: The sharp headcount increase post-2015 and declining average salaries suggest increased entry-level hiring — invest in structured career progression and longevity pay.
4. **Department Budgeting**: Use predicted salary distributions per agency to plan annual payroll budgets more accurately.

---

### 📁 Deliverables

| File | Description |
|---|---|
| `Texas_Salary_Prediction_Report.ipynb` | This project report |
| `Final_texas_salary_prediction.pkl` | Trained & tuned Random Forest model |

---
