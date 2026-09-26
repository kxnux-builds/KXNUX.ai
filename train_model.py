import os
import json
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Salary_Data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def find_column(df, aliases, required=True):
    normalized = {str(c).strip().lower().replace('_', ' '): c for c in df.columns}
    for alias in aliases:
        key = alias.strip().lower().replace('_', ' ')
        if key in normalized:
            return normalized[key]
    for c in df.columns:
        clean = str(c).strip().lower().replace('_', ' ')
        if any(alias.lower().replace('_', ' ') in clean for alias in aliases):
            return c
    if required:
        raise ValueError(f"Could not find a required column. Tried: {aliases}. Available: {list(df.columns)}")
    return None


def prepare_columns(df):
    age = find_column(df, ['age'])
    gender = find_column(df, ['gender', 'sex'])
    education = find_column(df, ['education level', 'education', 'degree'])
    experience = find_column(df, ['years of experience', 'experience', 'experience years'])
    job_title = find_column(df, ['job title', 'job', 'designation', 'role'])
    salary = find_column(df, ['salary', 'annual salary', 'salary usd', 'income'])

    rename_map = {
        age: 'Age', gender: 'Gender', education: 'Education Level',
        experience: 'Years of Experience', job_title: 'Job Title', salary: 'Salary'
    }
    out = df.rename(columns=rename_map).copy()
    return out[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title', 'Salary']]


def clean_basic(df):
    df = df.copy()
    for col in ['Age', 'Years of Experience', 'Salary']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['Gender', 'Education Level', 'Job Title']:
        df[col] = df[col].astype('object')
        df[col] = df[col].where(df[col].notna(), np.nan)
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(subset=['Salary'])
    df = df[(df['Salary'] >= 0)]
    return df

def save_descriptive_stats(df):
    stats = df.describe(include='all').transpose()
    stats.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_statistics.csv'))

def save_eda(df):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(9, 5))
    sns.histplot(df['Salary'].dropna(), kde=True, bins=30)
    plt.title('Salary Distribution')
    plt.xlabel('Salary')
    plt.ylabel('Number of Employees')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '01_salary_distribution.png'), dpi=180)
    plt.close()
    
    plt.figure(figsize=(9, 5))
    sns.scatterplot(data=df, x='Years of Experience', y='Salary', alpha=0.65)
    sns.regplot(data=df, x='Years of Experience', y='Salary', scatter=False, color='crimson')
    plt.title('Years of Experience vs Salary')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '02_experience_vs_salary.png'), dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    order = df['Education Level'].value_counts().index
    sns.countplot(data=df, y='Education Level', order=order)
    plt.title('Education Level Count')
    plt.xlabel('Employees')
    plt.ylabel('Education Level')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '03_education_level_count.png'), dpi=180)
    plt.close()

    top_jobs = df.groupby('Job Title')['Salary'].median().sort_values(ascending=False).head(15).index
    plt.figure(figsize=(11, 7))
    sns.boxplot(data=df[df['Job Title'].isin(top_jobs)], y='Job Title', x='Salary', order=top_jobs)
    plt.title('Salary by Job Title (Top 15 Median Salary)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '04_salary_by_job_title.png'), dpi=180)
    plt.close()

    numeric = df[['Age', 'Years of Experience', 'Salary']]
    plt.figure(figsize=(7, 5))
    sns.heatmap(numeric.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '05_correlation_heatmap.png'), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 3.8))
    sns.boxplot(x=df['Salary'])
    plt.title('Salary Boxplot')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '06_salary_boxplot.png'), dpi=180)
    plt.close()

def create_observations(df):
    obs = {}
    salary_skew = df['Salary'].skew()
    exp_corr = df[['Years of Experience', 'Salary']].corr().iloc[0, 1]
    edu_counts = df['Education Level'].value_counts()
    median_by_edu = df.groupby('Education Level')['Salary'].median().sort_values(ascending=False)

    if abs(salary_skew) < 0.5:
        dist = 'Salary is relatively balanced around its center.'
    elif salary_skew > 0:
        dist = 'Salary is right-skewed, indicating a smaller number of high-salary employees.'
    else:
        dist = 'Salary is left-skewed, indicating a smaller number of lower-salary employees.'
    obs['salary_distribution'] = [
        dist,
        f"Median salary is approximately {df['Salary'].median():,.0f}.",
        f"The salary range is about {df['Salary'].min():,.0f} to {df['Salary'].max():,.0f}."
    ]

    direction = 'positive' if exp_corr >= 0 else 'negative'
    obs['experience_vs_salary'] = [
        f"Years of experience has a {direction} correlation with salary (r ≈ {exp_corr:.2f}).",
        'Salary generally increases as experience increases, although individual variation remains.',
        'The spread suggests that job title and education can also affect salary.'
    ]

    obs['education_count'] = [
        f"The most common education level is {edu_counts.index[0]} ({edu_counts.iloc[0]} records).",
        f"There are {edu_counts.size} distinct education categories in the dataset.",
        f"The highest median salary by education category is associated with {median_by_edu.index[0]} in this dataset."
    ]

    top_job = df.groupby('Job Title')['Salary'].median().sort_values(ascending=False).index[0]
    low_job = df.groupby('Job Title')['Salary'].median().sort_values().index[0]
    obs['salary_by_job_title'] = [
        f"{top_job} has the highest median salary among job titles in this dataset.",
        f"{low_job} has the lowest median salary among the represented job titles.",
        'Salary distributions overlap, so job title alone should not be used to determine compensation.'
    ]

    corr = df[['Age', 'Years of Experience', 'Salary']].corr()['Salary'].drop('Salary').abs().sort_values(ascending=False)
    obs['correlation_heatmap'] = [
        f"Years of Experience has the strongest relationship with Salary among the numeric variables (absolute r ≈ {corr['Years of Experience']:.2f}).",
        f"Age has an absolute correlation of about {corr['Age']:.2f} with Salary.",
        'Correlation measures linear association and does not by itself prove causation.'
    ]

    q1, q3 = df['Salary'].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = ((df['Salary'] < q1 - 1.5*iqr) | (df['Salary'] > q3 + 1.5*iqr)).sum()
    obs['salary_boxplot'] = [
        f"The interquartile range of salary is approximately {iqr:,.0f}.",
        f"The boxplot flags approximately {outliers} observations as statistical outliers using the 1.5×IQR rule.",
        'Outliers can have a strong influence on regression metrics and should be inspected before deployment.'
    ]
    return obs


def build_model(df):
    features = ['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job Title']
    target = 'Salary'

    X = df[features]
    y = df[target]

    numeric_features = ['Age', 'Years of Experience']
    categorical_features = ['Gender', 'Education Level', 'Job Title']

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_features),
        ('cat', categorical_pipe, categorical_features)
    ])

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        'R2 Score': float(r2_score(y_test, pred)),
        'MAE': float(mean_absolute_error(y_test, pred)),
        'MSE': float(mean_squared_error(y_test, pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
        'Train Rows': int(len(X_train)),
        'Test Rows': int(len(X_test)),
        'Features': features
    }

    predictions = X_test.copy().reset_index(drop=True)
    predictions['Actual Salary'] = y_test.reset_index(drop=True)
    predictions['Predicted Salary'] = pred
    predictions['Absolute Error'] = np.abs(predictions['Actual Salary'] - predictions['Predicted Salary'])
    predictions.to_csv(os.path.join(OUTPUT_DIR, 'test_predictions.csv'), index=False)

    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'salary_prediction_pipeline.joblib'))
    with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    return metrics


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put your CSV file there and name it Salary_Data.csv."
        )

    print('=' * 72)
    print('SALARY PREDICTION SYSTEM')
    print('=' * 72)

    raw = pd.read_csv(DATA_PATH)
    print('\nDATA UNDERSTANDING')
    print('\nFirst 5 records:')
    print(raw.head())
    print(f'\nDataset dimensions: {raw.shape[0]} rows x {raw.shape[1]} columns')
    print('\nData types:')
    print(raw.dtypes)
    print('\nMissing values:')
    print(raw.isnull().sum())

    df = clean_basic(prepare_columns(raw))
    print('\nDescriptive statistics:')
    print(df.describe(include='all'))
    save_descriptive_stats(df)

    print('\nEDA')
    save_eda(df)
    observations = create_observations(df)
    with open(os.path.join(OUTPUT_DIR, 'eda_observations.json'), 'w', encoding='utf-8') as f:
        json.dump(observations, f, indent=4)
    print(f"EDA charts saved to: {PLOT_DIR}")

    print('\nPREPROCESSING AND MODEL')
    print('Numeric missing values: median imputation')
    print('Categorical missing values: most-frequent imputation')
    print('Categorical encoding: One-Hot Encoding')
    print('Target: Salary')
    print('Train/test split: 80/20')
    print('Model: Random Forest Regressor')

    metrics = build_model(df)
    print('\nMODEL EVALUATION')
    print(f"R² Score: {metrics['R2 Score']:.4f}")
    print(f"MAE: {metrics['MAE']:,.2f}")
    print(f"MSE: {metrics['MSE']:,.2f}")
    print(f"RMSE: {metrics['RMSE']:,.2f}")
    print('\nTest predictions saved to outputs/test_predictions.csv')
    print('Model saved to models/salary_prediction_pipeline.joblib')
    print('\nTraining completed successfully.')


if __name__ == '__main__':
    main()
