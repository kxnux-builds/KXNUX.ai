# 🚀 KXNUX.ai — AI-Powered Salary Prediction System

KXNUX.ai is an AI-powered salary prediction system for estimating annual salary from a candidate profile. It combines automated data cleaning, exploratory data analysis, preprocessing, a Random Forest regression pipeline, model evaluation, and a modern Streamlit dashboard for HR-focused salary analysis.

> **Important:** This application is a decision-support tool. Predictions are estimates based on the historical dataset and should be reviewed alongside location, market benchmarks, company policy, role scope, skills, and human HR judgment.

## ✨ Features

- Interactive Streamlit salary-prediction dashboard
- Random Forest regression pipeline with 400 trees
- Automatic numeric imputation and categorical one-hot encoding
- Flexible recognition of common dataset column-name variations
- 80/20 train/test split with reproducible results
- Evaluation with R², MAE, MSE, and RMSE
- Automated descriptive statistics and EDA observations
- Six generated visualizations for salary and feature analysis
- Test-set predictions with absolute errors
- Model metadata saved for display in the dashboard
- Modern, responsive HR-oriented interface

## 🧰 Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.8+ |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Regression model | RandomForestRegressor |
| Visualization | Matplotlib, Seaborn |
| Model persistence | Joblib |
| User interface | Streamlit |

## 📁 Project Structure

```text
KXNUX.ai/
├── data/
│   └── Salary_Data.csv                 # Add your training dataset here
├── models/                             # Created automatically by train_model.py
│   ├── salary_prediction_pipeline.joblib
│   └── model_metadata.json
├── outputs/                            # Created automatically by train_model.py
│   ├── plots/
│   │   ├── 01_salary_distribution.png
│   │   ├── 02_experience_vs_salary.png
│   │   ├── 03_education_level_count.png
│   │   ├── 04_salary_by_job_title.png
│   │   ├── 05_correlation_heatmap.png
│   │   └── 06_salary_boxplot.png
│   ├── descriptive_statistics.csv
│   ├── eda_observations.json
│   └── test_predictions.csv
├── app.py                              # Streamlit application
├── train_model.py                      # Data, EDA, and training pipeline
├── requirements.txt                    # Python dependencies
├── LICENSE
└── README.md
```

## 🚀 Quick Start

Run the following commands from the repository root:

```bash
git clone https://github.com/kxnux-builds/KXNUX.ai.git
cd KXNUX.ai

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 1. Read the dataset instructions first

Before training, read the [Dataset Format](#dataset-format) section below. The training script expects a CSV file at exactly:

```text
data/Salary_Data.csv
```

Create the folder if necessary and copy your dataset into it:

```bash
mkdir data
# Copy or move your file to: data/Salary_Data.csv
```

### 2. Train the model

After `Salary_Data.csv` has been added, run:

```bash
python train_model.py
```

The script performs data inspection, cleaning, EDA, model training, evaluation, and artifact generation. On successful completion it automatically creates the required `models/` and `outputs/` folders.

### 3. Start the dashboard

Only after training completes successfully, run:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, normally [http://localhost:8501](http://localhost:8501).

## 📊 Dataset Format

Your `data/Salary_Data.csv` file must contain six logical fields:

| Required field | Type | Purpose | Accepted examples |
|---|---|---|---|
| `Age` | Numeric | Candidate age | `25`, `34` |
| `Gender` | Categorical | Gender/sex value | `Male`, `Female` |
| `Education Level` | Categorical | Highest education level | `Bachelor's`, `Master's`, `PhD` |
| `Years of Experience` | Numeric | Professional experience | `0`, `4.5`, `12` |
| `Job Title` | Categorical | Candidate role | `Software Engineer`, `Data Scientist` |
| `Salary` | Numeric | Target annual salary | `65000`, `95000` |

The training script recognizes common aliases, including:

- Gender: `Gender`, `Sex`
- Education: `Education Level`, `Education`, `Degree`
- Experience: `Years of Experience`, `Experience`, `Experience Years`
- Job: `Job Title`, `Job`, `Designation`, `Role`
- Target: `Salary`, `Annual Salary`, `Salary USD`, `Income`

Example:

```csv
Age,Gender,Education Level,Years of Experience,Job Title,Salary
24,Male,High School,1,Junior Developer,45000
28,Female,Bachelor's,4,Software Engineer,70000
35,Male,Master's,10,Data Scientist,110000
42,Female,PhD,15,Engineering Manager,135000
```

### Dataset preparation notes

- Save the file as CSV, not Excel or JSON.
- Use the exact filename `Salary_Data.csv`.
- Put the file inside the root-level `data/` folder.
- The `Salary` column is required and must contain non-negative numeric values.
- Missing numeric values are median-imputed by the model pipeline.
- Missing categorical values are filled with the most frequent category.
- Rows without a valid salary are removed during cleaning.
- Salary is displayed in the same units and currency represented by your dataset; the application does not convert currencies.

## 🔄 What `train_model.py` Does

The training workflow is:

1. Loads `data/Salary_Data.csv`.
2. Detects and standardizes supported column-name variations.
3. Converts numeric columns and removes invalid target rows.
4. Generates descriptive statistics.
5. Creates EDA charts and data-driven observations.
6. Splits the cleaned data into 80% training and 20% test sets.
7. Imputes missing values and one-hot encodes categorical features.
8. Trains a `RandomForestRegressor`.
9. Calculates R², MAE, MSE, and RMSE.
10. Saves the trained pipeline, metadata, charts, and test predictions.

Training configuration currently uses:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
```

The model is configured with `n_estimators=400`, `min_samples_leaf=2`, `random_state=42`, and `n_jobs=-1`.

## 📈 Generated Outputs

After a successful training run:

### `models/`

- `salary_prediction_pipeline.joblib` — complete preprocessing and regression pipeline loaded by Streamlit.
- `model_metadata.json` — evaluation metrics, row counts, and feature names.

### `outputs/`

- `descriptive_statistics.csv` — descriptive statistics for the cleaned dataset.
- `test_predictions.csv` — test features, actual salary, predicted salary, and absolute error.
- `eda_observations.json` — generated observations about distributions, correlations, education, job titles, and outliers.
- `plots/` — salary distribution, experience relationship, education counts, job-title salary distributions, correlation heatmap, and salary boxplot.

## 🖥️ Streamlit Dashboard

The dashboard loads the trained model and dataset, then provides:

- Candidate profile controls for age, gender, education, experience, and job title
- Estimated annual salary in the dataset's original units
- Experience category: entry level, junior, mid-level, or senior/lead
- HR-oriented recommendation text
- R², MAE, MSE, and RMSE metrics
- A preview of the first 10 historical dataset records

If the model is missing, the dashboard instructs you to place the dataset in `data/Salary_Data.csv` and run `python train_model.py` first.

## 🧪 Evaluation Metrics

| Metric | Meaning |
|---|---|
| R² Score | Proportion of target variance explained by the model; higher is generally better. |
| MAE | Average absolute difference between actual and predicted salary. |
| MSE | Average squared prediction error; large errors receive more weight. |
| RMSE | Square root of MSE, expressed in the same units as salary. |

Metrics are calculated on the held-out 20% test set and should be interpreted in the context of the dataset size, quality, coverage, and salary units.

## 🛠️ Troubleshooting

### Dataset not found

```text
FileNotFoundError: Dataset not found at .../data/Salary_Data.csv
```

Confirm that the file exists at `data/Salary_Data.csv`, including capitalization and spelling.

### Required column not found

Check that the CSV includes all six logical fields and that the names match one of the supported aliases. Inspect the printed list of available columns in the error message.

### Model not found in Streamlit

Run the training command successfully before starting the dashboard:

```bash
python train_model.py
streamlit run app.py
```

### Missing package errors

Activate your virtual environment and reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

### Training is slow

Random Forest uses 400 trees and all available CPU cores. For experimentation, reduce `n_estimators` in `train_model.py`, then retrain the model.

## 🔐 Privacy and Responsible Use

Do not commit confidential employee information or personally identifiable information. Use anonymized or appropriately licensed datasets. Historical salary data may contain bias or reflect outdated compensation practices; predictions should support—not replace—human review and fair-pay policies.

## 🤝 Contributing

Contributions are welcome:

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Make and test your changes.
4. Commit them with a clear message.
5. Push the branch and open a pull request.

For significant changes, please open an issue first to discuss the proposed improvement.

## 📄 License

This project is available under the [MIT License](LICENSE).

## 👤 Author

- Author: Kishanu Mondal
- GitHub: https://github.com/kxnux-builds
- LinkedIn: https://www.linkedin.com/in/kishanu-mondal/
- X (Twitter): https://x.com/Kxnux_Dev

If this project is useful, consider giving the repository a star ⭐.
