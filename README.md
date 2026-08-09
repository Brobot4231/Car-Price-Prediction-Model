# 🚗 Car Price Prediction Model

> A Machine Learning pipeline that predicts used car selling prices using **Linear Regression** with **log transformation**, deployed as an interactive **Streamlit** web application.

---

## 📌 Overview

The used car market is highly dynamic, with prices influenced by age, mileage, engine specs, fuel type, and ownership history. This project builds an end-to-end **price prediction system** trained on Indian used car listings to estimate selling prices accurately. The pipeline addresses messy real-world data challenges—mixed units, missing values, duplicates, and skewed price distributions—to deliver reliable predictions.

**Key Improvement:** Log transformation on target variable boosts R² by compressing outliers and linearizing exponential depreciation.

---

## ✨ Features

- 🔧 **Robust Data Cleaning** — Custom parser extracts numeric values from mixed-unit strings (e.g., "1248 CC", "74 bhp", "23.4 kmpl")
- 🧹 **Missing Value & Duplicate Handling** — Drops incomplete records and ~1,189 duplicates for a clean training set
- 📈 **Log Transform Target** — Converts skewed price distribution into a normal distribution for better linear regression fit
- 🏷️ **Categorical Encoding** — Label/One-Hot encoding for fuel type, transmission, seller type, and ownership
- 🧮 **Real-Time Price Estimation** — Input car specs and get instant selling price predictions
- 🌐 **Interactive 3D Plot** — Visualize actual vs predicted prices with residual analysis

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.x |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Machine Learning** | Scikit-Learn |
| **Web Framework** | Streamlit |
| **Model Serialization** | Pickle |
| **Environment** | Jupyter Notebook |

---

## 📁 Dataset

- **Source:** `Cardetails.csv` (Used car market listings)
- **Initial Records:** 8,128 transactions
- **Final Records:** 6,717 transactions (after cleaning)
- **Features:** 12 columns (after dropping `torque`)

### Feature Description
| Feature | Description | Type |
|---------|-------------|------|
| `name` | Car model / brand name | Categorical |
| `year` | Year of manufacture | Numeric |
| `selling_price` | Listed selling price in INR | Numeric (Target) |
| `km_driven` | Total kilometers driven | Numeric |
| `fuel` | Fuel type (Petrol / Diesel / LPG / CNG) | Categorical |
| `seller_type` | Seller category (Individual / Dealer) | Categorical |
| `transmission` | Gearbox type (Manual / Automatic) | Categorical |
| `owner` | Ownership history (First / Second / Third) | Categorical |
| `mileage` | Fuel efficiency (kmpl / km/kg) | Numeric |
| `engine` | Engine displacement in CC | Numeric |
| `max_power` | Peak power output in bhp | Numeric |
| `seats` | Seating capacity | Numeric |

### Data Cleaning Summary
| Step | Details |
|------|---------|
| **Initial Shape** | 8,128 rows × 13 columns |
| **Column Dropped** | `torque` (complex mixed units + 221 missing values) |
| **Null Rows Removed** | 221 records |
| **Duplicates Removed** | 1,189 records |
| **Final Clean Dataset** | **6,717 rows × 12 columns** |

> ⚠️ **Note:** The `torque` column was dropped due to inconsistent formatting (e.g., "190Nm@ 2000rpm", "12.7@ 2700kgm@ rpm") and high parsing complexity.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Brobot4231/Car-Price-Prediction.git
cd Car-Price-Prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
pandas
numpy
matplotlib
plotly
scikit-learn
streamlit
```

---

## 🚀 Usage

### 1. Explore & Train the Model
Open `Car_Price_Prediction_Model.ipynb` in Jupyter Notebook and run all cells to:
- Load & inspect the `Cardetails.csv` dataset
- Drop the `torque` column and remove null values
- Eliminate duplicate records
- Clean mixed-unit columns (`mileage`, `engine`, `max_power`, `name`) using the custom `clean_data()` parser
- Convert data types and encode categorical variables
- Apply **log transformation** to `selling_price`
- Train the Linear Regression model
- Evaluate with 3D scatter plots (Actual vs Predicted)
- Save the model as `car_price_model.pkl` and encoders as `encoders.pkl`

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will start at `http://localhost:8501`

### 3. How to Use
| Step | Action |
|------|--------|
| **1** | Enter car details (year, km driven, mileage, engine, max power, seats) |
| **2** | Select categorical options (fuel, seller type, transmission, owner) |
| **3** | Click **Predict** to view the estimated selling price in INR |

---

## 📊 Model Performance

### Training Pipeline
| Step | Details |
|------|---------|
| **Initial Dataset** | 8,128 car listings |
| **After Cleaning** | 6,717 records |
| **Target Transform** | `log(selling_price)` |
| **Algorithm** | Linear Regression |
| **Categorical Encoding** | Label / One-Hot Encoding |
| **Model Serialization** | Pickle (`.pkl`) |

### Why Log Transformation?

| Problem | Solution |
|---------|----------|
| **Outliers** | Log compresses extreme luxury prices, shaping the target into a symmetrical normal distribution |
| **Exponential Decay** | Cars depreciate proportionally (e.g., ~15% annually); log converts this curved decay into a straight line |
| **Heteroscedasticity** | Forces the model to focus on percentage errors instead of absolute cash differences, stabilizing variance across all price ranges |

### Evaluation
- **3D Visualization:** Actual vs Predicted price scatter confirms balanced residuals across cheap and expensive cars
- **R² Improvement:** Significant score boost after log transformation
- **Error Spread:** Stable, trumpet-free variance across the full price spectrum

---

## 📂 Project Structure

```
Car-Price-Prediction/
│
├── 📓 Car_Price_Prediction_Model.ipynb      # Main Jupyter Notebook (EDA + Training)
├── 📓 Car_Price_Prediction_Model0.ipynb     # Backup / Alternative Notebook
├── 📊 Cardetails.csv                        # Raw Dataset
├── 🖥️ app.py                                # Streamlit Web Application
├── 🤖 car_price_model.pkl                   # Serialized Linear Regression Model
├── 🔧 encoders.pkl                          # Categorical Feature Encoders
├── 🖼️ Interface image.png                   # 3D Plot / App Screenshot
├── 📄 requirements.txt                      # Python Dependencies
└── 📄 README.md                             # Project Documentation
```

---

## 🖼️ Screenshot
![Interface](Interface%20image.png)

---

## 🔮 Future Scope

- Implement **Random Forest**, **XGBoost**, and **CatBoost** for non-linear patterns
- Hyperparameter tuning with **GridSearchCV** or **Optuna**
- Feature engineering: derive **car age**, extract **brand**, compute **power-to-weight ratio**
- Add **outlier detection** using IQR or Isolation Forest
- Deploy to **AWS / Heroku / Streamlit Cloud** for public access
- Add **SHAP values** for model interpretability and explainability
- Parse the `torque` column properly (RPM-based extraction) instead of dropping it
- Build a **model comparison dashboard** with residual plots and error metrics

---

## 🙏 Acknowledgements

- Used car dataset from Indian automotive market listings
- [Scikit-Learn](https://scikit-learn.org/) for the regression toolkit
- [Streamlit](https://streamlit.io/) for the rapid web application framework
- [Pandas](https://pandas.pydata.org/) for powerful data manipulation

---

## 🔗 Link

- [Car Price Predictor](https://car-price-prediction-model-4231.streamlit.app/)


> **Disclaimer:** This tool is for educational and research purposes only. Predicted prices are estimates based on historical data and should not replace professional vehicle valuation.


