# HR Analytics Dashboard (Under Development)

![HR Analytics Dashboard](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Overview

The HR Analytics Dashboard is a comprehensive analytical tool designed to support human resource management within organizations through data-driven insights. Combining expertise from certified social insurance and labor consultancy and experience in the HR industry with cutting-edge data science technologies, **this dashboard is currently under development. Its purpose is to provide analysis on various HR challenges within organizations.**

Furthermore, **we envision this tool functioning to** analyze and visualize a wide range of aspects, such as turnover prediction, recruitment efficiency, employee performance, compensation analysis, employee engagement, and labor productivity, **ultimately supporting** data-driven decision making.


## 🌟 Key Features

- **Attrition Analysis**: Turnover risk prediction and factor analysis using machine learning models
- **Talent Acquisition Analysis**: Recruitment source efficiency and cost analysis
- **Talent Development & Performance Analysis**: Performance evaluation trend monitoring and development impact measurement
- **Compensation Analysis**: Salary equity assessment and market competitiveness analysis
- **Employee Engagement**: Text mining and sentiment analysis of satisfaction survey data
- **Workforce Productivity/Work Style**: Correlation analysis between work patterns and productivity
- **Predictive Analytics & Simulation**: Prediction of HR initiative impacts and optimization recommendations

## 🔧 Technology Stack (Tentative)

- **Frontend**: Streamlit (interactive dashboard)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib
- **Machine Learning**: Scikit-learn, SHAP (model explainability)
- **Text Analysis**: NLTK, spaCy
- **Statistical Analysis**: SciPy, StatsModels
- **Documentation**: Sphinx, Read the Docs

## 🚀 Installation and Execution

### Prerequisites

- Python 3.8 or higher
- pip (package manager)

### Installation Steps

1. Clone the repository

```bash
git clone https://github.com/yf591/hr-analytics-dashboard.git
cd hr-analytics-dashboard
```

2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Prepare sample data

```bash
python src/data/prepare_sample_data.py
```

### Running the Application

```bash
streamlit run src/app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`.

## 📂 Repository Structure

```
hr-analytics-dashboard/
├── .github/                         # GitHub related settings
│   └── workflows/                   # GitHub Actions
├── data/                            # Data files
│   ├── processed/                   # Processed data
│   ├── interim/                     # Intermediate data
│   ├── external/                    # External reference data
│   └── raw/                         # Raw data sets
├── docs/                            # Documentation
│   ├── analysis_insights/           # Analysis reports
│   ├── assets/                      # Images and assets
│   ├── data_dictionary.md           # Data definitions
│   └── user_guide.md                # Usage guide
├── notebooks/                       # Jupyter Notebooks
│   ├── exploration/                 # Data exploration
│   ├── modeling/                    # Model construction
│   └── insights/                    # Insight extraction
├── src/                             # Source code
│   ├── data/                        # Data processing
│   ├── features/                    # Feature engineering
│   ├── models/                      # Analytical models
│   ├── visualization/               # Visualization components
│   ├── pages/                       # Dashboard pages
│   ├── utils/                       # Utilities
│   └── app.py                       # Main application
├── tests/                           # Test code
├── .gitignore                       # Git exclusion settings
├── LICENSE                          # License
├── README.md                        # Project overview
├── requirements.txt                 # Dependencies
├── setup.py                         # Package configuration
└── config.yaml                      # Application settings
```

## 📊 Detailed Analysis Pages

### 1. Attrition Analysis Page

Understand turnover trends and predict future attrition risks.

**Main Features:**
- Time-series trend analysis of turnover rates
- Turnover rate comparison by department/job role/manager
- Machine learning-based attrition prediction models
- Attrition factor analysis using SHAP values
- Identification of high-risk employees

**Implementation Technologies:**
- Random Forest/Gradient Boosting prediction models
- Time-series analysis for trend identification
- SHAP values for factor visualization
- Cohort analysis for retention rates by hire date

### 2. Talent Acquisition Analysis Page

Analyze recruitment process efficiency and source quality.

**Main Features:**
- Cost-efficiency evaluation by recruitment channel
- Recruitment process funnel analysis
- Correlation between recruitment sources and post-hire performance
- Recruitment timing and cost trend analysis
- Applicant demographic data visualization

**Implementation Technologies:**
- Funnel visualization
- Cost-efficiency analysis
- Correlation analysis
- Time-series aggregation

### 3. Talent Development & Performance Analysis Page

Analyze employee performance evaluations and development initiative impacts.

**Main Features:**
- Performance rating distribution and trend analysis
- Correlation evaluation between training investment and performance
- Skill map and skill gap visualization
- 9-box analysis (Performance × Potential)
- High performer characteristic analysis

**Implementation Technologies:**
- Radar charts for skill visualization
- Scatter plots for correlation analysis
- Clustering for employee grouping
- Time-series analysis for growth curve tracking

### 4. Compensation Analysis Page

Evaluate organizational compensation structure and pay equity.

**Main Features:**
- Salary distribution comparison with market benchmarks
- Relationship analysis between compensation and performance/tenure
- Statistical testing for pay gaps (by gender, age groups, etc.)
- Compensation ratio analysis
- Compensation optimization simulation

**Implementation Technologies:**
- Box plots for distribution comparisons
- Statistical tests (t-tests, ANOVA)
- Regression analysis
- Heat maps for salary band visualization

### 5. Employee Engagement Page

Analyze employee satisfaction and engagement levels.

**Main Features:**
- Time-series trends of engagement scores
- Engagement comparison by department/position
- Text mining of open comments
- Importance analysis of engagement factors
- Correlation between engagement and performance/turnover

**Implementation Technologies:**
- Sentiment analysis
- Topic modeling (LDA)
- Word clouds
- Factor analysis

### 6. Workforce Productivity/Work Style Analysis Page

Analyze the relationship between work styles and productivity.

**Main Features:**
- Relationship analysis between working hours and productivity
- Correlation between remote work ratio and performance
- Relationship between team composition and productivity
- Optimal working hours pattern estimation
- Productivity comparison by department

**Implementation Technologies:**
- Polynomial regression for optimal point estimation
- Scatter plots for correlation analysis
- Time-series aggregation
- Clustering for pattern discovery

### 7. Predictive Analytics & Simulation Page

Predict the effects of HR initiatives and recommend optimal interventions.

**Main Features:**
- Turnover reduction initiative simulation
- What-If analysis for compensation optimization
- Recruitment plan optimization recommendations
- Training ROI predictions
- Organizational change impact simulation

**Implementation Technologies:**
- Monte Carlo simulation
- Sensitivity analysis
- Optimization algorithms
- Predictive modeling for future forecasting

## 📈 Data Sources

This dashboard supports the following data sources:

1. **Internal HR Information Systems**: Basic employee information, compensation, attendance data, etc.
2. **Performance Evaluation Systems**: Evaluation results, feedback data
3. **Applicant Tracking Systems**: Candidate data, recruitment process information
4. **Employee Survey Data**: Engagement surveys, pulse survey results
5. **Public Datasets**: Industry benchmarks, market salary levels, etc.

Supported data formats:
- CSV/TSV files
- Excel spreadsheets
- SQL database connections
- API integration (optional)

## 📋 Use Cases

### Use Case 1: Attrition Risk Management

1. Understand organization-wide turnover trends through the attrition analysis page
2. Run the attrition prediction model to identify high-risk employees
3. Analyze turnover factors by department and develop countermeasures
4. Predict intervention effects using the simulation page

### Use Case 2: Compensation Structure Optimization

1. Analyze current compensation structure through the compensation analysis page
2. Evaluate correlation between compensation and performance
3. Check for disparities through equity analysis
4. Consider budget allocation plans using compensation optimization simulation

### Use Case 3: Engagement Improvement Planning

1. Review satisfaction score trends on the engagement page
2. Identify interdepartmental differences
3. Extract key issues through text mining
4. Analyze relationships between engagement factors and turnover/performance
5. Propose effective improvement measures

## 🔍 Analytical Methods Explained

### Machine Learning Models

#### Attrition Prediction Model
- **Algorithms**: Random Forest, Gradient Boosting, Logistic Regression, XGBoost, LightGBM
- **Features**: Tenure, compensation level, performance ratings, promotion intervals, working hours, etc.
- **Evaluation Metrics**: AUC-ROC, accuracy, recall
- **Explainability**: Model interpretation using SHAP values

#### Engagement Factor Analysis
- **Methodology**: Principal Component Analysis, Factor Analysis
- **Inputs**: Employee survey question responses
- **Outputs**: Key engagement factors and their importance

### Statistical Methods

#### Compensation Equity Analysis
- **Testing Methods**: t-tests (two-group comparison), ANOVA (multi-group comparison)
- **Adjustments**: Controls for position, years of experience, performance
- **Visualization**: Heat maps of adjusted compensation disparities

#### Recruitment Efficiency Evaluation
- **Metrics**: Recruitment cost efficiency, time-to-hire, offer acceptance rate
- **Benchmarking**: Comparison with industry standards
- **Trend Analysis**: Efficiency changes over time

### Text Analysis

#### Employee Feedback Analysis
- **Preprocessing**: Text cleaning, tokenization, stop word removal
- **Sentiment Analysis**: Positive/negative/neutral category classification
- **Topic Modeling**: Key topic extraction using LDA
- **Visualization**: Word clouds, topic distributions

## 🔒 Data Privacy and Security

This dashboard implements the following privacy protection measures:

1. **Data Anonymization**: Processing of personally identifiable information
2. **Access Control**: Role-based access management
3. **Data Encryption**: Encryption of stored data and communications
4. **Audit Logs**: Recording and tracking of analytical operations
5. **Synthetic Data Option**: Option to use statistically equivalent synthetic data instead of actual data


## 📝 License

This project is released under the MIT License. See the [LICENSE](./LICENSE) file for details.

## 👥 Developer

- [yf591](https://github.com/yf591)

---
<!-- 
## Contact

For questions or feedback, please contact us via [GitHub Issues](https://github.com/yf591/hr-analytics-dashboard/issues). -->
