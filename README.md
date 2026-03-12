# 🎓 Student Dropout, Retention and Re-Enrollment Analytics

### Corporación Universitaria Lasallista

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-purple?logo=plotly)
![ScikitLearn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-Academic-green)

---

# 📊 Project Overview

This project analyzes **student dropout, retention, and potential re-enrollment patterns** at **Corporación Universitaria Lasallista** using **data analytics, visualization, and machine learning**.

The dashboard allows institutional decision makers to:

* Identify students at risk of dropping out
* Monitor academic retention indicators
* Analyze historical enrollment behavior
* Explore geographic distribution of students
* Evaluate possible student re-entry within a 5-year window

The system was built using **Python, Streamlit, and interactive data visualizations**.

---

# 🧠 Analytical Approach

The project follows a **Data Science workflow**:

1. Data Extraction
2. Data Cleaning (ETL)
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Predictive Modeling
6. Visualization and Decision Support

The predictive component uses a **Random Forest model** to estimate dropout probability.

---

# 📂 Dataset

Dataset used:

```
DataSPSSReingreso.csv
```

Main variables include:

* Student ID
* Gender
* Academic Program
* Faculty
* Socioeconomic Stratum
* City of Residence
* Academic Status
* Cohort Year
* Academic Period

---

# 📈 Dashboard Features

The application provides several analytical modules:

### 📊 Institutional Overview

* Total students
* Active students
* Dropout students
* Dropout rate

### 📚 Academic Analysis

* Students by program
* Students by faculty
* Distribution by gender
* Distribution by socioeconomic stratum

### 🧭 Cohort Analysis

* Cohort retention matrix
* Academic progression visualization

### 🌍 Geographic Analysis

* Student distribution by city

### 🤖 Predictive Analytics

* Dropout probability model using **Random Forest**

---

# 🖥 Dashboard Preview

The dashboard is built using **Streamlit**, allowing interactive exploration with filters for:

* Academic Program
* Faculty
* Gender

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/your-repository/student-dropout-analytics.git
```

Move into the project folder:

```
cd student-dropout-analytics
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Launch the Streamlit dashboard:

```
streamlit run app.py
```

The dashboard will open automatically in your browser.

---

# 📁 Project Structure

```
student-dropout-analytics
│
├── app.py
├── requirements.txt
├── README.md
└── DataSPSSReingreso.csv
```

---

# 🎯 Expected Impact

This project supports **data-driven decision making** in higher education institutions by enabling:

* Early dropout detection
* Targeted student support strategies
* Re-enrollment campaigns
* Improved institutional planning

---

# 👨‍💻 Author

Feibert Alirio Guzmán Pérez
AI and Data Analytics Researcher
Corporación Universitaria Lasallista

---

# 📜 License

This project is intended for **academic and research purposes**.
