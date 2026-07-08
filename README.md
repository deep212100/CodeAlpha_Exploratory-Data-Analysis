# Task 2 — Exploratory Data Analysis (EDA): Titanic Dataset

This package contains a complete EDA solution covering all requirements from the task brief:

- ✅ Ask meaningful questions about the dataset before analysis
- ✅ Explore the data structure, including variables and data types
- ✅ Identify trends, patterns, and anomalies within the data
- ✅ Test hypotheses and validate assumptions using statistics and visualization
- ✅ Detect potential data issues or problems to address in further analysis

## Contents

```
├── EDA_Titanic.ipynb          # Main deliverable: full notebook (code + narrative + outputs)
├── EDA_Titanic_Report.html    # Same notebook, pre-rendered as a viewable HTML report
├── run_eda.py                 # Standalone script version (generates plots + prints stats)
├── data/
│   └── titanic.csv            # Dataset used (891 passengers, seaborn's Titanic sample)
├── plots/                     # All charts exported as standalone PNGs
│   ├── 01_survival_counts.png
│   ├── 02_survival_by_sex.png
│   ├── 03_survival_by_class.png
│   ├── 04_age_distribution.png
│   ├── 05_age_vs_survival.png
│   ├── 06_fare_by_class_boxplot.png
│   ├── 07_correlation_heatmap.png
│   ├── 08_missing_values.png
│   └── 09_family_size_vs_survival.png
└── eda_output.txt             # Raw console output from run_eda.py (stats summary)
```

## How to use

- **Just want to view it?** Open `EDA_Titanic_Report.html` in any browser — no setup needed.
- **Want to run/edit it?** Open `EDA_Titanic.ipynb` in Jupyter (requires `pandas`, `numpy`,
  `seaborn`, `matplotlib`, `scipy`).
- **Prefer a plain script?** Run `python run_eda.py` from this folder (needs the same libraries).

## Key findings (summary)

- Overall survival rate: **38.4%**
- **Sex** was the strongest predictor of survival (women survived far more than men,
  chi-square p < 0.001)
- **Passenger class** and **fare** were strongly linked to survival — 1st class passengers
  survived at much higher rates than 3rd class (p < 0.001)
- **Age** had a smaller but still statistically significant effect (p ≈ 0.04)
- **Data quality issues found:** `age` missing ~20%, `deck` missing ~77%, 107 duplicate rows,
  116 fare outliers (IQR method), and some £0 fares that need investigation before modeling

Full details, code, and charts are in the notebook/report.
