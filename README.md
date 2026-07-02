# ev-battery-degradation
EV Battery Degradation Monitoring Dashboard
# EV Battery Degradation — Industrial IoT Dashboard

Predictive maintenance system for monitoring electric vehicle battery health 
across a fleet of 10,000 vehicles using real sensor data.

## Dataset
[EV Battery Degradation Dataset on Kaggle](https://www.kaggle.com/datasets/bertnardomariouskono/electric-vehicle-ev-battery-degradation-and-charge)  
10,000 vehicles · NMC & LFP chemistries · 13 features · SoH as target variable

## 🔗 Live Demo
https://ev-battery-degradation.onrender.com
```

## Analysis Notebook
Open `battery_colab.ipynb` in [Google Colab](https://colab.research.google.com) 
and upload the CSV when prompted.

## What the Dashboard Shows
- SoH distribution by car model (Tesla, Ford, Hyundai, Wuling, BYD)
- SoH vs charging cycles, vehicle age, internal resistance
- Temperature impact on battery health
- Fast charging & driving style effects
- Fleet status — Healthy vs Replace Required
- NMC vs LFP chemistry comparison

## 🛠️ Tech Stack
- Python · Plotly Dash · Pandas · NumPy · Scikit-learn
- Google Colab for analysis
