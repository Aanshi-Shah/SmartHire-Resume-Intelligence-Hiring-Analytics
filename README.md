📊 SmartHire : Resume Intelligence & Hiring Analytics

 📌 Overview
SmartHire is a data analytics and machine learning project that analyzes candidate resume data and predicts Resume Score (0–100). The system demonstrates how data-driven insights can support smarter hiring decisions.

It Predict the Resume Score from 0-100 by giving input values as Experience Years , Technical Score and Soft Skills Score for new data.

--------------------------------------------------------------------------------------------------

🛠 Tech Stack:

Python | Pandas | NumPy | Matplotlib | Seaborn | Plotly | Scikit-learn | Streamlit
--------------------------------------------------------------------------------------------------

 📊 Exploratory Data Analysis (EDA)
- Resume Score distribution analysis  
- Correlation heatmap of numerical features  
- Experience-based segmentation  
- Feature importance analysis  

--------------------------------------------------------------------------------------------------

🤖 Prediction Model
- Model Used: Random Forest Regression  
- Target Variable: Resume Score (0–100)  
- Evaluation Metrics: R² Score & MAE  
- Identifies key factors influencing candidate evaluation  

---------------------------------------------------------------------------------------------------

🖥 Interactive Dashboard
Built using Streamlit & Plotly featuring:
- KPI metrics  
- Interactive visualizations  
- Correlation insights  
- Resume Score prediction tool  

---------------------------------------------------------------------------------------------------
Run locally:
```bash
pip install -r requirements.txt

streamlit run app.py

