
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Load dataset to find the end date
data = pd.read_excel("PJMW_MW_Hourly.xlsx")
data['Datetime'] = pd.to_datetime(data['Datetime'])
last_date = data['Datetime'].max()

# Load trained model
model = joblib.load("random_forest_model.pkl")
expected_features = list(model.feature_names_in_)

# Streamlit UI
st.title("🔌 PJM Energy Forecasting")
st.markdown("Forecast energy consumption for the next 30 days using a Random Forest model.")

# Interactive: Choose number of days to forecast
n_days = st.slider("Select number of days to forecast", min_value=1, max_value=30, value=30)

# Generate future dates starting from last date + 1
start_date = last_date + timedelta(days=1)
future_dates = [start_date + timedelta(days=i) for i in range(n_days)]

# Build features for prediction
features = []
for dt in future_dates:
    hour = 12
    weekday = dt.weekday()
    season = 0 if dt.month in [12, 1, 2] else 1 if dt.month in [3, 4, 5] else 2 if dt.month in [6, 7, 8] else 3
    is_weekend = 1 if weekday >= 5 else 0
    is_holiday = 0

    row = {
        'hour': hour,
        'day': dt.day,
        'month': dt.month,
        'year': dt.year,
        'weekday': weekday,
        'season': season,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday
    }
    features.append(row)

# Predict
future_df = pd.DataFrame(features)
future_df['Predicted'] = model.predict(future_df[expected_features])
future_df['Date'] = [dt.date() for dt in future_dates]

# Plot
fig, ax = plt.subplots()
ax.plot(future_df['Date'], future_df['Predicted'], marker='o', color='orange')
ax.set_title(f"Forecasted Power Demand (Next {n_days} Days)")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted Energy (MW)")
plt.xticks(rotation=45)
st.pyplot(fig)

# Table
st.markdown("### 📊 Tabular Forecast")
st.dataframe(future_df[['Date', 'Predicted']])

# Download
csv = future_df[['Date', 'Predicted']].to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Forecast as CSV", csv, "forecast.csv", "text/csv")

# Suggestions
st.markdown("### 🔧 Suggestions for Future Improvements")
st.markdown("""
- Dynamically adjust hour/day granularity.
- Add holiday detection from a calendar library.
- Visualize uncertainty bounds.
- Incorporate weather data as features.
""")

