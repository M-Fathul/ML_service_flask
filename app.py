from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
import pytz

app = Flask(__name__)

API_KEY = "your_api_key_here"

@app.route("/")
# def index():
#     time = datetime.now().date() - timedelta(days=1)
#     return f"Welcome to the Dummy Forecast API! Use /forecast endpoint to get predictions. Current time: {time}"

def fill_missing_dates(df):
    df['ds'] = pd.to_datetime(df['ds'])

    now = datetime.now(pytz.timezone('Asia/Jakarta')).date() - timedelta(days=1)

    full_range = pd.date_range(start=df['ds'].min(), end=now)

    df = df.set_index('ds').reindex(full_range).fillna(0).rename_axis('ds').reset_index()

    return df

def calculate_mape(actual, forecast):
    actual, forecast = np.array(actual), np.array(forecast)

    mask = actual != 0
    valid_actual = actual[mask]
    valid_forecast = forecast[mask]

    if len(valid_actual) > 0 and len(valid_forecast) > 0:
        return np.mean(np.abs((valid_forecast - valid_actual) / valid_actual)) * 100
    else:
        return 0  # atau nilai default lainnya


def generate_insight(mape):

    if mape < 10:
        return {
            "summary": "Sangat akurat.",
            "reason": "Data historis stabil dan pola mudah dipelajari.",
        }

    elif mape < 20:
        return {
            "summary": "Cukup akurat.",
            "reason": "Terdapat fluktuasi pada data historis.",
        }

    else:
        return {
            "summary": "Kurang akurat.",
            "reason": "Data sangat fluktuatif atau pola tidak konsisten akibat minimnya data historis."
        }

@app.route('/forecast', methods=['POST'])
def forecast():

    auth_header = request.headers.get('Authorization')
    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    df = pd.DataFrame(data['data'])
    periods = data.get('periods')

    df = fill_missing_dates(df)



    test_size = min(30, int(len(df) * 0.2))
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_mode='additive'
    )
    model.add_country_holidays(country_name='ID')

    model.fit(train_df)

    future_test = model.make_future_dataframe(periods=test_size)
    forecast_test = model.predict(future_test)

    mape = calculate_mape(
        df['y'].values,
        forecast_test['yhat'].values
    )

    model_full = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_mode='additive'
    )
    model_full.add_country_holidays(country_name='ID')

    model_full.fit(df)

    future = model_full.make_future_dataframe(periods=periods)
    forecast = model_full.predict(future)

    forecast = forecast.tail(periods)

    forecast_result = []
    for _, row in forecast.iterrows():
        forecast_result.append({
            "ds": row['ds'].strftime('%Y-%m-%d'),
            "month": row['ds'].strftime('%B'),
            "week": row['ds'].isocalendar()[1],
            "year": row['ds'].year,
            "yhat": max(0, float(round(row['yhat'], 0))),
            "yhat_lower": max(0, float(round(row['yhat_lower'], 0))),
            "yhat_upper": max(0, float(round(row['yhat_upper'], 0))),
        })

    validation_result = []

    for i in range(len(test_df)):
        validation_result.append({
            "ds": test_df.iloc[i]['ds'].strftime('%Y-%m-%d'),
            "month": test_df.iloc[i]['ds'].strftime('%B'),
            "week": test_df.iloc[i]['ds'].isocalendar()[1],
            "year": test_df.iloc[i]['ds'].year,
            "aktual": int(test_df.iloc[i]['y']),
            "yhat": max(0, float(round(forecast_test.iloc[i]['yhat'], 0))),
        })

    insight = generate_insight(mape)

    return jsonify({
        "forecast": forecast_result,
        "validation": validation_result,
        "mape": float(round(mape, 2)),
        "train_start": df['ds'].min().strftime('%Y-%m-%d'),
        "train_end": df['ds'].max().strftime('%Y-%m-%d'),
        "insight": insight
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)