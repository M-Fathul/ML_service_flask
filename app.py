from flask import Flask, request, jsonify
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
import pytz

app = Flask(__name__)

API_KEY = "your_api_key_here"

@app.route("/")
def index():
    time = datetime.now().date() - timedelta(days=1)
    return f"Welcome to the Dummy Forecast API! Use /forecast endpoint to get predictions. Current time: {time}"

def fill_missing_dates(df):
    date_col = df.select_dtypes(include=['object']).columns.tolist()
    df[date_col[0]] = pd.to_datetime(df[date_col[0]])

    now = datetime.now(pytz.timezone('Asia/Jakarta')).date() - timedelta(days=1)

    full_range = pd.date_range(start=df[date_col[0]].min(), end=df[date_col[0]].max())

    df = df.set_index(date_col[0]).reindex(full_range).fillna(0).rename_axis(date_col[0]).reset_index()

    return df

def transform_prophet(df):
    date_col = df.select_dtypes(include=['datetime64']).columns.tolist()
    data = df.select_dtypes(exclude=['datetime64']).columns.tolist()
    df = df.rename(columns={date_col[0]: 'ds', data[0]: 'y'})
    return df[['ds', 'y']]

def create_features(df):

    df = df.copy()
    date_col = df.select_dtypes(include=['datetime64']).columns.tolist()
    data = df.select_dtypes(exclude=['datetime64']).columns.tolist()

    df['year'] = df[date_col[0]].dt.year
    df['quarter'] = df[date_col[0]].dt.quarter
    df['month'] = df[date_col[0]].dt.month
    df['day'] = df[date_col[0]].dt.day
    df['day_of_week'] = df[date_col[0]].dt.dayofweek
    df['day_of_year'] = df[date_col[0]].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['lag_1'] = df[data[0]].shift(1)
    df['lag_3'] = df[data[0]].shift(3)
    df['lag_7'] = df[data[0]].shift(7)
    df['lag_14'] = df[data[0]].shift(14)

    df['rolling_mean_7'] = (
        df[data[0]]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df['rolling_mean_14'] = (
        df[data[0]]
        .shift(1)
        .rolling(14)
        .mean()
    )

    df['rolling_std_7'] = (
        df[data[0]]
        .shift(1)
        .rolling(7)
        .std()
    )

    df['rolling_std_14'] = (
        df[data[0]]
        .shift(1)
        .rolling(14)
        .std()
    )
    return df.drop(columns=date_col[0]).fillna(0).astype(float)

def calculate_mape(actual, forecast):
    actual = np.array(actual, dtype=float).flatten()
    forecast = np.array(forecast, dtype=float).flatten()

    mask = actual != 0
    valid_actual = actual[mask]
    valid_forecast = forecast[mask]

    if len(valid_actual) > 0 and len(valid_forecast) > 0:
        return np.mean(np.abs((valid_forecast - valid_actual) / valid_actual)) * 100
    else:
        return 0


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

    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "Invalid payload"}), 400

        if len(data['data'][0].keys()) > 3:
            return jsonify({"error": "Invalid data"}), 400
            
    except Exception as e:
        return jsonify({"error": "Invalid request" + str(e)}), 400

    df = pd.DataFrame(data['data'])
    periods = data.get('periods')

    df = fill_missing_dates(df)
    # df = transform_prophet(df)
    feature_df = create_features(df)

    feature_cols = [
        'year',
        'quarter',
        'month',
        'day',
        'day_of_week',
        'day_of_year',
        'is_weekend',
        'lag_1',
        'lag_3',
        'lag_7',
        'lag_14',
        'rolling_mean_7',
        'rolling_mean_14',
        'rolling_std_7',
        'rolling_std_14'
    ]

    test_size = min(30, int(len(df) * 0.2))
    train_df = feature_df[:-test_size]
    test_df = feature_df[-test_size:]

    X_train = train_df[feature_cols]
    y_train = train_df.drop(columns=feature_cols)

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    X_test = feature_df[feature_cols]
    y_test = feature_df.drop(columns=feature_cols)

    forecast_test = model.predict(X_test)

    mape = calculate_mape(
        y_test.values,
        forecast_test
    )
    validation_result = []
    validation_forecast = forecast_test[-test_size:]
    validation_df = df[-test_size:]
    date_col = df.select_dtypes(include=['datetime64']).columns.tolist()
    data = df.select_dtypes(exclude=['datetime64']).columns.tolist()

    for i in range(len(test_df)):
        validation_result.append({
            "tanggal_transaksi": validation_df.iloc[i][date_col[0]].strftime('%Y-%m-%d'),
            "month": validation_df.iloc[i][date_col[0]].strftime('%B'),
            "week": validation_df.iloc[i][date_col[0]].isocalendar()[1],
            "year": validation_df.iloc[i][date_col[0]].year,
            "aktual": int(validation_df.iloc[i][data[0]]),
            "prediction": max(0, float(round(validation_forecast[i], 0))),
        })

    X_full = feature_df[feature_cols]
    y_full = feature_df.drop(columns=feature_cols)

    model.fit(X_full, y_full)
    history = y_full.values.flatten().tolist()

    last_date = df[date_col[0]].max()

    forecast_result = []

    for i in range(periods):

        future_date = last_date + timedelta(days=i + 1)

        row = {
            'year': future_date.year,
            'quarter': ((future_date.month - 1) // 3) + 1,
            'month': future_date.month,
            'day': future_date.day,
            'day_of_week': future_date.dayofweek,
            'day_of_year': future_date.timetuple().tm_yday,
            'is_weekend': 1 if future_date.dayofweek >= 5 else 0,

            'lag_1': history[-1],
            'lag_3': history[-3],
            'lag_7': history[-7],
            'lag_14': history[-14],

            'rolling_mean_7': np.mean(history[-7:]),
            'rolling_mean_14': np.mean(history[-14:]),

            'rolling_std_7': np.std(history[-7:]),
            'rolling_std_14': np.std(history[-14:])
        }

        X_future = pd.DataFrame([row])
        
        # Convert lag and rolling columns to float to avoid XGBoost dtype error
        lag_cols = [col for col in X_future.columns if col.startswith('lag_') or col.startswith('rolling_')]
        for col in lag_cols:
            X_future[col] = X_future[col].astype(float)

        pred = float(model.predict(X_future)[0])

        pred = max(0, pred)

        forecast_result.append({
            "tanggal_transaksi": future_date.strftime('%Y-%m-%d'),
            "month": future_date.strftime('%B'),
            "week": future_date.isocalendar()[1],
            "year": future_date.year,
            "prediction": int(round(pred))
        })

        history.append(pred)

    insight = generate_insight(mape)

    return jsonify({
        "forecast": forecast_result,
        "validation": validation_result,
        "mape": float(round(mape, 2)),
        "train_start": df[date_col[0]].min().strftime('%Y-%m-%d'),
        "train_end": df[date_col[0]].max().strftime('%Y-%m-%d'),
        "insight": insight
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)