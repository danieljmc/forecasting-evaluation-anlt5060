import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# 1) Load and prepare data
file_path = r"ANLT5060_StAnthony-VilaHealth.csv"   # adjust as needed
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])
data = data.dropna(subset=['presentations']).copy()
data['presentations'] = data['presentations'].astype(float)
data = data.set_index('date').asfreq('D')          # daily frequency

# 2) Train/test split (chronological, 80/20)
split_idx = int(len(data) * 0.8)
train = data.iloc[:split_idx].copy()
test  = data.iloc[split_idx:].copy()

# 3) Linear Regression (time index as predictor)
train['t'] = np.arange(len(train))
test['t']  = np.arange(len(train), len(train) + len(test))

lin = LinearRegression().fit(train[['t']], train['presentations'])
pred_lr = lin.predict(test[['t']])

mape_lr = mean_absolute_percentage_error(test['presentations'], pred_lr) * 100
rmse_lr = np.sqrt(mean_squared_error(test['presentations'], pred_lr))
r2_lr   = r2_score(test['presentations'], pred_lr)

# 4) Exponential Smoothing (additive trend)
hw = ExponentialSmoothing(train['presentations'], trend='add', seasonal=None).fit()
pred_hw = hw.forecast(len(test))

mape_hw = mean_absolute_percentage_error(test['presentations'], pred_hw) * 100
rmse_hw = np.sqrt(mean_squared_error(test['presentations'], pred_hw))
r2_hw   = r2_score(test['presentations'], pred_hw)

# 5) ARIMA(1,1,1)
arima = ARIMA(train['presentations'], order=(1, 1, 1)).fit()
pred_ar = arima.forecast(len(test))

mape_ar = mean_absolute_percentage_error(test['presentations'], pred_ar) * 100
rmse_ar = np.sqrt(mean_squared_error(test['presentations'], pred_ar))
r2_ar   = r2_score(test['presentations'], pred_ar)

# 6) Results table (dynamic)
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Exponential Smoothing', 'ARIMA(1,1,1)'],
    'MAPE (%)': [mape_lr, mape_hw, mape_ar],
    'RMSE':     [rmse_lr, rmse_hw, rmse_ar],
    'R²':       [r2_lr,   r2_hw,   r2_ar]
})

print("\nForecast Evaluation Summary:")
print(results.round(3))

# 7) Visualization (Figure 1) using same dynamic results
models = results['Model']
x = np.arange(len(models))
width = 0.25

fig, ax1 = plt.subplots(figsize=(9, 5))

# --- Left axis for MAPE (%)
ax1.bar(x - width, results['MAPE (%)'], width, label='MAPE (%)', color='steelblue')
ax1.set_ylabel('MAPE (%)', color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# --- Right axis for RMSE
ax2 = ax1.twinx()
ax2.bar(x, results['RMSE'], width, label='RMSE (patients)', color='orange')
ax2.set_ylabel('RMSE (patients)', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# --- Secondary right axis for R²
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.bar(x + width, results['R²'], width, label='R²', color='green')
ax3.set_ylabel('R²', color='green')
ax3.tick_params(axis='y', labelcolor='green')
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')

# --- X-axis and labels
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=15)
ax1.set_xlabel('Model')

# --- Title and legend
fig.suptitle('Figure 1. Forecast Accuracy Comparison for Vila Health Models',
             fontsize=12, fontweight='bold')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
fig.tight_layout()

plt.show()
