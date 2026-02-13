import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import random
import os
from datetime import timedelta


os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

HISSE = 'NVDA'
GECMIS_GUN = 60          
TAHMIN_SURESI = 21       

print(f"{HISSE} verileri indiriliyor... (Yöntem: Yüzdelik Değişim Tahmini)")

veri_ham = yf.download(HISSE, start='2020-01-01', progress=False)


try:
    if isinstance(veri_ham.columns, pd.MultiIndex):
        veri = veri_ham.xs('Close', axis=1, level=0)
    elif 'Close' in veri_ham.columns:
        veri = veri_ham[['Close']]
    else:
        veri = veri_ham.iloc[:, 0].to_frame()
except Exception:
    veri = veri_ham.iloc[:, 0].to_frame()

if isinstance(veri, pd.Series):
    veri = veri.to_frame()
veri.columns = ['Close']

veri = veri.dropna()

veri['Degisim'] = veri['Close'].pct_change()
veri = veri.dropna() 
egitim_verisi = veri['Degisim'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(egitim_verisi)

x_train = []
y_train = []

for i in range(GECMIS_GUN, len(scaled_data)):
    x_train.append(scaled_data[i-GECMIS_GUN:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

print("Yapay zeka piyasa dalgalanmalarını öğreniyor...")
model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1)

inputs = scaled_data[-GECMIS_GUN:].reshape(1, GECMIS_GUN, 1)
tahmin_edilen_degisimler = []

for i in range(TAHMIN_SURESI):
    pred = model.predict(inputs, verbose=0)
    tahmin_edilen_degisimler.append(pred[0, 0])
    inputs = np.append(inputs[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

tahmin_edilen_degisimler = scaler.inverse_transform(np.array(tahmin_edilen_degisimler).reshape(-1, 1))
son_gercek_fiyat_array = veri['Close'].to_numpy().flatten()
mevcut_fiyat = float(son_gercek_fiyat_array[-1]) 

gelecek_fiyatlar = []

for degisim in tahmin_edilen_degisimler:
    yeni_fiyat = mevcut_fiyat * (1 + degisim[0])
    gelecek_fiyatlar.append(yeni_fiyat)
    mevcut_fiyat = yeni_fiyat

son_tarih = veri.index[-1]
if isinstance(son_tarih, pd.Timestamp):
    start_date = son_tarih
else:
    start_date = pd.to_datetime(str(son_tarih))

gelecek_tarihler = [start_date + timedelta(days=x) for x in range(1, TAHMIN_SURESI + 1)]
gelecek_tarihler.insert(0, start_date)

gelecek_fiyat_plot = [float(son_gercek_fiyat_array[-1])] + gelecek_fiyatlar

fig = go.Figure()


fig.add_trace(go.Scatter(
    x=veri.index[-150:], 
    y=son_gercek_fiyat_array[-150:], 
    mode='lines',
    name='Gerçek Fiyat',
    line=dict(color='black', width=2),
    hovertemplate='Tarih: %{x}<br>Fiyat: $%{y:.2f}'
))


fig.add_trace(go.Scatter(
    x=gelecek_tarihler, 
    y=gelecek_fiyat_plot,
    mode='lines+markers',
    name='Gerçekçi Tahmin (Değişim Bazlı)',
    line=dict(color='purple', width=3),
    marker=dict(size=5),
    hovertemplate='<b>Tahmin: %{x}</b><br>Fiyat: $%{y:.2f}'
))

fig.update_layout(
    title=f'{HISSE} - Piyasa Değişimlerine Göre Tahmin',
    xaxis_title='Tarih',
    yaxis_title='Fiyat (USD)',
    hovermode="x unified",
    template="plotly_white"
)

fig.show()


print(f"\n21. Günün Tahmini: {gelecek_fiyatlar[-1]:.2f} $")
