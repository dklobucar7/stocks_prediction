import numpy as np
import time as tm
import datetime as dt
# Data preparation
from yahoo_fin import stock_info as yf #povijesni podaci svih dionica
# Scikit-learn
from sklearn.preprocessing import MinMaxScaler #skaliranje vrednosti u određenom opsegu, obično od 0 do 1.
from collections import deque # je struktura podataka koja omogućava efikasno dodavanje i uklanjanje elemenata s oba kraja reda. 

# KERAS je sada dio TensorFlow-a od verzije 2.0
from keras.models import Sequential #je model koji se koristi za linearno slaganje slojeva (layer) u neuronskoj mreži, gdje svaki sloj ima točno jedan ulaz i jedan izlaz.
from keras.layers import Dense, LSTM, Dropout
#Dense- Ova klasa predstavlja potpuno povezani sloj (fully connected layer) u neuronskoj mreži. Svi neuroni u ovom sloju su povezani sa svakim neuronom u prethodnom i sledećem sloju.
#LSTM - Ova klasa predstavlja sloj Long Short-Term Memory (LSTM) u neuronskoj mreži. LSTM je vrsta rekurentnog sloja koja je posebno efikasna u radu sa sekvencijalnim podacima, kao što su vremenske serije.
#Dropout- va klasa predstavlja sloj za regularizaciju u neuronskoj mreži. Dropout se koristi kako bi se smanjila prenaučenost (overfitting) tako što se slučajno "isključuju" neki neuroni tokom treniranja, čime se sprečava prekomerno prilagođavanje treniranim podacima.
#high level api koji se koristi za tenserflow

# Graphics library
import matplotlib.pyplot as plt

# POSTAVLJANJE NEURONSKE MREŽE

# Window size or the sequence length, 7 (1 week)
# broj dana u prozoru, odnosno koliko duboko će ići neuronska mreža. 
# Postavljeno je na 7, što odgovara tjednu vremenskom periodu, odnosno analizirati će se podaci u tjednim intervalima.
N_STEPS = 7

# Lookup steps, 1 is the next day, 3 = after tomorrow
# Za koliko cemo broja dana predvidati cijene dionica.
LOOKUP_STEPS = [1, 2, 3]


# Simbol dionice Tesle je TSLA, META je Facebook, Apple je AAPL, Amazon je AMZN, Microsoft je MSFT
STOCK = 'TSLA'

# Current date
# Raspon unatrag 3 godine
date_now = tm.strftime('%Y-%m-%d')
date_3_years_back = (dt.date.today() - dt.timedelta(days=1104)).strftime('%Y-%m-%d')

# LOAD DATA 
# from yahoo_fin API
# for 1104 bars with interval = 1d (one day)
init_df = yf.get_data(
    STOCK, 
    start_date=date_3_years_back, 
    end_date=date_now, 
    interval='1d')

init_df

#Brisanje nepotrebnih podataka iz tablice, ostavit ćemo samo kolonu "CLOSE"
# remove columns which our neural network will not use
init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
# create the column 'date' based on index column
init_df['date'] = init_df.index

init_df

# Export the DataFrame to an Excel file
#excel_filename = 'TSLA_stock_data_.xlsx'  # Provide the desired filename

#init_df.to_excel(excel_filename, index=True)

# Crtanje dijagrama
# Let's preliminary see our data on the graphic
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(init_df['close'][-1000:])
plt.xlabel('days')
plt.ylabel('price')
plt.legend([f'Actual price for {STOCK}'])
plt.show()

# PRije TRENIRANJA MODELA STROJNOG UČENJA, SKALIRAT ćemo podatke 
# Skalirnaje podataka, koristit ćemo MinMaxScaler koji skalira sve vrijednosti u zadanom intervalu izmedu 1 i 0.
# Scale data for ML engine
scaler = MinMaxScaler()
init_df['scaled_close'] = scaler.fit_transform(np.expand_dims(init_df['close'].values, axis=1))

init_df

# Exportiranje u excel
#excel_filename = 'TSLA_stock_data_scaled.xlsx'  

#init_df.to_excel(excel_filename, index=True)

#PRIPREMA PODATAKA ZA TRENIRANJE NEUORNSKE MREŽE
# Sada je potrebno pripremiti te podatke za sljedeći postupak, jer je početni cilj predvidjeti cijenu dionica za naredna tri dana.
# To znači da se stupci moraju pomaknuti za broj dana za koje se želi predvidjeti cijena i u skladu s tim pripremiti podaci za model.

def PrepareData(days):
  # Inicijalizacija DataFrame-a:
  df = init_df.copy()

  #Dodavanje stupca future - Dodaje se nova kolona 'future' koja sadrži vrijednosti 'scaled_close' pomaknute unazad
  #za određeni broj dana (days). Ovo se često radi kako bi se modelu omogućilo da "uči" predviđati vrijednosti u budućnosti.
  df['future'] = df['scaled_close'].shift(-days)

  #Poslijednji red podataka:
  last_sequence = np.array(df[['scaled_close']].tail(days))

  #Uklanjanje redova sa nepostojećim vrijednostima:
  df.dropna(inplace=True)

  #Formiranje prazne sekvence
  sequence_data = []

  # Fromiranje sekvence s deque - sa maksimalnom dužinom od N_STEPS. Deque se koristi kao prozor koji klizi kroz podatke.
  sequences = deque(maxlen=N_STEPS)

  # Popunjavanje sekvenci i ciljnih vrednosti: Iterira se kroz DataFrame i popunjava deque (sequences) sa podacima,
  # a zatim dodaje sekvencu i odgovarajuća ciljna vrednost u sequence_data kada deque dostigne željenu dužinu N_STEPS.
  for entry, target in zip(df[['scaled_close'] + ['date']].values, df['future'].values):
      sequences.append(entry)
      if len(sequences) == N_STEPS:
          sequence_data.append([np.array(sequences), target])

  #Poslednja sekvenca: Dodaje se poslijednja sekvencija u niz last_sequence i konvertira se u numpy niz tipa float32.
  last_sequence = list([s[:len(['scaled_close'])] for s in sequences]) + list(last_sequence)
  last_sequence = np.array(last_sequence).astype(np.float32)

  # Formiraju se X i Y nizovi koji će se koristiti za treniranje neuronske mreže.
  # construct the X's and Y's
  X, Y = [], []
  for seq, target in sequence_data:
      X.append(seq)
      Y.append(target)

 # X i Y se konvertiraju u numpy nizove pre nego što se funkcija završi.
  # convert to numpy arrays
  X = np.array(X)
  Y = np.array(Y)

  return df, last_sequence, X, Y

PrepareData(3) # 3 days

#MODEL STROJNOG UČENJA
# Parametar „x_train“ je niz podataka, pripremljenih za model po kojem će se vršiti obuka. 
# Parametar „y_train“ pruža odgovor na obučavanje na temelju ciljane kolone 'close'. 
#  funkcija GetTrainedModel koristi Keras za definiranje, kompajliranje i treniranje neuronske mreže.
def GetTrainedModel(x_train, y_train):
  model = Sequential()

  # Dodaje LSTM sloj sa 60 jedinica, vraćanjem sekvenci i specificira oblik ulaza.
  model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['scaled_close']))))
  model.add(Dropout(0.3))
  model.add(LSTM(120, return_sequences=False))
  model.add(Dropout(0.3))
  model.add(Dense(20))
  model.add(Dense(1))

  BATCH_SIZE = 8
  EPOCHS = 80

  model.compile(loss='mean_squared_error', optimizer='adam')

  model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1)

  model.summary()

  return model

# GET PREDICTIONS
predictions = []

for step in LOOKUP_STEPS:
  df, last_sequence, x_train, y_train = PrepareData(step)
  x_train = x_train[:, :, :len(['scaled_close'])].astype(np.float32)

  model = GetTrainedModel(x_train, y_train)

  last_sequence = last_sequence[-N_STEPS:]
  last_sequence = np.expand_dims(last_sequence, axis=0)
  prediction = model.predict(last_sequence)
  predicted_price = scaler.inverse_transform(prediction)[0][0]

  predictions.append(round(float(predicted_price), 2))

if bool(predictions) == True and len(predictions) > 0:
  predictions_list = [str(d)+'$' for d in predictions]
  predictions_str = ', '.join(predictions_list)
  message = f'{STOCK} prediction for upcoming 3 days ({predictions_str})'
  
  print(message)

# Usporedba stvarne i predvidene vrijednosti dionica

last_3_y_df = init_df.copy()
y_predicted = model.predict(x_train)
y_predicted_transformed = np.squeeze(scaler.inverse_transform(y_predicted))
first_seq = scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
last_seq = scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
last_3_y_df[f'predicted_close'] = y_predicted_transformed

last_3_y_df

# export u excel
excel_filename = 'TSLA_stock_data_scaled_last_3_y.xlsx'  # Provide the desired filename

last_3_y_df.to_excel(excel_filename, index=True)

# Add predicted results to the table
date_now = dt.date.today()
date_tomorrow = dt.date.today() + dt.timedelta(days=1)
date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

last_3_y_df.loc[date_now] = [predictions[0], f'{date_now}', 0, 0]
last_3_y_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0, 0]
last_3_y_df.loc[date_after_tomorrow] = [predictions[2], f'{date_after_tomorrow}', 0, 0]

# Result chart
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(last_3_y_df['close'][-150:].head(147))
plt.plot(last_3_y_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed')
plt.plot(last_3_y_df['close'][-150:].tail(4))
plt.xlabel('days')
plt.ylabel('price')
plt.legend([f'Actual price for {STOCK}', 
            f'Predicted price for {STOCK}',
            f'Predicted price for future 3 days'])
plt.show()