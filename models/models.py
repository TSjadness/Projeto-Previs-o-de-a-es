from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Construção da rede MLP
def models_mlp(input_shape):
    modelo = Sequential()
    modelo.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dense(1))  # Saída com um único neurônio para prever o próximo valor

    modelo.compile(optimizer='adam', loss='mean_squared_error')
    return modelo
