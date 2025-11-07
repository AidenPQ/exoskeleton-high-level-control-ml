from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, GlorotUniform


init_he_normal = HeNormal()

init_glorot_uniform = GlorotUniform()


def create_model(input_shape, learning_rate=0.01, activation='relu', neurons=32, dropout_rate=0.0, regularizer=None, initializer='glorot_uniform', number_of_hidden_layers=4, number_outputs=24):
    # Choose the initializer
    if initializer == 'he_normal':
        init = init_he_normal
    elif initializer == 'glorot_uniform':
        init = init_glorot_uniform
    else:
        init = 'glorot_uniform'  # Default to glorot_uniform if unknown

    # Choose activation
    if activation == "relu":
        act = "relu"
    elif activation == "elu":
        act = "elu"
    elif activation == "prelu":
        act = PReLU()
    elif activation == "leakyrelu":
        act = LeakyReLU(alpha=0.01)

    model = Sequential()
    # Input Layer
    model.add(Dense(neurons, input_dim=input_shape, activation=act, kernel_initializer=init,  kernel_regularizer=regularizer))
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Hidden Layers
    for i in range(number_of_hidden_layers):
        model.add(Dense(neurons, activation=act, kernel_initializer=init,  kernel_regularizer=regularizer))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output Layer
    model.add(Dense(number_outputs, activation='linear', kernel_regularizer=regularizer))

    # Optimizer
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model
