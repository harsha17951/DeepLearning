import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense  # Corrected import statement

def train_model(model_instance, train_generator, validate_generator, epochs=5):
    mhistory = model_instance.fit(train_generator, validation_data=validate_generator, epochs=epochs)  # Corrected argument name
    return mhistory

def compare_models(models, train_generator, validate_generator, epochs=5):
    histories = []
    for model in models:
        history = train_model(model, train_generator, validate_generator, epochs=epochs)
        histories.append(history)

    plt.figure(figsize=(10, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Model {i + 1}')

    plt.title('model accuracy comparison')  # Corrected title spelling
    plt.xlabel('epochs')
    plt.ylabel('accuracy')  # Corrected spelling
    plt.legend()
    plt.show(block=True)

class DeepANN():
    def simple_model(self, input_shape=(28, 28, 3), optimizer='sgd'):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(2, activation="softmax"))
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metrics=["accuracy"])
        return model
