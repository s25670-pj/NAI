import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Wheat recognition neural network using TensorFlow and Keras
Authors: Mateusz Lech, Szymon Rosztajn
Required components:
- TensorFlow
- Matplotlib
- Scikit-learn
- Keras
"""

def load_data(file_path):
    """
    Loads the wheat dataset from a specified file.
    """
    wheat = pd.read_csv(file_path, delimiter='\t')
    print(wheat.head(5))
    return wheat

def preprocess_data(wheat):
    """
    Scales the features and prepares the data for training.
    """
    scaler = StandardScaler()
    scaler.fit(wheat.drop('class', axis=1))
    scaled_features = scaler.transform(wheat.drop('class', axis=1))
    df_features = pd.DataFrame(scaled_features, columns=wheat.columns[:-1])
    x = df_features
    y = wheat['class'] - 1
    return x, y

def build_model(input_shape, num_classes):
    """
    Builds and compiles a Keras sequential model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    """
    Main function to load data, preprocess it, train the model, and evaluate it.
    """
    # Load dataset
    wheat = load_data('seeds_dataset.csv')

    # Preprocess data
    x, y = preprocess_data(wheat)

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Build the model
    model = build_model(input_shape=x_train.shape[1], num_classes=3)

    # Train the model
    model.fit(x_train, y_train, epochs=30, batch_size=20, verbose=1)

    # Evaluate the model
    y_pred_probs = model.predict(x_test)
    y_pred = y_pred_probs.argmax(axis=1)

    # Display evaluation metrics
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
