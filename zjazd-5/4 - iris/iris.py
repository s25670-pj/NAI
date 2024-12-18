import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Iris recognition neural network
Authors: Mateusz Lech, Szymon Rosztajn
Required components:
- TensorFlow
- Pandas
- Sklearn

Iris dataset contains sepal and petal width and length.

Class options:
0 - Iris-setosa
1 - Iris-versicolor
2 - Iris-virginica
"""

def load_data(file_path):
    """
    Load the Iris dataset from the specified file path.
    """
    iris = pd.read_csv(file_path, delimiter=',')
    print(iris.head(5))
    return iris

def preprocess_data(iris):
    """
    Scale features and split data into training and testing sets.
    """
    scaler = StandardScaler()
    scaler.fit(iris.drop('class', axis=1))
    scaled_features = scaler.transform(iris.drop('class', axis=1))
    df_features = pd.DataFrame(scaled_features, columns=iris.columns[:-1])
    x = df_features
    y = iris['class']
    return train_test_split(x, y, test_size=0.3, random_state=42)

def build_model(input_shape, num_classes):
    """
    Build and compile a Keras neural network model.
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
    Main function to execute the Iris recognition pipeline.
    """
    # Load and preprocess the data
    iris = load_data('iris.csv')
    x_train, x_test, y_train, y_test = preprocess_data(iris)

    # Build and train the model
    model = build_model(input_shape=x_train.shape[1], num_classes=3)
    model.fit(x_train, y_train, epochs=30, batch_size=20, verbose=1)

    # Evaluate the model
    y_pred_probs = model.predict(x_test)
    y_pred = y_pred_probs.argmax(axis=1)

    # Print evaluation metrics
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
