import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

movies_train = pd.read_csv("train.csv")
X_train = movies_train.drop("rating", axis=1)
Y_train = movies_train["rating"]

movies_test = pd.read_csv("test.csv")
X_test = movies_test.drop("rating", axis=1)
Y_test = movies_test["rating"]

# Set up model
model = Sequential()
model.add(Dense(8, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)

model.fit(
    x=X_train,
    y=Y_train.values,
    validation_data=(X_test, Y_test.values),
    batch_size=128,
    epochs=400,
    callbacks=[early_stop],
)

# Predict movie ratings
predictions = model.predict(X_test)

pd.DataFrame(predictions).to_csv("results.csv")


# Compare outputs
for i, score in enumerate(predictions):
    print(f"Original score: {Y_test.iloc[i]} Predicted score: {score} \n")
    print(f"Difference is : {Y_test.iloc[i] - score}")

# Evaluate
acc = model.evaluate(X_test, Y_test)
print(acc)
with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")
