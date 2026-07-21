from data_preprocessing import load_data, clean_data, prepare_data
from model import train_model
from train import evaluate_model

print("Starting project...")

# Step 1: Load data
df = load_data()

# Step 2: Clean data
df = clean_data(df)

# Step 3: Prepare data
X_train, X_test, y_train, y_test, scaler = prepare_data(df)

# Step 4: Train model
model = train_model(X_train, y_train)

# Step 5: Evaluate model
predictions = evaluate_model(model, X_test, y_test)

import matplotlib.pyplot as plt

#  Actual vs Predicted
plt.scatter(y_test, predictions)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

#  Error visualization
errors = y_test - predictions

plt.hist(errors, bins=20)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()



area = float(input("Enter area: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))

# Put into array
user_data = [[area, bedrooms, bathrooms]]

#  IMPORTANT: scale it
user_data = scaler.transform(user_data)

# Predict
predicted_price = model.predict(user_data)

print(f"\n Predicted House Price: {predicted_price[0]:,.2f}")

print("Finished!")
input("Press Enter to exit...")
print("\n--- House Price Prediction ---")