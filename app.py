import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# 📌 Step 1: Generate Sample Data for Training
fuel = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Fuel in liters
mileage = np.array([40, 42, 43, 44, 44, 45, 46, 46, 47, 48])  # Mileage (km per liter)
distance = fuel.flatten() * mileage  # Distance = Fuel × Mileage

# 📌 Step 2: Prepare Data for Machine Learning
X = np.column_stack((fuel, mileage))  # Features: Fuel & Mileage
y = distance  # Target: Distance traveled
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 3: Train Machine Learning Models
linear_model = LinearRegression()
tree_model = DecisionTreeRegressor()

linear_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# 📌 Step 4: Get User Input for Prediction FIRST
print("\n🔹 Real-time Distance Prediction 🔹")
fuel_input = float(input("Enter fuel in liters: "))
mileage_input = float(input("Enter mileage (km per liter): "))

# Physics-Based Prediction
physics_predicted_distance = fuel_input * mileage_input

# AI-Based Prediction
ml_predicted_distance = linear_model.predict([[fuel_input, mileage_input]])[0]

# 📌 Step 5: Display Predictions Before Graph
print(f"\n📌 Estimated Distance Using ML (Hypothesis): {ml_predicted_distance:.2f} km")
print(f"📌 Estimated Distance Using Formula: {physics_predicted_distance:.2f} km")

# 📌 Step 6: Improved Graph After Predictions
sns.set_style("darkgrid")  # Apply Seaborn dark grid theme

plt.figure(figsize=(10, 5))

# Scatter plot for actual data points
plt.scatter(fuel, distance, color="blue", label="Actual Data", alpha=0.7)

# Smooth Prediction Line for AI-based Model
fuel_range = np.linspace(1, 10, 100).reshape(-1, 1)
mileage_range = np.full_like(fuel_range, np.mean(mileage))
predicted_curve = linear_model.predict(np.column_stack((fuel_range, mileage_range)))

plt.plot(fuel_range, predicted_curve, color="red", linestyle="dashed", label="ML Prediction Curve")

# User Input Predictions
plt.scatter(fuel_input, physics_predicted_distance, color="purple", marker="x", s=100, label="Physics Estimate")
plt.scatter(fuel_input, ml_predicted_distance, color="red", marker="o", s=100, label="ML Estimate")

plt.xlabel("Fuel (Liters)", fontsize=12)
plt.ylabel("Distance (Km)", fontsize=12)
plt.title("Fuel vs Distance Prediction with Machine Learning", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()