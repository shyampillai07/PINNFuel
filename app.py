import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

VEHICLES = {
    "hatchback": 0,
    "sedan": 1,
    "suv": 2,
    "pickup": 3,
    "van": 4,
    "motorcycle": 5,
}

TRAFFIC = {
    "light": 0,
    "moderate": 1,
    "heavy": 2,
}


CC_RANGES = {
    0: (800, 2000),
    1: (1200, 3500),
    2: (1500, 5000),
    3: (2000, 6000),
    4: (1500, 4000),
    5: (50, 1800),
}

def generate_realistic_dataset(sample_count=8000, random_state=42):
    rng = np.random.default_rng(random_state)

    vehicle_code = rng.integers(0, 6, sample_count)

    engine_cc = np.zeros(sample_count)
    for v_code, (min_cc, max_cc) in CC_RANGES.items():
        mask = vehicle_code == v_code
        engine_cc[mask] = rng.uniform(min_cc, max_cc, np.sum(mask))

    speed_kmh = rng.uniform(20, 130, sample_count)
    load_kg = rng.uniform(0, 600, sample_count)
    ac_on = rng.integers(0, 2, sample_count)
    tire_pressure_psi = rng.uniform(28, 42, sample_count)
    vehicle_age_years = rng.uniform(0, 20, sample_count)
    traffic_code = rng.integers(0, 3, sample_count)

    is_motorcycle = (vehicle_code == 5)

    cc_multiplier = np.where(is_motorcycle, 2.0, 3.5)
    base_consumption = (engine_cc / 1000.0) * cc_multiplier

    weight_penalty = np.array([0.0, 0.5, 2.0, 2.5, 2.0, 0.0])[vehicle_code]
    base_consumption += weight_penalty

    optimal_speed = np.where(is_motorcycle, 60.0, 75.0)
    speed_penalty = ((speed_kmh - optimal_speed) / 20.0) ** 2 * 0.4
    speed_penalty += np.where(speed_kmh < 40, (40 - speed_kmh) * 0.05, 0)

    load_penalty = load_kg * 0.003 * np.where(is_motorcycle, 0.5, 1.0)
    ac_penalty = ac_on * np.where(is_motorcycle, 0.0, 0.8 + (engine_cc / 5000.0))
    
    optimal_psi = np.where(is_motorcycle, 30.0, 35.0)
    under_pressure_penalty = np.clip(optimal_psi - tire_pressure_psi, 0, None) * 0.15
    tire_penalty = under_pressure_penalty

    age_penalty = vehicle_age_years * 0.08
    traffic_penalty = np.array([0.0, 1.0, 2.5])[traffic_code]

    noise = rng.normal(0, 0.4, sample_count)

    consumption_l_per_100km = (
        base_consumption
        + speed_penalty
        + load_penalty
        + ac_penalty
        + tire_penalty
        + age_penalty
        + traffic_penalty
        + noise
    )

    moto_min, moto_max = 1.5, 15.0
    car_min, car_max = 3.5, 35.0
    
    min_clip = np.where(is_motorcycle, moto_min, car_min)
    max_clip = np.where(is_motorcycle, moto_max, car_max)
    consumption_l_per_100km = np.clip(consumption_l_per_100km, min_clip, max_clip)

    features = np.column_stack(
        (
            vehicle_code,
            engine_cc,
            speed_kmh,
            load_kg,
            ac_on,
            tire_pressure_psi,
            vehicle_age_years,
            traffic_code,
        )
    )
    return features, consumption_l_per_100km


def train_consumption_model():
    print("Generating dataset and training model... Please wait.")
    X, y = generate_realistic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mae, r2


def read_float(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt).strip())
            if min_value is not None and value < min_value:
                print(f"Value must be at least {min_value}.")
                continue
            if max_value is not None and value > max_value:
                print(f"Value must be at most {max_value}.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")


def read_choice(prompt, choices):
    while True:
        value = input(prompt).strip().lower()
        if value in choices:
            return value
        print(f"Choose one of: {', '.join(choices)}")


def main():
    model, mae, r2 = train_consumption_model()

    print("\n--- Fuel Consumption and Driving Range Predictor ---")
    print(f"Model trained! Test Quality: MAE={mae:.2f} L/100km, R2={r2:.3f}\n")

    vehicle = read_choice("Enter vehicle type (hatchback/sedan/suv/pickup/van/motorcycle): ", VEHICLES.keys())
    is_moto = (vehicle == "motorcycle")
    v_code = VEHICLES[vehicle]
    
    min_cc, max_cc = CC_RANGES[v_code]
    engine_cc = read_float(f"Engine capacity in CC (range {min_cc}-{max_cc}): ", min_value=min_cc, max_value=max_cc)
    
    speed_kmh = read_float("Average driving speed (km/h): ", min_value=10, max_value=160)
    
    max_load_kg = 150 if is_moto else 1000
    load_kg = read_float(f"Extra load/passenger weight (kg, max {max_load_kg}): ", min_value=0, max_value=max_load_kg)
    
    if is_moto:
        ac_text = "no"
        print("Note: Skipping AC for motorcycles.")
    else:
        ac_text = read_choice("AC on? (yes/no): ", ["yes", "no"])
        
    min_psi = 20 if is_moto else 24
    max_psi = 45 if is_moto else 50
    tire_pressure_psi = read_float(f"Tire pressure (psi, range {min_psi}-{max_psi}): ", min_value=min_psi, max_value=max_psi)
    
    vehicle_age_years = read_float("Vehicle age (years): ", min_value=0, max_value=40)
    traffic = read_choice("Traffic conditions (light/moderate/heavy): ", TRAFFIC.keys())
    fuel_in_tank_l = read_float("Fuel available in tank (liters): ", min_value=0.5, max_value=200)

    feature_row = np.array(
        [[
            v_code,
            engine_cc,
            speed_kmh,
            load_kg,
            1 if ac_text == "yes" else 0,
            tire_pressure_psi,
            vehicle_age_years,
            TRAFFIC[traffic],
        ]]
    )

    predicted_l_per_100km = float(model.predict(feature_row)[0])
    predicted_km_per_l = 100.0 / predicted_l_per_100km
    predicted_range_km = fuel_in_tank_l * predicted_km_per_l

    print(f"\n==========================================")
    print(f" Prediction Results for {vehicle.capitalize()} ({engine_cc:.0f}cc)")
    print(f"==========================================")
    print(f"Estimated fuel consumption : {predicted_l_per_100km:.2f} L/100km")
    print(f"Estimated mileage          : {predicted_km_per_l:.2f} km/L")
    print(f"Max distance on current tank: {predicted_range_km:.1f} km")
    print(f"==========================================\n")

    print("Generating fuel consumption curve based on speed...")
    speed_axis = np.linspace(20, 140, 100)
    scenario_features = np.column_stack(
        (
            np.full_like(speed_axis, v_code),
            np.full_like(speed_axis, engine_cc),
            speed_axis,
            np.full_like(speed_axis, load_kg),
            np.full_like(speed_axis, 1 if ac_text == "yes" else 0),
            np.full_like(speed_axis, tire_pressure_psi),
            np.full_like(speed_axis, vehicle_age_years),
            np.full_like(speed_axis, TRAFFIC[traffic]),
        )
    )
    consumption_curve = model.predict(scenario_features)

    plt.figure(figsize=(10, 5))
    plt.plot(speed_axis, consumption_curve, color="blue", linewidth=2, label="Consumption Curve")
    plt.scatter([speed_kmh], [predicted_l_per_100km], color="red", s=100, zorder=5, label="Your Input Scenario")
    plt.xlabel("Average Speed (km/h)")
    plt.ylabel("Fuel Consumption (L/100km)")
    
    vehicle_label = f"{vehicle.capitalize()} ({int(engine_cc)}cc)"
    plt.title(f"Aerodynamic Speed Impact on Fuel Economy - {vehicle_label}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()