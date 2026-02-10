# Accra Traffic Prediction & ETA Engine

A comprehensive machine learning system for predicting traffic speeds and calculating estimated time of arrival (ETA) in Accra, Ghana. The engine uses XGBoost models trained on historical traffic data, weather conditions, and event information to provide accurate travel time predictions.

## Features

- **Real-time Traffic Speed Prediction**: ML-powered speed predictions for different road segments
- **Multi-Route ETA Calculation**: Calculate and compare ETAs for multiple route options
- **Route Optimization**: Find the fastest route based on current conditions
- **Traffic Pattern Analysis**: Analyze historical traffic patterns and trends
- **Departure Time Optimization**: Determine the best time to travel
- **Weather Integration**: Factor in weather conditions (rain, temperature, humidity) 
- **Event Awareness**: Account for special events that impact traffic
- **Prediction Caching**: Improved performance for repeated queries
- **Confidence Intervals**: Uncertainty estimates for all predictions

## Project Structure

```
eta/
├── eta.ipynb              # Main Jupyter notebook with the complete implementation
├── README.md              # Project documentation (this file)
└── data/
    ├── traffic_data.csv   # Historical traffic speed data
    ├── weather_data.csv   # Weather conditions data
    └── events_data.csv    # Special events data
```

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sedegah/eta.git
cd eta
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook eta.ipynb
```

## Usage

### Validate Data Before Training

Run the built-in validator to confirm the CSV files are complete and internally consistent:

```bash
python scripts/validate_data.py
```

This checks schema correctness, basic value ranges, and whether timestamps can be merged safely across traffic, weather, and events datasets.

### Basic Usage

```python
# Initialize the engine
engine = TrafficPredictionEngine()

# Load and prepare data
engine.load_data()
engine.engineer_features(engine.df_full)
engine.train_models()

# Define travel conditions
conditions = {
    'hour': 8,          # 8 AM
    'weekday': 0,       # Monday
    'rain': 0,          # No rain
    'temperature': 30,  # 30°C
    'humidity': 70,     # 70%
    'event_type': 'none'
}

# Predict ETA for a single route
result = engine.predict_route_eta('Circle Rd', distance_km=5.0, conditions=conditions)
print(f"ETA: {result['eta_minutes']:.1f} minutes")
print(f"Speed: {result['speed']:.1f} km/h")
```

### Compare Multiple Routes

```python
# Compare alternative routes
routes = [
    {'road': 'Circle Rd', 'distance': 5.0},
    {'road': 'Spintex Rd', 'distance': 6.5},
    {'road': 'Independence Ave', 'distance': 4.8}
]

comparison = engine.compare_routes(routes, conditions)
for route in comparison:
    print(f"{route['road']}: {route['eta_minutes']:.1f} min")
```

### Find Best Departure Time

```python
# Find optimal departure time
best_time = engine.find_best_departure_time('Circle Rd', 5.0, base_conditions)
print(f"Best time to leave: {best_time['hour']}:00")
print(f"Expected ETA: {best_time['eta_minutes']:.1f} minutes")
```

## Technical Details

### Machine Learning Approach

- **Algorithm**: XGBoost (Gradient Boosting)
- **Models**: Separate model trained for each road segment
- **Features**: Time encoding (cyclical), weather conditions, lag features, rolling statistics
- **Evaluation Metrics**: MAE, RMSE, R²

### Feature Engineering

The engine performs sophisticated feature engineering including:

- **Temporal Features**: Hour of day, day of week, weekend indicator, rush hour flag
- **Cyclical Encoding**: Sine/cosine transformations for time features
- **Weather Features**: Rain categories, temperature bins, humidity levels
- **Event Features**: One-hot encoding for special events
- **Lag Features**: Previous time period's traffic speeds
- **Rolling Statistics**: Moving averages of traffic patterns

### Road Segments

The system currently supports predictions for:
- Circle Rd (8.5 km)
- Spintex Rd (12.3 km)
- Independence Ave (6.7 km)

## Data

### Traffic Data
Historical traffic speed measurements including:
- Timestamp
- Road segment
- Average speed (km/h)

### Weather Data
Weather conditions including:
- Rain (mm)
- Temperature (°C)
- Humidity (%)

### Events Data
Special events that impact traffic:
- Event type (none, concert, sports, festival, market_day, accident)
- Event timestamps

## Future Enhancements

1. Real-time API integration for live traffic data
2. Multi-segment route planning (connecting multiple roads)
3. Historical pattern visualization and dashboards
4. Mobile app integration via REST API
5. Traffic alert notifications
6. Integration with Google Maps/Waze
7. Deep learning models (LSTM/GRU) for better temporal predictions
8. Integration with public transportation data
9. Incident detection and reporting

## Model Performance

The XGBoost models achieve strong performance on historical data:
- Road-specific training and evaluation
- Confidence intervals for uncertainty quantification
- Regular retraining to maintain accuracy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Development Notes

- Use `requirements.txt` for dependency installation to keep environments consistent.
- Run `python scripts/validate_data.py` before retraining models if datasets are modified.
- Keep the notebook outputs lightweight where possible to reduce repository churn.

## License

This project is open source and available under the MIT License.

## Author

**sedegah**
- GitHub: [@sedegah](https://github.com/sedegah)

## Acknowledgments

- Traffic data collected from Accra road network
- Weather data from local meteorological services
- Event data from public sources

---

**Note**: This is a research/educational project. For production use, consider additional validation, real-time data integration, and safety measures.
