from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import warnings

warnings.filterwarnings('ignore')

# Import your LSTM model class
from lstm_predictive_model import ImprovedLSTMPredictiveMaintenance

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'


class MillSimulator:
    """Mill Machine Simulator for SPPA-T3000"""

    def __init__(self):
        self.model = None
        self.lstm_model = None
        self.preprocessing_data = None
        self.load_trained_model()

    def load_trained_model(self):
        """Load the trained LSTM model and preprocessing data"""
        try:
            # Load the LSTM model
            self.model = load_model('guaranteed_3class_lstm_model.h5')
            print("✅ LSTM model loaded successfully")

            # Load preprocessing data
            with open('guaranteed_3class_lstm_model_preprocessing.pkl', 'rb') as f:
                self.preprocessing_data = pickle.load(f)
            print("✅ Preprocessing data loaded successfully")

            # Initialize LSTM model instance for feature engineering
            self.lstm_model = ImprovedLSTMPredictiveMaintenance()
            self.lstm_model.scaler = self.preprocessing_data['scaler']
            self.lstm_model.base_features = self.preprocessing_data['base_features']
            self.lstm_model.all_feature_columns = self.preprocessing_data['all_feature_columns']
            self.lstm_model.sequence_length = self.preprocessing_data['sequence_length']
            self.lstm_model.prediction_horizon = self.preprocessing_data['prediction_horizon']

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Using dummy model for demonstration")
            self.setup_dummy_model()

    def setup_dummy_model(self):
        """Setup dummy model for demonstration when real model is not available"""
        self.model = None
        self.lstm_model = ImprovedLSTMPredictiveMaintenance()

    def generate_realistic_sensor_data(self, days, base_condition='normal'):
        """Generate realistic sensor data with proper correlations and patterns"""
        hours = days * 24
        timestamps = pd.date_range(start=datetime.now(), periods=hours, freq='H')

        # More realistic parameter ranges based on industrial mill operations
        base_params = {
            'mill_motor_air_temp': {
                'normal': (45, 65),
                'warning': (65, 85),
                'critical': (85, 120)
            },
            'coal_feed_flow': {
                'normal': (90, 110),
                'warning': (70, 90),
                'critical': (50, 70)
            },
            'mill_inlet_temp': {
                'normal': (25, 40),
                'warning': (40, 55),
                'critical': (55, 75)
            },
            'mill_diff_pressure': {
                'normal': (2.5, 3.5),
                'warning': (3.5, 5.0),
                'critical': (5.0, 8.0)
            },
            'mill_motor_current': {
                'normal': (180, 220),
                'warning': (220, 270),
                'critical': (270, 350)
            },
            'vibrations_velocity': {
                'normal': (1.5, 2.5),
                'warning': (2.5, 4.0),
                'critical': (4.0, 7.0)
            },
            'mill_outlet_temp': {
                'normal': (55, 70),
                'warning': (70, 85),
                'critical': (85, 110)
            },
            'machine_loading': {
                'normal': (80, 95),
                'warning': (95, 110),
                'critical': (110, 130)
            }
        }

        data = {'timestamp': timestamps}

        # Generate base sensor patterns
        for param, ranges in base_params.items():
            min_val, max_val = ranges[base_condition]

            # Create more realistic patterns
            # 1. Base trend (slight drift over time)
            if base_condition == 'normal':
                base_trend = np.random.uniform(min_val, min_val + (max_val - min_val) * 0.3, hours)
            elif base_condition == 'warning':
                base_trend = np.random.uniform(min_val + (max_val - min_val) * 0.2,
                                               min_val + (max_val - min_val) * 0.8, hours)
            else:  # critical
                base_trend = np.random.uniform(min_val + (max_val - min_val) * 0.5, max_val, hours)

            # 2. Daily operational cycles (8-hour shifts)
            daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * (max_val - min_val) * 0.1
            shift_cycle = np.sin(np.arange(hours) * 2 * np.pi / 8) * (max_val - min_val) * 0.05

            # 3. Random operational noise
            noise = np.random.normal(0, (max_val - min_val) * 0.03, hours)

            # 4. Occasional operational events (startup, shutdown, load changes)
            events = np.random.poisson(0.1, hours)  # Average 0.1 events per hour
            event_magnitude = np.where(events > 0,
                                       np.random.normal(0, (max_val - min_val) * 0.1, hours), 0)

            # Combine all components
            values = base_trend + daily_cycle + shift_cycle + noise + event_magnitude
            values = np.clip(values, min_val * 0.9, max_val * 1.1)

            data[param] = values

        # Generate correlated failure indicator based on actual sensor values
        failure_indicator = self.calculate_failure_indicator(data, base_condition)
        data['failure_indicator'] = failure_indicator

        return pd.DataFrame(data)

    def calculate_failure_indicator(self, sensor_data, base_condition):
        """Calculate failure indicator based on sensor correlations"""
        hours = len(sensor_data['mill_motor_air_temp'])

        # Define critical thresholds for each sensor
        thresholds = {
            'mill_motor_air_temp': 70,
            'mill_motor_current': 250,
            'vibrations_velocity': 3.0,
            'mill_diff_pressure': 4.0,
            'machine_loading': 105
        }

        # Calculate normalized risk scores for each parameter
        risk_scores = []

        for param, threshold in thresholds.items():
            if param in sensor_data:
                values = np.array(sensor_data[param])
                # Normalize risk: 0 when well below threshold, 1 when well above
                risk = np.clip((values - threshold * 0.8) / (threshold * 0.4), 0, 1)
                risk_scores.append(risk)

        # Combine risk scores (weighted average)
        if risk_scores:
            combined_risk = np.mean(risk_scores, axis=0)
        else:
            combined_risk = np.zeros(hours)

        # Add some base noise and ensure realistic ranges
        base_noise = np.random.normal(0, 0.02, hours)

        # Adjust based on condition
        if base_condition == 'normal':
            # Normal: mostly low values, occasional small spikes
            failure_indicator = combined_risk * 0.3 + base_noise + 0.05
            failure_indicator = np.clip(failure_indicator, 0.0, 0.4)

        elif base_condition == 'warning':
            # Warning: moderate values with more variation
            failure_indicator = 0.3 + combined_risk * 0.4 + base_noise
            failure_indicator = np.clip(failure_indicator, 0.25, 0.75)

        else:  # critical
            # Critical: high values with some variation
            failure_indicator = 0.6 + combined_risk * 0.35 + base_noise
            failure_indicator = np.clip(failure_indicator, 0.6, 1.0)

        return failure_indicator

    def simulate_gradual_degradation(self, days):
        """Simulate realistic gradual machine degradation over time"""
        hours = days * 24
        timestamps = pd.date_range(start=datetime.now(), periods=hours, freq='H')

        # Create degradation timeline (S-curve for realistic wear patterns)
        time_factor = np.arange(hours) / hours
        # S-curve: slow start, faster middle, slow end
        degradation_curve = 1 / (1 + np.exp(-10 * (time_factor - 0.5)))

        data = {'timestamp': timestamps}

        # Parameters that degrade over time with different rates
        degradation_params = {
            'mill_motor_air_temp': {
                'start': (45, 55),
                'end': (65, 80),
                'rate': 1.0  # Normal degradation rate
            },
            'mill_motor_current': {
                'start': (180, 200),
                'end': (230, 270),
                'rate': 1.2  # Faster degradation
            },
            'vibrations_velocity': {
                'start': (1.5, 2.0),
                'end': (3.0, 4.5),
                'rate': 1.5  # Bearing wear accelerates
            },
            'mill_diff_pressure': {
                'start': (2.5, 3.0),
                'end': (4.0, 5.5),
                'rate': 0.8  # Slower pressure buildup
            },
            'coal_feed_flow': {
                'start': (100, 110),
                'end': (75, 85),
                'rate': 0.9  # Gradual flow reduction
            },
            'machine_loading': {
                'start': (80, 90),
                'end': (100, 115),
                'rate': 1.1  # Load increases with inefficiency
            }
        }

        # Stable parameters (less affected by wear)
        stable_params = {
            'mill_inlet_temp': (25, 40),
            'mill_outlet_temp': (55, 75)
        }

        # Generate degrading parameters
        for param, config in degradation_params.items():
            start_range = config['start']
            end_range = config['end']
            rate = config['rate']

            # Base values at start and end of period
            start_val = np.random.uniform(*start_range)
            end_val = np.random.uniform(*end_range)

            # Apply degradation curve with individual rate
            adjusted_curve = np.clip(degradation_curve ** (1 / rate), 0, 1)
            base_trend = start_val + (end_val - start_val) * adjusted_curve

            # Add operational patterns
            daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * abs(end_val - start_val) * 0.05
            noise = np.random.normal(0, abs(end_val - start_val) * 0.03, hours)

            # Add maintenance events (temporary improvements)
            if days > 30:  # Only for longer simulations
                maintenance_intervals = hours // (7 * 24)  # Weekly maintenance
                maintenance_effect = np.zeros(hours)
                for i in range(1, maintenance_intervals + 1):
                    maint_time = i * 7 * 24
                    if maint_time < hours:
                        # Temporary improvement after maintenance
                        improvement = np.exp(-(np.arange(hours) - maint_time) / (2 * 24))
                        improvement = np.where(np.arange(hours) >= maint_time, improvement, 0)
                        maintenance_effect -= improvement * (end_val - start_val) * 0.1
            else:
                maintenance_effect = np.zeros(hours)

            data[param] = base_trend + daily_cycle + noise + maintenance_effect

        # Generate stable parameters
        for param, (min_val, max_val) in stable_params.items():
            base_val = np.random.uniform(min_val, max_val)
            daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * (max_val - min_val) * 0.1
            noise = np.random.normal(0, (max_val - min_val) * 0.05, hours)
            data[param] = base_val + daily_cycle + noise

        # Calculate realistic failure indicator based on degradation
        failure_indicator = self.calculate_degradation_failure_indicator(
            data, degradation_curve, days
        )
        data['failure_indicator'] = failure_indicator

        return pd.DataFrame(data)

    def calculate_degradation_failure_indicator(self, sensor_data, degradation_curve, days):
        """Calculate failure indicator for degradation simulation"""
        hours = len(degradation_curve)

        # Base degradation follows the curve but stays realistic
        base_degradation = 0.05 + degradation_curve * 0.4  # 0.05 to 0.45 range

        # Add sensor-based risk assessment
        risk_factors = []

        # Temperature risk
        if 'mill_motor_air_temp' in sensor_data:
            temp_risk = np.clip((np.array(sensor_data['mill_motor_air_temp']) - 50) / 30, 0, 1)
            risk_factors.append(temp_risk * 0.2)

        # Current risk
        if 'mill_motor_current' in sensor_data:
            current_risk = np.clip((np.array(sensor_data['mill_motor_current']) - 200) / 50, 0, 1)
            risk_factors.append(current_risk * 0.2)

        # Vibration risk (most critical for degradation)
        if 'vibrations_velocity' in sensor_data:
            vib_risk = np.clip((np.array(sensor_data['vibrations_velocity']) - 2.0) / 2.0, 0, 1)
            risk_factors.append(vib_risk * 0.3)

        # Combine risks
        if risk_factors:
            additional_risk = np.sum(risk_factors, axis=0)
        else:
            additional_risk = np.zeros(hours)

        # Final failure indicator
        failure_indicator = base_degradation + additional_risk

        # Add realistic noise and daily variations
        noise = np.random.normal(0, 0.01, hours)
        daily_variation = np.sin(np.arange(hours) * 2 * np.pi / 24) * 0.02

        failure_indicator += noise + daily_variation

        # Ensure realistic bounds
        failure_indicator = np.clip(failure_indicator, 0.0, 0.8)  # Cap at 0.8 for gradual degradation

        return failure_indicator

    def predict_machine_status(self, df):
        """Improved prediction method with better thresholds"""
        predictions = []

        # Define more realistic thresholds based on failure_indicator
        # These should match your training data distribution
        normal_threshold = 0.33
        warning_threshold = 0.67

        # Process each row
        for idx, row in df.iterrows():
            failure_score = row['failure_indicator']

            # Determine status based on thresholds
            if failure_score < normal_threshold:
                status = 'Normal'
                color = 'green'
                probs = [0.8 - failure_score, 0.15 + failure_score / 2, 0.05 + failure_score / 4]
            elif failure_score < warning_threshold:
                status = 'Warning'
                color = 'orange'
                probs = [0.3 - (failure_score - normal_threshold),
                         0.6 + (failure_score - normal_threshold),
                         0.1 + (failure_score - normal_threshold) / 2]
            else:
                status = 'Critical'
                color = 'red'
                probs = [0.1, 0.2, 0.7 + (failure_score - warning_threshold)]

            # Normalize probabilities
            prob_sum = sum(probs)
            probs = [p / prob_sum for p in probs]

            predictions.append({
                'timestamp': row['timestamp'],
                'status': status,
                'color': color,
                'confidence': max(probs),
                'probabilities': {
                    'normal': probs[0],
                    'warning': probs[1],
                    'critical': probs[2]
                },
                'failure_score': failure_score
            })

        return predictions

    def debug_simulation(self, df, predictions):
        """Debug method to analyze simulation results"""
        print(f"\n=== SIMULATION DEBUG ANALYSIS ===")
        print(f"Total data points: {len(df)}")
        print(f"Simulation period: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Analyze failure indicator distribution
        failure_stats = df['failure_indicator'].describe()
        print(f"\nFailure Indicator Statistics:")
        for stat, value in failure_stats.items():
            print(f"  {stat}: {value:.4f}")

        # Analyze prediction distribution
        if predictions:
            pred_counts = {'Normal': 0, 'Warning': 0, 'Critical': 0}
            confidence_sum = 0

            for pred in predictions:
                pred_counts[pred['status']] += 1
                confidence_sum += pred['confidence']

            total_preds = len(predictions)
            print(f"\nPrediction Distribution:")
            for status, count in pred_counts.items():
                percentage = (count / total_preds) * 100
                print(f"  {status}: {count} ({percentage:.1f}%)")

            avg_confidence = confidence_sum / total_preds
            print(f"  Average Confidence: {avg_confidence:.3f}")

        # Sensor value ranges
        sensor_columns = [col for col in df.columns if col not in ['timestamp', 'failure_indicator']]
        print(f"\nSensor Value Ranges:")
        for sensor in sensor_columns:
            min_val = df[sensor].min()
            max_val = df[sensor].max()
            mean_val = df[sensor].mean()
            print(f"  {sensor}: {min_val:.2f} - {max_val:.2f} (avg: {mean_val:.2f})")

        print("=" * 50)

    # def generate_realistic_sensor_data(self, days, base_condition='normal'):
    #     """Generate realistic sensor data for simulation"""
    #     hours = days * 24
    #     timestamps = pd.date_range(start=datetime.now(), periods=hours, freq='H')
    #
    #     # Base parameter ranges for SPPA-T3000 Mill
    #     base_params = {
    #         'mill_motor_air_temp': {'normal': (45, 65), 'warning': (65, 85), 'critical': (85, 120)},
    #         'coal_feed_flow': {'normal': (80, 120), 'warning': (60, 80), 'critical': (40, 60)},
    #         'mill_inlet_temp': {'normal': (25, 45), 'warning': (45, 65), 'critical': (65, 85)},
    #         'mill_diff_pressure': {'normal': (2.5, 4.0), 'warning': (4.0, 6.0), 'critical': (6.0, 10.0)},
    #         'mill_motor_current': {'normal': (180, 220), 'warning': (220, 280), 'critical': (280, 350)},
    #         'vibrations_velocity': {'normal': (1.5, 3.0), 'warning': (3.0, 5.0), 'critical': (5.0, 8.0)},
    #         'mill_outlet_temp': {'normal': (55, 75), 'warning': (75, 95), 'critical': (95, 120)},
    #         'machine_loading': {'normal': (75, 95), 'warning': (95, 110), 'critical': (110, 130)}
    #     }
    #
    #     data = {'timestamp': timestamps}
    #
    #     # Generate sensor data with realistic patterns
    #     for param, ranges in base_params.items():
    #         min_val, max_val = ranges[base_condition]
    #
    #         # Create trend with some randomness
    #         trend = np.linspace(min_val, max_val, hours)
    #         noise = np.random.normal(0, (max_val - min_val) * 0.1, hours)
    #         daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * (max_val - min_val) * 0.1
    #
    #         values = trend + noise + daily_cycle
    #         values = np.clip(values, min_val * 0.8, max_val * 1.2)  # Allow some variation
    #
    #         data[param] = values
    #
    #     # Add failure indicator based on conditions
    #     if base_condition == 'normal':
    #         failure_values = np.random.uniform(0.0, 0.3, hours)
    #     elif base_condition == 'warning':
    #         failure_values = np.random.uniform(0.3, 0.7, hours)
    #     else:  # critical
    #         failure_values = np.random.uniform(0.7, 1.0, hours)
    #
    #     data['failure_indicator'] = failure_values
    #
    #     return pd.DataFrame(data)
    #
    # def simulate_gradual_degradation(self, days):
    #     """Simulate gradual machine degradation over time"""
    #     hours = days * 24
    #     timestamps = pd.date_range(start=datetime.now(), periods=hours, freq='H')
    #
    #     # Simulate gradual degradation
    #     degradation_factor = np.linspace(0, 1, hours)  # 0 to 1 over the period
    #
    #     data = {'timestamp': timestamps}
    #
    #     # Parameters that worsen over time
    #     degrading_params = {
    #         'mill_motor_air_temp': (45, 85),  # Increases with wear
    #         'mill_diff_pressure': (2.5, 6.0),  # Increases due to blockages
    #         'mill_motor_current': (180, 280),  # Increases with load
    #         'vibrations_velocity': (1.5, 5.0),  # Increases with bearing wear
    #         'coal_feed_flow': (120, 70),  # Decreases due to blockages
    #         'machine_loading': (85, 110)  # Increases as efficiency drops
    #     }
    #
    #     stable_params = {
    #         'mill_inlet_temp': (25, 45),
    #         'mill_outlet_temp': (55, 75)
    #     }
    #
    #     # Generate degrading parameters
    #     for param, (start_val, end_val) in degrading_params.items():
    #         base_trend = start_val + (end_val - start_val) * degradation_factor
    #         noise = np.random.normal(0, abs(end_val - start_val) * 0.05, hours)
    #         daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * abs(end_val - start_val) * 0.05
    #
    #         data[param] = base_trend + noise + daily_cycle
    #
    #     # Generate stable parameters
    #     for param, (min_val, max_val) in stable_params.items():
    #         values = np.random.uniform(min_val, max_val, hours)
    #         daily_cycle = np.sin(np.arange(hours) * 2 * np.pi / 24) * (max_val - min_val) * 0.1
    #         data[param] = values + daily_cycle
    #
    #     failure_indicator = (
    #             np.linspace(0.0, 0.33, hours)  # ramps linearly from 0.0 up to your Normal cutoff
    #             + np.random.normal(loc=0.0, scale=0.02, size=hours)  # small jitter so it’s not a perfect line
    #     )
    #     data['failure_indicator'] = np.clip(failure_indicator, 0.0, 1.0)
    #
    #     return pd.DataFrame(data)


    def predict_machine_status_old(self, df):
        """Predict machine status using the trained model"""
        if self.model is None or self.lstm_model is None:
            # Return dummy predictions for demonstration
            return self.generate_dummy_predictions(len(df))

        try:
            # Prepare data using the same preprocessing as training
            X, _ = self.lstm_model.prepare_data(df, is_training=False)

            if len(X) == 0:
                return self.generate_dummy_predictions(len(df))

            # Make predictions
            predictions = self.model.predict(X, verbose=0)

            # Convert to status format
            results = []
            for i, pred in enumerate(predictions):
                predicted_class = int(np.argmax(pred))
                confidence = float(np.max(pred))

                status_mapping = {0: 'Normal', 1: 'Warning', 2: 'Critical'}
                color_mapping = {0: 'green', 1: 'orange', 2: 'red'}

                results.append({
                    'timestamp': df.iloc[i + self.lstm_model.sequence_length][
                        'timestamp'] if i + self.lstm_model.sequence_length < len(df) else df.iloc[-1]['timestamp'],
                    'status': status_mapping[predicted_class],
                    'color': color_mapping[predicted_class],
                    'confidence': confidence,
                    'probabilities': {
                        'normal': float(pred[0]),
                        'warning': float(pred[1]) if len(pred) > 1 else 0.0,
                        'critical': float(pred[2]) if len(pred) > 2 else 0.0
                    }
                })

            return results

        except Exception as e:
            print(f"Prediction error: {e}")
            return self.generate_dummy_predictions(len(df))

    def generate_dummy_predictions(self, n_samples):
        """Generate dummy predictions for demonstration"""
        results = []
        for i in range(min(n_samples, 100)):  # Limit to 100 predictions
            # Simulate realistic prediction distribution
            if i < n_samples * 0.7:  # 70% normal
                status, color = 'Normal', 'green'
                probs = [0.8, 0.15, 0.05]
            elif i < n_samples * 0.9:  # 20% warning
                status, color = 'Warning', 'orange'
                probs = [0.2, 0.7, 0.1]
            else:  # 10% critical
                status, color = 'Critical', 'red'
                probs = [0.1, 0.2, 0.7]

            results.append({
                'timestamp': datetime.now() + timedelta(hours=i),
                'status': status,
                'color': color,
                'confidence': max(probs),
                'probabilities': {
                    'normal': probs[0],
                    'warning': probs[1],
                    'critical': probs[2]
                }
            })

        return results

    def generate_maintenance_recommendations(self, predictions):
        """Generate maintenance recommendations based on predictions"""
        recommendations = []

        # Count status occurrences
        status_counts = {'Normal': 0, 'Warning': 0, 'Critical': 0}
        for pred in predictions:
            status_counts[pred['status']] += 1

        total_predictions = len(predictions)

        # Generate recommendations
        if status_counts['Critical'] > 0:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Immediate Shutdown Required',
                'description': f'Critical conditions detected in {status_counts["Critical"]} instances. Immediate maintenance required.',
                'timeline': 'Within 24 hours',
                'components': ['Mill motor', 'Bearing system', 'Vibration dampeners']
            })

        if status_counts['Warning'] > total_predictions * 0.3:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'Scheduled Maintenance',
                'description': f'Warning conditions detected in {status_counts["Warning"]} instances. Schedule maintenance within next week.',
                'timeline': 'Within 7 days',
                'components': ['Coal feed system', 'Temperature sensors', 'Pressure monitoring']
            })

        if status_counts['Normal'] > total_predictions * 0.8:
            recommendations.append({
                'priority': 'LOW',
                'action': 'Routine Inspection',
                'description': 'System operating normally. Continue with routine maintenance schedule.',
                'timeline': 'Next scheduled maintenance',
                'components': ['All systems', 'Preventive checks']
            })

        return recommendations


# Initialize simulator
simulator = MillSimulator()


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    """Handle simulation request"""
    try:
        data = request.get_json()

        # Extract parameters
        simulation_type = data.get('type', 'days')
        duration = int(data.get('duration', 14))
        condition = data.get('condition', 'normal')

        # Convert duration to days
        if simulation_type == 'months':
            days = duration * 30
        elif simulation_type == 'years':
            days = duration * 365
        else:  # days
            days = duration

        # Limit simulation size for performance
        days = min(days, 365)  # Max 1 year

        # Generate sensor data
        if condition == 'degradation':
            df = simulator.simulate_gradual_degradation(days)
        else:
            df = simulator.generate_realistic_sensor_data(days, condition)

        # Make predictions
        predictions = simulator.predict_machine_status(df)

        # Generate maintenance recommendations
        recommendations = simulator.generate_maintenance_recommendations(predictions)

        # Prepare response data
        response_data = {
            'success': True,
            'predictions': predictions[:100],  # Limit to first 100 for performance
            'recommendations': recommendations,
            'summary': {
                'total_periods': len(predictions),
                'normal_count': sum(1 for p in predictions if p['status'] == 'Normal'),
                'warning_count': sum(1 for p in predictions if p['status'] == 'Warning'),
                'critical_count': sum(1 for p in predictions if p['status'] == 'Critical'),
                'simulation_days': days
            },
            'sensor_data': {
                'timestamps': [str(ts) for ts in df['timestamp'].head(100)],
                'mill_motor_current': df['mill_motor_current'].head(100).tolist(),
                'vibrations_velocity': df['vibrations_velocity'].head(100).tolist(),
                'mill_motor_air_temp': df['mill_motor_air_temp'].head(100).tolist(),
                'failure_indicator': df['failure_indicator'].head(100).tolist()
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/real-time-data')
def real_time_data():
    """Generate real-time sensor data"""
    try:
        # Generate current sensor reading
        current_time = datetime.now()

        # Simulate current readings with some randomness
        current_data = {
            'timestamp': current_time.isoformat(),
            'mill_motor_air_temp': np.random.uniform(45, 75),
            'coal_feed_flow': np.random.uniform(80, 120),
            'mill_inlet_temp': np.random.uniform(25, 45),
            'mill_diff_pressure': np.random.uniform(2.5, 4.5),
            'mill_motor_current': np.random.uniform(180, 240),
            'vibrations_velocity': np.random.uniform(1.5, 3.5),
            'mill_outlet_temp': np.random.uniform(55, 80),
            'machine_loading': np.random.uniform(75, 100),
            'failure_indicator': np.random.uniform(0.1, 0.4)
        }

        # Determine status based on readings
        if (current_data['mill_motor_air_temp'] > 70 or
                current_data['vibrations_velocity'] > 3.2 or
                current_data['mill_motor_current'] > 230):
            status = 'Warning'
            color = 'orange'
        elif (current_data['mill_motor_air_temp'] > 80 or
              current_data['vibrations_velocity'] > 4.5 or
              current_data['mill_motor_current'] > 280):
            status = 'Critical'
            color = 'red'
        else:
            status = 'Normal'
            color = 'green'

        current_data['status'] = status
        current_data['color'] = color

        return jsonify({
            'success': True,
            'data': current_data
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/export-report', methods=['POST'])
def export_report():
    """Export simulation report"""
    try:
        data = request.get_json()

        # Generate a simple report
        report_content = f"""
SPPA-T3000 Mill Machine Simulation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SIMULATION SUMMARY:
- Total Periods: {data.get('total_periods', 0)}
- Normal Conditions: {data.get('normal_count', 0)}
- Warning Conditions: {data.get('warning_count', 0)}
- Critical Conditions: {data.get('critical_count', 0)}

RECOMMENDATIONS:
{chr(10).join([f"- {rec['action']}: {rec['description']}" for rec in data.get('recommendations', [])])}
        """

        # Create a text file in memory
        output = io.StringIO()
        output.write(report_content)
        output.seek(0)

        return jsonify({
            'success': True,
            'report_content': report_content
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4001)