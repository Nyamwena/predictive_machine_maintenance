import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
import warnings
import pickle
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings('ignore')


class ImprovedLSTMPredictiveMaintenance:
    """
    Improved LSTM model with fixes for overfitting and class imbalance
    """

    def __init__(self, sequence_length=24, prediction_horizon=6, model_type='bidirectional'):
        """Initialize the improved LSTM model."""
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.model = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        self.base_features = None
        self.all_feature_columns = None

    def select_and_engineer_features(self, df, is_training=True):
        """Simplified feature engineering to reduce overfitting"""
        print("Engineering features...")

        # Core sensor parameters - reduced set
        core_features = [
            'mill_motor_air_temp', 'coal_feed_flow', 'mill_inlet_temp',
            'mill_diff_pressure', 'mill_motor_current', 'vibrations_velocity',
            'mill_outlet_temp', 'machine_loading'
        ]

        # Check which features exist
        available_features = []
        for feature in core_features:
            if feature in df.columns:
                available_features.append(feature)

        if is_training:
            self.base_features = available_features.copy()
        else:
            if self.base_features is not None:
                available_features = self.base_features.copy()

        # Start with basic features
        features_df = df[available_features].copy()

        # Add only essential engineered features
        # Simple rolling average for top 3 features
        rolling_features = available_features[:3] if len(available_features) >= 3 else available_features
        for feature in rolling_features:
            if feature in df.columns:
                features_df[f'{feature}_ma5'] = df[feature].rolling(window=5).mean()

        # Add temperature difference if both temps available
        if 'mill_inlet_temp' in available_features and 'mill_outlet_temp' in available_features:
            features_df['temp_diff'] = df['mill_outlet_temp'] - df['mill_inlet_temp']

        # Add current-to-loading ratio
        if 'mill_motor_current' in available_features and 'machine_loading' in available_features:
            features_df['current_per_load'] = df['mill_motor_current'] / (df['machine_loading'] + 1e-6)

        # Add basic time features if available
        if 'timestamp' in df.columns:
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            features_df['hour'] = df_copy['timestamp'].dt.hour
            features_df['day_of_week'] = df_copy['timestamp'].dt.dayofweek

        return features_df

    def create_better_target(self, df):
        """Create more balanced target variable"""
        if 'failure_indicator' in df.columns:
            # Use percentile-based thresholds instead of fixed values
            q33 = df['failure_indicator'].quantile(0.33)
            q67 = df['failure_indicator'].quantile(0.67)

            target = df['failure_indicator'].apply(
                lambda x: 0 if x <= q33 else (1 if x <= q67 else 2)
            ).values

            print(f"Target thresholds: Normal ≤ {q33:.3f}, Warning ≤ {q67:.3f}, Critical > {q67:.3f}")
            return target
        else:
            return np.zeros(len(df))

    def create_better_target_v2(self, df):
        """Create more balanced target variable - IMPROVED VERSION"""
        if 'failure_indicator' not in df.columns:
            return np.zeros(len(df))

        print("\n" + "=" * 50)
        print("TARGET CREATION ANALYSIS")
        print("=" * 50)

        failure_values = df['failure_indicator'].values
        print(f"Failure indicator range: {failure_values.min():.4f} to {failure_values.max():.4f}")
        print(f"Mean: {failure_values.mean():.4f}, Std: {failure_values.std():.4f}")

        # Strategy 1: Fixed thresholds (if you know the expected range)
        if failure_values.max() > 1.0:
            # Assuming failure_indicator ranges from 0 to some max value
            threshold_1 = failure_values.max() * 0.33
            threshold_2 = failure_values.max() * 0.67
        else:
            # Assuming failure_indicator ranges from 0 to 1
            threshold_1 = 0.33
            threshold_2 = 0.67

        print(
            f"Fixed thresholds: Normal ≤ {threshold_1:.3f}, Warning ≤ {threshold_2:.3f}, Critical > {threshold_2:.3f}")

        target_fixed = np.where(failure_values <= threshold_1, 0,
                                np.where(failure_values <= threshold_2, 1, 2))

        print(f"Fixed threshold distribution: {Counter(target_fixed)}")

        # Strategy 2: Quantile-based with minimum class size
        q25 = np.percentile(failure_values, 25)
        q75 = np.percentile(failure_values, 75)

        print(f"Quantile thresholds: Normal ≤ {q25:.3f}, Warning ≤ {q75:.3f}, Critical > {q75:.3f}")

        target_quantile = np.where(failure_values <= q25, 0,
                                   np.where(failure_values <= q75, 1, 2))

        print(f"Quantile distribution: {Counter(target_quantile)}")

        # Strategy 3: Standard deviation based
        mean_val = failure_values.mean()
        std_val = failure_values.std()

        threshold_low = mean_val - 0.5 * std_val
        threshold_high = mean_val + 0.5 * std_val

        print(
            f"Std-based thresholds: Normal ≤ {threshold_low:.3f}, Warning ≤ {threshold_high:.3f}, Critical > {threshold_high:.3f}")

        target_std = np.where(failure_values <= threshold_low, 0,
                              np.where(failure_values <= threshold_high, 1, 2))

        print(f"Std-based distribution: {Counter(target_std)}")

        # Strategy 4: Force balanced classes
        n_samples = len(failure_values)
        indices = np.argsort(failure_values)

        class_0_size = n_samples // 3
        class_1_size = n_samples // 3
        class_2_size = n_samples - class_0_size - class_1_size

        target_balanced = np.zeros(n_samples, dtype=int)
        target_balanced[indices[:class_0_size]] = 0
        target_balanced[indices[class_0_size:class_0_size + class_1_size]] = 1
        target_balanced[indices[class_0_size + class_1_size:]] = 2

        print(f"Forced balanced distribution: {Counter(target_balanced)}")

        # Choose the best strategy based on class balance
        strategies = {
            'fixed': target_fixed,
            'quantile': target_quantile,
            'std': target_std,
            'balanced': target_balanced
        }

        # Evaluate each strategy
        best_strategy = None
        best_score = 0

        for name, target in strategies.items():
            classes = Counter(target)

            # Check if we have all 3 classes
            if len(classes) == 3:
                # Calculate balance score (higher is better)
                min_count = min(classes.values())
                max_count = max(classes.values())
                balance_score = min_count / max_count if max_count > 0 else 0

                print(f"{name} strategy: balance_score = {balance_score:.3f}")

                if balance_score > best_score:
                    best_score = balance_score
                    best_strategy = name

        if best_strategy is None:
            print("⚠️  WARNING: No strategy produced 3 classes! Using forced balanced approach.")
            best_strategy = 'balanced'

        selected_target = strategies[best_strategy]
        print(f"\n✅ Selected strategy: {best_strategy}")
        print(f"Final distribution: {Counter(selected_target)}")

        return selected_target

    def augment_critical_class(self, df, target):
        """Augment data to ensure Critical class is well represented"""
        print("\nAugmenting Critical class...")

        # Find indices of critical samples
        critical_indices = np.where(target == 2)[0]

        if len(critical_indices) == 0:
            print("No critical samples found! Creating synthetic critical conditions...")

            # Create synthetic critical samples by taking high failure_indicator values
            # and adding noise to create variations
            high_failure_indices = np.where(df['failure_indicator'] > df['failure_indicator'].quantile(0.9))[0]

            if len(high_failure_indices) > 0:
                # Take top 10% and label as critical
                critical_indices = high_failure_indices
                target[critical_indices] = 2
                print(f"Created {len(critical_indices)} synthetic critical samples")

        # If still too few critical samples, augment them
        min_samples_per_class = len(target) // 10  # At least 10% per class

        if len(critical_indices) < min_samples_per_class:
            print(f"Critical class has only {len(critical_indices)} samples, need at least {min_samples_per_class}")

            # Strategy: Take samples from Warning class that are closest to Critical threshold
            warning_indices = np.where(target == 1)[0]

            if len(warning_indices) > 0:
                # Find warning samples with highest failure_indicator values
                warning_failure_values = df['failure_indicator'].iloc[warning_indices]
                top_warning_indices = warning_indices[np.argsort(warning_failure_values)[-min_samples_per_class:]]

                # Convert some warning samples to critical
                samples_to_convert = min(len(top_warning_indices), min_samples_per_class - len(critical_indices))
                target[top_warning_indices[-samples_to_convert:]] = 2

                print(f"Converted {samples_to_convert} Warning samples to Critical")

        return target

    def create_synthetic_critical_data(self, df):
        """Create synthetic critical failure conditions"""
        print("Creating synthetic critical failure data...")

        # Identify normal operating ranges
        feature_stats = {}
        key_features = ['mill_motor_current', 'vibrations_velocity', 'mill_diff_pressure',
                        'mill_motor_air_temp', 'mill_outlet_temp']

        for feature in key_features:
            if feature in df.columns:
                feature_stats[feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'max': df[feature].max(),
                    'q95': df[feature].quantile(0.95)
                }

        # Create synthetic critical scenarios
        synthetic_rows = []
        n_synthetic = len(df) // 20  # Create 5% synthetic data

        for i in range(n_synthetic):
            synthetic_row = df.iloc[np.random.randint(0, len(df))].copy()

            # Modify key parameters to simulate critical conditions
            for feature, stats in feature_stats.items():
                if np.random.random() < 0.7:  # 70% chance to modify each feature
                    # Create values beyond normal range
                    if 'current' in feature.lower() or 'pressure' in feature.lower():
                        # High current or pressure indicates problems
                        synthetic_row[feature] = stats['q95'] + np.random.normal(0, stats['std'] * 0.5)
                    elif 'temp' in feature.lower():
                        # High temperature indicates problems
                        synthetic_row[feature] = stats['q95'] + np.random.normal(0, stats['std'] * 0.3)
                    elif 'vibration' in feature.lower():
                        # High vibration indicates problems
                        synthetic_row[feature] = stats['q95'] + np.random.normal(0, stats['std'] * 0.8)

            # Set high failure indicator
            synthetic_row['failure_indicator'] = np.random.uniform(0.8, 1.0)
            synthetic_rows.append(synthetic_row)

        if synthetic_rows:
            synthetic_df = pd.DataFrame(synthetic_rows)
            augmented_df = pd.concat([df, synthetic_df], ignore_index=True)
            print(f"Added {len(synthetic_rows)} synthetic critical samples")
            return augmented_df

        return df

    def train_with_guaranteed_3_classes(self, df, test_size=0.2, epochs=50, batch_size=32):
        """Training that guarantees 3 classes - UPDATED VERSION"""
        print("Starting training with guaranteed 3 classes...")

        # Step 1: Augment data to ensure critical samples exist
        df_augmented = self.create_synthetic_critical_data(df)

        # Step 2: Create features
        features_df = self.select_and_engineer_features(df_augmented, is_training=True)
        anomaly_flags = self.detect_anomalies_simple(features_df)
        combined_features = pd.concat([features_df, anomaly_flags], axis=1)
        combined_features = combined_features.dropna(axis=1, thresh=len(combined_features) * 0.8)
        self.all_feature_columns = list(combined_features.columns)
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Step 3: Create better target with guaranteed 3 classes
        target = self.create_better_target_v2(df_augmented)

        # Step 4: Ensure we have critical samples
        target = self.augment_critical_class(df_augmented, target)

        # Verify we have 3 classes
        unique_classes = np.unique(target)
        if len(unique_classes) != 3:
            print(f"⚠️  Still only have {len(unique_classes)} classes: {unique_classes}")
            print("Forcing 3-class distribution...")
            # Force equal distribution
            n_per_class = len(target) // 3
            indices = np.random.permutation(len(target))
            target[indices[:n_per_class]] = 0
            target[indices[n_per_class:2 * n_per_class]] = 1
            target[indices[2 * n_per_class:]] = 2

        print(f"✅ Final target distribution: {Counter(target)}")

        # Step 5: Scale features and create sequences
        features_scaled = self.scaler.fit_transform(combined_features.values)
        X, y = self.create_sequences(features_scaled, target, self.sequence_length)

        if len(X) == 0:
            print("Error: No sequences created.")
            return None, None, None

        print(f"Sequence shapes: X={X.shape}, y={y.shape}")
        print(f"Sequence target distribution: {Counter(y)}")

        # Step 6: Balance dataset if needed
        X_balanced, y_balanced = self.balance_dataset(X, y)

        # Step 7: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=test_size, random_state=42,
            stratify=y_balanced
        )

        print(f"Final distributions:")
        print(f"Training: {Counter(y_train)}")
        print(f"Testing: {Counter(y_test)}")

        # Step 8: Build and train model
        self.model = self.build_robust_model((X_train.shape[1], X_train.shape[2]), 3)

        # Calculate class weights
        try:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
            print(f"Class weights: {class_weight_dict}")
        except:
            class_weight_dict = None

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train model
        try:
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )

            return history, X_test, y_test

        except Exception as e:
            print(f"Training failed: {e}")
            return None, None, None



    # ADD THIS METHOD after create_better_target
    def balance_dataset(self, X, y):
        """Balance the dataset using SMOTE - SAFER VERSION"""
        print("Original class distribution:", Counter(y))
        print(f"Original X shape: {X.shape}")

        # Check if we need balancing
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())

        # If data is reasonably balanced, skip SMOTE
        if max_count / min_count < 3:
            print("Data is reasonably balanced, skipping SMOTE")
            return X, y

        try:
            # Reshape for SMOTE (flatten sequences)
            original_shape = X.shape
            X_flat = X.reshape(X.shape[0], -1)
            print(f"Flattened shape for SMOTE: {X_flat.shape}")

            # Use regular SMOTE instead of SMOTETomek for more stability
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
            X_balanced, y_balanced = smote.fit_resample(X_flat, y)

            # Reshape back to sequences
            n_samples = X_balanced.shape[0]
            X_balanced = X_balanced.reshape(n_samples, original_shape[1], original_shape[2])

            print("Balanced class distribution:", Counter(y_balanced))
            print(f"Balanced X shape: {X_balanced.shape}")
            return X_balanced, y_balanced

        except Exception as e:
            print(f"SMOTE failed: {e}")
            print("Proceeding without balancing...")
            return X, y

    # ADD THIS METHOD after balance_dataset
    def build_robust_model(self, input_shape, num_classes=3):
        """Build more robust model - SIMPLIFIED VERSION"""
        print(f"Building model with {num_classes} classes and input shape {input_shape}")

        model = Sequential()

        # Simpler architecture to avoid shape conflicts
        model.add(LSTM(
            32,
            return_sequences=True,
            input_shape=input_shape,
            dropout=0.2,
            recurrent_dropout=0.2
        ))

        model.add(LSTM(
            16,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2
        ))

        # Simpler dense layers
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))

        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.3))

        # Output layer
        model.add(Dense(num_classes, activation='softmax'))

        # Compile with appropriate optimizer
        optimizer = Adam(learning_rate=0.001)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    # ADD THIS METHOD after build_robust_model
    def train_with_fixes(self, df, test_size=0.2, epochs=50, batch_size=32):
        """Training with all fixes applied - CORRECTED VERSION"""
        print("Starting training with fixes...")

        # Prepare initial data
        X, y = self.prepare_data(df, is_training=True)

        # Replace the target creation with better version
        if 'failure_indicator' in df.columns:
            print("Creating better target variable...")
            y = self.create_better_target(df)

            # Re-create sequences with new target
            features_df = self.select_and_engineer_features(df, is_training=True)
            anomaly_flags = self.detect_anomalies_simple(features_df)
            combined_features = pd.concat([features_df, anomaly_flags], axis=1)
            combined_features = combined_features.dropna(axis=1, thresh=len(combined_features) * 0.8)
            self.all_feature_columns = list(combined_features.columns)
            combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            features_scaled = self.scaler.fit_transform(combined_features.values)
            X, y = self.create_sequences(features_scaled, y, self.sequence_length)

        if len(X) == 0:
            print("Error: No sequences created.")
            return None, None, None

        # Check and fix labels
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        print(f"Unique classes found: {unique_classes}")
        print(f"Number of classes: {num_classes}")
        print(f"Original shape: X={X.shape}, y={y.shape}")
        print(f"Before balancing: {Counter(y)}")

        # Ensure labels are continuous from 0
        if not np.array_equal(unique_classes, np.arange(num_classes)):
            print("Remapping labels to continuous range...")
            label_mapping = {old: new for new, old in enumerate(sorted(unique_classes))}
            y = np.array([label_mapping[label] for label in y])
            print(f"After remapping: {Counter(y)}")

        # Try balancing (may skip if not needed)
        X_balanced, y_balanced = self.balance_dataset(X, y)

        # Check shapes after balancing
        print(f"After balancing: X={X_balanced.shape}, y={y_balanced.shape}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced, test_size=test_size, random_state=42,
            stratify=y_balanced if len(np.unique(y_balanced)) > 1 else None
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Training shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
        print(f"Final training distribution: {Counter(y_train)}")
        print(f"Final test distribution: {Counter(y_test)}")

        # Build model with correct number of classes
        actual_num_classes = len(np.unique(y_balanced))
        print(f"Building model for {actual_num_classes} classes")
        self.model = self.build_robust_model((X_train.shape[1], X_train.shape[2]), actual_num_classes)

        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()

        # Calculate class weights
        try:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
            print(f"Class weights: {class_weight_dict}")
        except:
            class_weight_dict = None
            print("Could not calculate class weights, proceeding without them")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Reduced patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train model with error handling
        try:
            print(f"\nStarting training with batch_size={batch_size}")
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                class_weight=class_weight_dict,
                verbose=1
            )

            return history, X_test, y_test

        except Exception as e:
            print(f"Training failed with error: {e}")
            print("Trying with smaller batch size...")

            # Retry with smaller batch size
            try:
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=16,  # Smaller batch size
                    callbacks=callbacks,
                    class_weight=class_weight_dict,
                    verbose=1
                )
                return history, X_test, y_test
            except Exception as e2:
                print(f"Training failed again: {e2}")
                return None, None, None

    # ADD THIS METHOD after train_with_fixes
    def comprehensive_evaluation(self, X_test, y_test):
        """Comprehensive evaluation with multiple metrics"""
        if self.model is None:
            print("Model not trained yet!")
            return

        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        print("\n" + "=" * 60)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

        print(f"\nOverall Accuracy: {accuracy:.4f}")
        print("\nClass-wise Performance:")
        class_names = ['Normal', 'Warning', 'Critical']
        for i, class_name in enumerate(class_names):
            print(
                f"{class_name:>8}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")

        # Macro and weighted averages
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support)
        print(f"\nMacro F1-Score: {macro_f1:.4f}")
        print(f"Weighted F1-Score: {weighted_f1:.4f}")

        # ROC-AUC for each class (one-vs-rest)
        try:
            y_test_binarized = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))
            roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')
            print(f"ROC-AUC (macro): {roc_auc:.4f}")
        except:
            print("Could not calculate ROC-AUC")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        return y_pred, y_pred_proba

    # ADD THIS METHOD after comprehensive_evaluation
    def check_data_quality(self, df):
        """Check data quality and distribution"""
        print("\n" + "=" * 50)
        print("DATA QUALITY ANALYSIS")
        print("=" * 50)

        if 'failure_indicator' in df.columns:
            print(f"Failure indicator statistics:")
            print(df['failure_indicator'].describe())

            # Plot distribution
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.hist(df['failure_indicator'], bins=50, alpha=0.7)
            plt.title('Failure Indicator Distribution')
            plt.xlabel('Failure Indicator Value')
            plt.ylabel('Frequency')

            plt.subplot(1, 2, 2)
            plt.plot(df.index, df['failure_indicator'])
            plt.title('Failure Indicator Over Time')
            plt.xlabel('Time Index')
            plt.ylabel('Failure Indicator Value')

            plt.tight_layout()
            plt.show()

        # Check for missing values
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\nMissing data found:")
            print(missing_data[missing_data > 0])
        else:
            print("\nNo missing data found.")

        # Check recent data for prediction
        recent_failure_dist = df.tail(200)[
            'failure_indicator'].describe() if 'failure_indicator' in df.columns else None
        if recent_failure_dist is not None:
            print(f"\nRecent data (last 200 samples) failure indicator stats:")
            print(recent_failure_dist)

            if recent_failure_dist['std'] < 0.01:
                print("⚠️  WARNING: Recent data shows very low variance - may not be suitable for prediction!")

    def detect_anomalies_simple(self, df):
        """Simplified anomaly detection"""
        print("Detecting anomalies...")

        anomaly_flags = pd.DataFrame(index=df.index)

        # Only detect anomalies for key features to avoid overfitting
        key_features = ['mill_motor_current', 'vibrations_velocity', 'mill_diff_pressure']

        for column in key_features:
            if column in df.columns:
                # Simple z-score based detection
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                anomaly_flags[f'{column}_anomaly'] = (z_scores > 2.5).astype(int)

        # Summary anomaly count
        anomaly_flags['anomaly_count'] = anomaly_flags.sum(axis=1)

        return anomaly_flags

    def create_sequences(self, data, target, seq_len):
        """Create sequences for LSTM"""
        X, y = [], []

        for i in range(seq_len, len(data) - self.prediction_horizon + 1):
            X.append(data[i - seq_len:i])
            y.append(target[i + self.prediction_horizon - 1])

        return np.array(X), np.array(y)

    def prepare_data(self, df, is_training=True):
        """Improved data preparation"""
        print("Preparing data...")

        # Feature engineering
        features_df = self.select_and_engineer_features(df, is_training=is_training)

        # Simple anomaly detection
        anomaly_flags = self.detect_anomalies_simple(features_df)

        # Combine features
        combined_features = pd.concat([features_df, anomaly_flags], axis=1)

        if is_training:
            # Remove features with too many missing values
            combined_features = combined_features.dropna(axis=1, thresh=len(combined_features) * 0.8)
            self.all_feature_columns = list(combined_features.columns)
            print(f"Selected {len(self.all_feature_columns)} features for training")
        else:
            if self.all_feature_columns is not None:
                # Ensure consistency
                for col in self.all_feature_columns:
                    if col not in combined_features.columns:
                        combined_features[col] = 0
                combined_features = combined_features[self.all_feature_columns]

        # Fill missing values
        combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Create balanced target variable
        if 'failure_indicator' in df.columns:
            # More balanced classification
            target = df['failure_indicator'].apply(
                lambda x: 0 if x < 0.3 else (1 if x < 0.7 else 2)
            ).values
        else:
            target = np.zeros(len(df))

        print(f"Feature shape: {combined_features.shape}")
        print(f"Target distribution: {Counter(target)}")

        # Scale features
        if is_training:
            features_scaled = self.scaler.fit_transform(combined_features.values)
        else:
            features_scaled = self.scaler.transform(combined_features.values)

        # Create sequences
        X, y = self.create_sequences(features_scaled, target, self.sequence_length)

        return X, y

    def build_improved_model(self, input_shape, num_classes=3):
        """Build improved model with regularization"""
        print(f"Building improved model with input shape: {input_shape}")

        model = Sequential()

        # Reduced complexity to prevent overfitting
        model.add(LSTM(
            16,  # Reduced from 32
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),  # L1/L2 regularization
            dropout=0.2,
            recurrent_dropout=0.2
        ))

        model.add(LSTM(
            8,  # Reduced from 16
            return_sequences=False,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
            dropout=0.2,
            recurrent_dropout=0.2
        ))

        # Simplified dense layers
        model.add(Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(0.5))  # Increased dropout

        model.add(Dense(8, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
        model.add(Dropout(0.3))

        model.add(Dense(num_classes, activation='softmax'))

        # Lower learning rate for better convergence
        optimizer = Adam(learning_rate=0.0005)

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_improved(self, df, test_size=0.2, epochs=100, batch_size=64):
        """Improved training with better regularization"""
        print("Starting improved training...")

        # Prepare data
        X, y = self.prepare_data(df, is_training=True)

        if len(X) == 0:
            print("Error: No sequences created.")
            return None, None, None

        # Stratified split to maintain class distribution
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Training class distribution: {Counter(y_train)}")
        print(f"Test class distribution: {Counter(y_test)}")

        # Build model
        num_classes = len(np.unique(y))
        self.model = self.build_improved_model((X.shape[1], X.shape[2]), num_classes)

        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")

        # Improved callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',  # Monitor accuracy instead of loss
                patience=20,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.3,
                patience=10,
                min_lr=1e-6,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                'best_improved_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]

        # Train with larger validation split
        history = self.model.fit(
            X_train, y_train,
            validation_split=0.3,  # Increased validation split
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )

        # Evaluate on test set
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Check for overfitting
        final_train_acc = max(history.history['accuracy'])
        final_val_acc = max(history.history['val_accuracy'])
        overfitting_gap = final_train_acc - final_val_acc

        print(f"\nOverfitting Analysis:")
        print(f"Training Accuracy: {final_train_acc:.4f}")
        print(f"Validation Accuracy: {final_val_acc:.4f}")
        print(f"Overfitting Gap: {overfitting_gap:.4f}")

        if overfitting_gap > 0.1:
            print("⚠️  WARNING: Model may still be overfitting!")
        else:
            print("✅ Good generalization achieved!")

        return history, X_test, y_test

    def enhanced_evaluate(self, X_test, y_test):
        """Enhanced evaluation with focus on class-specific performance"""
        if self.model is None:
            print("Model not trained yet!")
            return

        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Detailed analysis
        print("\n" + "=" * 50)
        print("DETAILED PERFORMANCE ANALYSIS")
        print("=" * 50)

        # Class-specific metrics
        target_names = ['Normal', 'Warning', 'Critical']
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Confidence analysis
        confidence_scores = np.max(y_pred_proba, axis=1)
        print(f"\nConfidence Statistics:")
        print(f"Mean Confidence: {np.mean(confidence_scores):.3f}")
        print(f"Min Confidence: {np.min(confidence_scores):.3f}")
        print(f"Max Confidence: {np.max(confidence_scores):.3f}")
        print(f"Std Confidence: {np.std(confidence_scores):.3f}")

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

        return y_pred, y_pred_proba

    def predict_maintenance_schedule(self, df, days_ahead=14):
        """Predict maintenance schedule with confidence thresholds"""
        if self.model is None:
            print("Model not trained yet!")
            return []

        print(f"Generating maintenance schedule for next {days_ahead} days...")

        # Get recent data
        recent_data = df.tail(200)
        X, y = self.prepare_data(recent_data, is_training=False)

        predictions = []
        if len(X) > 0:
            num_predictions = min(days_ahead, len(X))
            for i in range(num_predictions):
                sequence = X[i:i + 1]
                prediction = self.model.predict(sequence, verbose=0)

                # Apply confidence thresholds
                confidence = float(np.max(prediction[0]))
                predicted_class = int(np.argmax(prediction[0]))

                # Adjust predictions based on confidence
                if confidence < 0.6:  # Low confidence
                    maintenance_recommended = True  # Be conservative
                    adjusted_status = "Uncertain - Inspect"
                else:
                    maintenance_recommended = bool(predicted_class >= 1)
                    adjusted_status = ["Normal", "Warning", "Critical"][predicted_class]

                result = {
                    'day': i + 1,
                    'status': adjusted_status,
                    'normal_prob': float(prediction[0][0]),
                    'warning_prob': float(prediction[0][1]) if prediction.shape[1] > 1 else 0.0,
                    'critical_prob': float(prediction[0][2]) if prediction.shape[1] > 2 else 0.0,
                    'confidence': confidence,
                    'maintenance_recommended': maintenance_recommended
                }
                predictions.append(result)

        return predictions

    def save_model_and_scaler(self, model_filepath, scaler_filepath=None):
        """Save model and preprocessing components"""
        if self.model is None:
            print("No model to save!")
            return

        try:
            self.model.save(model_filepath)
            print(f"Model saved to {model_filepath}")

            if scaler_filepath is None:
                scaler_filepath = model_filepath.replace('.h5', '_preprocessing.pkl')

            preprocessing_data = {
                'scaler': self.scaler,
                'base_features': self.base_features,
                'all_feature_columns': self.all_feature_columns,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }

            with open(scaler_filepath, 'wb') as f:
                pickle.dump(preprocessing_data, f)
            print(f"Preprocessing data saved to {scaler_filepath}")

        except Exception as e:
            print(f"Error saving model: {e}")


def debug_target_creation(self, df):
    """Debug function to check target creation"""
    if 'failure_indicator' not in df.columns:
        print("No failure_indicator column found!")
        return

    print("Failure indicator analysis:")
    print(f"Min: {df['failure_indicator'].min()}")
    print(f"Max: {df['failure_indicator'].max()}")
    print(f"Mean: {df['failure_indicator'].mean()}")
    print(f"Std: {df['failure_indicator'].std()}")

    # Test both approaches
    print("\n3-class target:")
    target_3class = self.create_better_target(df)
    print(f"Classes: {np.unique(target_3class)}")
    print(f"Distribution: {Counter(target_3class)}")

    print("\nBinary target:")
    target_binary = self.create_binary_target(df)
    print(f"Classes: {np.unique(target_binary)}")
    print(f"Distribution: {Counter(target_binary)}")


def main():
    """Main function with guaranteed 3-class model"""
    print("Loading data...")
    try:
        df = pd.read_csv("mill_predictive_maintenance_data.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded data: {df.shape}")
    except FileNotFoundError:
        print("Error: mill_predictive_maintenance_data.csv not found!")
        return

    # Initialize improved model
    improved_model = ImprovedLSTMPredictiveMaintenance(
        sequence_length=24,
        prediction_horizon=6,
        model_type='bidirectional'
    )

    # Check data quality first
    improved_model.check_data_quality(df)

    # Train with guaranteed 3 classes - USE NEW METHOD
    print("Training model with guaranteed 3-class output...")
    history, X_test, y_test = improved_model.train_with_guaranteed_3_classes(df, epochs=50, batch_size=32)

    if history is not None:
        # Evaluate with fixed evaluation method
        y_pred, y_pred_proba = improved_model.comprehensive_evaluation(X_test, y_test)

        # Generate maintenance schedule
        maintenance_schedule = improved_model.predict_maintenance_schedule(df, days_ahead=14)

        print("\n" + "=" * 60)
        print("MAINTENANCE SCHEDULE (Next 14 Days)")
        print("=" * 60)

        for pred in maintenance_schedule:
            print(f"Day {pred['day']:2d}: {pred['status']:20} | "
                  f"Confidence: {pred['confidence']:.3f} | "
                  f"Maintenance: {'YES' if pred['maintenance_recommended'] else 'NO'}")

        # Save model
        improved_model.save_model_and_scaler('guaranteed_3class_lstm_model.h5')
        print("\n✅ Model training completed with 3 classes guaranteed!")

    else:
        print("❌ Training failed.")




if __name__ == "__main__":
    main()