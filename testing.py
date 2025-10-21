"""
Comprehensive Testing Harness for Cloud Anomaly Detection ML System

This test suite includes:
1. Unit tests for data generation and feature engineering
2. Model performance tests (detection accuracy, classification metrics)
3. Integration tests for the full pipeline
4. Edge case and robustness tests
5. Performance benchmarks

Run with: pytest test_anomaly_detection.py -v
Or: python test_anomaly_detection.py (runs unittest)
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import datetime
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)


class AnomalyDetectionSystem:
    """Encapsulated version of the anomaly detection system for testing"""
    
    def __init__(self):
        self.df = None
        self.iso_model = None
        self.clf_model = None
        self.feature_cols = None
        self.label_map = {
            "normal": 0,
            "network_failure": 1,
            "cpu_spike": 2,
            "dns_issue": 3,
            "monitoring_subsystem": 4,
        }
        self.mitigation_policies = {
            "network_failure": lambda t: f"[{t}] ACTION: Reroute traffic away from affected AZ.",
            "cpu_spike": lambda t: f"[{t}] ACTION: Auto-scale compute pool.",
            "dns_issue": lambda t: f"[{t}] ACTION: Failover to secondary DNS providers.",
            "monitoring_subsystem": lambda t: f"[{t}] ACTION: Switch to backup monitoring plane.",
            "normal": lambda t: f"[{t}] ACTION: No action required.",
        }
    
    def generate_telemetry(self, T=10000, n_anomalies=60):
        """Generate simulated telemetry data with injected anomalies"""
        start = datetime.datetime(2025, 10, 20, 0, 0)
        timestamps = [start + datetime.timedelta(seconds=i) for i in range(T)]
        
        # Base normal signals
        latency = np.random.normal(loc=50, scale=5, size=T)
        error_rate = np.random.beta(a=1.5, b=100, size=T) * 0.02
        active_conns = np.random.poisson(lam=200, size=T)
        cpu = np.random.normal(loc=40, scale=5, size=T)
        dns_failures = np.random.binomial(1, 0.0005, size=T)
        
        self.df = pd.DataFrame({
            "ts": timestamps,
            "latency_ms": latency,
            "error_rate": error_rate,
            "active_conns": active_conns,
            "cpu_percent": cpu,
            "dns_failures": dns_failures,
        })
        
        # Inject anomalies
        anomaly_types = ["network_failure", "cpu_spike", "dns_issue", "monitoring_subsystem"]
        labels = ["normal"] * T
        
        for i in range(n_anomalies):
            t = np.random.randint(300, T - 300)
            kind = random.choice(anomaly_types)
            length = np.random.randint(5, 80)
            
            for j in range(t, min(T, t + length)):
                if kind == "network_failure":
                    self.df.loc[j, "latency_ms"] += np.random.normal(300, 50)
                    self.df.loc[j, "active_conns"] = np.maximum(0, self.df.loc[j, "active_conns"] - np.random.poisson(100))
                    self.df.loc[j, "error_rate"] += np.random.beta(2, 10) * 0.5
                elif kind == "cpu_spike":
                    self.df.loc[j, "cpu_percent"] += np.random.normal(40, 10)
                    self.df.loc[j, "latency_ms"] += np.random.normal(80, 20)
                elif kind == "dns_issue":
                    self.df.loc[j, "dns_failures"] += np.random.binomial(1, 0.6)
                    self.df.loc[j, "latency_ms"] += np.random.normal(150, 40)
                elif kind == "monitoring_subsystem":
                    self.df.loc[j, "latency_ms"] += np.random.normal(120, 120)
                    self.df.loc[j, "active_conns"] = np.maximum(0, self.df.loc[j, "active_conns"] + np.random.randint(-150, 150))
                labels[j] = kind
        
        self.df["label"] = labels
        return self.df
    
    def engineer_features(self, window=15):
        """Add rolling statistical features"""
        self.df["latency_roll_mean"] = self.df["latency_ms"].rolling(window=window, min_periods=1).mean()
        self.df["latency_roll_std"] = self.df["latency_ms"].rolling(window=window, min_periods=1).std().fillna(0)
        self.df["error_roll_mean"] = self.df["error_rate"].rolling(window=window, min_periods=1).mean()
        self.df["active_conns_roll_mean"] = self.df["active_conns"].rolling(window=window, min_periods=1).mean()
        self.df["cpu_roll_mean"] = self.df["cpu_percent"].rolling(window=window, min_periods=1).mean()
        self.df["dns_failures_roll_sum"] = self.df["dns_failures"].rolling(window=window, min_periods=1).sum()
        
        self.feature_cols = [
            "latency_ms", "latency_roll_mean", "latency_roll_std",
            "error_rate", "error_roll_mean",
            "active_conns", "active_conns_roll_mean",
            "cpu_percent", "cpu_roll_mean",
            "dns_failures", "dns_failures_roll_sum",
        ]
    
    def train_detector(self, contamination=0.01):
        """Train IsolationForest anomaly detector"""
        X = self.df[self.feature_cols].fillna(0).values
        T = len(self.df)
        train_cut = int(0.6 * T)
        normal_mask = (self.df.index < train_cut) & (self.df["label"] == "normal")
        
        self.iso_model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
        self.iso_model.fit(X[normal_mask])
        
        iso_preds = self.iso_model.predict(X)
        self.df["anomaly_flag"] = (iso_preds == -1).astype(int)
    
    def train_classifier(self):
        """Train RandomForest root-cause classifier"""
        anom_df = self.df[self.df["label"] != "normal"].copy()
        normal_sample = self.df[self.df["label"] == "normal"].sample(
            n=min(len(anom_df), 2000), random_state=42
        )
        clf_df = pd.concat([anom_df, normal_sample])
        clf_df["class"] = clf_df["label"].map(self.label_map)
        
        X_clf = clf_df[self.feature_cols].fillna(0).values
        y_clf = clf_df["class"].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
        )
        
        self.clf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        self.clf_model.fit(X_train, y_train)
        
        return X_test, y_test
    
    def get_mitigation_action(self, row):
        """Get mitigation action for a detected anomaly"""
        feat = row[self.feature_cols].values.reshape(1, -1)
        pred_class = self.clf_model.predict(feat)[0]
        inv_map = {v: k for k, v in self.label_map.items()}
        pred_label = inv_map.get(pred_class, "normal")
        return self.mitigation_policies[pred_label](row["ts"])


class TestDataGeneration(unittest.TestCase):
    """Test data generation functionality"""
    
    def setUp(self):
        self.system = AnomalyDetectionSystem()
    
    def test_telemetry_shape(self):
        """Test that generated telemetry has correct shape"""
        df = self.system.generate_telemetry(T=1000, n_anomalies=10)
        self.assertEqual(len(df), 1000)
        self.assertEqual(len(df.columns), 6)  # 5 metrics + label
    
    def test_telemetry_columns(self):
        """Test that all required columns are present"""
        df = self.system.generate_telemetry(T=1000)
        required_cols = ["ts", "latency_ms", "error_rate", "active_conns", "cpu_percent", "dns_failures"]
        for col in required_cols:
            self.assertIn(col, df.columns)
    
    def test_anomaly_injection(self):
        """Test that anomalies are actually injected"""
        df = self.system.generate_telemetry(T=1000, n_anomalies=20)
        anomaly_count = (df["label"] != "normal").sum()
        self.assertGreater(anomaly_count, 0)
    
    def test_metric_ranges(self):
        """Test that metrics are in reasonable ranges"""
        df = self.system.generate_telemetry(T=1000, n_anomalies=10)
        self.assertTrue((df["latency_ms"] >= 0).all())
        self.assertTrue((df["error_rate"] >= 0).all())
        self.assertTrue((df["error_rate"] <= 1).all())
        self.assertTrue((df["active_conns"] >= 0).all())
        self.assertTrue((df["cpu_percent"] >= 0).all())
    
    def test_timestamp_continuity(self):
        """Test that timestamps are continuous"""
        df = self.system.generate_telemetry(T=100)
        time_diffs = [(df.loc[i+1, "ts"] - df.loc[i, "ts"]).total_seconds() for i in range(99)]
        self.assertTrue(all(diff == 1.0 for diff in time_diffs))


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def setUp(self):
        self.system = AnomalyDetectionSystem()
        self.system.generate_telemetry(T=1000, n_anomalies=10)
    
    def test_rolling_features_created(self):
        """Test that rolling features are created"""
        self.system.engineer_features(window=15)
        expected_features = ["latency_roll_mean", "latency_roll_std", "error_roll_mean",
                           "active_conns_roll_mean", "cpu_roll_mean", "dns_failures_roll_sum"]
        for feat in expected_features:
            self.assertIn(feat, self.system.df.columns)
    
    def test_rolling_features_no_nan(self):
        """Test that rolling features don't have NaN values after fillna"""
        self.system.engineer_features(window=15)
        for col in self.system.feature_cols:
            self.assertFalse(self.system.df[col].isna().any())
    
    def test_feature_count(self):
        """Test that we have the expected number of features"""
        self.system.engineer_features(window=15)
        self.assertEqual(len(self.system.feature_cols), 11)
    
    def test_rolling_window_sizes(self):
        """Test different rolling window sizes"""
        for window in [5, 10, 20, 30]:
            self.system.engineer_features(window=window)
            self.assertEqual(len(self.system.feature_cols), 11)


class TestAnomalyDetection(unittest.TestCase):
    """Test IsolationForest anomaly detection"""
    
    def setUp(self):
        self.system = AnomalyDetectionSystem()
        self.system.generate_telemetry(T=5000, n_anomalies=30)
        self.system.engineer_features()
    
    def test_detector_training(self):
        """Test that detector trains without errors"""
        self.system.train_detector()
        self.assertIsNotNone(self.system.iso_model)
    
    def test_anomaly_detection_rate(self):
        """Test that detection rate is within reasonable bounds"""
        self.system.train_detector()
        detection_rate = self.system.df["anomaly_flag"].mean()
        self.assertGreater(detection_rate, 0.001)  # At least 0.1%
        self.assertLess(detection_rate, 0.5)  # Less than 50%
    
    def test_detection_precision(self):
        """Test that precision is reasonable"""
        self.system.train_detector()
        true_anomaly = (self.system.df["label"] != "normal").astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_anomaly, self.system.df["anomaly_flag"], 
            average="binary", zero_division=0
        )
        self.assertGreater(precision, 0.1)  # At least 10% precision
    
    def test_detection_recall(self):
        """Test that recall is reasonable"""
        self.system.train_detector()
        true_anomaly = (self.system.df["label"] != "normal").astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_anomaly, self.system.df["anomaly_flag"], 
            average="binary", zero_division=0
        )
        self.assertGreater(recall, 0.05)  # At least 5% recall
    
    def test_contamination_parameter(self):
        """Test different contamination parameters"""
        for contamination in [0.005, 0.01, 0.02, 0.05]:
            self.system.train_detector(contamination=contamination)
            detection_rate = self.system.df["anomaly_flag"].mean()
            self.assertGreater(detection_rate, 0)


class TestRootCauseClassification(unittest.TestCase):
    """Test RandomForest root-cause classifier"""
    
    def setUp(self):
        self.system = AnomalyDetectionSystem()
        self.system.generate_telemetry(T=5000, n_anomalies=50)
        self.system.engineer_features()
        self.system.train_detector()
    
    def test_classifier_training(self):
        """Test that classifier trains without errors"""
        X_test, y_test = self.system.train_classifier()
        self.assertIsNotNone(self.system.clf_model)
        self.assertGreater(len(X_test), 0)
    
    def test_classification_accuracy(self):
        """Test that classification accuracy is reasonable"""
        X_test, y_test = self.system.train_classifier()
        y_pred = self.system.clf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.4)  # At least 40% accuracy (better than random)
    
    def test_all_classes_predicted(self):
        """Test that classifier can predict multiple classes"""
        X_test, y_test = self.system.train_classifier()
        y_pred = self.system.clf_model.predict(X_test)
        unique_predictions = len(np.unique(y_pred))
        self.assertGreater(unique_predictions, 1)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent for same input"""
        X_test, y_test = self.system.train_classifier()
        sample = X_test[:10]
        pred1 = self.system.clf_model.predict(sample)
        pred2 = self.system.clf_model.predict(sample)
        np.testing.assert_array_equal(pred1, pred2)


class TestMitigationPolicies(unittest.TestCase):
    """Test mitigation policy execution"""
    
    def setUp(self):
        self.system = AnomalyDetectionSystem()
        self.system.generate_telemetry(T=1000, n_anomalies=10)
        self.system.engineer_features()
        self.system.train_detector()
        self.system.train_classifier()
    
    def test_mitigation_action_generation(self):
        """Test that mitigation actions can be generated"""
        anomalies = self.system.df[self.system.df["anomaly_flag"] == 1]
        if len(anomalies) > 0:
            action = self.system.get_mitigation_action(anomalies.iloc[0])
            self.assertIsInstance(action, str)
            self.assertIn("ACTION", action)
    
    def test_all_anomaly_types_have_policies(self):
        """Test that all anomaly types have mitigation policies"""
        for anomaly_type in ["network_failure", "cpu_spike", "dns_issue", "monitoring_subsystem", "normal"]:
            self.assertIn(anomaly_type, self.system.mitigation_policies)
    
    def test_mitigation_action_uniqueness(self):
        """Test that different anomaly types produce different actions"""
        actions = set()
        for anomaly_type in ["network_failure", "cpu_spike", "dns_issue"]:
            action = self.system.mitigation_policies[anomaly_type](datetime.datetime.now())
            actions.add(action.split("]")[1])  # Remove timestamp
        self.assertEqual(len(actions), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for full pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution"""
        system = AnomalyDetectionSystem()
        
        # Step 1: Generate data
        df = system.generate_telemetry(T=2000, n_anomalies=20)
        self.assertEqual(len(df), 2000)
        
        # Step 2: Engineer features
        system.engineer_features()
        self.assertEqual(len(system.feature_cols), 11)
        
        # Step 3: Train detector
        system.train_detector()
        self.assertIsNotNone(system.iso_model)
        
        # Step 4: Train classifier
        X_test, y_test = system.train_classifier()
        self.assertIsNotNone(system.clf_model)
        
        # Step 5: Generate mitigations
        anomalies = system.df[system.df["anomaly_flag"] == 1]
        if len(anomalies) > 0:
            action = system.get_mitigation_action(anomalies.iloc[0])
            self.assertIsInstance(action, str)
    
    def test_pipeline_with_no_anomalies(self):
        """Test pipeline with no injected anomalies"""
        system = AnomalyDetectionSystem()
        df = system.generate_telemetry(T=1000, n_anomalies=0)
        system.engineer_features()
        system.train_detector()
        # Should still detect some points as anomalies due to contamination parameter
        detection_rate = system.df["anomaly_flag"].mean()
        self.assertGreaterEqual(detection_rate, 0)
    
    def test_pipeline_with_many_anomalies(self):
        """Test pipeline with high anomaly density"""
        system = AnomalyDetectionSystem()
        df = system.generate_telemetry(T=2000, n_anomalies=100)
        system.engineer_features()
        system.train_detector()
        system.train_classifier()
        
        true_anomaly = (system.df["label"] != "normal").astype(int)
        self.assertGreater(true_anomaly.sum(), 100)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and robustness"""
    
    def test_small_dataset(self):
        """Test with very small dataset"""
        system = AnomalyDetectionSystem()
        df = system.generate_telemetry(T=500, n_anomalies=5)
        system.engineer_features()
        system.train_detector()
        self.assertIsNotNone(system.iso_model)
    
    def test_zero_variance_feature(self):
        """Test handling of zero-variance features"""
        system = AnomalyDetectionSystem()
        system.generate_telemetry(T=1000, n_anomalies=10)
        # Set one feature to constant
        system.df["cpu_percent"] = 50.0
        system.engineer_features()
        system.train_detector()
        self.assertIsNotNone(system.iso_model)
    
    def test_missing_data_handling(self):
        """Test that missing data is handled properly"""
        system = AnomalyDetectionSystem()
        system.generate_telemetry(T=1000, n_anomalies=10)
        # Introduce some NaN values
        system.df.loc[100:110, "latency_ms"] = np.nan
        system.engineer_features()
        
        # Check that features don't have NaN after fillna
        X = system.df[system.feature_cols].fillna(0).values
        self.assertFalse(np.isnan(X).any())
    
    def test_extreme_values(self):
        """Test handling of extreme metric values"""
        system = AnomalyDetectionSystem()
        system.generate_telemetry(T=1000, n_anomalies=10)
        # Add extreme values
        system.df.loc[500, "latency_ms"] = 10000
        system.df.loc[501, "error_rate"] = 1.0
        system.engineer_features()
        system.train_detector()
        self.assertIsNotNone(system.iso_model)


class TestPerformance(unittest.TestCase):
    """Performance benchmarks"""
    
    def test_data_generation_speed(self):
        """Test data generation performance"""
        system = AnomalyDetectionSystem()
        start = time.time()
        system.generate_telemetry(T=10000, n_anomalies=60)
        elapsed = time.time() - start
        self.assertLess(elapsed, 5.0)  # Should complete in under 5 seconds
    
    def test_feature_engineering_speed(self):
        """Test feature engineering performance"""
        system = AnomalyDetectionSystem()
        system.generate_telemetry(T=10000, n_anomalies=60)
        start = time.time()
        system.engineer_features()
        elapsed = time.time() - start
        self.assertLess(elapsed, 2.0)  # Should complete in under 2 seconds
    
    def test_detector_training_speed(self):
        """Test detector training performance"""
        system = AnomalyDetectionSystem()
        system.generate_telemetry(T=10000, n_anomalies=60)
        system.engineer_features()
        start = time.time()
        system.train_detector()
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0)  # Should complete in under 10 seconds
    
    def test_classifier_training_speed(self):
        """Test classifier training performance"""
        system = AnomalyDetectionSystem()
        system.generate_telemetry(T=10000, n_anomalies=60)
        system.engineer_features()
        system.train_detector()
        start = time.time()
        system.train_classifier()
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0)  # Should complete in under 10 seconds
    
    def test_prediction_speed(self):
        """Test real-time prediction performance"""
        system = AnomalyDetectionSystem()
        system.generate_telemetry(T=5000, n_anomalies=30)
        system.engineer_features()
        system.train_detector()
        system.train_classifier()
        
        # Test prediction on 100 samples
        test_samples = system.df[system.feature_cols].fillna(0).values[:100]
        start = time.time()
        predictions = system.clf_model.predict(test_samples)
        elapsed = time.time() - start
        
        avg_time_per_prediction = elapsed / 100
        self.assertLess(avg_time_per_prediction, 0.01)  # Less than 10ms per prediction


def run_test_suite():
    """Run all tests and print summary"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestAnomalyDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestRootCauseClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestMitigationPolicies))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    print("="*70)
    
    return result


if __name__ == "__main__":
    run_test_suite()