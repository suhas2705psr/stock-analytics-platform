"""
Ensemble model combining multiple ML approaches.
"""

import numpy as np
import pandas as pd
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb


# ------------------------------
# Logging Configuration
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------
# Ensemble Predictor Class
# ------------------------------
class EnsemblePredictor:
    def __init__(self, model_weights=None):
        self.models = {
            "logistic": LogisticRegression(random_state=42, max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "xgboost": xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            ),
            "gradient_boost": GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

        self.weights = model_weights or {
            "logistic": 0.2,
            "random_forest": 0.3,
            "xgboost": 0.3,
            "gradient_boost": 0.2,
        }

        self.feature_names = None
        self.is_trained = False

    # ------------------------------
    # Feature Preparation
    # ------------------------------
    def prepare_features(self, df: pd.DataFrame):
        feature_cols = [
            "Returns",
            "SMA_20",
            "SMA_50",
            "Volatility",
            "RSI",
            "MACD",
            "Signal",
            "Volume",
        ]

        feature_cols = [col for col in feature_cols if col in df.columns]

        X = df[feature_cols].copy()
        X = X.ffill().fillna(0)

        y = df["Target"] if "Target" in df.columns else None

        self.feature_names = feature_cols
        return X, y

    # ------------------------------
    # Training
    # ------------------------------
    def train(self, df: pd.DataFrame, test_size: float = 0.2):
        logger.info("Training ensemble models...")

        X, y = self.prepare_features(df)

        if y is None:
            raise ValueError("Target column 'Target' not found")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        metrics = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            metrics[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
            }

            logger.info(f"{name} - Accuracy: {metrics[name]['accuracy']:.4f}")

        self.is_trained = True

        # Ensemble performance
        ensemble_result = self.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_result["prediction"])

        logger.info(f"\n🎯 Ensemble Accuracy: {ensemble_accuracy:.4f}")

        metrics["ensemble"] = {
            "accuracy": ensemble_accuracy,
            "individual_models": {k: v for k, v in metrics.items() if k != "ensemble"},
        }

        return metrics

    # ------------------------------
    # Batch Prediction
    # ------------------------------
    def predict(self, X: pd.DataFrame):
        if not self.is_trained:
            raise ValueError("Models must be trained first")

        predictions = {}

        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        weighted_votes = np.zeros(len(X))

        for name, pred in predictions.items():
            weighted_votes += pred * self.weights[name]

        final_prediction = (weighted_votes >= 0.5).astype(int)
        confidence = np.abs(weighted_votes - 0.5) * 200

        return {
            "prediction": final_prediction,
            "confidence": confidence,
            "signal": ["SELL" if p == 0 else "BUY" for p in final_prediction],
            "individual_votes": predictions,
            "weighted_score": weighted_votes,
        }

    # ------------------------------
    # Single Prediction
    # ------------------------------
    def predict_single(self, features: dict):
        X = pd.DataFrame([features], columns=self.feature_names)

        result = self.predict(X)

        return {
            "signal": result["signal"][0],
            "confidence": float(result["confidence"][0]),
            "prediction": int(result["prediction"][0]),
            "recommendation": self._get_recommendation(
                result["signal"][0], result["confidence"][0]
            ),
        }

    # ------------------------------
    # Recommendation Logic
    # ------------------------------
    def _get_recommendation(self, signal: str, confidence: float):
        if confidence < 60:
            return "HOLD (Low confidence)"
        elif confidence < 75:
            return f"{signal} (Moderate confidence)"
        else:
            return f"STRONG {signal} (High confidence)"

    # ------------------------------
    # Feature Importance
    # ------------------------------
    def get_feature_importance(self):
        importance_data = []

        for name in ["random_forest", "xgboost", "gradient_boost"]:
            model = self.models[name]
            if hasattr(model, "feature_importances_"):
                importance_data.append(model.feature_importances_)

        if not importance_data:
            return pd.DataFrame()

        avg_importance = np.mean(importance_data, axis=0)

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": avg_importance,
            }
        ).sort_values("importance", ascending=False)

        return importance_df


# ------------------------------
# Standalone Testing
# ------------------------------
if __name__ == "__main__":
    from src.live.stream import LiveDataStreamer

    streamer = LiveDataStreamer("AAPL")
    df = streamer.get_historical_with_features(period="90d")

    if df is not None:
        ensemble = EnsemblePredictor()
        metrics = ensemble.train(df)

        print("\n📊 Model Performance:")
        for model, scores in metrics.items():
            if model != "ensemble":
                print(f"{model}: {scores['accuracy']:.4f}")

        latest_features = df.iloc[-1][ensemble.feature_names].to_dict()
        prediction = ensemble.predict_single(latest_features)

        print("\n🎯 Latest Prediction:")
        print(f"Signal: {prediction['signal']}")
        print(f"Confidence: {prediction['confidence']:.1f}%")
        print(f"Recommendation: {prediction['recommendation']}")

        importance = ensemble.get_feature_importance()
        print("\n📈 Top Features:")
        print(importance.head())
