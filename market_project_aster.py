import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- ደረጃ 1: ሲስተሙን በክፍል (Class) ማዋቀር ---
class StockPredictionSystem:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        # XGBoost ለከፍተኛ ትክክለኛነት የተመረጠ ሞዴል ነው
        self.model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6)

    def download_data(self):
        """መረጃን ከYahoo Finance ያወርዳል"""
        print(f"[{self.ticker}] መረጃ እየወረደ ነው...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        return df

    def engineer_features(self, df):
        """ደረጃ 2: ቴክኒካዊ አመልካቾችን በእጅ መስራት (No 'ta' library needed)"""
        # RSI (Relative Strength Index) - የገበያውን ጥንካሬ ይለካል
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD (Momentum) - የዋጋ ለውጥ ፍጥነትን ያሳያል
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2

        # Bollinger Bands - የዋጋ መዋዠቅን (Volatility) ያሳያል
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD20'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)

        return df.dropna()

    def preprocess_and_split(self, df):
        """ደረጃ 3: መረጃውን ለሞዴሉ ማዘጋጀት"""
        features = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'MA20', 'Upper_Band', 'Lower_Band']
        X = df[features].values
        y = df['Close'].values

        # Scaling (ከ0-1 ማስተካከል)
        X_scaled = self.scaler.fit_transform(X)

        # Time-Series Split (መረጃውን ሳይቀላቅሉ መከፋፈል)
        split = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]

        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """ደረጃ 4: ሞዴሉን ማሰልጠን እና ትንበያ መስራት"""
        print("XGBoost ሞዴል እየሰለጠነ ነው...")
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        
        # የውጤት መለኪያ
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"\n--- ውጤት ---")
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"Accuracy (R2 Score): {r2:.4f}")
        
        return preds

# --- ሲስተሙን ማስነሳት ---
if __name__ == "__main__":
    # አፕል (AAPL) አክሲዮንን ለምሳሌ እንውሰድ
    system = StockPredictionSystem("AAPL", "2018-01-01", "2024-01-01")
    
    data = system.download_data()
    featured_data = system.engineer_features(data)
    X_train, X_test, y_train, y_test = system.preprocess_and_split(featured_data)
    
    predictions = system.train_and_evaluate(X_train, X_test, y_train, y_test)

    # ውጤቱን በግራፍ ማሳየት
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label="Actual Price", color='blue', alpha=0.6)
    plt.plot(predictions, label="Predicted Price", color='red', linestyle='--')
    plt.title("Stock Market Prediction - Enterprise System")
    plt.legend()
    plt.show()