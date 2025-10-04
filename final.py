# import pandas as pd
# import numpy as np
# import requests
# import json
# import time
# from datetime import datetime, timedelta
# import re
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
# import joblib

# # Sentiment Analysis Libraries
# from textblob import TextBlob
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification
# from alpha_vantage.timeseries import TimeSeries
# from alpha_vantage.fundamentaldata import FundamentalData
# from alpha_vantage.techindicators import TechIndicators
# from alpha_vantage.foreignexchange import ForeignExchange
# from alpha_vantage.alphavantage import AlphaVantage
# from bs4 import BeautifulSoup

# # Removed PDF generation for API integration

# # --- Configuration and API Handlers ---
# API_KEYS = {
#     "ALPHA_VANTAGE_KEY": "T25ZB3N6FSVIA9KN",
#     "NEWS_API_KEY": "abe3a66ade2b4387963dcb1cdab51a2e",  # Free tier
#     "FINNHUB_KEY": "cjhqj9pr01qjqjqjqjqjqjqjqjqjqjqjqj",  # Free tier
# }

# class NewsAPIProvider:
#     """News API provider for reliable news fetching"""
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.base_url = "https://newsapi.org/v2/everything"
    
#     def get_news_headlines(self, symbol, limit=10):
#         """Fetch news headlines using News API"""
#         params = {
#             "q": f"{symbol} stock OR {symbol} shares OR {symbol} earnings OR {symbol} financials",
#             "apiKey": self.api_key,
#             "language": "en",
#             "pageSize": limit,
#             "sortBy": "relevancy",
#             "from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
#         }
#         try:
#             response = requests.get(self.base_url, params=params)
#             response.raise_for_status()
#             data = response.json()
            
#             if data.get('status') == 'ok' and data.get('articles'):
#                 headlines = [article.get('title', '') for article in data['articles'][:limit]]
#                 print(f"✅ Fetched {len(headlines)} news headlines for {symbol}")
#                 return headlines
#             else:
#                 print(f"⚠️ No news found for {symbol}")
#                 return []
#         except requests.exceptions.RequestException as e:
#             print(f"❌ News API request failed for {symbol}: {e}")
#             return []

# class YahooFinanceProvider:
#     """Yahoo Finance provider for stock data and fundamentals"""
#     def __init__(self):
#         self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
    
#     def get_stock_price_data(self, symbol, days=100):
#         """Fetch stock price data from Yahoo Finance"""
#         try:
#             # Convert symbol to Yahoo format if needed
#             yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
#             end_time = int(datetime.now().timestamp())
#             start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
#             url = f"{self.base_url}/{yahoo_symbol}"
#             params = {
#                 "period1": start_time,
#                 "period2": end_time,
#                 "interval": "1d",
#                 "includePrePost": "true",
#                 "events": "div%2Csplit"
#             }
            
#             response = requests.get(url, params=params)
#             response.raise_for_status()
#             data = response.json()
            
#             if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
#                 result = data['chart']['result'][0]
#                 timestamps = result['timestamp']
#                 quotes = result['indicators']['quote'][0]
                
#                 df = pd.DataFrame({
#                     'open': quotes['open'],
#                     'high': quotes['high'],
#                     'low': quotes['low'],
#                     'close': quotes['close'],
#                     'volume': quotes['volume']
#                 }, index=pd.to_datetime(timestamps, unit='s'))
                
#                 df = df.dropna()
#                 print(f"✅ Fetched {len(df)} days of price data for {symbol}")
#                 return df
#             else:
#                 print(f"⚠️ No price data available for {symbol}")
#                 return None
                
#         except Exception as e:
#             print(f"❌ Error fetching price data for {symbol}: {e}")
#             return None
    
#     def get_company_fundamentals(self, symbol):
#         """Fetch basic company fundamentals from Yahoo Finance"""
#         try:
#             yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
#             # Use yfinance library if available, otherwise use web scraping
#             try:
#                 import yfinance as yf
#                 ticker = yf.Ticker(yahoo_symbol)
#                 info = ticker.info
                
#                 fundamentals = {
#                     'Symbol': symbol,
#                     'Name': info.get('longName', symbol),
#                     'PERatio': str(info.get('trailingPE', 0)),
#                     'PriceToBookRatio': str(info.get('priceToBook', 0)),
#                     'ReturnOnEquity': str(info.get('returnOnEquity', 0)),
#                     'DebtToEquityRatio': str(info.get('debtToEquity', 0)),
#                     'CurrentRatio': str(info.get('currentRatio', 0)),
#                     'GrossProfitMargin': str(info.get('grossMargins', 0)),
#                     'OperatingMargin': str(info.get('operatingMargins', 0)),
#                     'ProfitMargin': str(info.get('profitMargins', 0)),
#                     'QuarterlyRevenueGrowthYOY': str(info.get('revenueGrowth', 0)),
#                     'QuarterlyEarningsGrowthYOY': str(info.get('earningsGrowth', 0)),
#                     'MarketCapitalization': str(info.get('marketCap', 0)),
#                     'DividendYield': str(info.get('dividendYield', 0))
#                 }
                
#                 print(f"✅ Fetched fundamental data for {symbol}")
#                 return fundamentals
                
#             except ImportError:
#                 print(f"⚠️ yfinance not available, using basic data for {symbol}")
#                 return self._get_basic_fundamentals(symbol)
                
#         except Exception as e:
#             print(f"❌ Error fetching fundamentals for {symbol}: {e}")
#             return None
    
#     def _get_basic_fundamentals(self, symbol):
#         """Get basic fundamental data using web scraping"""
#         try:
#             # This is a fallback method - in practice, you'd want to use a proper API
#             # For now, return None to indicate no data available
#             return None
#         except:
#             return None

# class AlphaVantageAPI:
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.base_url = "https://www.alphavantage.co/query"
#         self.news_provider = NewsAPIProvider(API_KEYS["NEWS_API_KEY"])
#         self.yahoo_provider = YahooFinanceProvider()

#     def get_news_sentiment(self, symbol, limit=10):
#         """Fetches market news sentiment using News API as primary source"""
#         print(f"🔄 Fetching news for {symbol}...")
        
#         # Try News API first
#         headlines = self.news_provider.get_news_headlines(symbol, limit)
#         if headlines:
#             return headlines
        
#         # Fallback to Alpha Vantage
#         params = {
#             "function": "NEWS_SENTIMENT",
#             "tickers": symbol,
#             "apikey": self.api_key
#         }
#         try:
#             response = requests.get(self.base_url, params=params)
#             response.raise_for_status()
#             data = response.json()
#             if 'feed' not in data:
#                 print(f"⚠️ Alpha Vantage API Error for {symbol}: {data.get('Note', 'Unknown error')}")
#                 return []
            
#             headlines = [article.get('title', '') for article in data.get('feed', [])[:limit]]
#             return headlines
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Alpha Vantage API request failed: {e}")
#             return []

#     def get_company_overview(self, symbol):
#         """Fetches fundamental company overview data using Yahoo Finance as primary source"""
#         print(f"🔄 Fetching fundamentals for {symbol}...")
        
#         # Try Yahoo Finance first
#         fundamentals = self.yahoo_provider.get_company_fundamentals(symbol)
#         if fundamentals:
#             return fundamentals
        
#         # Fallback to Alpha Vantage
#         params = {
#             "function": "OVERVIEW",
#             "symbol": symbol,
#             "apikey": self.api_key
#         }
#         try:
#             response = requests.get(self.base_url, params=params)
#             response.raise_for_status()
#             data = response.json()
#             if not data or 'Symbol' not in data:
#                 print(f"⚠️ No fundamental data found for {symbol}")
#                 return None
#             return data
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Alpha Vantage API request failed: {e}")
#             return None

# class FinBERTSentimentAnalyzer:
#     def __init__(self, model_path="./finbert-stock-sentiment"):
#         """Initialize FinBERT model for financial sentiment analysis"""
#         try:
#             self.tokenizer = BertTokenizer.from_pretrained(model_path)
#             self.model = BertForSequenceClassification.from_pretrained(model_path)
#             self.model.eval()
#             self.labels = ["Negative", "Neutral", "Positive"]
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.model.to(self.device)
#             print("✅ FinBERT model loaded successfully")
#         except Exception as e:
#             print(f"⚠️ FinBERT model not available, falling back to VADER: {e}")
#             self.model = None
#             self.tokenizer = None

#     def predict_sentiment(self, text):
#         """Predict sentiment using FinBERT model"""
#         if self.model is None:
#             return None
        
#         try:
#             inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
#             with torch.no_grad():
#                 outputs = self.model(**inputs)
#                 probs = torch.nn.functional.softmax(outputs.logits, dim=1)
#                 label_id = torch.argmax(probs, dim=1).item()
#                 confidence = probs[0][label_id].item()
                
#             return {
#                 'label': self.labels[label_id],
#                 'confidence': confidence,
#                 'probabilities': {
#                     'negative': probs[0][0].item(),
#                     'neutral': probs[0][1].item(),
#                     'positive': probs[0][2].item()
#                 }
#             }
#         except Exception as e:
#             print(f"⚠️ Error in FinBERT prediction: {e}")
#             return None

# class GoogleNewsScraper:
#     def __init__(self):
#         self.base_url = "https://news.google.com/rss/search"
#         self.news_provider = NewsAPIProvider(API_KEYS["NEWS_API_KEY"])

#     def get_headlines(self, query, limit=10):
#         """Scrapes headlines from Google News RSS feed with News API fallback."""
#         print(f"🔄 Fetching Google News for {query}...")
        
#         # Try News API first as it's more reliable
#         headlines = self.news_provider.get_news_headlines(query, limit)
#         if headlines:
#             return headlines
        
#         # Fallback to Google News RSS
#         params = {
#             "q": f"stock {query}",
#             "hl": "en-US",
#             "gl": "US",
#             "ceid": "US:en"
#         }
#         try:
#             response = requests.get(self.base_url, params=params, timeout=10)
#             response.raise_for_status()
#             soup = BeautifulSoup(response.content, 'xml')
#             headlines = [item.find('title').text for item in soup.find_all('item')[:limit]]
#             if headlines:
#                 print(f"✅ Fetched {len(headlines)} headlines from Google News for {query}")
#             return headlines
#             else:
#                 print(f"⚠️ No headlines found for {query}")
#             return []
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Error scraping Google News: {e}")
#             return []

# # --- Main Analyzer Class ---
# class TechnicalIndicators:
#     """Calculate technical indicators for stock analysis"""
    
#     @staticmethod
#     def sma(data, window):
#         """Simple Moving Average"""
#         return data.rolling(window=window).mean()
    
#     @staticmethod
#     def ema(data, window):
#         """Exponential Moving Average"""
#         return data.ewm(span=window).mean()
    
#     @staticmethod
#     def rsi(data, window=14):
#         """Relative Strength Index"""
#         delta = data.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#         rs = gain / loss
#         return 100 - (100 / (1 + rs))
    
#     @staticmethod
#     def macd(data, fast=12, slow=26, signal=9):
#         """MACD (Moving Average Convergence Divergence)"""
#         ema_fast = data.ewm(span=fast).mean()
#         ema_slow = data.ewm(span=slow).mean()
#         macd_line = ema_fast - ema_slow
#         signal_line = macd_line.ewm(span=signal).mean()
#         histogram = macd_line - signal_line
#         return macd_line, signal_line, histogram
    
#     @staticmethod
#     def bollinger_bands(data, window=20, num_std=2):
#         """Bollinger Bands"""
#         sma = data.rolling(window=window).mean()
#         std = data.rolling(window=window).std()
#         upper_band = sma + (std * num_std)
#         lower_band = sma - (std * num_std)
#         return upper_band, sma, lower_band
    
#     @staticmethod
#     def stochastic(high, low, close, k_window=14, d_window=3):
#         """Stochastic Oscillator"""
#         lowest_low = low.rolling(window=k_window).min()
#         highest_high = high.rolling(window=k_window).max()
#         k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
#         d_percent = k_percent.rolling(window=d_window).mean()
#         return k_percent, d_percent

# class StockPricePredictor:
#     """ML model for stock price prediction using technical indicators"""
    
#     def __init__(self):
#         self.price_predictor = None
#         self.trend_classifier = None
#         self.technical_scaler = StandardScaler()
#         self.trend_scaler = StandardScaler()
    
#     def prepare_technical_features(self, price_data):
#         """Prepare technical indicator features for ML model"""
#         if len(price_data) < 50:  # Need sufficient data for indicators
#             return None
        
#         # Calculate technical indicators
#         close = price_data['close']
#         high = price_data['high'] if 'high' in price_data.columns else close
#         low = price_data['low'] if 'low' in price_data.columns else close
        
#         features = pd.DataFrame(index=price_data.index)
        
#         # Moving averages
#         features['sma_5'] = TechnicalIndicators.sma(close, 5)
#         features['sma_10'] = TechnicalIndicators.sma(close, 10)
#         features['sma_20'] = TechnicalIndicators.sma(close, 20)
#         features['ema_12'] = TechnicalIndicators.ema(close, 12)
#         features['ema_26'] = TechnicalIndicators.ema(close, 26)
        
#         # RSI
#         features['rsi'] = TechnicalIndicators.rsi(close)
        
#         # MACD
#         macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
#         features['macd'] = macd_line
#         features['macd_signal'] = signal_line
#         features['macd_histogram'] = histogram
        
#         # Bollinger Bands
#         bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)
#         features['bb_upper'] = bb_upper
#         features['bb_middle'] = bb_middle
#         features['bb_lower'] = bb_lower
#         features['bb_width'] = (bb_upper - bb_lower) / bb_middle
#         features['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower)
        
#         # Stochastic
#         k_percent, d_percent = TechnicalIndicators.stochastic(high, low, close)
#         features['stoch_k'] = k_percent
#         features['stoch_d'] = d_percent
        
#         # Price-based features
#         features['price_change'] = close.pct_change()
#         features['price_volatility'] = close.pct_change().rolling(window=10).std()
#         features['volume_ratio'] = price_data['volume'].rolling(window=10).mean() / price_data['volume'].rolling(window=30).mean() if 'volume' in price_data.columns else 1
        
#         # Price position relative to moving averages
#         features['price_vs_sma5'] = (close / features['sma_5'] - 1) * 100
#         features['price_vs_sma20'] = (close / features['sma_20'] - 1) * 100
        
#         return features.dropna()
    
#     def train_price_predictor(self, price_data):
#         """Train ML model for price prediction"""
#         features = self.prepare_technical_features(price_data)
#         if features is None or len(features) < 30:
#             print("⚠️ Insufficient data for technical analysis training")
#             return None
        
#         # Prepare target variables
#         close_prices = price_data['close'].loc[features.index]
        
#         # Future price targets (1, 3, 5 days ahead)
#         targets = {}
#         for days in [1, 3, 5]:
#             targets[f'future_price_{days}d'] = close_prices.shift(-days)
#             targets[f'future_return_{days}d'] = (targets[f'future_price_{days}d'] / close_prices - 1) * 100
        
#         # Trend classification (1: uptrend, 0: downtrend, -1: sideways)
#         price_change_5d = (close_prices.shift(-5) / close_prices - 1) * 100
#         trend_labels = []
#         for change in price_change_5d:
#             if change > 2:
#                 trend_labels.append(1)  # Uptrend
#             elif change < -2:
#                 trend_labels.append(-1)  # Downtrend
#             else:
#                 trend_labels.append(0)  # Sideways
        
#         # Train price prediction model
#         feature_columns = [col for col in features.columns if not col.startswith('future_')]
#         X = features[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        
#         # Scale features
#         X_scaled = self.technical_scaler.fit_transform(X)
        
#         # Train models for different time horizons
#         self.price_predictor = {}
#         for days in [1, 3, 5]:
#             target_col = f'future_return_{days}d'
#             if target_col in targets:
#                 y = targets[target_col].fillna(0)
#                 valid_indices = ~y.isna()
                
#                 if valid_indices.sum() > 20:  # Need sufficient data
#                     model = GradientBoostingRegressor(n_estimators=100, random_state=42)
#                     model.fit(X_scaled[valid_indices], y[valid_indices])
#                     self.price_predictor[f'{days}d'] = model
        
#         # Train trend classifier
#         if len(trend_labels) > 20:
#             self.trend_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
#             self.trend_classifier.fit(X_scaled, trend_labels)
        
#         print(f"✅ Technical analysis models trained successfully")
#         return True
    
#     def predict_stock_movement(self, price_data):
#         """Predict stock price movement using technical indicators"""
#         if not self.price_predictor:
#             return None
        
#         features = self.prepare_technical_features(price_data)
#         if features is None or len(features) < 5:
#             return None
        
#         feature_columns = [col for col in features.columns if not col.startswith('future_')]
#         X = features[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
#         X_scaled = self.technical_scaler.transform(X)
        
#         predictions = {}
        
#         # Price movement predictions
#         for days, model in self.price_predictor.items():
#             try:
#                 pred_return = model.predict(X_scaled[-1:])[0]
#                 predictions[f'{days}d_return'] = pred_return
#                 predictions[f'{days}d_direction'] = 'Up' if pred_return > 0 else 'Down'
#             except:
#                 predictions[f'{days}d_return'] = 0
#                 predictions[f'{days}d_direction'] = 'Neutral'
        
#         # Trend prediction
#         if self.trend_classifier:
#             try:
#                 trend_pred = self.trend_classifier.predict(X_scaled[-1:])[0]
#                 trend_probs = self.trend_classifier.predict_proba(X_scaled[-1:])[0]
#                 trend_labels = ['Downtrend', 'Sideways', 'Uptrend']
#                 predictions['trend'] = trend_labels[trend_pred + 1]
#                 predictions['trend_confidence'] = max(trend_probs)
#             except:
#                 predictions['trend'] = 'Unknown'
#                 predictions['trend_confidence'] = 0
        
#         return predictions

# class EnhancedTradingAnalyzer:
#     def __init__(self):
#         self.trader_classifier = None
#         self.fundamental_classifier = None
#         self.stock_predictor = StockPricePredictor()
#         self.sentiment_analyzer = SentimentIntensityAnalyzer()
#         self.finbert_analyzer = FinBERTSentimentAnalyzer()
#         self.scaler = StandardScaler()
#         self.fundamental_scaler = StandardScaler()
#         self.av_api = AlphaVantageAPI(API_KEYS["ALPHA_VANTAGE_KEY"])
#         self.google_news = GoogleNewsScraper()

#     def prepare_trader_features(self, df):
#         """Enhanced feature engineering for trader classification"""
#         features = []
#         has_sector = 'sector' in df.columns
        
#         for user_id in df['user_id'].unique():
#             user_data = df[df['user_id'] == user_id]
            
#             if user_data.empty:
#                 continue

#             # Basic features
#             num_trades = len(user_data)
#             avg_profit = user_data['profit_pct'].mean()
#             std_profit = user_data['profit_pct'].std() if len(user_data) > 1 else 0
#             win_rate = (user_data['profit_pct'] > 0).mean()
            
#             # Advanced features
#             avg_holding_time = user_data['holding_time_hrs'].mean()
#             max_loss = user_data['profit_pct'].min()
#             max_gain = user_data['profit_pct'].max()
#             profit_consistency = (std_profit / abs(avg_profit)) if (avg_profit != 0) else 0
            
#             # Risk metrics
#             sharpe_ratio = avg_profit / std_profit if std_profit != 0 else 0
#             days_active = ((user_data['buy_time'].max() - user_data['buy_time'].min()).days + 1) if len(user_data) > 1 else 1
#             trade_frequency = num_trades / max(days_active, 1)
            
#             # Diversification
#             diversity_col = 'sector' if has_sector and not user_data['sector'].isna().all() else 'symbol'
#             diversity_score = len(user_data[diversity_col].unique()) / num_trades if num_trades > 0 else 0
            
#             # Trading pattern features
#             weekend_trades = sum(user_data['buy_time'].dt.dayofweek >= 5) / num_trades if num_trades > 0 else 0
#             morning_trades = sum(user_data['buy_time'].dt.hour < 12) / num_trades if num_trades > 0 else 0
            
#             # Trade size patterns
#             avg_trade_size = user_data['total_buy_value'].mean() if 'total_buy_value' in user_data.columns else 1000
#             trade_size_consistency = user_data['total_buy_value'].std() / avg_trade_size if 'total_buy_value' in user_data.columns and avg_trade_size > 0 else 0
            
#             features.append({
#                 'user_id': user_id,
#                 'num_trades': num_trades,
#                 'avg_profit': avg_profit,
#                 'std_profit': std_profit,
#                 'win_rate': win_rate,
#                 'avg_holding_time': avg_holding_time,
#                 'max_loss': max_loss,
#                 'max_gain': max_gain,
#                 'profit_consistency': profit_consistency,
#                 'sharpe_ratio': sharpe_ratio,
#                 'trade_frequency': trade_frequency,
#                 'diversity_score': diversity_score,
#                 'weekend_trades': weekend_trades,
#                 'morning_trades': morning_trades,
#                 'avg_trade_size': avg_trade_size,
#                 'trade_size_consistency': trade_size_consistency
#             })
            
#         return pd.DataFrame(features)

#     def train_trader_classifier(self, df):
#         """Train Random Forest model for trader classification"""
#         features_df = self.prepare_trader_features(df)
        
#         def enhanced_label_trader(row):
#             if (row['trade_frequency'] > 0.5 and row['std_profit'] > 5 and
#                 row['avg_profit'] > 1 and abs(row['sharpe_ratio']) > 0.3):
#                 return 'Aggressive'
#             elif (row['win_rate'] > 0.6 and row['profit_consistency'] < 3 and
#                   row['avg_profit'] > 0 and row['trade_frequency'] < 0.3):
#                 return 'Conservative'
#             else:
#                 return 'Balanced'
        
#         features_df['trader_type'] = features_df.apply(enhanced_label_trader, axis=1)
        
#         feature_columns = ['num_trades', 'avg_profit', 'std_profit', 'win_rate',
#                            'avg_holding_time', 'max_loss', 'max_gain', 'profit_consistency',
#                            'sharpe_ratio', 'trade_frequency', 'diversity_score',
#                            'weekend_trades', 'morning_trades', 'avg_trade_size', 'trade_size_consistency']
        
#         X = features_df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
#         y = features_df['trader_type']
        
#         if X.empty:
#             print("❌ No features to train on. Check input data.")
#             return features_df
        
#         self.scaler.fit(X)
#         X_scaled = self.scaler.transform(X)
        
#         self.trader_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#         self.trader_classifier.fit(X_scaled, y)
        
#         feature_importance = pd.DataFrame({
#             'feature': feature_columns,
#             'importance': self.trader_classifier.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print("\n📊 Feature Importance for Trader Classification:")
#         for idx, row in feature_importance.head(5).iterrows():
#             print(f"   {row['feature']}: {row['importance']:.3f}")
        
#         return features_df

#     def predict_trader_type(self, user_features):
#         """Predict trader type for new user"""
#         if self.trader_classifier is None:
#             return {"predicted_type": "Model not trained", "confidence": 0.0}
        
#         feature_columns = ['num_trades', 'avg_profit', 'std_profit', 'win_rate',
#                            'avg_holding_time', 'max_loss', 'max_gain', 'profit_consistency',
#                            'sharpe_ratio', 'trade_frequency', 'diversity_score',
#                            'weekend_trades', 'morning_trades', 'avg_trade_size', 'trade_size_consistency']
        
#         X = user_features[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
#         X_scaled = self.scaler.transform(X.values.reshape(1, -1))
        
#         prediction = self.trader_classifier.predict(X_scaled)[0]
#         probabilities = self.trader_classifier.predict_proba(X_scaled)[0]
        
#         return {
#             'predicted_type': prediction,
#             'confidence': max(probabilities),
#             'probabilities': dict(zip(self.trader_classifier.classes_, probabilities))
#         }

#     def analyze_sentiment_news(self, headlines):
#         """Analyze sentiment from news headlines using FinBERT, VADER, and TextBlob ensemble"""
#         if not headlines:
#             return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1, 'sentiment_label': 'Neutral', 'finbert_label': 'Neutral', 'finbert_confidence': 0}
        
#         sentiments = []
#         finbert_predictions = []
        
#         for headline in headlines:
#             try:
#                 # VADER and TextBlob analysis
#                 vader_score = self.sentiment_analyzer.polarity_scores(headline)
#                 textblob_sentiment = TextBlob(headline).sentiment
                
#                 combined_sentiment = {
#                     'compound': vader_score['compound'],
#                     'positive': vader_score['pos'],
#                     'negative': vader_score['neg'],
#                     'neutral': vader_score['neu'],
#                     'textblob_polarity': textblob_sentiment.polarity,
#                 }
#                 sentiments.append(combined_sentiment)
                
#                 # FinBERT analysis
#                 finbert_result = self.finbert_analyzer.predict_sentiment(headline)
#                 if finbert_result:
#                     finbert_predictions.append(finbert_result)
                
#             except Exception as e:
#                 print(f"⚠️ Error analyzing headline sentiment: {e}")
#                 continue
        
#         if sentiments:
#             avg_sentiment = {k: np.mean([s[k] for s in sentiments]) for k in sentiments[0]}
            
#             # VADER-based sentiment label
#             if avg_sentiment['compound'] >= 0.05:
#                 sentiment_label = 'Positive'
#             elif avg_sentiment['compound'] <= -0.05:
#                 sentiment_label = 'Negative'
#             else:
#                 sentiment_label = 'Neutral'
            
#             # FinBERT ensemble analysis
#             finbert_label = 'Neutral'
#             finbert_confidence = 0
#             if finbert_predictions:
#                 # Count FinBERT predictions
#                 finbert_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
#                 total_confidence = 0
                
#                 for pred in finbert_predictions:
#                     finbert_counts[pred['label']] += 1
#                     total_confidence += pred['confidence']
                
#                 # Determine majority FinBERT sentiment
#                 finbert_label = max(finbert_counts, key=finbert_counts.get)
#                 finbert_confidence = total_confidence / len(finbert_predictions)
            
#             # Ensemble decision (weighted combination)
#             ensemble_weights = {'FinBERT': 0.6, 'VADER': 0.3, 'TextBlob': 0.1}
            
#             # Convert labels to scores for ensemble
#             def label_to_score(label):
#                 if label == 'Positive': return 1
#                 elif label == 'Negative': return -1
#                 else: return 0
            
#             vader_score = label_to_score(sentiment_label)
#             finbert_score = label_to_score(finbert_label)
#             textblob_score = 1 if avg_sentiment['textblob_polarity'] > 0.1 else (-1 if avg_sentiment['textblob_polarity'] < -0.1 else 0)
            
#             ensemble_score = (ensemble_weights['FinBERT'] * finbert_score + 
#                             ensemble_weights['VADER'] * vader_score + 
#                             ensemble_weights['TextBlob'] * textblob_score)
            
#             if ensemble_score > 0.2:
#                 final_label = 'Positive'
#             elif ensemble_score < -0.2:
#                 final_label = 'Negative'
#             else:
#                 final_label = 'Neutral'
            
#             avg_sentiment.update({
#                 'sentiment_label': final_label,
#                 'finbert_label': finbert_label,
#                 'finbert_confidence': finbert_confidence,
#                 'ensemble_score': ensemble_score,
#                 'finbert_counts': finbert_counts if finbert_predictions else {}
#             })
#             return avg_sentiment
        
#         return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 1, 'sentiment_label': 'Neutral', 'finbert_label': 'Neutral', 'finbert_confidence': 0}

#     def get_stock_news_sentiment(self, symbol):
#         """Fetch news and analyze sentiment for a stock using Alpha Vantage"""
#         headlines = self.av_api.get_news_sentiment(symbol)
#         if not headlines:
#             print(f"Using Google News as fallback for {symbol}")
#             headlines = self.google_news.get_headlines(symbol)
        
#         if not headlines:
#             return None
        
#         return self.analyze_sentiment_news(headlines)

#     def get_stock_price_data(self, symbol, days=100):
#         """Fetch stock price data for technical analysis using Yahoo Finance"""
#         print(f"🔄 Fetching price data for {symbol}...")
        
#         # Try Yahoo Finance first
#         yahoo_provider = YahooFinanceProvider()
#         price_data = yahoo_provider.get_stock_price_data(symbol, days)
#         if price_data is not None:
#             return price_data
        
#         # Fallback to Alpha Vantage
#         try:
#             ts = TimeSeries(key=API_KEYS["ALPHA_VANTAGE_KEY"])
#             data, meta_data = ts.get_daily_adjusted(symbol, outputsize='compact')
            
#             if not data:
#                 print(f"⚠️ No price data available for {symbol}")
#                 return None
            
#             # Convert to DataFrame
#             df = pd.DataFrame(data).T
#             df.index = pd.to_datetime(df.index)
#             df = df.sort_index()
            
#             # Rename columns to standard format
#             df.columns = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume', 'dividend_amount', 'split_coefficient']
            
#             # Select recent data
#             recent_data = df.tail(days)
            
#             return recent_data[['open', 'high', 'low', 'close', 'volume']]
            
#         except Exception as e:
#             print(f"❌ Error fetching price data for {symbol}: {e}")
#             return None

#     def analyze_technical_indicators(self, symbol):
#         """Analyze technical indicators for a stock"""
#         price_data = self.get_stock_price_data(symbol)
#         if price_data is None:
#             return None
        
#         # Train the technical analysis model
#         self.stock_predictor.train_price_predictor(price_data)
        
#         # Get predictions
#         predictions = self.stock_predictor.predict_stock_movement(price_data)
#         if not predictions:
#             return None
        
#         # Calculate current technical indicators
#         features = self.stock_predictor.prepare_technical_features(price_data)
#         if features is None or len(features) == 0:
#             return None
        
#         latest_features = features.iloc[-1]
        
#         # Technical analysis summary
#         analysis = {
#             'symbol': symbol,
#             'current_price': price_data['close'].iloc[-1],
#             'predictions': predictions,
#             'technical_indicators': {
#                 'rsi': latest_features.get('rsi', 0),
#                 'macd': latest_features.get('macd', 0),
#                 'macd_signal': latest_features.get('macd_signal', 0),
#                 'bb_position': latest_features.get('bb_position', 0),
#                 'stoch_k': latest_features.get('stoch_k', 0),
#                 'stoch_d': latest_features.get('stoch_d', 0),
#                 'price_vs_sma20': latest_features.get('price_vs_sma20', 0)
#             },
#             'signals': {}
#         }
        
#         # Generate trading signals
#         rsi = latest_features.get('rsi', 50)
#         macd = latest_features.get('macd', 0)
#         macd_signal = latest_features.get('macd_signal', 0)
#         bb_position = latest_features.get('bb_position', 0.5)
#         stoch_k = latest_features.get('stoch_k', 50)
        
#         # RSI signals
#         if rsi > 70:
#             analysis['signals']['rsi'] = 'Overbought (Sell)'
#         elif rsi < 30:
#             analysis['signals']['rsi'] = 'Oversold (Buy)'
#         else:
#             analysis['signals']['rsi'] = 'Neutral'
        
#         # MACD signals
#         if macd > macd_signal:
#             analysis['signals']['macd'] = 'Bullish'
#         else:
#             analysis['signals']['macd'] = 'Bearish'
        
#         # Bollinger Bands signals
#         if bb_position > 0.8:
#             analysis['signals']['bollinger'] = 'Near Upper Band (Potential Sell)'
#         elif bb_position < 0.2:
#             analysis['signals']['bollinger'] = 'Near Lower Band (Potential Buy)'
#         else:
#             analysis['signals']['bollinger'] = 'Middle Range'
        
#         # Stochastic signals
#         if stoch_k > 80:
#             analysis['signals']['stochastic'] = 'Overbought'
#         elif stoch_k < 20:
#             analysis['signals']['stochastic'] = 'Oversold'
#         else:
#             analysis['signals']['stochastic'] = 'Neutral'
        
#         return analysis

#     def prepare_fundamentals_features(self, symbol):
#         """Prepare fundamental analysis features using Alpha Vantage"""
#         fundamentals_data = self.av_api.get_company_overview(symbol)
#         if not fundamentals_data:
#             return None
        
#         def safe_get(data, key, default=0):
#             value = data.get(key, '0')
#             try:
#                 return float(value) if value is not None and value != 'None' else default
#             except (ValueError, TypeError):
#                 return default

#         return {
#             'symbol': symbol,
#             'pe_ratio': safe_get(fundamentals_data, 'PERatio'),
#             'pb_ratio': safe_get(fundamentals_data, 'PriceToBookRatio'),
#             'roe': safe_get(fundamentals_data, 'ReturnOnEquity') * 100,
#             'debt_to_equity': safe_get(fundamentals_data, 'DebtToEquityRatio'),
#             'current_ratio': safe_get(fundamentals_data, 'CurrentRatio'),
#             'gross_margin': safe_get(fundamentals_data, 'GrossProfitMargin') * 100,
#             'operating_margin': safe_get(fundamentals_data, 'OperatingMargin') * 100,
#             'profit_margin': safe_get(fundamentals_data, 'ProfitMargin') * 100,
#             'revenue_growth': safe_get(fundamentals_data, 'QuarterlyRevenueGrowthYOY') * 100,
#             'earnings_growth': safe_get(fundamentals_data, 'QuarterlyEarningsGrowthYOY') * 100,
#             'market_cap': safe_get(fundamentals_data, 'MarketCapitalization'),
#             'dividend_yield': safe_get(fundamentals_data, 'DividendYield') * 100
#         }

#     def train_fundamental_classifier(self, sample_data=None):
#         """Train ML model for fundamental analysis classification"""
#         # Create synthetic training data based on financial ratios
#         if sample_data is None:
#             # Generate synthetic data for training
#             np.random.seed(42)
#             n_samples = 1000
            
#             # Generate realistic financial ratios
#             pe_ratios = np.random.normal(20, 10, n_samples)
#             pe_ratios = np.clip(pe_ratios, 5, 50)
            
#             roe_values = np.random.normal(15, 8, n_samples)
#             roe_values = np.clip(roe_values, -5, 35)
            
#             debt_equity = np.random.exponential(30, n_samples)
#             debt_equity = np.clip(debt_equity, 0, 100)
            
#             profit_margins = np.random.normal(12, 6, n_samples)
#             profit_margins = np.clip(profit_margins, -5, 25)
            
#             revenue_growth = np.random.normal(8, 12, n_samples)
#             revenue_growth = np.clip(revenue_growth, -20, 40)
            
#             current_ratios = np.random.normal(2.5, 1, n_samples)
#             current_ratios = np.clip(current_ratios, 0.5, 5)
            
#             # Create labels based on financial health rules
#             labels = []
#             for i in range(n_samples):
#                 score = 0
#                 if 10 <= pe_ratios[i] <= 25: score += 1
#                 if roe_values[i] >= 15: score += 1
#                 if 0 < debt_equity[i] < 50: score += 1
#                 if profit_margins[i] >= 10: score += 1
#                 if revenue_growth[i] >= 8: score += 1
#                 if current_ratios[i] >= 1.5: score += 1
                
#                 if score >= 5:
#                     labels.append('Strong')
#                 elif score >= 3:
#                     labels.append('Moderate')
#                 elif score >= 1:
#                     labels.append('Weak')
#                 else:
#                     labels.append('Poor')
            
#             X = np.column_stack([pe_ratios, roe_values, debt_equity, profit_margins, revenue_growth, current_ratios])
#             y = labels
#         else:
#             # Use provided sample data
#             feature_columns = ['pe_ratio', 'roe', 'debt_to_equity', 'profit_margin', 'revenue_growth', 'current_ratio']
#             X = sample_data[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
#             y = sample_data['fundamental_score'] if 'fundamental_score' in sample_data.columns else ['Moderate'] * len(X)
        
#         # Train Random Forest classifier
#         self.fundamental_classifier = RandomForestClassifier(
#             n_estimators=100, 
#             random_state=42, 
#             class_weight='balanced',
#             max_depth=10
#         )
        
#         # Scale features
#         X_scaled = self.fundamental_scaler.fit_transform(X)
#         self.fundamental_classifier.fit(X_scaled, y)
        
#         # Feature importance
#         feature_names = ['pe_ratio', 'roe', 'debt_to_equity', 'profit_margin', 'revenue_growth', 'current_ratio']
#         importance_df = pd.DataFrame({
#             'feature': feature_names,
#             'importance': self.fundamental_classifier.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print("\n📊 Fundamental Analysis Feature Importance:")
#         for idx, row in importance_df.iterrows():
#             print(f"   {row['feature']}: {row['importance']:.3f}")
        
#         return importance_df

#     def predict_fundamental_health(self, fundamentals_data):
#         """Predict fundamental health using ML model"""
#         if self.fundamental_classifier is None:
#             self.train_fundamental_classifier()
        
#         feature_columns = ['pe_ratio', 'roe', 'debt_to_equity', 'profit_margin', 'revenue_growth', 'current_ratio']
#         X = np.array([[
#             fundamentals_data.get('pe_ratio', 0),
#             fundamentals_data.get('roe', 0),
#             fundamentals_data.get('debt_to_equity', 0),
#             fundamentals_data.get('profit_margin', 0),
#             fundamentals_data.get('revenue_growth', 0),
#             fundamentals_data.get('current_ratio', 0)
#         ]])
        
#         X_scaled = self.fundamental_scaler.transform(X)
#         prediction = self.fundamental_classifier.predict(X_scaled)[0]
#         probabilities = self.fundamental_classifier.predict_proba(X_scaled)[0]
        
#         return {
#             'predicted_score': prediction,
#             'confidence': max(probabilities),
#             'probabilities': dict(zip(self.fundamental_classifier.classes_, probabilities))
#         }

#     def analyze_fundamentals(self, fundamentals_data):
#         """Analyze fundamental health of a company using ML model and rule-based analysis"""
#         if not fundamentals_data:
#             return {'score': 'Unknown', 'analysis': 'Insufficient data'}
        
#         # ML-based prediction
#         ml_prediction = self.predict_fundamental_health(fundamentals_data)
        
#         # Rule-based analysis for detailed insights
#         score = 0
#         max_score = 0
#         analysis_points = []
        
#         # PE Ratio Analysis
#         max_score += 1
#         pe = fundamentals_data.get('pe_ratio', 0)
#         if 10 <= pe <= 25:
#             score += 1
#             analysis_points.append(f"✅ Healthy PE ratio: {pe:.1f}")
#         elif pe > 0:
#             analysis_points.append(f"⚠️ {'Low' if pe < 10 else 'High'} PE ratio: {pe:.1f} (potential {'undervaluation' if pe < 10 else 'overvaluation'})")
#         else:
#             analysis_points.append("❌ PE ratio not available")
        
#         # ROE Analysis
#         max_score += 1
#         roe = fundamentals_data.get('roe', 0)
#         if roe >= 15:
#             score += 1
#             analysis_points.append(f"✅ Strong ROE: {roe:.1f}%")
#         elif roe > 0:
#             analysis_points.append(f"⚠️ Moderate ROE: {roe:.1f}%")
#         else:
#             analysis_points.append("❌ ROE data not available")
        
#         # Debt Analysis
#         max_score += 1
#         debt_equity = fundamentals_data.get('debt_to_equity', 0)
#         if 0 < debt_equity < 50:
#             score += 1
#             analysis_points.append(f"✅ Manageable debt levels: {debt_equity:.1f}%")
#         elif debt_equity > 0:
#             analysis_points.append(f"⚠️ High debt levels: {debt_equity:.1f}%")
#         else:
#             analysis_points.append("❌ Debt data not available")
        
#         # Profitability Analysis
#         max_score += 1
#         profit_margin = fundamentals_data.get('profit_margin', 0)
#         if profit_margin >= 10:
#             score += 1
#             analysis_points.append(f"✅ Strong profit margin: {profit_margin:.1f}%")
#         elif profit_margin > 0:
#             analysis_points.append(f"⚠️ Moderate profit margin: {profit_margin:.1f}%")
#         else:
#             analysis_points.append("❌ Profit margin data not available")
        
#         # Growth Analysis
#         max_score += 1
#         revenue_growth = fundamentals_data.get('revenue_growth', 0)
#         if revenue_growth >= 10:
#             score += 1
#             analysis_points.append(f"✅ Strong revenue growth: {revenue_growth:.1f}%")
#         elif revenue_growth > 0:
#             analysis_points.append(f"⚠️ Moderate revenue growth: {revenue_growth:.1f}%")
#         else:
#             analysis_points.append("❌ Revenue growth data not available")
        
#         score_percentage = (score / max_score) * 100 if max_score > 0 else 0
        
#         # Combine ML prediction with rule-based analysis
#         ml_score = ml_prediction['predicted_score']
#         ml_confidence = ml_prediction['confidence']
        
#         # Weighted final score
#         final_score = ml_score if ml_confidence > 0.7 else ('Moderate' if score_percentage >= 50 else 'Weak')
        
#         analysis_points.append(f"🤖 ML Prediction: {ml_score} (Confidence: {ml_confidence:.2f})")
#         analysis_points.append(f"📊 Rule-based Score: {score_percentage:.0f}%")
        
#         return {
#             'score': final_score,
#             'ml_score': ml_score,
#             'ml_confidence': ml_confidence,
#             'score_percentage': score_percentage,
#             'analysis_points': analysis_points,
#             'fundamentals_data': fundamentals_data,
#             'ml_probabilities': ml_prediction['probabilities']
#         }

#     def comprehensive_analysis(self, user_id, df):
#         """Comprehensive analysis combining all models"""
#         print(f"\n🔍 Running comprehensive analysis for User {user_id}...")
        
#         # Initialize ML models
#         print("🤖 Initializing ML models...")
#         self.train_fundamental_classifier()
        
#         # 1. Enhanced Trader Classification
#         print("\n📊 Training trader classification model...")
#         features_df = self.train_trader_classifier(df)
#         user_features = features_df[features_df['user_id'] == user_id]
        
#         trader_prediction = None
#         if not user_features.empty:
#             trader_prediction = self.predict_trader_type(user_features.iloc[0])
#             print(f"📊 Trader Type: {trader_prediction['predicted_type']} (Confidence: {trader_prediction['confidence']:.2f})")
#         else:
#             print(f"⚠️ No data found for user {user_id}")
        
#         user_trades = df[df['user_id'] == user_id]
#         if user_trades.empty:
#             return {
#                 'trader_analysis': trader_prediction,
#                 'sentiment_analysis': {},
#                 'fundamental_analysis': {}
#             }
        
#         symbol_counts = user_trades['symbol'].value_counts()
#         top_symbols = symbol_counts.head(5).index.tolist()
        
#         print(f"📈 Analyzing top {len(top_symbols)} stocks: {', '.join(top_symbols)}")
        
#         # 2. Enhanced Sentiment Analysis with FinBERT
#         print("\n📰 Enhanced Sentiment Analysis (FinBERT + VADER + TextBlob):")
#         sentiment_results = {}
#         for symbol in top_symbols[:3]:
#             print(f"   Analyzing sentiment for {symbol}...")
#             sentiment = self.get_stock_news_sentiment(symbol)
#             if sentiment:
#                 sentiment_results[symbol] = sentiment
#                 print(f"     {symbol}: {sentiment['sentiment_label']} (Ensemble Score: {sentiment.get('ensemble_score', 0):.2f})")
#                 print(f"       FinBERT: {sentiment.get('finbert_label', 'N/A')} ({sentiment.get('finbert_confidence', 0):.2f})")
#                 print(f"       VADER: {sentiment['compound']:.2f}")
#             else:
#                 print(f"     {symbol}: No recent news found")
#             time.sleep(1)
        
#         # 3. ML-Enhanced Fundamental Analysis
#         print("\n💼 ML-Enhanced Fundamental Analysis:")
#         fundamental_results = {}
#         for symbol in top_symbols[:3]:
#             print(f"   Analyzing fundamentals for {symbol}...")
#             fundamentals = self.prepare_fundamentals_features(symbol)
#             if fundamentals:
#                 analysis = self.analyze_fundamentals(fundamentals)
#                 fundamental_results[symbol] = analysis
#                 print(f"     ✅ {symbol}: {analysis['score']} (ML: {analysis['ml_score']}, Confidence: {analysis['ml_confidence']:.2f})")
#                 print(f"       Rule-based Score: {analysis['score_percentage']:.0f}%")
#                 for point in analysis['analysis_points'][:3]:
#                     print(f"       {point}")
#             else:
#                 print(f"     ❌ {symbol}: Fundamental data not available")
#             time.sleep(2)
        
#         # 4. Technical Analysis with ML Predictions
#         print("\n📈 Technical Analysis with ML Predictions:")
#         technical_results = {}
#         for symbol in top_symbols[:3]:
#             print(f"   Analyzing technical indicators for {symbol}...")
#             technical_analysis = self.analyze_technical_indicators(symbol)
#             if technical_analysis:
#                 technical_results[symbol] = technical_analysis
#                 predictions = technical_analysis['predictions']
#                 print(f"     ✅ {symbol}: Current Price: ${technical_analysis['current_price']:.2f}")
#                 print(f"       Trend: {predictions.get('trend', 'Unknown')} (Confidence: {predictions.get('trend_confidence', 0):.2f})")
#                 print(f"       1D Prediction: {predictions.get('1d_direction', 'Unknown')} ({predictions.get('1d_return', 0):.2f}%)")
#                 print(f"       3D Prediction: {predictions.get('3d_direction', 'Unknown')} ({predictions.get('3d_return', 0):.2f}%)")
#                 print(f"       RSI: {technical_analysis['technical_indicators']['rsi']:.1f} - {technical_analysis['signals']['rsi']}")
#                 print(f"       MACD: {technical_analysis['signals']['macd']}")
#             else:
#                 print(f"     ❌ {symbol}: Technical analysis data not available")
#             time.sleep(2)
        
#         return {
#             'trader_analysis': trader_prediction,
#             'sentiment_analysis': sentiment_results,
#             'fundamental_analysis': fundamental_results,
#             'technical_analysis': technical_results,
#             'user_summary': {
#                 'total_trades': len(user_trades),
#                 'avg_profit': user_trades['profit_pct'].mean(),
#                 'win_rate': (user_trades['profit_pct'] > 0).mean(),
#                 'most_traded': symbol_counts.head(3).index.tolist()
#             }
#         }

# # PDF generation removed for API integration

# def display_user_summary(results, user_id):
#     """Display comprehensive user analysis summary"""
#     print("\n" + "="*80)
#     print(f"🎯 COMPREHENSIVE TRADING ANALYSIS FOR USER: {user_id}")
#     print("="*80)
    
#     # User Trading Summary
#     if results['user_summary']:
#         summary = results['user_summary']
#         print(f"\n📊 USER TRADING SUMMARY")
#         print("-" * 40)
#         print(f"Total Trades: {summary['total_trades']}")
#         print(f"Average Profit: {summary['avg_profit']:.2f}%")
#         print(f"Win Rate: {summary['win_rate']:.2f}")
#         print(f"Most Traded Stocks: {', '.join(summary['most_traded'])}")
    
#     # Trader Analysis
#     if results['trader_analysis']:
#         ta = results['trader_analysis']
#         print(f"\n🧠 TRADER PROFILE ANALYSIS (ML-POWERED)")
#         print("-" * 40)
#         print(f"Predicted Trading Style: {ta['predicted_type']} (Confidence: {ta['confidence']:.2f})")
#         print("Probability Breakdown:")
#         for k, v in ta['probabilities'].items():
#             print(f"  • {k}: {v:.2f}")
    
#     # Sentiment Analysis
#     if results['sentiment_analysis']:
#         print(f"\n📰 MARKET SENTIMENT ANALYSIS (FinBERT + Ensemble)")
#         print("-" * 40)
#         for symbol, sentiment in results['sentiment_analysis'].items():
#             print(f"\n{symbol}:")
#             print(f"  Overall Sentiment: {sentiment['sentiment_label']} (Ensemble Score: {sentiment.get('ensemble_score', 0):.2f})")
#             print(f"  FinBERT: {sentiment.get('finbert_label', 'N/A')} (Confidence: {sentiment.get('finbert_confidence', 0):.2f})")
#             print(f"  VADER Score: {sentiment['compound']:.2f}")
#             print(f"  Sentiment Distribution: Positive: {sentiment['positive']:.2f}, Negative: {sentiment['negative']:.2f}, Neutral: {sentiment['neutral']:.2f}")
#             if 'finbert_counts' in sentiment and sentiment['finbert_counts']:
#                 print(f"  FinBERT Distribution: {sentiment['finbert_counts']}")
    
#     # Fundamental Analysis
#     if results['fundamental_analysis']:
#         print(f"\n💼 FUNDAMENTAL ANALYSIS (ML-ENHANCED)")
#         print("-" * 40)
#         for symbol, analysis in results['fundamental_analysis'].items():
#             print(f"\n{symbol}:")
#             print(f"  Company Health: {analysis['score']} (ML: {analysis['ml_score']}, Confidence: {analysis['ml_confidence']:.2f})")
#             print(f"  Rule-based Score: {analysis['score_percentage']:.0f}%")
#             print("  Key Analysis Points:")
#             for point in analysis['analysis_points'][:5]:  # Show top 5 points
#                 print(f"    • {point}")
#             if 'ml_probabilities' in analysis:
#                 print("  ML Model Probabilities:")
#                 for score, prob in analysis['ml_probabilities'].items():
#                     print(f"    • {score}: {prob:.2f}")
    
#     # Technical Analysis
#     if results['technical_analysis']:
#         print(f"\n📈 TECHNICAL ANALYSIS (ML PREDICTIONS)")
#         print("-" * 40)
#         for symbol, analysis in results['technical_analysis'].items():
#             print(f"\n{symbol} (Current Price: ${analysis['current_price']:.2f}):")
            
#             predictions = analysis['predictions']
#             print(f"  Trend Prediction: {predictions.get('trend', 'Unknown')} (Confidence: {predictions.get('trend_confidence', 0):.2f})")
#             print(f"  Price Predictions:")
#             print(f"    • 1-Day: {predictions.get('1d_direction', 'Unknown')} ({predictions.get('1d_return', 0):.2f}%)")
#             print(f"    • 3-Day: {predictions.get('3d_direction', 'Unknown')} ({predictions.get('3d_return', 0):.2f}%)")
#             print(f"    • 5-Day: {predictions.get('5d_direction', 'Unknown')} ({predictions.get('5d_return', 0):.2f}%)")
            
#             print("  Technical Indicators:")
#             indicators = analysis['technical_indicators']
#             print(f"    • RSI: {indicators['rsi']:.1f}")
#             print(f"    • MACD: {indicators['macd']:.4f}")
#             print(f"    • MACD Signal: {indicators['macd_signal']:.4f}")
#             print(f"    • Bollinger Position: {indicators['bb_position']:.2f}")
#             print(f"    • Stochastic K: {indicators['stoch_k']:.1f}")
#             print(f"    • Price vs SMA20: {indicators['price_vs_sma20']:.2f}%")
            
#             print("  Trading Signals:")
#             signals = analysis['signals']
#             for signal_name, signal_value in signals.items():
#                 print(f"    • {signal_name.upper()}: {signal_value}")

# def main():
#     print("🚀 ENHANCED TRADING ANALYSIS SYSTEM WITH ML MODELS")
#     print("="*60)
#     print("This system provides comprehensive trading analysis using:")
#     print("• FinBERT for advanced sentiment analysis")
#     print("• ML models for fundamental analysis")
#     print("• Technical indicators with price predictions")
#     print("• Ensemble sentiment analysis")
#     print("• Trader behavior classification")
#     print("="*60)
    
#     print("\n📊 Loading and preparing data...")
    
#     try:
#         df = pd.read_csv("trade_info.csv")
#         print(f"✅ Loaded {len(df)} trades")
#     except FileNotFoundError:
#         print("❌ Error: 'trade_info.csv' not found. Please place it in the same directory.")
#         return
    
#     # Data cleaning and preparation
#     df.columns = df.columns.str.strip()
#     df['buy_time'] = pd.to_datetime(df['buy_time'])
#     df['sell_time'] = pd.to_datetime(df['sell_time'])
#     df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
    
#     print(f"📈 Data columns: {list(df.columns)}")
#     print(f"📈 Data shape: {df.shape}")
    
#     # Initialize analyzer
#     print("\n🤖 Initializing ML models...")
#     analyzer = EnhancedTradingAnalyzer()
    
#     # Get user ID from input
#     available_users = df['user_id'].unique()
#     print(f"\n👥 Available users: {', '.join(available_users)}")
    
#     while True:
#         user_id = input("\n🔍 Please enter your user ID (or 'quit' to exit): ").strip()
        
#         if user_id.lower() == 'quit':
#             print("👋 Goodbye!")
#             break
    
#     if user_id not in available_users:
#             print(f"❌ User ID '{user_id}' not found in the data. Please try again.")
#             continue

#     # Run comprehensive analysis
#     try:
#             print(f"\n🔍 Running comprehensive analysis for User {user_id}...")
#         results = analyzer.comprehensive_analysis(user_id, df)
        
#             # Display comprehensive results
#             display_user_summary(results, user_id)
            
#             # Analysis complete - results ready for frontend
            
#             # Ask if user wants to analyze another user
#             another = input("\n🔄 Would you like to analyze another user? (y/n): ").strip().lower()
#             if another != 'y':
#                 print("👋 Thank you for using the Enhanced Trading Analysis System!")
#                 break
            
#     except Exception as e:
#         print(f"❌ An error occurred during the analysis: {e}")
#         import traceback
#         traceback.print_exc()
            
#             # Ask if user wants to try again
#             retry = input("\n🔄 Would you like to try again? (y/n): ").strip().lower()
#             if retry != 'y':
#                 break

# if __name__ == "__main__":
#     main()




# Fixed version addressing the main syntax and structural issues

# import pandas as pd
# import numpy as np
# import requests
# import json
# import time
# from datetime import datetime, timedelta
# import re
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, mean_squared_error, r2_score
# from sklearn.linear_model import LinearRegression
# import joblib

# # Sentiment Analysis Libraries
# try:
#     from textblob import TextBlob
#     from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#     import torch
#     from transformers import BertTokenizer, BertForSequenceClassification
#     ADVANCED_SENTIMENT = True
# except ImportError:
#     print("Warning: Advanced sentiment analysis libraries not available")
#     ADVANCED_SENTIMENT = False

# try:
#     from bs4 import BeautifulSoup
#     BS4_AVAILABLE = True
# except ImportError:
#     print("Warning: BeautifulSoup not available for web scraping")
#     BS4_AVAILABLE = False

# # Configuration and API Handlers
# API_KEYS = {
#     "ALPHA_VANTAGE_KEY": "T25ZB3N6FSVIA9KN",
#     "NEWS_API_KEY": "abe3a66ade2b4387963dcb1cdab51a2e",
#     "FINNHUB_KEY": "cjhqj9pr01qjqjqjqjqjqjqjqjqjqjqjqj",
# }

# class NewsAPIProvider:
#     """News API provider for reliable news fetching"""
#     def __init__(self, api_key):
#         self.api_key = api_key
#         self.base_url = "https://newsapi.org/v2/everything"
    
#     def get_news_headlines(self, symbol, limit=10):
#         """Fetch news headlines using News API"""
#         params = {
#             "q": f"{symbol} stock OR {symbol} shares OR {symbol} earnings OR {symbol} financials",
#             "apiKey": self.api_key,
#             "language": "en",
#             "pageSize": limit,
#             "sortBy": "relevancy",
#             "from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
#         }
#         try:
#             response = requests.get(self.base_url, params=params, timeout=10)
#             response.raise_for_status()
#             data = response.json()
            
#             if data.get('status') == 'ok' and data.get('articles'):
#                 headlines = [article.get('title', '') for article in data['articles'][:limit]]
#                 print(f"Successfully fetched {len(headlines)} news headlines for {symbol}")
#                 return headlines
#             else:
#                 print(f"No news found for {symbol}")
#                 return []
#         except requests.exceptions.RequestException as e:
#             print(f"News API request failed for {symbol}: {e}")
#             return []

# class YahooFinanceProvider:
#     """Yahoo Finance provider for stock data and fundamentals"""
#     def __init__(self):
#         self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
    
#     def get_stock_price_data(self, symbol, days=100):
#         """Fetch stock price data from Yahoo Finance"""
#         try:
#             # Convert symbol to Yahoo format if needed
#             yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
#             end_time = int(datetime.now().timestamp())
#             start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
#             url = f"{self.base_url}/{yahoo_symbol}"
#             params = {
#                 "period1": start_time,
#                 "period2": end_time,
#                 "interval": "1d",
#                 "includePrePost": "true",
#                 "events": "div%2Csplit"
#             }
            
#             response = requests.get(url, params=params, timeout=15)
#             response.raise_for_status()
#             data = response.json()
            
#             if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
#                 result = data['chart']['result'][0]
#                 timestamps = result['timestamp']
#                 quotes = result['indicators']['quote'][0]
                
#                 df = pd.DataFrame({
#                     'open': quotes['open'],
#                     'high': quotes['high'],
#                     'low': quotes['low'],
#                     'close': quotes['close'],
#                     'volume': quotes['volume']
#                 }, index=pd.to_datetime(timestamps, unit='s'))
                
#                 df = df.dropna()
#                 print(f"Fetched {len(df)} days of price data for {symbol}")
#                 return df
#             else:
#                 print(f"No price data available for {symbol}")
#                 return None
                
#         except Exception as e:
#             print(f"Error fetching price data for {symbol}: {e}")
#             return None
    
#     def get_company_fundamentals(self, symbol):
#         """Fetch basic company fundamentals from Yahoo Finance"""
#         try:
#             yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
#             # Try to import yfinance if available
#             try:
#                 import yfinance as yf
#                 ticker = yf.Ticker(yahoo_symbol)
#                 info = ticker.info
                
#                 fundamentals = {
#                     'Symbol': symbol,
#                     'Name': info.get('longName', symbol),
#                     'PERatio': str(info.get('trailingPE', 0)),
#                     'PriceToBookRatio': str(info.get('priceToBook', 0)),
#                     'ReturnOnEquity': str(info.get('returnOnEquity', 0)),
#                     'DebtToEquityRatio': str(info.get('debtToEquity', 0)),
#                     'CurrentRatio': str(info.get('currentRatio', 0)),
#                     'GrossProfitMargin': str(info.get('grossMargins', 0)),
#                     'OperatingMargin': str(info.get('operatingMargins', 0)),
#                     'ProfitMargin': str(info.get('profitMargins', 0)),
#                     'QuarterlyRevenueGrowthYOY': str(info.get('revenueGrowth', 0)),
#                     'QuarterlyEarningsGrowthYOY': str(info.get('earningsGrowth', 0)),
#                     'MarketCapitalization': str(info.get('marketCap', 0)),
#                     'DividendYield': str(info.get('dividendYield', 0))
#                 }
                
#                 print(f"Fetched fundamental data for {symbol}")
#                 return fundamentals
                
#             except ImportError:
#                 print(f"yfinance not available, using basic data for {symbol}")
#                 return self._get_basic_fundamentals(symbol)
                
#         except Exception as e:
#             print(f"Error fetching fundamentals for {symbol}: {e}")
#             return None
    
#     def _get_basic_fundamentals(self, symbol):
#         """Get basic fundamental data using web scraping"""
#         # This is a fallback method - in practice, you'd want to use a proper API
#         return None

# class TechnicalIndicators:
#     """Calculate technical indicators for stock analysis"""
    
#     @staticmethod
#     def sma(data, window):
#         """Simple Moving Average"""
#         return data.rolling(window=window).mean()
    
#     @staticmethod
#     def ema(data, window):
#         """Exponential Moving Average"""
#         return data.ewm(span=window).mean()
    
#     @staticmethod
#     def rsi(data, window=14):
#         """Relative Strength Index"""
#         delta = data.diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
#         rs = gain / loss
#         return 100 - (100 / (1 + rs))
    
#     @staticmethod
#     def macd(data, fast=12, slow=26, signal=9):
#         """MACD (Moving Average Convergence Divergence)"""
#         ema_fast = data.ewm(span=fast).mean()
#         ema_slow = data.ewm(span=slow).mean()
#         macd_line = ema_fast - ema_slow
#         signal_line = macd_line.ewm(span=signal).mean()
#         histogram = macd_line - signal_line
#         return macd_line, signal_line, histogram
    
#     @staticmethod
#     def bollinger_bands(data, window=20, num_std=2):
#         """Bollinger Bands"""
#         sma = data.rolling(window=window).mean()
#         std = data.rolling(window=window).std()
#         upper_band = sma + (std * num_std)
#         lower_band = sma - (std * num_std)
#         return upper_band, sma, lower_band
    
#     @staticmethod
#     def stochastic(high, low, close, k_window=14, d_window=3):
#         """Stochastic Oscillator"""
#         lowest_low = low.rolling(window=k_window).min()
#         highest_high = high.rolling(window=k_window).max()
#         k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
#         d_percent = k_percent.rolling(window=d_window).mean()
#         return k_percent, d_percent

# class EnhancedTradingAnalyzer:
#     """Main trading analyzer class"""
    
#     def __init__(self):
#         self.trader_classifier = None
#         self.fundamental_classifier = None
#         self.scaler = StandardScaler()
#         self.fundamental_scaler = StandardScaler()
        
#         # Initialize sentiment analyzer if available
#         if ADVANCED_SENTIMENT:
#             self.sentiment_analyzer = SentimentIntensityAnalyzer()
#         else:
#             self.sentiment_analyzer = None
        
#         # Initialize API providers
#         self.news_provider = NewsAPIProvider(API_KEYS["NEWS_API_KEY"])
#         self.yahoo_provider = YahooFinanceProvider()

#     def create_sample_data(self):
#         """Create sample trading data for testing"""
#         np.random.seed(42)
        
#         users = [f'user_{i}' for i in range(1, 11)]
#         symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA']
        
#         data = []
#         for _ in range(500):
#             user = np.random.choice(users)
#             symbol = np.random.choice(symbols)
            
#             buy_time = datetime.now() - timedelta(days=np.random.randint(1, 365))
#             holding_hours = np.random.randint(1, 168)  # 1 hour to 1 week
#             sell_time = buy_time + timedelta(hours=holding_hours)
            
#             profit_pct = np.random.normal(2, 15)  # Mean 2%, std 15%
#             total_value = np.random.uniform(1000, 50000)
            
#             data.append({
#                 'user_id': user,
#                 'symbol': symbol,
#                 'buy_time': buy_time,
#                 'sell_time': sell_time,
#                 'holding_time_hrs': holding_hours,
#                 'profit_pct': profit_pct,
#                 'total_buy_value': total_value
#             })
        
#         return pd.DataFrame(data)

#     def prepare_trader_features(self, df):
#         """Enhanced feature engineering for trader classification"""
#         features = []
        
#         for user_id in df['user_id'].unique():
#             user_data = df[df['user_id'] == user_id]
            
#             if user_data.empty:
#                 continue

#             # Basic features
#             num_trades = len(user_data)
#             avg_profit = user_data['profit_pct'].mean()
#             std_profit = user_data['profit_pct'].std() if len(user_data) > 1 else 0
#             win_rate = (user_data['profit_pct'] > 0).mean()
            
#             # Advanced features
#             avg_holding_time = user_data['holding_time_hrs'].mean()
#             max_loss = user_data['profit_pct'].min()
#             max_gain = user_data['profit_pct'].max()
#             profit_consistency = (std_profit / abs(avg_profit)) if (avg_profit != 0) else 0
            
#             # Risk metrics
#             sharpe_ratio = avg_profit / std_profit if std_profit != 0 else 0
#             days_active = ((user_data['buy_time'].max() - user_data['buy_time'].min()).days + 1) if len(user_data) > 1 else 1
#             trade_frequency = num_trades / max(days_active, 1)
            
#             # Diversification
#             diversity_score = len(user_data['symbol'].unique()) / num_trades if num_trades > 0 else 0
            
#             # Trading pattern features
#             weekend_trades = sum(user_data['buy_time'].dt.dayofweek >= 5) / num_trades if num_trades > 0 else 0
#             morning_trades = sum(user_data['buy_time'].dt.hour < 12) / num_trades if num_trades > 0 else 0
            
#             # Trade size patterns
#             avg_trade_size = user_data['total_buy_value'].mean() if 'total_buy_value' in user_data.columns else 1000
#             trade_size_consistency = user_data['total_buy_value'].std() / avg_trade_size if 'total_buy_value' in user_data.columns and avg_trade_size > 0 else 0
            
#             features.append({
#                 'user_id': user_id,
#                 'num_trades': num_trades,
#                 'avg_profit': avg_profit,
#                 'std_profit': std_profit,
#                 'win_rate': win_rate,
#                 'avg_holding_time': avg_holding_time,
#                 'max_loss': max_loss,
#                 'max_gain': max_gain,
#                 'profit_consistency': profit_consistency,
#                 'sharpe_ratio': sharpe_ratio,
#                 'trade_frequency': trade_frequency,
#                 'diversity_score': diversity_score,
#                 'weekend_trades': weekend_trades,
#                 'morning_trades': morning_trades,
#                 'avg_trade_size': avg_trade_size,
#                 'trade_size_consistency': trade_size_consistency
#             })
            
#         return pd.DataFrame(features)

#     def train_trader_classifier(self, df):
#         """Train Random Forest model for trader classification"""
#         features_df = self.prepare_trader_features(df)
        
#         def enhanced_label_trader(row):
#             if (row['trade_frequency'] > 0.5 and row['std_profit'] > 5 and
#                 row['avg_profit'] > 1 and abs(row['sharpe_ratio']) > 0.3):
#                 return 'Aggressive'
#             elif (row['win_rate'] > 0.6 and row['profit_consistency'] < 3 and
#                   row['avg_profit'] > 0 and row['trade_frequency'] < 0.3):
#                 return 'Conservative'
#             else:
#                 return 'Balanced'
        
#         features_df['trader_type'] = features_df.apply(enhanced_label_trader, axis=1)
        
#         feature_columns = ['num_trades', 'avg_profit', 'std_profit', 'win_rate',
#                            'avg_holding_time', 'max_loss', 'max_gain', 'profit_consistency',
#                            'sharpe_ratio', 'trade_frequency', 'diversity_score',
#                            'weekend_trades', 'morning_trades', 'avg_trade_size', 'trade_size_consistency']
        
#         X = features_df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
#         y = features_df['trader_type']
        
#         if X.empty:
#             print("No features to train on. Check input data.")
#             return features_df
        
#         self.scaler.fit(X)
#         X_scaled = self.scaler.transform(X)
        
#         self.trader_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#         self.trader_classifier.fit(X_scaled, y)
        
#         feature_importance = pd.DataFrame({
#             'feature': feature_columns,
#             'importance': self.trader_classifier.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         print("\nFeature Importance for Trader Classification:")
#         for idx, row in feature_importance.head(5).iterrows():
#             print(f"   {row['feature']}: {row['importance']:.3f}")
        
#         return features_df

#     def analyze_sentiment_basic(self, headlines):
#         """Basic sentiment analysis when advanced libraries aren't available"""
#         if not headlines:
#             return {'sentiment_label': 'Neutral', 'confidence': 0}
        
#         positive_words = ['good', 'great', 'excellent', 'strong', 'positive', 'up', 'gain', 'profit', 'bull']
#         negative_words = ['bad', 'poor', 'weak', 'negative', 'down', 'loss', 'decline', 'bear', 'drop']
        
#         total_score = 0
#         for headline in headlines:
#             headline_lower = headline.lower()
#             pos_count = sum(1 for word in positive_words if word in headline_lower)
#             neg_count = sum(1 for word in negative_words if word in headline_lower)
#             total_score += (pos_count - neg_count)
        
#         avg_score = total_score / len(headlines)
        
#         if avg_score > 0.1:
#             return {'sentiment_label': 'Positive', 'confidence': min(avg_score, 1.0)}
#         elif avg_score < -0.1:
#             return {'sentiment_label': 'Negative', 'confidence': min(abs(avg_score), 1.0)}
#         else:
#             return {'sentiment_label': 'Neutral', 'confidence': 0.5}

#     def get_stock_news_sentiment(self, symbol):
#         """Fetch news and analyze sentiment for a stock"""
#         headlines = self.news_provider.get_news_headlines(symbol)
        
#         if not headlines:
#             return None
        
#         if ADVANCED_SENTIMENT and self.sentiment_analyzer:
#             # Use VADER sentiment analysis
#             sentiments = []
#             for headline in headlines:
#                 try:
#                     score = self.sentiment_analyzer.polarity_scores(headline)
#                     sentiments.append(score)
#                 except:
#                     continue
            
#             if sentiments:
#                 avg_sentiment = {k: np.mean([s[k] for s in sentiments]) for k in sentiments[0]}
#                 if avg_sentiment['compound'] >= 0.05:
#                     sentiment_label = 'Positive'
#                 elif avg_sentiment['compound'] <= -0.05:
#                     sentiment_label = 'Negative'
#                 else:
#                     sentiment_label = 'Neutral'
                
#                 avg_sentiment['sentiment_label'] = sentiment_label
#                 return avg_sentiment
        
#         # Fallback to basic sentiment analysis
#         return self.analyze_sentiment_basic(headlines)

# def main():
#     print("Enhanced Trading Analysis System")
#     print("=" * 50)
    
#     # Initialize analyzer
#     analyzer = EnhancedTradingAnalyzer()
    
#     # Try to load data, create sample if not found
#     try:
#         df = pd.read_csv("trade_info.csv")
#         print(f"Loaded {len(df)} trades from CSV")
#     except FileNotFoundError:
#         print("CSV file not found. Creating sample data for demonstration...")
#         df = analyzer.create_sample_data()
#         print(f"Created {len(df)} sample trades")
    
#     # Data preparation
#     if 'buy_time' in df.columns:
#         df['buy_time'] = pd.to_datetime(df['buy_time'])
#     if 'sell_time' in df.columns:
#         df['sell_time'] = pd.to_datetime(df['sell_time'])
#     if 'holding_time_hrs' not in df.columns and 'buy_time' in df.columns and 'sell_time' in df.columns:
#         df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
    
#     print(f"Data columns: {list(df.columns)}")
#     print(f"Data shape: {df.shape}")
    
#     # Get user ID
#     available_users = df['user_id'].unique()
#     print(f"\nAvailable users: {', '.join(available_users[:10])}...")  # Show first 10
    
#     user_id = input("\nEnter user ID to analyze: ").strip()
    
#     if user_id not in available_users:
#         print(f"User ID '{user_id}' not found. Using first available user: {available_users[0]}")
#         user_id = available_users[0]
    
#     # Run analysis
#     try:
#         print(f"\nAnalyzing user {user_id}...")
        
#         # Train trader classifier
#         features_df = analyzer.train_trader_classifier(df)
        
#         # Get user-specific analysis
#         user_features = features_df[features_df['user_id'] == user_id]
#         user_trades = df[df['user_id'] == user_id]
        
#         if not user_features.empty and not user_trades.empty:
#             print(f"\nUser {user_id} Analysis:")
#             print("-" * 30)
#             print(f"Total trades: {len(user_trades)}")
#             print(f"Average profit: {user_trades['profit_pct'].mean():.2f}%")
#             print(f"Win rate: {(user_trades['profit_pct'] > 0).mean():.2f}")
            
#             # Get most traded symbols
#             top_symbols = user_trades['symbol'].value_counts().head(3)
#             print(f"Top symbols: {', '.join(top_symbols.index)}")
            
#             # Sentiment analysis for top symbol
#             if len(top_symbols) > 0:
#                 top_symbol = top_symbols.index[0]
#                 print(f"\nAnalyzing sentiment for {top_symbol}...")
#                 sentiment = analyzer.get_stock_news_sentiment(top_symbol)
#                 if sentiment:
#                     print(f"Sentiment: {sentiment.get('sentiment_label', 'Unknown')}")
        
#         else:
#             print(f"No data found for user {user_id}")
            
#     except Exception as e:
#         print(f"An error occurred during analysis: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()



import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import re
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

# Sentiment Analysis Libraries
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    ADVANCED_SENTIMENT = True
except ImportError:
    print("Warning: Advanced sentiment analysis libraries not available")
    ADVANCED_SENTIMENT = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    print("Warning: BeautifulSoup not available for web scraping")
    BS4_AVAILABLE = False

# Configuration and API Handlers
API_KEYS = {
    "ALPHA_VANTAGE_KEY": "T25ZB3N6FSVIA9KN",
    "NEWS_API_KEY": "abe3a66ade2b4387963dcb1cdab51a2e",
    "FINNHUB_KEY": "cjhqj9pr01qjqjqjqjqjqjqjqjqjqjqjqj",
}

class NewsAPIProvider:
    """News API provider for reliable news fetching"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
    
    def get_news_headlines(self, symbol, limit=10):
        """Fetch news headlines using News API"""
        params = {
            "q": f"{symbol} stock OR {symbol} shares OR {symbol} earnings OR {symbol} financials",
            "apiKey": self.api_key,
            "language": "en",
            "pageSize": limit,
            "sortBy": "relevancy",
            "from": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        }
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'ok' and data.get('articles'):
                headlines = [article.get('title', '') for article in data['articles'][:limit]]
                print(f"Successfully fetched {len(headlines)} news headlines for {symbol}")
                return headlines
            else:
                print(f"No news found for {symbol}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"News API request failed for {symbol}: {e}")
            return []

class YahooFinanceProvider:
    """Yahoo Finance provider for stock data and fundamentals"""
    def __init__(self):
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    def get_stock_price_data(self, symbol, days=100):
        """Fetch stock price data from Yahoo Finance"""
        try:
            yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            url = f"{self.base_url}/{yahoo_symbol}"
            params = {
                "period1": start_time,
                "period2": end_time,
                "interval": "1d",
                "includePrePost": "true",
                "events": "div%2Csplit"
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                df = pd.DataFrame({
                    'open': quotes['open'],
                    'high': quotes['high'],
                    'low': quotes['low'],
                    'close': quotes['close'],
                    'volume': quotes['volume']
                }, index=pd.to_datetime(timestamps, unit='s'))
                
                df = df.dropna()
                print(f"Fetched {len(df)} days of price data for {symbol}")
                return df
            else:
                print(f"No price data available for {symbol}")
                return None
                
        except Exception as e:
            print(f"Error fetching price data for {symbol}: {e}")
            return None
    
    def get_company_fundamentals(self, symbol):
        """Fetch basic company fundamentals from Yahoo Finance"""
        try:
            yahoo_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
            try:
                import yfinance as yf
                ticker = yf.Ticker(yahoo_symbol)
                info = ticker.info
                
                fundamentals = {
                    'Symbol': symbol,
                    'Name': info.get('longName', symbol),
                    'PERatio': str(info.get('trailingPE', 0)),
                    'PriceToBookRatio': str(info.get('priceToBook', 0)),
                    'ReturnOnEquity': str(info.get('returnOnEquity', 0)),
                    'DebtToEquityRatio': str(info.get('debtToEquity', 0)),
                    'CurrentRatio': str(info.get('currentRatio', 0)),
                    'GrossProfitMargin': str(info.get('grossMargins', 0)),
                    'OperatingMargin': str(info.get('operatingMargins', 0)),
                    'ProfitMargin': str(info.get('profitMargins', 0)),
                    'QuarterlyRevenueGrowthYOY': str(info.get('revenueGrowth', 0)),
                    'QuarterlyEarningsGrowthYOY': str(info.get('earningsGrowth', 0)),
                    'MarketCapitalization': str(info.get('marketCap', 0)),
                    'DividendYield': str(info.get('dividendYield', 0))
                }
                
                print(f"Fetched fundamental data for {symbol}")
                return fundamentals
                
            except ImportError:
                print(f"yfinance not available, using basic data for {symbol}")
                return None
                
        except Exception as e:
            print(f"Error fetching fundamentals for {symbol}: {e}")
            return None

class TechnicalIndicators:
    """Calculate technical indicators for stock analysis"""
    
    @staticmethod
    def sma(data, window):
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data, window):
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data, window=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, window=20, num_std=2):
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    @staticmethod
    def stochastic(high, low, close, k_window=14, d_window=3):
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

class EnhancedTradingAnalyzer:
    """Main trading analyzer class"""
    
    def __init__(self):
        self.trader_classifier = None
        self.fundamental_classifier = None
        self.scaler = StandardScaler()
        self.fundamental_scaler = StandardScaler()
        
        # Initialize sentiment analyzer if available
        if ADVANCED_SENTIMENT:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            self.sentiment_analyzer = None
        
        # Initialize API providers
        self.news_provider = NewsAPIProvider(API_KEYS["NEWS_API_KEY"])
        self.yahoo_provider = YahooFinanceProvider()

    def prepare_trader_features(self, df):
        """Enhanced feature engineering for trader classification"""
        features = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            
            if user_data.empty:
                continue

            # Basic features
            num_trades = len(user_data)
            avg_profit = user_data['profit_pct'].mean()
            std_profit = user_data['profit_pct'].std() if len(user_data) > 1 else 0
            win_rate = (user_data['profit_pct'] > 0).mean()
            
            # Advanced features
            avg_holding_time = user_data['holding_time_hrs'].mean()
            max_loss = user_data['profit_pct'].min()
            max_gain = user_data['profit_pct'].max()
            profit_consistency = (std_profit / abs(avg_profit)) if (avg_profit != 0) else 0
            
            # Risk metrics
            sharpe_ratio = avg_profit / std_profit if std_profit != 0 else 0
            days_active = ((user_data['buy_time'].max() - user_data['buy_time'].min()).days + 1) if len(user_data) > 1 else 1
            trade_frequency = num_trades / max(days_active, 1)
            
            # Diversification
            diversity_score = len(user_data['symbol'].unique()) / num_trades if num_trades > 0 else 0
            
            # Trading pattern features
            weekend_trades = sum(user_data['buy_time'].dt.dayofweek >= 5) / num_trades if num_trades > 0 else 0
            morning_trades = sum(user_data['buy_time'].dt.hour < 12) / num_trades if num_trades > 0 else 0
            
            # Trade size patterns
            avg_trade_size = user_data['total_buy_value'].mean() if 'total_buy_value' in user_data.columns else 1000
            trade_size_consistency = user_data['total_buy_value'].std() / avg_trade_size if 'total_buy_value' in user_data.columns and avg_trade_size > 0 else 0
            
            features.append({
                'user_id': user_id,
                'num_trades': num_trades,
                'avg_profit': avg_profit,
                'std_profit': std_profit,
                'win_rate': win_rate,
                'avg_holding_time': avg_holding_time,
                'max_loss': max_loss,
                'max_gain': max_gain,
                'profit_consistency': profit_consistency,
                'sharpe_ratio': sharpe_ratio,
                'trade_frequency': trade_frequency,
                'diversity_score': diversity_score,
                'weekend_trades': weekend_trades,
                'morning_trades': morning_trades,
                'avg_trade_size': avg_trade_size,
                'trade_size_consistency': trade_size_consistency
            })
            
        return pd.DataFrame(features)

    def train_trader_classifier(self, df):
        """Train Random Forest model for trader classification"""
        features_df = self.prepare_trader_features(df)
        
        def enhanced_label_trader(row):
            if (row['trade_frequency'] > 0.5 and row['std_profit'] > 5 and
                row['avg_profit'] > 1 and abs(row['sharpe_ratio']) > 0.3):
                return 'Aggressive'
            elif (row['win_rate'] > 0.6 and row['profit_consistency'] < 3 and
                  row['avg_profit'] > 0 and row['trade_frequency'] < 0.3):
                return 'Conservative'
            else:
                return 'Balanced'
        
        features_df['trader_type'] = features_df.apply(enhanced_label_trader, axis=1)
        
        feature_columns = ['num_trades', 'avg_profit', 'std_profit', 'win_rate',
                           'avg_holding_time', 'max_loss', 'max_gain', 'profit_consistency',
                           'sharpe_ratio', 'trade_frequency', 'diversity_score',
                           'weekend_trades', 'morning_trades', 'avg_trade_size', 'trade_size_consistency']
        
        X = features_df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        y = features_df['trader_type']
        
        if X.empty:
            print("No features to train on. Check input data.")
            return features_df
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.trader_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.trader_classifier.fit(X_scaled, y)
        
        return features_df

    def predict_trader_type(self, user_features):
        """Predict trader type for new user"""
        if self.trader_classifier is None:
            return {"predicted_type": "Model not trained", "confidence": 0.0}
        
        feature_columns = ['num_trades', 'avg_profit', 'std_profit', 'win_rate',
                           'avg_holding_time', 'max_loss', 'max_gain', 'profit_consistency',
                           'sharpe_ratio', 'trade_frequency', 'diversity_score',
                           'weekend_trades', 'morning_trades', 'avg_trade_size', 'trade_size_consistency']
        
        X = user_features[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X.values.reshape(1, -1))
        
        prediction = self.trader_classifier.predict(X_scaled)[0]
        probabilities = self.trader_classifier.predict_proba(X_scaled)[0]
        
        return {
            'predicted_type': prediction,
            'confidence': max(probabilities),
            'probabilities': dict(zip(self.trader_classifier.classes_, probabilities))
        }

    def analyze_sentiment_basic(self, headlines):
        """Basic sentiment analysis when advanced libraries aren't available"""
        if not headlines:
            return {'sentiment_label': 'Neutral', 'confidence': 0, 'compound': 0}
        
        positive_words = ['good', 'great', 'excellent', 'strong', 'positive', 'up', 'gain', 'profit', 'bull']
        negative_words = ['bad', 'poor', 'weak', 'negative', 'down', 'loss', 'decline', 'bear', 'drop']
        
        total_score = 0
        for headline in headlines:
            headline_lower = headline.lower()
            pos_count = sum(1 for word in positive_words if word in headline_lower)
            neg_count = sum(1 for word in negative_words if word in headline_lower)
            total_score += (pos_count - neg_count)
        
        avg_score = total_score / len(headlines)
        
        if avg_score > 0.1:
            return {'sentiment_label': 'Positive', 'confidence': min(avg_score, 1.0), 'compound': avg_score}
        elif avg_score < -0.1:
            return {'sentiment_label': 'Negative', 'confidence': min(abs(avg_score), 1.0), 'compound': avg_score}
        else:
            return {'sentiment_label': 'Neutral', 'confidence': 0.5, 'compound': 0}

    def get_stock_news_sentiment(self, symbol):
        """Fetch news and analyze sentiment for a stock"""
        headlines = self.news_provider.get_news_headlines(symbol)
        
        if not headlines:
            return None
        
        if ADVANCED_SENTIMENT and self.sentiment_analyzer:
            sentiments = []
            for headline in headlines:
                try:
                    score = self.sentiment_analyzer.polarity_scores(headline)
                    sentiments.append(score)
                except:
                    continue
            
            if sentiments:
                avg_sentiment = {k: np.mean([s[k] for s in sentiments]) for k in sentiments[0]}
                if avg_sentiment['compound'] >= 0.05:
                    sentiment_label = 'Positive'
                elif avg_sentiment['compound'] <= -0.05:
                    sentiment_label = 'Negative'
                else:
                    sentiment_label = 'Neutral'
                
                avg_sentiment['sentiment_label'] = sentiment_label
                return avg_sentiment
        
        return self.analyze_sentiment_basic(headlines)

    def prepare_fundamentals_features(self, symbol):
        """Prepare fundamental analysis features"""
        fundamentals_data = self.yahoo_provider.get_company_fundamentals(symbol)
        if not fundamentals_data:
            return None
        
        def safe_get(data, key, default=0):
            value = data.get(key, '0')
            try:
                return float(value) if value is not None and value != 'None' else default
            except (ValueError, TypeError):
                return default

        return {
            'symbol': symbol,
            'pe_ratio': safe_get(fundamentals_data, 'PERatio'),
            'pb_ratio': safe_get(fundamentals_data, 'PriceToBookRatio'),
            'roe': safe_get(fundamentals_data, 'ReturnOnEquity') * 100,
            'debt_to_equity': safe_get(fundamentals_data, 'DebtToEquityRatio'),
            'current_ratio': safe_get(fundamentals_data, 'CurrentRatio'),
            'gross_margin': safe_get(fundamentals_data, 'GrossProfitMargin') * 100,
            'operating_margin': safe_get(fundamentals_data, 'OperatingMargin') * 100,
            'profit_margin': safe_get(fundamentals_data, 'ProfitMargin') * 100,
            'revenue_growth': safe_get(fundamentals_data, 'QuarterlyRevenueGrowthYOY') * 100,
            'earnings_growth': safe_get(fundamentals_data, 'QuarterlyEarningsGrowthYOY') * 100,
            'market_cap': safe_get(fundamentals_data, 'MarketCapitalization'),
            'dividend_yield': safe_get(fundamentals_data, 'DividendYield') * 100
        }

    def train_fundamental_classifier(self):
        """Train ML model for fundamental analysis classification"""
        np.random.seed(42)
        n_samples = 1000
        
        pe_ratios = np.clip(np.random.normal(20, 10, n_samples), 5, 50)
        roe_values = np.clip(np.random.normal(15, 8, n_samples), -5, 35)
        debt_equity = np.clip(np.random.exponential(30, n_samples), 0, 100)
        profit_margins = np.clip(np.random.normal(12, 6, n_samples), -5, 25)
        revenue_growth = np.clip(np.random.normal(8, 12, n_samples), -20, 40)
        current_ratios = np.clip(np.random.normal(2.5, 1, n_samples), 0.5, 5)
        
        labels = []
        for i in range(n_samples):
            score = 0
            if 10 <= pe_ratios[i] <= 25: score += 1
            if roe_values[i] >= 15: score += 1
            if 0 < debt_equity[i] < 50: score += 1
            if profit_margins[i] >= 10: score += 1
            if revenue_growth[i] >= 8: score += 1
            if current_ratios[i] >= 1.5: score += 1
            
            if score >= 5:
                labels.append('Strong')
            elif score >= 3:
                labels.append('Moderate')
            else:
                labels.append('Weak')
        
        X = np.column_stack([pe_ratios, roe_values, debt_equity, profit_margins, revenue_growth, current_ratios])
        y = labels
        
        self.fundamental_classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            max_depth=10
        )
        
        X_scaled = self.fundamental_scaler.fit_transform(X)
        self.fundamental_classifier.fit(X_scaled, y)

    def analyze_fundamentals(self, fundamentals_data):
        """Analyze fundamental health of a company"""
        if not fundamentals_data:
            return {'score': 'Unknown', 'analysis': 'Insufficient data'}
        
        if self.fundamental_classifier is None:
            self.train_fundamental_classifier()
        
        feature_columns = ['pe_ratio', 'roe', 'debt_to_equity', 'profit_margin', 'revenue_growth', 'current_ratio']
        X = np.array([[
            fundamentals_data.get('pe_ratio', 0),
            fundamentals_data.get('roe', 0),
            fundamentals_data.get('debt_to_equity', 0),
            fundamentals_data.get('profit_margin', 0),
            fundamentals_data.get('revenue_growth', 0),
            fundamentals_data.get('current_ratio', 0)
        ]])
        
        X_scaled = self.fundamental_scaler.transform(X)
        prediction = self.fundamental_classifier.predict(X_scaled)[0]
        probabilities = self.fundamental_classifier.predict_proba(X_scaled)[0]
        
        analysis_points = []
        score = 0
        max_score = 6
        
        pe = fundamentals_data.get('pe_ratio', 0)
        if 10 <= pe <= 25:
            score += 1
            analysis_points.append(f"Healthy PE ratio: {pe:.1f}")
        
        roe = fundamentals_data.get('roe', 0)
        if roe >= 15:
            score += 1
            analysis_points.append(f"Strong ROE: {roe:.1f}%")
        
        score_percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        return {
            'score': prediction,
            'ml_score': prediction,
            'ml_confidence': max(probabilities),
            'score_percentage': score_percentage,
            'analysis_points': analysis_points,
            'fundamentals_data': fundamentals_data,
            'ml_probabilities': dict(zip(self.fundamental_classifier.classes_, probabilities))
        }

    def analyze_technical_indicators(self, symbol):
        """Analyze technical indicators for a stock"""
        price_data = self.yahoo_provider.get_stock_price_data(symbol)
        if price_data is None or len(price_data) < 50:
            return None
        
        close = price_data['close']
        high = price_data['high']
        low = price_data['low']
        
        # Calculate indicators
        rsi = TechnicalIndicators.rsi(close).iloc[-1]
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)
        stoch_k, stoch_d = TechnicalIndicators.stochastic(high, low, close)
        
        current_price = close.iloc[-1]
        sma_20 = TechnicalIndicators.sma(close, 20).iloc[-1]
        price_vs_sma20 = ((current_price / sma_20 - 1) * 100) if sma_20 > 0 else 0
        
        bb_position = ((current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])) if (bb_upper.iloc[-1] - bb_lower.iloc[-1]) > 0 else 0.5
        
        # Generate signals
        signals = {}
        if rsi > 70:
            signals['rsi'] = 'Overbought (Sell)'
        elif rsi < 30:
            signals['rsi'] = 'Oversold (Buy)'
        else:
            signals['rsi'] = 'Neutral'
        
        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            signals['macd'] = 'Bullish'
        else:
            signals['macd'] = 'Bearish'
        
        if bb_position > 0.8:
            signals['bollinger'] = 'Near Upper Band (Potential Sell)'
        elif bb_position < 0.2:
            signals['bollinger'] = 'Near Lower Band (Potential Buy)'
        else:
            signals['bollinger'] = 'Middle Range'
        
        if stoch_k.iloc[-1] > 80:
            signals['stochastic'] = 'Overbought'
        elif stoch_k.iloc[-1] < 20:
            signals['stochastic'] = 'Oversold'
        else:
            signals['stochastic'] = 'Neutral'
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predictions': {
                'trend': 'Uptrend' if macd_line.iloc[-1] > signal_line.iloc[-1] else 'Downtrend',
                'trend_confidence': 0.75,
                '1d_direction': 'Up' if rsi < 70 and macd_line.iloc[-1] > signal_line.iloc[-1] else 'Down',
                '1d_return': 0.5,
                '3d_direction': 'Up' if signals['macd'] == 'Bullish' else 'Down',
                '3d_return': 1.2,
                '5d_direction': 'Up' if signals['macd'] == 'Bullish' else 'Down',
                '5d_return': 2.1
            },
            'technical_indicators': {
                'rsi': rsi,
                'macd': macd_line.iloc[-1],
                'macd_signal': signal_line.iloc[-1],
                'bb_position': bb_position,
                'stoch_k': stoch_k.iloc[-1],
                'stoch_d': stoch_d.iloc[-1],
                'price_vs_sma20': price_vs_sma20
            },
            'signals': signals
        }

    def comprehensive_analysis(self, user_id, df):
        """Comprehensive analysis combining all models"""
        print(f"\nRunning comprehensive analysis for User {user_id}...")
        
        # Train models
        self.train_fundamental_classifier()
        
        # Trader Classification
        features_df = self.train_trader_classifier(df)
        user_features = features_df[features_df['user_id'] == user_id]
        
        trader_prediction = None
        if not user_features.empty:
            trader_prediction = self.predict_trader_type(user_features.iloc[0])
        
        user_trades = df[df['user_id'] == user_id]
        if user_trades.empty:
            return {
                'trader_analysis': trader_prediction,
                'sentiment_analysis': {},
                'fundamental_analysis': {},
                'technical_analysis': {},
                'user_summary': {}
            }
        
        symbol_counts = user_trades['symbol'].value_counts()
        top_symbols = symbol_counts.head(5).index.tolist()
        
        # Sentiment Analysis
        sentiment_results = {}
        for symbol in top_symbols[:3]:
            sentiment = self.get_stock_news_sentiment(symbol)
            if sentiment:
                sentiment_results[symbol] = sentiment
            time.sleep(1)
        
        # Fundamental Analysis
        fundamental_results = {}
        for symbol in top_symbols[:3]:
            fundamentals = self.prepare_fundamentals_features(symbol)
            if fundamentals:
                analysis = self.analyze_fundamentals(fundamentals)
                fundamental_results[symbol] = analysis
            time.sleep(1)
        
        # Technical Analysis
        technical_results = {}
        for symbol in top_symbols[:3]:
            technical_analysis = self.analyze_technical_indicators(symbol)
            if technical_analysis:
                technical_results[symbol] = technical_analysis
            time.sleep(1)
        
        return {
            'trader_analysis': trader_prediction,
            'sentiment_analysis': sentiment_results,
            'fundamental_analysis': fundamental_results,
            'technical_analysis': technical_results,
            'user_summary': {
                'total_trades': len(user_trades),
                'avg_profit': user_trades['profit_pct'].mean(),
                'win_rate': (user_trades['profit_pct'] > 0).mean(),
                'most_traded': symbol_counts.head(3).index.tolist()
            }
        }


def main():
    print("Enhanced Trading Analysis System")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = EnhancedTradingAnalyzer()
    
    # Try to load data
    try:
        df = pd.read_csv("userbehaviour/trade_info.csv")
        print(f"Loaded {len(df)} trades from CSV")
    except FileNotFoundError:
        print("Error: trade_info.csv not found")
        return
    
    # Data preparation
    df.columns = df.columns.str.strip()
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    if 'sell_time' in df.columns:
        df['sell_time'] = pd.to_datetime(df['sell_time'])
    if 'holding_time_hrs' not in df.columns and 'buy_time' in df.columns and 'sell_time' in df.columns:
        df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
    
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Get user ID
    available_users = df['user_id'].unique()
    print(f"\nAvailable users: {', '.join(available_users[:10])}...")
    
    user_id = input("\nEnter user ID to analyze: ").strip()
    
    if user_id not in available_users:
        print(f"User ID '{user_id}' not found. Using first available user: {available_users[0]}")
        user_id = available_users[0]
    
    # Run comprehensive analysis
    try:
        results = analyzer.comprehensive_analysis(user_id, df)
        
        # Display results
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE ANALYSIS FOR USER: {user_id}")
        print(f"{'='*60}")
        
        if results['trader_analysis']:
            ta = results['trader_analysis']
            print(f"\nTrader Profile:")
            print(f"  Type: {ta['predicted_type']} (Confidence: {ta['confidence']:.2f})")
        
        if results['user_summary']:
            summary = results['user_summary']
            print(f"\nTrading Summary:")
            print(f"  Total Trades: {summary['total_trades']}")
            print(f"  Average Profit: {summary['avg_profit']:.2f}%")
            print(f"  Win Rate: {summary['win_rate']:.2f}")
            print(f"  Most Traded: {', '.join(summary['most_traded'])}")
        
        if results['sentiment_analysis']:
            print(f"\nSentiment Analysis:")
            for symbol, sentiment in results['sentiment_analysis'].items():
                print(f"  {symbol}: {sentiment.get('sentiment_label', 'Unknown')}")
        
        if results['fundamental_analysis']:
            print(f"\nFundamental Analysis:")
            for symbol, analysis in results['fundamental_analysis'].items():
                print(f"  {symbol}: {analysis['score']} (Confidence: {analysis['ml_confidence']:.2f})")
        
        if results['technical_analysis']:
            print(f"\nTechnical Analysis:")
            for symbol, analysis in results['technical_analysis'].items():
                print(f"  {symbol}: Current Price: ${analysis['current_price']:.2f}")
                print(f"    Trend: {analysis['predictions']['trend']}")
                print(f"    RSI Signal: {analysis['signals']['rsi']}")
        
        print(f"\n{'='*60}")
        print("Analysis complete!")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
