from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global analyzer instance
analyzer = None
df = None
ML_AVAILABLE = False

# Try to import the ML analyzer, use simple fallback if it fails
try:
    from final import EnhancedTradingAnalyzer
    ML_AVAILABLE = True
    logger.info("ML analyzer imported successfully")
except Exception as e:
    logger.warning(f"Could not import ML analyzer: {e}")
    logger.warning("Running in simplified mode without ML features")
    EnhancedTradingAnalyzer = None

def initialize_analyzer():
    """Initialize the analyzer with data"""
    global analyzer, df
    try:
        # Load your trading data
        df = pd.read_csv("userbehaviour/trade_info.csv")
        df.columns = df.columns.str.strip()
        df['buy_time'] = pd.to_datetime(df['buy_time'])
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
        
        if ML_AVAILABLE:
            analyzer = EnhancedTradingAnalyzer()
            logger.info("Full ML analyzer initialized successfully")
        else:
            analyzer = None
            logger.info("Data loaded successfully (ML features disabled)")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return False

# Initialize when module loads
initialize_analyzer()

def simple_trader_classification(user_trades):
    """Simple rule-based trader classification when ML is not available"""
    avg_profit = user_trades['profit_pct'].mean()
    win_rate = (user_trades['profit_pct'] > 0).mean()
    avg_holding_time = user_trades['holding_time_hrs'].mean()
    num_trades = len(user_trades)
    trade_frequency = num_trades / 30  # Assume 30-day period
    
    # Simple classification rules
    if win_rate > 0.7 and avg_profit > 2 and avg_holding_time > 48:
        trader_type = "Conservative"
        confidence = 0.8
    elif num_trades > 20 and avg_holding_time < 24 and trade_frequency > 1:
        trader_type = "Aggressive"  
        confidence = 0.75
    else:
        trader_type = "Balanced"
        confidence = 0.7
    
    return {
        'predicted_type': trader_type,
        'confidence': confidence,
        'probabilities': {
            trader_type: confidence,
            'Other': 1 - confidence
        }
    }

# Error handler
@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'analyzer_ready': analyzer is not None,
        'ml_features_available': ML_AVAILABLE,
        'data_loaded': df is not None
    })

@app.route('/api/users', methods=['GET'])
def get_users():
    """Get list of available users"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        users = df['user_id'].unique().tolist()
        user_stats = []
        
        for user_id in users:
            user_trades = df[df['user_id'] == user_id]
            if not user_trades.empty:
                user_stats.append({
                    'user_id': user_id,
                    'total_trades': len(user_trades),
                    'avg_profit': float(user_trades['profit_pct'].mean()),
                    'win_rate': float((user_trades['profit_pct'] > 0).mean()),
                    'most_traded_stock': user_trades['symbol'].value_counts().index[0] if not user_trades.empty else None
                })
        
        return jsonify({
            'users': user_stats,
            'total_users': len(users),
            'ml_features_available': ML_AVAILABLE
        })
    
    except Exception as e:
        logger.error(f"Error in get_users: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/trader/<user_id>', methods=['GET'])
def analyze_trader(user_id):
    """Analyze trader behavior for specific user"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        if user_id not in df['user_id'].unique():
            return jsonify({'error': f'User {user_id} not found'}), 404
        
        user_trades = df[df['user_id'] == user_id]
        
        # Get user summary
        summary = {
            'total_trades': len(user_trades),
            'avg_profit': float(user_trades['profit_pct'].mean()),
            'win_rate': float((user_trades['profit_pct'] > 0).mean()),
            'avg_holding_time': float(user_trades['holding_time_hrs'].mean()),
            'most_traded_stocks': user_trades['symbol'].value_counts().head(5).to_dict()
        }
        
        # Use ML model if available, otherwise use simple classification
        if ML_AVAILABLE and analyzer:
            try:
                # Train trader classifier and get predictions
                features_df = analyzer.train_trader_classifier(df)
                user_features = features_df[features_df['user_id'] == user_id]
                
                if user_features.empty:
                    prediction = simple_trader_classification(user_trades)
                    prediction['method'] = 'fallback_rules'
                else:
                    prediction = analyzer.predict_trader_type(user_features.iloc[0])
                    prediction['method'] = 'ml_model'
            except Exception as e:
                logger.warning(f"ML analysis failed, using fallback: {e}")
                prediction = simple_trader_classification(user_trades)
                prediction['method'] = 'fallback_rules'
        else:
            prediction = simple_trader_classification(user_trades)
            prediction['method'] = 'simple_rules'
        
        return jsonify({
            'user_id': user_id,
            'trader_prediction': prediction,
            'user_summary': summary,
            'analysis_timestamp': datetime.now().isoformat(),
            'ml_features_used': ML_AVAILABLE
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_trader: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/sentiment/<symbol>', methods=['GET'])
def analyze_sentiment(symbol):
    """Analyze sentiment for a specific stock symbol"""
    try:
        if not ML_AVAILABLE or analyzer is None:
            return jsonify({
                'symbol': symbol.upper(),
                'sentiment': None,
                'message': 'Sentiment analysis requires ML features (fix final.py syntax error)',
                'ml_features_available': False
            })
        
        sentiment = analyzer.get_stock_news_sentiment(symbol.upper())
        
        if sentiment is None:
            return jsonify({
                'symbol': symbol.upper(),
                'sentiment': None,
                'message': 'No recent news found for this symbol'
            })
        
        return jsonify({
            'symbol': symbol.upper(),
            'sentiment': sentiment,
            'analysis_timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/comprehensive/<user_id>', methods=['GET'])
def comprehensive_analysis(user_id):
    """Run comprehensive analysis for a user"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        if user_id not in df['user_id'].unique():
            return jsonify({'error': f'User {user_id} not found'}), 404
        
        if not ML_AVAILABLE or analyzer is None:
            # Return basic analysis without ML features
            user_trades = df[df['user_id'] == user_id]
            basic_results = {
                'trader_analysis': simple_trader_classification(user_trades),
                'user_summary': {
                    'total_trades': len(user_trades),
                    'avg_profit': float(user_trades['profit_pct'].mean()),
                    'win_rate': float((user_trades['profit_pct'] > 0).mean()),
                    'most_traded': user_trades['symbol'].value_counts().head(3).index.tolist()
                }
            }
            
            return jsonify({
                'user_id': user_id,
                'results': basic_results,
                'analysis_timestamp': datetime.now().isoformat(),
                'note': 'Limited analysis - ML features disabled due to syntax error in final.py',
                'ml_features_available': False
            })
        
        # Run comprehensive analysis with ML
        results = analyzer.comprehensive_analysis(user_id, df)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results = convert_numpy_types(results)
        
        return jsonify({
            'user_id': user_id,
            'results': results,
            'analysis_timestamp': datetime.now().isoformat(),
            'ml_features_available': True
        })
    
    except Exception as e:
        logger.error(f"Error in comprehensive_analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stocks/popular', methods=['GET'])
def get_popular_stocks():
    """Get most popular stocks from trading data"""
    try:
        if df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Get stock popularity metrics
        stock_stats = df.groupby('symbol').agg({
            'profit_pct': ['mean', 'count'],
            'user_id': 'nunique'
        }).round(2)
        
        stock_stats.columns = ['avg_profit', 'total_trades', 'unique_traders']
        stock_stats = stock_stats.reset_index()
        stock_stats = stock_stats.sort_values('total_trades', ascending=False)
        
        popular_stocks = []
        for _, row in stock_stats.head(20).iterrows():
            popular_stocks.append({
                'symbol': row['symbol'],
                'avg_profit': float(row['avg_profit']),
                'total_trades': int(row['total_trades']),
                'unique_traders': int(row['unique_traders'])
            })
        
        return jsonify({
            'popular_stocks': popular_stocks,
            'total_stocks': len(stock_stats)
        })
    
    except Exception as e:
        logger.error(f"Error in get_popular_stocks: {e}")
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     print("=" * 60)
#     print("TRADING ANALYSIS API SERVER")
#     print("=" * 60)
#     print(f"ML Features Available: {ML_AVAILABLE}")
#     print(f"Data Loaded: {df is not None}")
#     if df is not None:
#         print(f"Total Trades in Database: {len(df)}")
#         print(f"Unique Users: {len(df['user_id'].unique())}")
#     print("=" * 60)
#     print("Starting server on http://localhost:5000")
#     print("API Endpoints:")
#     print("  GET  /api/health")
#     print("  GET  /api/users") 
#     print("  GET  /api/analyze/trader/{user_id}")
#     print("  GET  /api/analyze/comprehensive/{user_id}")
#     print("  GET  /api/stocks/popular")
#     print("=" * 60)
    
#     app.run(debug=True, host='0.0.0.0', port=5001)



if __name__ == '__main__':
    print("=" * 60)
    print("TRADING ANALYSIS API SERVER")
    print("=" * 60)
    print(f"ML Features Available: {ML_AVAILABLE}")
    print(f"Data Loaded: {df is not None}")
    if df is not None:
        print(f"Total Trades in Database: {len(df)}")
        print(f"Unique Users: {len(df['user_id'].unique())}")
    print("=" * 60)
    
    # Get port from environment variable or default to 5000 for local development
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    
    print("API Endpoints:")
    print("  GET  /api/health")
    print("  GET  /api/users") 
    print("  GET  /api/analyze/trader/{user_id}")
    print("  GET  /api/analyze/comprehensive/{user_id}")
    print("  GET  /api/stocks/popular")
    print("=" * 60)
    
    # Use PORT environment variable for Render deployment
    app.run(debug=False, host='0.0.0.0', port=port)
