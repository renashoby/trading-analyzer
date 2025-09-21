import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5001/api"

def test_api():
    """Test all API endpoints"""
    
    print("=" * 60)
    print("TESTING TRADING ANALYSIS API")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Get users
    print("\n2. Testing get users...")
    try:
        response = requests.get(f"{BASE_URL}/users")
        data = response.json()
        print(f"Status: {response.status_code}")
        print(f"Found {data.get('total_users', 0)} users")
        
        # Get first user for further testing
        if 'users' in data and data['users']:
            test_user = data['users'][0]['user_id']
            print(f"Using test user: {test_user}")
        else:
            print("No users found - check your CSV data")
            return
            
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Test 3: Trader analysis
    print(f"\n3. Testing trader analysis for {test_user}...")
    try:
        response = requests.get(f"{BASE_URL}/analyze/trader/{test_user}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('trader_prediction', {})
            print(f"Trader Type: {prediction.get('predicted_type', 'Unknown')}")
            print(f"Confidence: {prediction.get('confidence', 0):.2f}")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Sentiment analysis
    print("\n4. Testing sentiment analysis for AAPL...")
    try:
        response = requests.get(f"{BASE_URL}/analyze/sentiment/AAPL")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            sentiment = data.get('sentiment', {})
            if sentiment:
                print(f"Sentiment: {sentiment.get('sentiment_label', 'Unknown')}")
                print(f"Score: {sentiment.get('compound', 0):.2f}")
            else:
                print("No sentiment data available")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)  # Rate limiting
    
    # Test 5: Fundamental analysis
    print("\n5. Testing fundamental analysis for AAPL...")
    try:
        response = requests.get(f"{BASE_URL}/analyze/fundamentals/AAPL")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            analysis = data.get('analysis', {})
            if analysis:
                print(f"Health Score: {analysis.get('score', 'Unknown')}")
                print(f"ML Score: {analysis.get('ml_score', 'Unknown')}")
            else:
                print("No fundamental data available")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    time.sleep(2)  # Rate limiting
    
    # Test 6: Technical analysis
    print("\n6. Testing technical analysis for AAPL...")
    try:
        response = requests.get(f"{BASE_URL}/analyze/technical/AAPL")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            analysis = data.get('analysis', {})
            if analysis:
                predictions = analysis.get('predictions', {})
                print(f"Current Price: ${analysis.get('current_price', 0):.2f}")
                print(f"Trend: {predictions.get('trend', 'Unknown')}")
                print(f"1D Prediction: {predictions.get('1d_direction', 'Unknown')} ({predictions.get('1d_return', 0):.2f}%)")
            else:
                print("No technical analysis data available")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 7: Comprehensive analysis
    print(f"\n7. Testing comprehensive analysis for {test_user}...")
    try:
        response = requests.get(f"{BASE_URL}/analyze/comprehensive/{test_user}")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', {})
            print("Comprehensive analysis completed successfully!")
            print(f"Components analyzed: {list(results.keys())}")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 8: Popular stocks
    print("\n8. Testing popular stocks...")
    try:
        response = requests.get(f"{BASE_URL}/stocks/popular")
        data = response.json()
        print(f"Status: {response.status_code}")
        if 'popular_stocks' in data:
            print(f"Found {len(data['popular_stocks'])} popular stocks")
            for stock in data['popular_stocks'][:3]:
                print(f"  {stock['symbol']}: {stock['total_trades']} trades, {stock['avg_profit']:.2f}% avg profit")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 9: Batch analysis
    print("\n9. Testing batch analysis...")
    try:
        payload = {
            "symbols": ["AAPL", "GOOGL"],
            "analysis_types": ["sentiment"]
        }
        response = requests.post(
            f"{BASE_URL}/analyze/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            results = data.get('results', {})
            print(f"Batch analysis completed for {len(results)} symbols")
        else:
            print(f"Error: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("API TESTING COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_api()