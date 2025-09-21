# # # import pandas as pd
# # # import requests
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # import os
# # # import json
# # # import time
# # # from datetime import datetime, timedelta
# # # import numpy as np

# # # CACHE_FILE = "symbol_sector_cache.json"
# # # IDEAL_CSV_PATH = "user_ideals.csv"
# # # FULL_EVOLUTION_CSV_PATH = "full_trader_evolution.csv"

# # # if os.path.exists(CACHE_FILE):
# # #     with open(CACHE_FILE, 'r') as f:
# # #         symbol_sector_cache = json.load(f)
# # # else:
# # #     symbol_sector_cache = {}

# # # def get_search_id(symbol):
# # #     url = f"https://groww.in/v1/api/search/v3/query/global/st_query?query={symbol}"
# # #     try:
# # #         r = requests.get(url)
# # #         r.raise_for_status() 
# # #         data = r.json()
# # #         for item in data['data']['content']:
# # #             if item['entity_type'] == 'Stocks' and item.get('nse_scrip_code', '').upper() == symbol.upper():
# # #                 return item['search_id']
# # #     except requests.exceptions.RequestException as e:
# # #         print(f"Network or API error finding searchId for {symbol}: {e}")
# # #     except json.JSONDecodeError:
# # #         print(f"JSON decode error for {symbol}'s searchId response.")
# # #     except Exception as e:
# # #         print(f"An unexpected error occurred finding searchId for {symbol}: {e}")
# # #     return None

# # # def get_sector(search_id):
# # #     url = f"https://groww.in/v1/api/stocks_data/v1/company/search_id/{search_id}?fields=COMPANY_HEADER"
# # #     try:
# # #         r = requests.get(url)
# # #         r.raise_for_status() 
# # #         data = r.json()
# # #         return data['header'].get('industryName', 'Unknown')
# # #     except requests.exceptions.RequestException as e:
# # #         print(f"Network or API error getting sector for {search_id}: {e}")
# # #     except json.JSONDecodeError:
# # #         print(f"JSON decode error for {search_id}'s sector response.")
# # #     except Exception as e:
# # #         print(f"An unexpected error occurred getting sector for {search_id}: {e}")
# # #     return "Unknown"

# # # def label_trader(row):
# # #     if row['num_trades'] > 10 and row['std_profit_pct'] is not None and row['std_profit_pct'] > 5 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] > 2.5:
# # #         return 'Aggressive'
# # #     elif row['num_trades'] <= 5 and row['win_rate'] is not None and row['win_rate'] > 0.7 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] >= 1:
# # #         return 'Conservative'
# # #     else:
# # #         return 'Balanced'

# # # try:
# # #     df = pd.read_csv("trade_info.csv")
# # #     df.columns = df.columns.str.strip() 
# # #     df['buy_time'] = pd.to_datetime(df['buy_time'])
# # #     df['sell_time'] = pd.to_datetime(df['sell_time'])
# # #     df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
# # # except FileNotFoundError:
# # #     print("Error: 'trade_info.csv' not found. Please ensure the file is in the same directory.")
# # #     exit()
# # # except Exception as e:
# # #     print(f"Error loading or processing 'trade_info.csv': {e}")
# # #     exit()

# # # unique_symbols = df['symbol'].unique()
# # # print("Starting sector data fetching and caching...")
# # # for symbol in unique_symbols:
# # #     if symbol not in symbol_sector_cache:
# # #         # print(f"Fetching sector for {symbol}...")
# # #         search_id = get_search_id(symbol)
# # #         time.sleep(1)
# # #         if search_id:
# # #             sector = get_sector(search_id)
# # #         else:
# # #             sector = "Unknown"
# # #         symbol_sector_cache[symbol] = sector
# # #         time.sleep(1)

# # # with open(CACHE_FILE, 'w') as f:
# # #     json.dump(symbol_sector_cache, f, indent=2)
# # # print("Sector cache updated and saved.")

# # # df['sector'] = df['symbol'].map(symbol_sector_cache)

# # # if os.path.exists(IDEAL_CSV_PATH):
# # #     user_ideals_df = pd.read_csv(IDEAL_CSV_PATH)
# # # else:
# # #     user_ideals_df = pd.DataFrame(columns=["user_id", "ideal_type"])

# # # label_color_map = {
# # #     'Conservative': '#f39c12', 
# # #     'Balanced': '#2ecc71',     
# # #     'Aggressive': '#e74c3c'  
# # # }
# # # type_to_num = {'Conservative': 0, 'Balanced': 1, 'Aggressive': 2}

# # # evolution = [] 
# # # WINDOW_SIZE = 14 

# # # user_id = input("Enter the user_id you want to analyze (e.g., U001): ").strip()

# # # if user_id not in df['user_id'].values:
# # #     print(f"User ID '{user_id}' not found in dataset. Exiting.")
# # #     exit()

# # # user_trades = df[df['user_id'] == user_id].sort_values(by='buy_time')
# # # total_trades = len(user_trades)

# # # if total_trades <= 5:
# # #     print(f"User {user_id} has {total_trades} trades, which is too few for behavior evolution analysis.")
# # #     evolution.append({
# # #         'user_id': user_id,
# # #         'start_time': None, 'end_time': None,
# # #         'num_trades': total_trades, 'avg_profit_pct': None,
# # #         'win_rate': None, 'avg_gain_pct': None,
# # #         'avg_holding_time': None, 'avg_trade_size': None,
# # #         'std_profit_pct': None,
# # #         'trader_type': 'Unclassified'
# # #     })
# # #     plot_evolution = False
# # # else:
# # #     plot_evolution = True
# # #     if user_id not in user_ideals_df['user_id'].values:
# # #         print(f"\nEnter ideal trader type for user {user_id} (Conservative / Balanced / Aggressive):")
# # #         while True:
# # #             ideal = input(">> ").strip().capitalize()
# # #             if ideal in ['Conservative', 'Balanced', 'Aggressive']:
# # #                 break
# # #             print("Invalid input. Please enter Conservative, Balanced, or Aggressive.")

# # #         new_row = pd.DataFrame([{'user_id': user_id, 'ideal_type': ideal}])
# # #         user_ideals_df = pd.concat([user_ideals_df, new_row], ignore_index=True)
# # #         user_ideals_df = user_ideals_df.sort_values(by='user_id').reset_index(drop=True)
# # #         user_ideals_df.to_csv(IDEAL_CSV_PATH, index=False)
# # #         print(f"Ideal trader type for {user_id} saved to '{IDEAL_CSV_PATH}'.")
# # #     else:
# # #         print(f"Ideal trader type for {user_id} already recorded.")

# # #     ideal_type = user_ideals_df[user_ideals_df['user_id'] == user_id]['ideal_type'].values[0]
# # #     ideal_num = type_to_num[ideal_type]

# # #     user_trades = user_trades.reset_index(drop=True)
# # #     start_date = user_trades['buy_time'].min()
# # #     end_date = user_trades['buy_time'].max()
# # #     current_start = start_date

# # #     while current_start <= end_date:
# # #         current_end = current_start + timedelta(days=WINDOW_SIZE)
# # #         window = user_trades[(user_trades['buy_time'] >= current_start) & (user_trades['buy_time'] < current_end)]

# # #         if len(window) >= 1:
# # #             stats = {
# # #                 'user_id': user_id,
# # #                 'start_time': current_start,
# # #                 'end_time': current_end,
# # #                 'num_trades': len(window),
# # #                 'avg_profit_pct': window['profit_pct'].mean(),
# # #                 'win_rate': (window['profit_pct'] > 0).mean(),
# # #                 'avg_gain_pct': window[window['profit_pct'] > 0]['profit_pct'].mean() if any(window['profit_pct'] > 0) else 0,
# # #                 'avg_holding_time': window['holding_time_hrs'].mean(),
# # #                 'avg_trade_size': window['total_buy_value'].mean(),
# # #                 'std_profit_pct': window['profit_pct'].std() if len(window) > 1 else 0, 
# # #             }

# # #             stats['trader_type'] = label_trader(stats)
# # #             evolution.append(stats)

# # #         current_start += timedelta(days=7) 

# # # evolution_df = pd.DataFrame(evolution)
# # # evolution_df.to_csv(FULL_EVOLUTION_CSV_PATH, index=False)
# # # print(f"Full trader evolution data saved to '{FULL_EVOLUTION_CSV_PATH}'.")

# # # sns.set(style="whitegrid")

# # # fig, axes = plt.subplots(1, 2, figsize=(16, 8)) 

# # # if plot_evolution:
# # #     plot_df = evolution_df[evolution_df['trader_type'] != 'Unclassified'].copy()
# # #     plot_df['label_num'] = plot_df['trader_type'].map(type_to_num)
# # #     user_df_plot = plot_df[plot_df['user_id'] == user_id]

# # #     if not user_df_plot.empty:
# # #         for i in range(len(user_df_plot) - 1):
# # #             x_vals = [user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i + 1]['start_time']]
# # #             y_vals = [user_df_plot.iloc[i]['label_num'], user_df_plot.iloc[i + 1]['label_num']]
# # #             label = user_df_plot.iloc[i]['trader_type']
# # #             color = label_color_map.get(label, 'blue')
# # #             axes[0].plot(x_vals, y_vals, linestyle='-', color=color, linewidth=2)
# # #             axes[0].scatter(user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i]['label_num'], color=color, s=50, zorder=5)

# # #         last_row = user_df_plot.iloc[-1]
# # #         axes[0].scatter(last_row['start_time'], last_row['label_num'],
# # #                         color=label_color_map.get(last_row['trader_type'], 'gray'), s=50, zorder=5)

# # #         axes[0].plot(user_df_plot['start_time'], [ideal_num] * len(user_df_plot),
# # #                      linestyle=':', color='red', linewidth=2, label=f'Ideal ({ideal_type})')

# # #         axes[0].set_title(f"Trader Behavior Evolution - User {user_id}", fontsize=14)
# # #         axes[0].set_yticks([0, 1, 2])
# # #         axes[0].set_yticklabels(['Conservative', 'Balanced', 'Aggressive'], fontsize=10)
# # #         axes[0].set_xlabel("Time", fontsize=12)
# # #         axes[0].set_ylabel("Trader Type", fontsize=12)
# # #         axes[0].set_ylim(-0.5, 2.5)
# # #         axes[0].legend(loc='lower right', fontsize=10)
# # #         axes[0].grid(False) 
# # #         axes[0].tick_params(axis='x', rotation=45)
# # #         axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
# # #     else:
# # #         axes[0].set_title(f"Trader Behavior Evolution - User {user_id}\n(Not enough data to classify)", fontsize=14)
# # #         axes[0].axis('off') 
# # # else:
# # #     axes[0].set_title(f"Trader Behavior Evolution - User {user_id}\n(Not enough trades for analysis)", fontsize=14)
# # #     axes[0].axis('off') 

# # # user_df_for_pie = df[df['user_id'] == user_id]
# # # if not user_df_for_pie.empty:
# # #     sector_counts = user_df_for_pie['sector'].value_counts()

# # #     if not sector_counts.empty:
# # #         axes[1].pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
# # #         axes[1].set_title(f"Sector Distribution - User {user_id}", fontsize=14)
# # #         axes[1].axis('equal') 
# # #     else:
# # #         axes[1].set_title(f"Sector Distribution - User {user_id}\n(No sector data available)", fontsize=14)
# # #         axes[1].axis('off') 
# # # else:
# # #     axes[1].set_title(f"Sector Distribution - User {user_id}\n(No trade data available)", fontsize=14)
# # #     axes[1].axis('off') 

# # # plt.tight_layout() 
# # # plt.show() 

# # # print("\nAnalysis complete. Combined plots displayed.")



# # import pandas as pd
# # import requests
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import os
# # import json
# # import time
# # from datetime import datetime, timedelta
# # import numpy as np

# # # Import ReportLab modules for PDF generation
# # from reportlab.lib.pagesizes import letter
# # from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
# # from reportlab.lib.styles import getSampleStyleSheet
# # from reportlab.lib.units import inch

# # # --- Configuration and File Paths ---
# # CACHE_FILE = "symbol_sector_cache.json"
# # IDEAL_CSV_PATH = "user_ideals.csv"
# # FULL_EVOLUTION_CSV_PATH = "full_trader_evolution.csv"

# # # --- Load or create symbol-to-sector cache ---
# # # This cache helps avoid repeated API calls for stock sector information.
# # if os.path.exists(CACHE_FILE):
# #     with open(CACHE_FILE, 'r') as f:
# #         symbol_sector_cache = json.load(f)
# # else:
# #     symbol_sector_cache = {}

# # # --- Groww API Helpers ---
# # # Functions to interact with the Groww API to fetch stock details.
# # def get_search_id(symbol):
# #     """
# #     Fetches the search_id for a given stock symbol from Groww's search API.
# #     This search_id is crucial for subsequent API calls to get detailed stock info.
# #     """
# #     url = f"https://groww.in/v1/api/search/v3/query/global/st_query?query={symbol}"
# #     try:
# #         r = requests.get(url)
# #         r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
# #         data = r.json()
# #         for item in data['data']['content']:
# #             # Look for stock entities and match the NSE scrip code
# #             if item['entity_type'] == 'Stocks' and item.get('nse_scrip_code', '').upper() == symbol.upper():
# #                 return item['search_id']
# #     except requests.exceptions.RequestException as e:
# #         print(f"Network or API error finding searchId for {symbol}: {e}")
# #     except json.JSONDecodeError:
# #         print(f"JSON decode error for {symbol}'s searchId response.")
# #     except Exception as e:
# #         print(f"An unexpected error occurred finding searchId for {symbol}: {e}")
# #     return None

# # def get_sector(search_id):
# #     """
# #     Fetches the industry (sector) name for a given search_id from Groww's company data API.
# #     """
# #     url = f"https://groww.in/v1/api/stocks_data/v1/company/search_id/{search_id}?fields=COMPANY_HEADER"
# #     try:
# #         r = requests.get(url)
# #         r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
# #         data = r.json()
# #         # Extract industry name from the header
# #         return data['header'].get('industryName', 'Unknown')
# #     except requests.exceptions.RequestException as e:
# #         print(f"Network or API error getting sector for {search_id}: {e}")
# #     except json.JSONDecodeError:
# #         print(f"JSON decode error for {search_id}'s sector response.")
# #     except Exception as e:
# #         print(f"An unexpected error occurred getting sector for {search_id}: {e}")
# #     return "Unknown"

# # # --- Trader Classification Logic ---
# # def label_trader(row):
# #     """
# #     Classifies a trader's behavior based on their trading statistics.
# #     'Aggressive': High number of trades, high standard deviation of profit, good average profit.
# #     'Conservative': Fewer trades, high win rate, decent average profit.
# #     'Balanced': Falls between aggressive and conservative.
# #     """
# #     if row['num_trades'] > 10 and row['std_profit_pct'] is not None and row['std_profit_pct'] > 5 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] > 2.5:
# #         return 'Aggressive'
# #     elif row['num_trades'] <= 5 and row['win_rate'] is not None and row['win_rate'] > 0.7 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] >= 1:
# #         return 'Conservative'
# #     else:
# #         return 'Balanced'

# # # --- Gemini LLM API Call Function ---
# # def call_gemini_api(prompt_text, max_retries=5, initial_delay=1):
# #     """
# #     Calls the Gemini LLM API to generate text based on the provided prompt.
# #     Implements exponential backoff for robust API calls.
# #     """
# #     api_key = "" # As per instructions, leave this empty. Canvas will provide it.
# #     api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
# #     headers = {'Content-Type': 'application/json'}
# #     payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}

# #     for i in range(max_retries):
# #         try:
# #             response = requests.post(api_url, headers=headers, json=payload)
# #             response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
# #             result = response.json()
# #             if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
# #                 return result['candidates'][0]['content']['parts'][0]['text']
# #             else:
# #                 return "Could not generate explanation due to unexpected API response structure."
# #         except requests.exceptions.RequestException as e:
# #             if response.status_code in [429, 500, 503] and i < max_retries - 1: # Retry on Too Many Requests, Internal Server Error, Service Unavailable
# #                 delay = initial_delay * (2 ** i)
# #                 time.sleep(delay)
# #             else:
# #                 print(f"LLM API call failed after {i+1} retries: {e}")
# #                 return "Could not generate explanation due to API error."
# #         except json.JSONDecodeError:
# #             print(f"LLM API response is not valid JSON. Response: {response.text}")
# #             return "Could not generate explanation due to invalid API response."
# #         except Exception as e:
# #             print(f"An unexpected error occurred during LLM API call: {e}")
# #             return "Could not generate explanation due to an unexpected error."
# #     return "Could not generate explanation after multiple retries."

# # # --- Main Data Processing ---
# # # Load the trade information and preprocess columns.
# # try:
# #     df = pd.read_csv("trade_info.csv")
# #     df.columns = df.columns.str.strip() # Clean column names
# #     df['buy_time'] = pd.to_datetime(df['buy_time'])
# #     df['sell_time'] = pd.to_datetime(df['sell_time'])
# #     # Calculate holding time in hours
# #     df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
# # except FileNotFoundError:
# #     print("Error: 'trade_info.csv' not found. Please ensure the file is in the same directory.")
# #     exit()
# # except Exception as e:
# #     print(f"Error loading or processing 'trade_info.csv': {e}")
# #     exit()

# # # --- Fill in missing sectors using Groww API and cache ---
# # unique_symbols = df['symbol'].unique()
# # print("Starting sector data fetching and caching...")
# # for symbol in unique_symbols:
# #     if symbol not in symbol_sector_cache:
# #         print(f"🔍 Fetching sector for {symbol}...")
# #         search_id = get_search_id(symbol)
# #         time.sleep(1) # Be respectful to the API, add a small delay
# #         if search_id:
# #             sector = get_sector(search_id)
# #         else:
# #             sector = "Unknown"
# #         symbol_sector_cache[symbol] = sector
# #         time.sleep(1) # Another small delay after getting sector

# # # Save updated cache
# # with open(CACHE_FILE, 'w') as f:
# #     json.dump(symbol_sector_cache, f, indent=2)
# # print("Sector cache updated and saved.")

# # # Add sector column to DataFrame
# # df['sector'] = df['symbol'].map(symbol_sector_cache)

# # # --- Trader Evolution Setup ---
# # # Load or create user ideals CSV
# # if os.path.exists(IDEAL_CSV_PATH):
# #     user_ideals_df = pd.read_csv(IDEAL_CSV_PATH)
# # else:
# #     user_ideals_df = pd.DataFrame(columns=["user_id", "ideal_type"])

# # # Define color mapping for trader types for consistent plotting
# # label_color_map = {
# #     'Conservative': '#f39c12', # Orange
# #     'Balanced': '#2ecc71',     # Green
# #     'Aggressive': '#e74c3c'    # Red
# # }
# # # Numerical mapping for y-axis plotting
# # type_to_num = {'Conservative': 0, 'Balanced': 1, 'Aggressive': 2}

# # evolution = [] # To store periodic trader statistics
# # WINDOW_SIZE = 14 # Days for the sliding window analysis

# # # --- User Input for Analysis ---
# # user_id = input("Enter the user_id you want to analyze (e.g., U001): ").strip()

# # if user_id not in df['user_id'].values:
# #     print(f"User ID '{user_id}' not found in dataset. Exiting.")
# #     exit()

# # # Filter trades for the selected user and sort by buy time
# # user_trades = df[df['user_id'] == user_id].sort_values(by='buy_time')
# # total_trades = len(user_trades)

# # # Handle users with very few trades (cannot classify behavior evolution)
# # if total_trades <= 5:
# #     print(f"User {user_id} has {total_trades} trades, which is too few for behavior evolution analysis.")
# #     evolution.append({
# #         'user_id': user_id,
# #         'start_time': None, 'end_time': None,
# #         'num_trades': total_trades, 'avg_profit_pct': None,
# #         'win_rate': None, 'avg_gain_pct': None,
# #         'avg_holding_time': None, 'avg_trade_size': None,
# #         'std_profit_pct': None,
# #         'trader_type': 'Unclassified'
# #     })
# #     plot_evolution = False
# # else:
# #     plot_evolution = True
# #     # Prompt for ideal trader type if not already recorded for this user
# #     if user_id not in user_ideals_df['user_id'].values:
# #         print(f"\nEnter ideal trader type for user {user_id} (Conservative / Balanced / Aggressive):")
# #         while True:
# #             ideal = input(">> ").strip().capitalize()
# #             if ideal in ['Conservative', 'Balanced', 'Aggressive']:
# #                 break
# #             print("Invalid input. Please enter Conservative, Balanced, or Aggressive.")

# #         # Add new ideal type and save to CSV
# #         new_row = pd.DataFrame([{'user_id': user_id, 'ideal_type': ideal}])
# #         user_ideals_df = pd.concat([user_ideals_df, new_row], ignore_index=True)
# #         user_ideals_df = user_ideals_df.sort_values(by='user_id').reset_index(drop=True)
# #         user_ideals_df.to_csv(IDEAL_CSV_PATH, index=False)
# #         print(f"Ideal trader type for {user_id} saved to '{IDEAL_CSV_PATH}'.")
# #     else:
# #         print(f"Ideal trader type for {user_id} already recorded.")

# #     ideal_type = user_ideals_df[user_ideals_df['user_id'] == user_id]['ideal_type'].values[0]
# #     ideal_num = type_to_num[ideal_type]

# #     user_trades = user_trades.reset_index(drop=True)
# #     start_date = user_trades['buy_time'].min()
# #     end_date = user_trades['buy_time'].max()
# #     current_start = start_date

# #     # Sliding window analysis for trader evolution
# #     while current_start <= end_date:
# #         current_end = current_start + timedelta(days=WINDOW_SIZE)
# #         window = user_trades[(user_trades['buy_time'] >= current_start) & (user_trades['buy_time'] < current_end)]

# #         if len(window) >= 1:
# #             # Calculate statistics for the current window
# #             stats = {
# #                 'user_id': user_id,
# #                 'start_time': current_start,
# #                 'end_time': current_end,
# #                 'num_trades': len(window),
# #                 'avg_profit_pct': window['profit_pct'].mean(),
# #                 'win_rate': (window['profit_pct'] > 0).mean(),
# #                 'avg_gain_pct': window[window['profit_pct'] > 0]['profit_pct'].mean() if any(window['profit_pct'] > 0) else 0,
# #                 'avg_holding_time': window['holding_time_hrs'].mean(),
# #                 'avg_trade_size': window['total_buy_value'].mean(),
# #                 'std_profit_pct': window['profit_pct'].std() if len(window) > 1 else 0, # std needs at least 2 points
# #             }

# #             stats['trader_type'] = label_trader(stats)
# #             evolution.append(stats)

# #         current_start += timedelta(days=7) # Move window by 7 days

# # evolution_df = pd.DataFrame(evolution)
# # evolution_df.to_csv(FULL_EVOLUTION_CSV_PATH, index=False)
# # print(f"Full trader evolution data saved to '{FULL_EVOLUTION_CSV_PATH}'.")

# # # --- Plot 1: Trader Behavior Evolution (Separate Figure) ---
# # trader_evolution_explanation = "No evolution data to analyze."
# # trader_evolution_image_path = None # Initialize to None
# # if plot_evolution:
# #     plot_df = evolution_df[evolution_df['trader_type'] != 'Unclassified'].copy()
# #     plot_df['label_num'] = plot_df['trader_type'].map(type_to_num)
# #     user_df_plot = plot_df[plot_df['user_id'] == user_id]

# #     if not user_df_plot.empty:
# #         fig_evolution, ax_evolution = plt.subplots(figsize=(10, 7)) # Create a new figure for this plot

# #         # Plot lines connecting the trader types over time
# #         for i in range(len(user_df_plot) - 1):
# #             x_vals = [user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i + 1]['start_time']]
# #             y_vals = [user_df_plot.iloc[i]['label_num'], user_df_plot.iloc[i + 1]['label_num']]
# #             label = user_df_plot.iloc[i]['trader_type']
# #             color = label_color_map.get(label, 'blue')
# #             ax_evolution.plot(x_vals, y_vals, linestyle='-', color=color, linewidth=2)
# #             ax_evolution.scatter(user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i]['label_num'], color=color, s=50, zorder=5)

# #         # Plot the last point
# #         last_row = user_df_plot.iloc[-1]
# #         ax_evolution.scatter(last_row['start_time'], last_row['label_num'],
# #                         color=label_color_map.get(last_row['trader_type'], 'gray'), s=50, zorder=5)

# #         # Plot the ideal trader type as a horizontal dashed line
# #         ax_evolution.plot(user_df_plot['start_time'], [ideal_num] * len(user_df_plot),
# #                      linestyle=':', color='red', linewidth=2, label=f'Ideal ({ideal_type})')

# #         ax_evolution.set_title(f"Trader Behavior Evolution - User {user_id}", fontsize=14)
# #         ax_evolution.set_yticks([0, 1, 2])
# #         ax_evolution.set_yticklabels(['Conservative', 'Balanced', 'Aggressive'], fontsize=10)
# #         ax_evolution.set_xlabel("Time", fontsize=12)
# #         ax_evolution.set_ylabel("Trader Type", fontsize=12)
# #         ax_evolution.set_ylim(-0.5, 2.5)
# #         ax_evolution.legend(loc='lower right', fontsize=10)
# #         ax_evolution.grid(False) # Remove grid for cleaner look
# #         # Format x-axis dates
# #         ax_evolution.tick_params(axis='x', rotation=45)
# #         ax_evolution.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))

# #         trader_evolution_image_path = f"user_{user_id}_evolution.png"
# #         fig_evolution.tight_layout() # Ensure tight layout for this specific figure
# #         fig_evolution.savefig(trader_evolution_image_path)
# #         plt.close(fig_evolution) # Close the figure to free up memory

# #         # Prepare prompt for LLM for Trader Evolution
# #         current_trader_type = last_row['trader_type']
# #         trader_evolution_prompt = (
# #             f"Analyze the trading behavior for user {user_id}. "
# #             f"Their ideal trader type is '{ideal_type}'. "
# #             f"Their current trading pattern reflects a '{current_trader_type}' trader type. "
# #             f"Explain what it means to have '{ideal_type}' as an ideal trader type. "
# #             f"Describe what their current '{current_trader_type}' pattern reflects. "
# #             f"Provide actionable suggestions on how they can better their trading behavior "
# #             f"and move closer to their ideal '{ideal_type}' type. Keep the explanation concise and to the point."
# #         )
# #         trader_evolution_explanation = call_gemini_api(trader_evolution_prompt)

# #     else:
# #         trader_evolution_explanation = f"User {user_id} does not have enough classified trading data for behavior evolution analysis."
# # else:
# #     trader_evolution_explanation = f"User {user_id} has too few trades ({total_trades}) for behavior evolution analysis."


# # # --- Plot 2: Sector Distribution Pie Chart (Separate Figure) ---
# # sector_distribution_explanation = "No sector data to analyze."
# # sector_pie_image_path = None # Initialize to None
# # user_df_for_pie = df[df['user_id'] == user_id]
# # if not user_df_for_pie.empty:
# #     sector_counts = user_df_for_pie['sector'].value_counts()

# #     if not sector_counts.empty:
# #         fig_pie, ax_pie = plt.subplots(figsize=(8, 8)) # Create a new figure for the pie chart
# #         ax_pie.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
# #         ax_pie.set_title(f"Sector Distribution - User {user_id}", fontsize=14)
# #         ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

# #         sector_pie_image_path = f"user_{user_id}_sector_pie.png"
# #         fig_pie.tight_layout() # Ensure tight layout for this specific figure
# #         fig_pie.savefig(sector_pie_image_path)
# #         plt.close(fig_pie) # Close the figure to free up memory

# #         # Prepare prompt for LLM for Sector Distribution
# #         sector_info_list = [f"{sector}: {percentage:.1f}%" for sector, percentage in (sector_counts / sector_counts.sum() * 100).items()]
# #         sector_info_str = ", ".join(sector_info_list)

# #         # Heuristic to determine spread/concentration
# #         max_percentage = (sector_counts / sector_counts.sum() * 100).max()
# #         if max_percentage > 70:
# #             concentration_comment = "Their portfolio is highly concentrated in a few sectors."
# #         elif len(sector_counts) > 5 and max_percentage < 40:
# #             concentration_comment = "Their portfolio is quite diversified across multiple sectors."
# #         else:
# #             concentration_comment = "Their portfolio shows a balanced distribution across sectors."

# #         sector_distribution_prompt = (
# #             f"Analyze the sector distribution for user {user_id}. "
# #             f"Their portfolio is distributed as follows: {sector_info_str}. "
# #             f"{concentration_comment} "
# #             f"Explain the benefits of their current distribution (whether concentrated or diversified). "
# #             f"Suggest if they need to make changes to their sector allocation, and if so, what kind of changes and why. "
# #             f"Keep the explanation concise and to the point."
# #         )
# #         sector_distribution_explanation = call_gemini_api(sector_distribution_prompt)

# #     else:
# #         sector_distribution_explanation = f"User {user_id} has no sector data available for analysis."
# # else:
# #     sector_distribution_explanation = f"User {user_id} has no trade data available for sector distribution analysis."

# # # --- PDF Report Generation ---
# # pdf_filename = f"user_{user_id}_report.pdf"
# # doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
# # styles = getSampleStyleSheet()
# # story = []

# # # Add Trader Behavior Evolution Graph and Explanation
# # if trader_evolution_image_path and os.path.exists(trader_evolution_image_path):
# #     img_evolution = Image(trader_evolution_image_path)
# #     # Scale image to fit within page width, maintaining aspect ratio
# #     img_width, img_height = img_evolution.drawWidth, img_evolution.drawHeight
# #     aspect_ratio = img_height / img_width
# #     max_img_width = 6.5 * inch # Max width for image in PDF (leaving margins)
# #     img_evolution.drawWidth = max_img_width
# #     img_evolution.drawHeight = max_img_width * aspect_ratio

# #     story.append(img_evolution)
# #     story.append(Spacer(1, 0.2 * inch)) # Add some space
# #     story.append(Paragraph("<b>Trader Behavior Analysis</b>", styles['h2']))
# #     story.append(Paragraph(trader_evolution_explanation.replace('\n', '<br/>'), styles['Normal']))
# #     story.append(Spacer(1, 0.4 * inch)) # Add more space between sections
# # else:
# #     story.append(Paragraph("<b>Trader Behavior Analysis</b>", styles['h2']))
# #     story.append(Paragraph(trader_evolution_explanation.replace('\n', '<br/>'), styles['Normal']))
# #     story.append(Spacer(1, 0.4 * inch))

# # # Add Sector Distribution Pie Chart and Explanation
# # if sector_pie_image_path and os.path.exists(sector_pie_image_path):
# #     img_pie = Image(sector_pie_image_path)
# #     # Scale image to fit within page width, maintaining aspect ratio
# #     img_width, img_height = img_pie.drawWidth, img_pie.drawHeight
# #     aspect_ratio = img_height / img_width
# #     max_img_width = 6.5 * inch # Keep consistent width
# #     img_pie.drawWidth = max_img_width
# #     img_pie.drawHeight = max_img_width * aspect_ratio

# #     story.append(img_pie)
# #     story.append(Spacer(1, 0.2 * inch)) # Add some space
# #     story.append(Paragraph("<b>Sector Distribution Analysis</b>", styles['h2']))
# #     story.append(Paragraph(sector_distribution_explanation.replace('\n', '<br/>'), styles['Normal']))
# #     story.append(Spacer(1, 0.4 * inch))
# # else:
# #     story.append(Paragraph("<b>Sector Distribution Analysis</b>", styles['h2']))
# #     story.append(Paragraph(sector_distribution_explanation.replace('\n', '<br/>'), styles['Normal']))
# #     story.append(Spacer(1, 0.4 * inch))

# # doc.build(story)
# # print(f"\n✅ PDF report generated: '{pdf_filename}'")

# # # --- Clean up temporary image files ---
# # if trader_evolution_image_path and os.path.exists(trader_evolution_image_path):
# #     os.remove(trader_evolution_image_path)
# #     print(f"Cleaned up temporary image: {trader_evolution_image_path}")
# # if sector_pie_image_path and os.path.exists(sector_pie_image_path):
# #     os.remove(sector_pie_image_path)
# #     print(f"Cleaned up temporary image: {sector_pie_image_path}")

# # print("\n🎉 Analysis complete. PDF report generated.")












# import pandas as pd
# import requests
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import json
# import time
# from datetime import datetime, timedelta
# import numpy as np

# # Import ReportLab modules for PDF generation
# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
# from reportlab.lib.styles import getSampleStyleSheet
# from reportlab.lib.units import inch

# # --- Configuration and File Paths ---
# CACHE_FILE = "symbol_sector_cache.json"
# IDEAL_CSV_PATH = "user_ideals.csv"
# FULL_EVOLUTION_CSV_PATH = "full_trader_evolution.csv"

# # --- Load or create symbol-to-sector cache ---
# # This cache helps avoid repeated API calls for stock sector information.
# if os.path.exists(CACHE_FILE):
#     with open(CACHE_FILE, 'r') as f:
#         symbol_sector_cache = json.load(f)
# else:
#     symbol_sector_cache = {}

# # --- Groww API Helpers ---
# # Functions to interact with the Groww API to fetch stock details.
# def get_search_id(symbol):
#     """
#     Fetches the search_id for a given stock symbol from Groww's search API.
#     This search_id is crucial for subsequent API calls to get detailed stock info.
#     """
#     url = f"https://groww.in/v1/api/search/v3/query/global/st_query?query={symbol}"
#     try:
#         r = requests.get(url)
#         r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
#         data = r.json()
#         for item in data['data']['content']:
#             # Look for stock entities and match the NSE scrip code
#             if item['entity_type'] == 'Stocks' and item.get('nse_scrip_code', '').upper() == symbol.upper():
#                 return item['search_id']
#     except requests.exceptions.RequestException as e:
#         print(f"Network or API error finding searchId for {symbol}: {e}")
#     except json.JSONDecodeError:
#         print(f"JSON decode error for {symbol}'s searchId response.")
#     except Exception as e:
#         print(f"An unexpected error occurred finding searchId for {symbol}: {e}")
#     return None

# def get_sector(search_id):
#     """
#     Fetches the industry (sector) name for a given search_id from Groww's company data API.
#     """
#     url = f"https://groww.in/v1/api/stocks_data/v1/company/search_id/{search_id}?fields=COMPANY_HEADER"
#     try:
#         r = requests.get(url)
#         r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
#         data = r.json()
#         # Extract industry name from the header
#         return data['header'].get('industryName', 'Unknown')
#     except requests.exceptions.RequestException as e:
#         print(f"Network or API error getting sector for {search_id}: {e}")
#     except json.JSONDecodeError:
#         print(f"JSON decode error for {search_id}'s sector response.")
#     except Exception as e:
#         print(f"An unexpected error occurred getting sector for {search_id}: {e}")
#     return "Unknown"

# # --- Trader Classification Logic ---
# def label_trader(row):
#     """
#     Classifies a trader's behavior based on their trading statistics.
#     'Aggressive': High number of trades, high standard deviation of profit, good average profit.
#     'Conservative': Fewer trades, high win rate, decent average profit.
#     'Balanced': Falls between aggressive and conservative.
#     """
#     if row['num_trades'] > 10 and row['std_profit_pct'] is not None and row['std_profit_pct'] > 5 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] > 2.5:
#         return 'Aggressive'
#     elif row['num_trades'] <= 5 and row['win_rate'] is not None and row['win_rate'] > 0.7 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] >= 1:
#         return 'Conservative'
#     else:
#         return 'Balanced'

# # --- Gemini LLM API Call Function ---
# def call_gemini_api(prompt_text, max_retries=5, initial_delay=1):
#     """
#     Calls the Gemini LLM API to generate text based on the provided prompt.
#     Implements exponential backoff for robust API calls.
#     """
#     # Replace "YOUR_API_KEY_HERE" with your actual Gemini API key.
#     # IMPORTANT: Hardcoding API keys is not recommended for production environments.
#     # Consider using environment variables for better security.
#     api_key = "AIzaSyA-djBCrj6HEsFAMCGSdYnYiEpYHoXZHg0" 
#     api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
#     headers = {'Content-Type': 'application/json'}
#     payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}

#     for i in range(max_retries):
#         try:
#             response = requests.post(api_url, headers=headers, json=payload)
#             response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
#             result = response.json()
#             if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
#                 return result['candidates'][0]['content']['parts'][0]['text']
#             else:
#                 return "Could not generate explanation due to unexpected API response structure."
#         except requests.exceptions.RequestException as e:
#             if response.status_code in [429, 500, 503] and i < max_retries - 1: # Retry on Too Many Requests, Internal Server Error, Service Unavailable
#                 delay = initial_delay * (2 ** i)
#                 time.sleep(delay)
#             else:
#                 print(f"LLM API call failed after {i+1} retries: {e}")
#                 return "Could not generate explanation due to API error."
#         except json.JSONDecodeError:
#             print(f"LLM API response is not valid JSON. Response: {response.text}")
#             return "Could not generate explanation due to invalid API response."
#         except Exception as e:
#             print(f"An unexpected error occurred during LLM API call: {e}")
#             return "Could not generate explanation due to an unexpected error."
#     return "Could not generate explanation after multiple retries."

# # --- Main Data Processing ---
# # Load the trade information and preprocess columns.
# try:
#     df = pd.read_csv("trade_info.csv")
#     df.columns = df.columns.str.strip() # Clean column names
#     df['buy_time'] = pd.to_datetime(df['buy_time'])
#     df['sell_time'] = pd.to_datetime(df['sell_time'])
#     # Calculate holding time in hours
#     df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
# except FileNotFoundError:
#     print("Error: 'trade_info.csv' not found. Please ensure the file is in the same directory.")
#     exit()
# except Exception as e:
#     print(f"Error loading or processing 'trade_info.csv': {e}")
#     exit()

# # --- Fill in missing sectors using Groww API and cache ---
# unique_symbols = df['symbol'].unique()
# print("Starting sector data fetching and caching...")
# for symbol in unique_symbols:
#     if symbol not in symbol_sector_cache:
#         print(f"🔍 Fetching sector for {symbol}...")
#         search_id = get_search_id(symbol)
#         time.sleep(1) # Be respectful to the API, add a small delay
#         if search_id:
#             sector = get_sector(search_id)
#         else:
#             sector = "Unknown"
#         symbol_sector_cache[symbol] = sector
#         time.sleep(1) # Another small delay after getting sector

# # Save updated cache
# with open(CACHE_FILE, 'w') as f:
#     json.dump(symbol_sector_cache, f, indent=2)
# print("Sector cache updated and saved.")

# # Add sector column to DataFrame
# df['sector'] = df['symbol'].map(symbol_sector_cache)

# # --- Trader Evolution Setup ---
# # Load or create user ideals CSV
# if os.path.exists(IDEAL_CSV_PATH):
#     user_ideals_df = pd.read_csv(IDEAL_CSV_PATH)
# else:
#     user_ideals_df = pd.DataFrame(columns=["user_id", "ideal_type"])

# # Define color mapping for trader types for consistent plotting
# label_color_map = {
#     'Conservative': '#f39c12', # Orange
#     'Balanced': '#2ecc71',     # Green
#     'Aggressive': '#e74c3c'    # Red
# }
# # Numerical mapping for y-axis plotting
# type_to_num = {'Conservative': 0, 'Balanced': 1, 'Aggressive': 2}

# evolution = [] # To store periodic trader statistics
# WINDOW_SIZE = 14 # Days for the sliding window analysis

# # --- User Input for Analysis ---
# user_id = input("Enter the user_id you want to analyze (e.g., U001): ").strip()

# if user_id not in df['user_id'].values:
#     print(f"User ID '{user_id}' not found in dataset. Exiting.")
#     exit()

# # Filter trades for the selected user and sort by buy time
# user_trades = df[df['user_id'] == user_id].sort_values(by='buy_time')
# total_trades = len(user_trades)

# # Handle users with very few trades (cannot classify behavior evolution)
# if total_trades <= 5:
#     print(f"User {user_id} has {total_trades} trades, which is too few for behavior evolution analysis.")
#     evolution.append({
#         'user_id': user_id,
#         'start_time': None, 'end_time': None,
#         'num_trades': total_trades, 'avg_profit_pct': None,
#         'win_rate': None, 'avg_gain_pct': None,
#         'avg_holding_time': None, 'avg_trade_size': None,
#         'std_profit_pct': None,
#         'trader_type': 'Unclassified'
#     })
#     plot_evolution = False
# else:
#     plot_evolution = True
#     # Prompt for ideal trader type if not already recorded for this user
#     if user_id not in user_ideals_df['user_id'].values:
#         print(f"\nEnter ideal trader type for user {user_id} (Conservative / Balanced / Aggressive):")
#         while True:
#             ideal = input(">> ").strip().capitalize()
#             if ideal in ['Conservative', 'Balanced', 'Aggressive']:
#                 break
#             print("Invalid input. Please enter Conservative, Balanced, or Aggressive.")

#         # Add new ideal type and save to CSV
#         new_row = pd.DataFrame([{'user_id': user_id, 'ideal_type': ideal}])
#         user_ideals_df = pd.concat([user_ideals_df, new_row], ignore_index=True)
#         user_ideals_df = user_ideals_df.sort_values(by='user_id').reset_index(drop=True)
#         user_ideals_df.to_csv(IDEAL_CSV_PATH, index=False)
#         print(f"Ideal trader type for {user_id} saved to '{IDEAL_CSV_PATH}'.")
#     else:
#         print(f"Ideal trader type for {user_id} already recorded.")

#     ideal_type = user_ideals_df[user_ideals_df['user_id'] == user_id]['ideal_type'].values[0]
#     ideal_num = type_to_num[ideal_type]

#     user_trades = user_trades.reset_index(drop=True)
#     start_date = user_trades['buy_time'].min()
#     end_date = user_trades['buy_time'].max()
#     current_start = start_date

#     # Sliding window analysis for trader evolution
#     while current_start <= end_date:
#         current_end = current_start + timedelta(days=WINDOW_SIZE)
#         window = user_trades[(user_trades['buy_time'] >= current_start) & (user_trades['buy_time'] < current_end)]

#         if len(window) >= 1:
#             # Calculate statistics for the current window
#             stats = {
#                 'user_id': user_id,
#                 'start_time': current_start,
#                 'end_time': current_end,
#                 'num_trades': len(window),
#                 'avg_profit_pct': window['profit_pct'].mean(),
#                 'win_rate': (window['profit_pct'] > 0).mean(),
#                 'avg_gain_pct': window[window['profit_pct'] > 0]['profit_pct'].mean() if any(window['profit_pct'] > 0) else 0,
#                 'avg_holding_time': window['holding_time_hrs'].mean(),
#                 'avg_trade_size': window['total_buy_value'].mean(),
#                 'std_profit_pct': window['profit_pct'].std() if len(window) > 1 else 0, # std needs at least 2 points
#             }

#             stats['trader_type'] = label_trader(stats)
#             evolution.append(stats)

#         current_start += timedelta(days=7) # Move window by 7 days

# evolution_df = pd.DataFrame(evolution)
# evolution_df.to_csv(FULL_EVOLUTION_CSV_PATH, index=False)
# print(f"Full trader evolution data saved to '{FULL_EVOLUTION_CSV_PATH}'.")

# # --- Plot 1: Trader Behavior Evolution (Separate Figure) ---
# trader_evolution_explanation = "No evolution data to analyze."
# trader_evolution_image_path = None # Initialize to None
# if plot_evolution:
#     plot_df = evolution_df[evolution_df['trader_type'] != 'Unclassified'].copy()
#     plot_df['label_num'] = plot_df['trader_type'].map(type_to_num)
#     user_df_plot = plot_df[plot_df['user_id'] == user_id]

#     if not user_df_plot.empty:
#         fig_evolution, ax_evolution = plt.subplots(figsize=(10, 7)) # Create a new figure for this plot

#         # Plot lines connecting the trader types over time
#         for i in range(len(user_df_plot) - 1):
#             x_vals = [user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i + 1]['start_time']]
#             y_vals = [user_df_plot.iloc[i]['label_num'], user_df_plot.iloc[i + 1]['label_num']]
#             label = user_df_plot.iloc[i]['trader_type']
#             color = label_color_map.get(label, 'blue')
#             ax_evolution.plot(x_vals, y_vals, linestyle='-', color=color, linewidth=2)
#             ax_evolution.scatter(user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i]['label_num'], color=color, s=50, zorder=5)

#         # Plot the last point
#         last_row = user_df_plot.iloc[-1]
#         ax_evolution.scatter(last_row['start_time'], last_row['label_num'],
#                         color=label_color_map.get(last_row['trader_type'], 'gray'), s=50, zorder=5)

#         # Plot the ideal trader type as a horizontal dashed line
#         ax_evolution.plot(user_df_plot['start_time'], [ideal_num] * len(user_df_plot),
#                      linestyle=':', color='red', linewidth=2, label=f'Ideal ({ideal_type})')

#         ax_evolution.set_title(f"Trader Behavior Evolution - User {user_id}", fontsize=14)
#         ax_evolution.set_yticks([0, 1, 2])
#         ax_evolution.set_yticklabels(['Conservative', 'Balanced', 'Aggressive'], fontsize=10)
#         ax_evolution.set_xlabel("Time", fontsize=12)
#         ax_evolution.set_ylabel("Trader Type", fontsize=12)
#         ax_evolution.set_ylim(-0.5, 2.5)
#         ax_evolution.legend(loc='lower right', fontsize=10)
#         ax_evolution.grid(False) # Remove grid for cleaner look
#         # Format x-axis dates
#         ax_evolution.tick_params(axis='x', rotation=45)
#         ax_evolution.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))

#         trader_evolution_image_path = f"user_{user_id}_evolution.png"
#         fig_evolution.tight_layout() # Ensure tight layout for this specific figure
#         fig_evolution.savefig(trader_evolution_image_path)
#         plt.close(fig_evolution) # Close the figure to free up memory

#         # Prepare prompt for LLM for Trader Evolution
#         current_trader_type = last_row['trader_type']
#         trader_evolution_prompt = (
#             f"Analyze the trading behavior for user {user_id}. "
#             f"Their ideal trader type is '{ideal_type}'. "
#             f"Their current trading pattern reflects a '{current_trader_type}' trader type. "
#             f"Explain what it means to have '{ideal_type}' as an ideal trader type. "
#             f"Describe what their current '{current_trader_type}' pattern reflects. "
#             f"Provide actionable suggestions on how they can better their trading behavior "
#             f"and move closer to their ideal '{ideal_type}' type. Keep the explanation concise and to the point."
#         )
#         trader_evolution_explanation = call_gemini_api(trader_evolution_prompt)

#     else:
#         trader_evolution_explanation = f"User {user_id} does not have enough classified trading data for behavior evolution analysis."
# else:
#     trader_evolution_explanation = f"User {user_id} has too few trades ({total_trades}) for behavior evolution analysis."


# # --- Plot 2: Sector Distribution Pie Chart (Separate Figure) ---
# sector_distribution_explanation = "No sector data to analyze."
# sector_pie_image_path = None # Initialize to None
# user_df_for_pie = df[df['user_id'] == user_id]
# if not user_df_for_pie.empty:
#     sector_counts = user_df_for_pie['sector'].value_counts()

#     if not sector_counts.empty:
#         fig_pie, ax_pie = plt.subplots(figsize=(8, 8)) # Create a new figure for the pie chart
#         ax_pie.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
#         ax_pie.set_title(f"Sector Distribution - User {user_id}", fontsize=14)
#         ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

#         sector_pie_image_path = f"user_{user_id}_sector_pie.png"
#         fig_pie.tight_layout() # Ensure tight layout for this specific figure
#         fig_pie.savefig(sector_pie_image_path)
#         plt.close(fig_pie) # Close the figure to free up memory

#         # Prepare prompt for LLM for Sector Distribution
#         sector_info_list = [f"{sector}: {percentage:.1f}%" for sector, percentage in (sector_counts / sector_counts.sum() * 100).items()]
#         sector_info_str = ", ".join(sector_info_list)

#         # Heuristic to determine spread/concentration
#         max_percentage = (sector_counts / sector_counts.sum() * 100).max()
#         if max_percentage > 70:
#             concentration_comment = "Their portfolio is highly concentrated in a few sectors."
#         elif len(sector_counts) > 5 and max_percentage < 40:
#             concentration_comment = "Their portfolio is quite diversified across multiple sectors."
#         else:
#             concentration_comment = "Their portfolio shows a balanced distribution across sectors."

#         sector_distribution_prompt = (
#             f"Analyze the sector distribution for user {user_id}. "
#             f"Their portfolio is distributed as follows: {sector_info_str}. "
#             f"{concentration_comment} "
#             f"Explain the benefits of their current distribution (whether concentrated or diversified). "
#             f"Suggest if they need to make changes to their sector allocation, and if so, what kind of changes and why. "
#             f"Keep the explanation concise and to the point."
#         )
#         sector_distribution_explanation = call_gemini_api(sector_distribution_prompt)

#     else:
#         sector_distribution_explanation = f"User {user_id} has no sector data available for analysis."
# else:
#     sector_distribution_explanation = f"User {user_id} has no trade data available for sector distribution analysis."

# # --- PDF Report Generation ---
# pdf_filename = f"user_{user_id}_report.pdf"
# doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
# styles = getSampleStyleSheet()
# story = []

# # Add Trader Behavior Evolution Graph and Explanation
# if trader_evolution_image_path and os.path.exists(trader_evolution_image_path):
#     img_evolution = Image(trader_evolution_image_path)
#     # Scale image to fit within page width, maintaining aspect ratio
#     img_width, img_height = img_evolution.drawWidth, img_evolution.drawHeight
#     aspect_ratio = img_height / img_width
#     max_img_width = 6.5 * inch # Max width for image in PDF (leaving margins)
#     img_evolution.drawWidth = max_img_width
#     img_evolution.drawHeight = max_img_width * aspect_ratio

#     story.append(img_evolution)
#     story.append(Spacer(1, 0.2 * inch)) # Add some space
#     story.append(Paragraph("<b>Trader Behavior Analysis</b>", styles['h2']))
#     story.append(Paragraph(trader_evolution_explanation.replace('\n', '<br/>'), styles['Normal']))
#     story.append(Spacer(1, 0.4 * inch)) # Add more space between sections
# else:
#     story.append(Paragraph("<b>Trader Behavior Analysis</b>", styles['h2']))
#     story.append(Paragraph(trader_evolution_explanation.replace('\n', '<br/>'), styles['Normal']))
#     story.append(Spacer(1, 0.4 * inch))

# # Add Sector Distribution Pie Chart and Explanation
# if sector_pie_image_path and os.path.exists(sector_pie_image_path):
#     img_pie = Image(sector_pie_image_path)
#     # Scale image to fit within page width, maintaining aspect ratio
#     img_width, img_height = img_pie.drawWidth, img_pie.drawHeight
#     aspect_ratio = img_height / img_width
#     max_img_width = 6.5 * inch # Keep consistent width
#     img_pie.drawWidth = max_img_width
#     img_pie.drawHeight = max_img_width * aspect_ratio

#     story.append(img_pie)
#     story.append(Spacer(1, 0.2 * inch)) # Add some space
#     story.append(Paragraph("<b>Sector Distribution Analysis</b>", styles['h2']))
#     story.append(Paragraph(sector_distribution_explanation.replace('\n', '<br/>'), styles['Normal']))
#     story.append(Spacer(1, 0.4 * inch))
# else:
#     story.append(Paragraph("<b>Sector Distribution Analysis</b>", styles['h2']))
#     story.append(Paragraph(sector_distribution_explanation.replace('\n', '<br/>'), styles['Normal']))
#     story.append(Spacer(1, 0.4 * inch))

# doc.build(story)
# print(f"\n✅ PDF report generated: '{pdf_filename}'")

# # --- Clean up temporary image files ---
# if trader_evolution_image_path and os.path.exists(trader_evolution_image_path):
#     os.remove(trader_evolution_image_path)
#     print(f"Cleaned up temporary image: {trader_evolution_image_path}")
# if sector_pie_image_path and os.path.exists(sector_pie_image_path):
#     os.remove(sector_pie_image_path)
#     print(f"Cleaned up temporary image: {sector_pie_image_path}")

# print("\n🎉 Analysis complete. PDF report generated.")






import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from datetime import datetime, timedelta
import numpy as np
import re # Import regex module

# Import ReportLab modules for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.colors import green, red # Import colors

# --- Configuration and File Paths ---
CACHE_FILE = "symbol_sector_cache.json"
IDEAL_CSV_PATH = "user_ideals.csv"
FULL_EVOLUTION_CSV_PATH = "full_trader_evolution.csv"

# --- Load or create symbol-to-sector cache ---
# This cache helps avoid repeated API calls for stock sector information.
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, 'r') as f:
        symbol_sector_cache = json.load(f)
else:
    symbol_sector_cache = {}

# --- Groww API Helpers ---
# Functions to interact with the Groww API to fetch stock details.
def get_search_id(symbol):
    """
    Fetches the search_id for a given stock symbol from Groww's search API.
    This search_id is crucial for subsequent API calls to get detailed stock info.
    """
    url = f"https://groww.in/v1/api/search/v3/query/global/st_query?query={symbol}"
    try:
        r = requests.get(url)
        r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = r.json()
        for item in data['data']['content']:
            # Look for stock entities and match the NSE scrip code
            if item['entity_type'] == 'Stocks' and item.get('nse_scrip_code', '').upper() == symbol.upper():
                return item['search_id']
    except requests.exceptions.RequestException as e:
        print(f"Network or API error finding searchId for {symbol}: {e}")
    except json.JSONDecodeError:
        print(f"JSON decode error for {symbol}'s searchId response.")
    except Exception as e:
        print(f"An unexpected error occurred finding searchId for {symbol}: {e}")
    return None

def get_sector(search_id):
    """
    Fetches the industry (sector) name for a given search_id from Groww's company data API.
    """
    url = f"https://groww.in/v1/api/stocks_data/v1/company/search_id/{search_id}?fields=COMPANY_HEADER"
    try:
        r = requests.get(url)
        r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = r.json()
        # Extract industry name from the header
        return data['header'].get('industryName', 'Unknown')
    except requests.exceptions.RequestException as e:
        print(f"Network or API error getting sector for {search_id}: {e}")
    except json.JSONDecodeError:
        print(f"JSON decode error for {search_id}'s sector response.")
    except Exception as e:
        print(f"An unexpected error occurred getting sector for {search_id}: {e}")
    return "Unknown"

# --- Trader Classification Logic ---
def label_trader(row):
    """
    Classifies a trader's behavior based on their trading statistics.
    'Aggressive': High number of trades, high standard deviation of profit, good average profit.
    'Conservative': Fewer trades, high win rate, decent average profit.
    'Balanced': Falls between aggressive and conservative.
    """
    if row['num_trades'] > 10 and row['std_profit_pct'] is not None and row['std_profit_pct'] > 5 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] > 2.5:
        return 'Aggressive'
    elif row['num_trades'] <= 5 and row['win_rate'] is not None and row['win_rate'] > 0.7 and row['avg_profit_pct'] is not None and row['avg_profit_pct'] >= 1:
        return 'Conservative'
    else:
        return 'Balanced'

# --- Gemini LLM API Call Function ---
def call_gemini_api(prompt_text, max_retries=5, initial_delay=1):
    """
    Calls the Gemini LLM API to generate text based on the provided prompt.
    Implements exponential backoff for robust API calls.
    """
    # Replace "YOUR_API_KEY_HERE" with your actual Gemini API key.
    # IMPORTANT: Hardcoding API keys is not recommended for production environments.
    # Consider using environment variables for better security.
    api_key = "AIzaSyA-djBCrj6HEsFAMCGSdYnYiEpYHoXZHg0"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}]}

    for i in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
                raw_text = result['candidates'][0]['content']['parts'][0]['text']
                # Convert Markdown bold to HTML bold for ReportLab
                # Handles **text** and *text*
                html_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', raw_text)
                html_text = re.sub(r'\*(.*?)\*', r'<b>\1</b>', html_text)
                return html_text
            else:
                return "Could not generate explanation due to unexpected API response structure."
        except requests.exceptions.RequestException as e:
            if response.status_code in [429, 500, 503] and i < max_retries - 1: # Retry on Too Many Requests, Internal Server Error, Service Unavailable
                delay = initial_delay * (2 ** i)
                time.sleep(delay)
            else:
                print(f"LLM API call failed after {i+1} retries: {e}")
                return "Could not generate explanation due to API error."
        except json.JSONDecodeError:
            print(f"LLM API response is not valid JSON. Response: {response.text}")
            return "Could not generate explanation due to invalid API response."
        except Exception as e:
            print(f"An unexpected error occurred during LLM API call: {e}")
            return "Could not generate explanation due to an unexpected error."
    return "Could not generate explanation after multiple retries."

# --- Main Data Processing ---
# Load the trade information and preprocess columns.
try:
    df = pd.read_csv("userbehaviour/trade_info.csv")
    df.columns = df.columns.str.strip() # Clean column names
    df['buy_time'] = pd.to_datetime(df['buy_time'])
    df['sell_time'] = pd.to_datetime(df['sell_time'])
    # Calculate holding time in hours
    df['holding_time_hrs'] = (df['sell_time'] - df['buy_time']).dt.total_seconds() / 3600
except FileNotFoundError:
    print("Error: 'trade_info.csv' not found. Please ensure the file is in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading or processing 'trade_info.csv': {e}")
    exit()

# --- Fill in missing sectors using Groww API and cache ---
unique_symbols = df['symbol'].unique()
print("Starting sector data fetching and caching...")
for symbol in unique_symbols:
    if symbol not in symbol_sector_cache:
        print(f"🔍 Fetching sector for {symbol}...")
        search_id = get_search_id(symbol)
        time.sleep(1) # Be respectful to the API, add a small delay
        if search_id:
            sector = get_sector(search_id)
        else:
            sector = "Unknown"
        symbol_sector_cache[symbol] = sector
        time.sleep(1) # Another small delay after getting sector

# Save updated cache
with open(CACHE_FILE, 'w') as f:
    json.dump(symbol_sector_cache, f, indent=2)
print("Sector cache updated and saved.")

# Add sector column to DataFrame
df['sector'] = df['symbol'].map(symbol_sector_cache)

# --- Trader Evolution Setup ---
# Load or create user ideals CSV
if os.path.exists(IDEAL_CSV_PATH):
    user_ideals_df = pd.read_csv(IDEAL_CSV_PATH)
else:
    user_ideals_df = pd.DataFrame(columns=["user_id", "ideal_type"])

# Define color mapping for trader types for consistent plotting
label_color_map = {
    'Conservative': '#f39c12', # Orange
    'Balanced': '#2ecc71',     # Green
    'Aggressive': '#e74c3c'    # Red
}
# Numerical mapping for y-axis plotting
type_to_num = {'Conservative': 0, 'Balanced': 1, 'Aggressive': 2}

evolution = [] # To store periodic trader statistics
WINDOW_SIZE = 14 # Days for the sliding window analysis

# --- User Input for Analysis ---
user_id = input("Enter the user_id you want to analyze (e.g., U001): ").strip()

if user_id not in df['user_id'].values:
    print(f"User ID '{user_id}' not found in dataset. Exiting.")
    exit()

# Filter trades for the selected user and sort by buy time
user_trades = df[df['user_id'] == user_id].sort_values(by='buy_time')
total_trades = len(user_trades)

# Handle users with very few trades (cannot classify behavior evolution)
if total_trades <= 5:
    print(f"User {user_id} has {total_trades} trades, which is too few for behavior evolution analysis.")
    evolution.append({
        'user_id': user_id,
        'start_time': None, 'end_time': None,
        'num_trades': total_trades, 'avg_profit_pct': None,
        'win_rate': None, 'avg_gain_pct': None,
        'avg_holding_time': None, 'avg_trade_size': None,
        'std_profit_pct': None,
        'trader_type': 'Unclassified'
    })
    plot_evolution = False
else:
    plot_evolution = True
    # Prompt for ideal trader type if not already recorded for this user
    if user_id not in user_ideals_df['user_id'].values:
        print(f"\nEnter ideal trader type for user {user_id} (Conservative / Balanced / Aggressive):")
        while True:
            ideal = input(">> ").strip().capitalize()
            if ideal in ['Conservative', 'Balanced', 'Aggressive']:
                break
            print("Invalid input. Please enter Conservative, Balanced, or Aggressive.")

        # Add new ideal type and save to CSV
        new_row = pd.DataFrame([{'user_id': user_id, 'ideal_type': ideal}])
        user_ideals_df = pd.concat([user_ideals_df, new_row], ignore_index=True)
        user_ideals_df = user_ideals_df.sort_values(by='user_id').reset_index(drop=True)
        user_ideals_df.to_csv(IDEAL_CSV_PATH, index=False)
        print(f"Ideal trader type for {user_id} saved to '{IDEAL_CSV_PATH}'.")
    else:
        print(f"Ideal trader type for {user_id} already recorded.")

    ideal_type = user_ideals_df[user_ideals_df['user_id'] == user_id]['ideal_type'].values[0]
    ideal_num = type_to_num[ideal_type]

    user_trades = user_trades.reset_index(drop=True)
    start_date = user_trades['buy_time'].min()
    end_date = user_trades['buy_time'].max()
    current_start = start_date

    # Sliding window analysis for trader evolution
    while current_start <= end_date:
        current_end = current_start + timedelta(days=WINDOW_SIZE)
        window = user_trades[(user_trades['buy_time'] >= current_start) & (user_trades['buy_time'] < current_end)]

        if len(window) >= 1:
            # Calculate statistics for the current window
            stats = {
                'user_id': user_id,
                'start_time': current_start,
                'end_time': current_end,
                'num_trades': len(window),
                'avg_profit_pct': window['profit_pct'].mean(),
                'win_rate': (window['profit_pct'] > 0).mean(),
                'avg_gain_pct': window[window['profit_pct'] > 0]['profit_pct'].mean() if any(window['profit_pct'] > 0) else 0,
                'avg_holding_time': window['holding_time_hrs'].mean(),
                'avg_trade_size': window['total_buy_value'].mean(),
                'std_profit_pct': window['profit_pct'].std() if len(window) > 1 else 0, # std needs at least 2 points
            }

            stats['trader_type'] = label_trader(stats)
            evolution.append(stats)

        current_start += timedelta(days=7) # Move window by 7 days

evolution_df = pd.DataFrame(evolution)
evolution_df.to_csv(FULL_EVOLUTION_CSV_PATH, index=False)
print(f"Full trader evolution data saved to '{FULL_EVOLUTION_CSV_PATH}'.")

# --- Plot 1: Trader Behavior Evolution (Separate Figure) ---
trader_evolution_explanation = "No evolution data to analyze."
trader_evolution_right = ""
trader_evolution_wrong = ""
trader_evolution_image_path = None
if plot_evolution:
    plot_df = evolution_df[evolution_df['trader_type'] != 'Unclassified'].copy()
    plot_df['label_num'] = plot_df['trader_type'].map(type_to_num)
    user_df_plot = plot_df[plot_df['user_id'] == user_id]

    if not user_df_plot.empty:
        fig_evolution, ax_evolution = plt.subplots(figsize=(10, 7)) # Create a new figure for this plot

        # Plot lines connecting the trader types over time
        for i in range(len(user_df_plot) - 1):
            x_vals = [user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i + 1]['start_time']]
            y_vals = [user_df_plot.iloc[i]['label_num'], user_df_plot.iloc[i + 1]['label_num']]
            label = user_df_plot.iloc[i]['trader_type']
            color = label_color_map.get(label, 'blue')
            ax_evolution.plot(x_vals, y_vals, linestyle='-', color=color, linewidth=2)
            ax_evolution.scatter(user_df_plot.iloc[i]['start_time'], user_df_plot.iloc[i]['label_num'], color=color, s=50, zorder=5)

        # Plot the last point
        last_row = user_df_plot.iloc[-1]
        ax_evolution.scatter(last_row['start_time'], last_row['label_num'],
                        color=label_color_map.get(last_row['trader_type'], 'gray'), s=50, zorder=5)

        # Plot the ideal trader type as a horizontal dashed line
        ax_evolution.plot(user_df_plot['start_time'], [ideal_num] * len(user_df_plot),
                     linestyle=':', color='red', linewidth=2, label=f'Ideal ({ideal_type})')

        ax_evolution.set_title(f"Trader Behavior Evolution - User {user_id}", fontsize=14)
        ax_evolution.set_yticks([0, 1, 2])
        ax_evolution.set_yticklabels(['Conservative', 'Balanced', 'Aggressive'], fontsize=10)
        ax_evolution.set_xlabel("Time", fontsize=12)
        ax_evolution.set_ylabel("Trader Type", fontsize=12)
        ax_evolution.set_ylim(-0.5, 2.5)
        ax_evolution.legend(loc='lower right', fontsize=10)
        ax_evolution.grid(False) # Remove grid for cleaner look
        # Format x-axis dates
        ax_evolution.tick_params(axis='x', rotation=45)
        ax_evolution.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))

        trader_evolution_image_path = f"user_{user_id}_evolution.png"
        fig_evolution.tight_layout() # Ensure tight layout for this specific figure
        fig_evolution.savefig(trader_evolution_image_path)
        plt.close(fig_evolution) # Close the figure to free up memory

        # Prepare prompt for LLM for Trader Evolution
        current_trader_type = last_row['trader_type']
        trader_evolution_prompt = (
            f"Analyze the trading behavior for user {user_id}. "
            f"Their ideal trader type is '{ideal_type}'. "
            f"Their current trading pattern reflects a '{current_trader_type}' trader type. "
            f"Please provide your analysis in the following structured format:\n\n"
            f"<b>Overall Analysis:</b> Explain what it means to have '{ideal_type}' as an ideal trader type and describe what their current '{current_trader_type}' pattern reflects. "
            f"<b>What you are doing right:</b> Provide positive aspects of their current trading behavior. "
            f"<b>What you are doing wrong:</b> Highlight areas for improvement in their current trading behavior. "
            f"Keep each section concise and to the point. Do not include explicit suggestions in this response."
        )
        llm_response = call_gemini_api(trader_evolution_prompt)

        # Parse LLM response
        analysis_match = re.search(r'<b>Overall Analysis:</b>(.*?)<b>What you are doing right:</b>', llm_response, re.DOTALL)
        right_match = re.search(r'<b>What you are doing right:</b>(.*?)<b>What you are doing wrong:</b>', llm_response, re.DOTALL)
        wrong_match = re.search(r'<b>What you are doing wrong:</b>(.*)', llm_response, re.DOTALL)

        trader_evolution_explanation = analysis_match.group(1).strip() if analysis_match else llm_response
        trader_evolution_right = right_match.group(1).strip() if right_match else "N/A"
        trader_evolution_wrong = wrong_match.group(1).strip() if wrong_match else "N/A"

    else:
        trader_evolution_explanation = f"User {user_id} does not have enough classified trading data for behavior evolution analysis."
else:
    trader_evolution_explanation = f"User {user_id} has too few trades ({total_trades}) for behavior evolution analysis."


# --- Plot 2: Sector Distribution Pie Chart (Separate Figure) ---
sector_distribution_explanation = "No sector data to analyze."
sector_distribution_right = ""
sector_distribution_wrong = ""
sector_pie_image_path = None # Initialize to None
user_df_for_pie = df[df['user_id'] == user_id]
if not user_df_for_pie.empty:
    sector_counts = user_df_for_pie['sector'].value_counts()

    if not sector_counts.empty:
        fig_pie, ax_pie = plt.subplots(figsize=(8, 8)) # Create a new figure for the pie chart
        ax_pie.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=140, pctdistance=0.85)
        ax_pie.set_title(f"Sector Distribution - User {user_id}", fontsize=14)
        ax_pie.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

        sector_pie_image_path = f"user_{user_id}_sector_pie.png"
        fig_pie.tight_layout() # Ensure tight layout for this specific figure
        fig_pie.savefig(sector_pie_image_path)
        plt.close(fig_pie) # Close the figure to free up memory

        # Prepare prompt for LLM for Sector Distribution
        sector_info_list = [f"{sector}: {percentage:.1f}%" for sector, percentage in (sector_counts / sector_counts.sum() * 100).items()]
        sector_info_str = ", ".join(sector_info_list)

        # Heuristic to determine spread/concentration
        max_percentage = (sector_counts / sector_counts.sum() * 100).max()
        if max_percentage > 70:
            concentration_comment = "Their portfolio is highly concentrated in a few sectors."
        elif len(sector_counts) > 5 and max_percentage < 40:
            concentration_comment = "Their portfolio is quite diversified across multiple sectors."
        else:
            concentration_comment = "Their portfolio shows a balanced distribution across sectors."

        sector_distribution_prompt = (
            f"Analyze the sector distribution for user {user_id}. "
            f"Their portfolio is distributed as follows: {sector_info_str}. "
            f"{concentration_comment} "
            f"Please provide your analysis in the following structured format:\n\n"
            f"<b>Overall Analysis:</b> Explain the benefits of their current distribution (whether concentrated or diversified). "
            f"<b>What you are doing right:</b> Provide positive aspects of their current sector allocation. "
            f"<b>What you are doing wrong:</b> Highlight areas for improvement in their current sector allocation. "
            f"Keep each section concise and to the point. Do not include explicit suggestions in this response."
        )
        llm_response = call_gemini_api(sector_distribution_prompt)

        analysis_match = re.search(r'<b>Overall Analysis:</b>(.*?)<b>What you are doing right:</b>', llm_response, re.DOTALL)
        right_match = re.search(r'<b>What you are doing right:</b>(.*?)<b>What you are doing wrong:</b>', llm_response, re.DOTALL)
        wrong_match = re.search(r'<b>What you are doing wrong:</b>(.*)', llm_response, re.DOTALL)

        sector_distribution_explanation = analysis_match.group(1).strip() if analysis_match else llm_response
        sector_distribution_right = right_match.group(1).strip() if right_match else "N/A"
        sector_distribution_wrong = wrong_match.group(1).strip() if wrong_match else "N/A"

    else:
        sector_distribution_explanation = f"User {user_id} has no sector data available for analysis."
else:
    sector_distribution_explanation = f"User {user_id} has no trade data available for sector distribution analysis."

# --- Combined Suggestions for LLM ---
# combined_suggestion_prompt = (
#     f"Based on the user's trading behavior (ideal: {ideal_type}, current: {current_trader_type if plot_evolution else 'Unclassified'}) "
#     f"and their sector distribution ({sector_info_str if 'sector_info_str' in locals() else 'N/A'}). "
#     f"Provide a combined set of actionable suggestions for the user to improve their overall trading strategy, "
#     f"considering both their behavior and portfolio diversification. "
#     f"Focus on what changes to make. Be concise and provide bullet points if appropriate."
# )
# combined_suggestions = call_gemini_api(combined_suggestion_prompt)


# --- PDF Report Generation ---
pdf_filename = f"user_{user_id}_report.pdf"
doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
styles = getSampleStyleSheet()

# Define custom styles for green and red text
styles.add(styles['Normal'].clone('GreenText'))
styles['GreenText'].textColor = green
styles.add(styles['Normal'].clone('RedText'))
styles['RedText'].textColor = red

story = []

# Add Trader Behavior Evolution Graph and Explanation
story.append(Paragraph(f"<b>Trader Behavior Evolution - User {user_id}</b>", styles['h1']))
if trader_evolution_image_path and os.path.exists(trader_evolution_image_path):
    img_evolution = Image(trader_evolution_image_path)
    # img_width, img_height = img_evolution.drawWidth, img_evolution.drawHeight
    # aspect_ratio = img_height / img_width
    # max_img_width = 6.5 * inch
    # img_evolution.drawWidth = max_img_width
    # img_evolution.drawHeight = max_img_width * aspect_ratio
    img_evolution.drawWidth = 8.0 * inch  # Example: 4 inches wide
    img_evolution.drawHeight = 4.0 * inch # Example: 4 inches high
    story.append(img_evolution)
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>Overall Analysis:</b>", styles['h3']))
story.append(Paragraph(trader_evolution_explanation.replace('\n', '<br/>'), styles['Normal']))
story.append(Spacer(1, 0.1 * inch))

story.append(Paragraph("<b>What you are doing right:</b>", styles['h3']))
story.append(Paragraph(trader_evolution_right.replace('\n', '<br/>'), styles['GreenText']))
story.append(Spacer(1, 0.1 * inch))

story.append(Paragraph("<b>What you are doing wrong:</b>", styles['h3']))
story.append(Paragraph(trader_evolution_wrong.replace('\n', '<br/>'), styles['RedText']))
story.append(Spacer(1, 0.4 * inch))

# Add Sector Distribution Pie Chart and Explanation
story.append(Paragraph(f"<b>Sector Distribution - User {user_id}</b>", styles['h1']))
# if sector_pie_image_path and os.path.exists(sector_pie_image_path):
#     img_pie = Image(sector_pie_image_path)
#     img_width, img_height = img_pie.drawWidth, img_pie.drawHeight
#     aspect_ratio = img_height / img_width
#     max_img_width = 6.5 * inch
#     img_pie.drawWidth = max_img_width
#     img_pie.drawHeight = max_img_width * aspect_ratio
#     story.append(img_pie)
# story.append(Spacer(1, 0.2 * inch))

if sector_pie_image_path and os.path.exists(sector_pie_image_path):
    img_pie = Image(sector_pie_image_path)
    # Set specific width and height for the pie chart image
    img_pie.drawWidth = 4.0 * inch  # Example: 4 inches wide
    img_pie.drawHeight = 4.0 * inch # Example: 4 inches high
    story.append(img_pie)
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("<b>Overall Analysis:</b>", styles['h3']))
story.append(Paragraph(sector_distribution_explanation.replace('\n', '<br/>'), styles['Normal']))
story.append(Spacer(1, 0.1 * inch))

story.append(Paragraph("<b>What you are doing right:</b>", styles['h3']))
story.append(Paragraph(sector_distribution_right.replace('\n', '<br/>'), styles['GreenText']))
story.append(Spacer(1, 0.1 * inch))

story.append(Paragraph("<b>What you are doing wrong:</b>", styles['h3']))
story.append(Paragraph(sector_distribution_wrong.replace('\n', '<br/>'), styles['RedText']))
story.append(Spacer(1, 0.4 * inch))

# Add Combined Suggestions
story.append(Paragraph("---", styles['Normal'])) # Horizontal line
story.append(Spacer(1, 0.2 * inch))
# story.append(Paragraph("<b>Combined Suggestions: What Changes to Make</b>", styles['h1']))
# story.append(Paragraph(combined_suggestions.replace('\n', '<br/>'), styles['Normal']))
story.append(Spacer(1, 0.4 * inch))

doc.build(story)
print(f"\n✅ PDF report generated: '{pdf_filename}'")

# --- Clean up temporary image files ---
if trader_evolution_image_path and os.path.exists(trader_evolution_image_path):
    os.remove(trader_evolution_image_path)
    print(f"Cleaned up temporary image: {trader_evolution_image_path}")
if sector_pie_image_path and os.path.exists(sector_pie_image_path):
    os.remove(sector_pie_image_path)
    print(f"Cleaned up temporary image: {sector_pie_image_path}")

print("\n🎉 Analysis complete. PDF report generated.")


