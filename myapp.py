import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class StockDataAnalyzer:
    def __init__(self, stock_codes, period='1y', rsi_period=14):
        self.stock_codes = stock_codes
        self.period = period
        self.rsi_period = rsi_period
        self.stock_data_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()

    def fetch_stock_data(self):
        all_stock_data = []
        
        for stock_code in self.stock_codes:
            try:
                yahoo_ticker = f"{stock_code}.NS"
                stock = yf.Ticker(yahoo_ticker)
                data = stock.history(period=self.period)
                
                if not data.empty:
                    stock_data = {
                        'Date': data.index.tolist(),
                        'Ticker': [stock_code] * len(data),
                        'Open': data['Open'].tolist(),
                        'High': data['High'].tolist(),
                        'Low': data['Low'].tolist(),
                        'Close': data['Close'].tolist(),
                        'Volume': data['Volume'].tolist(),
                        'Sector': [stock.info.get('industry', 'N/A')] * len(data)
                    }
                    all_stock_data.append(pd.DataFrame(stock_data))
                else:
                    print(f"No data available for {stock_code}")
            except Exception as e:
                print(f"Error fetching data for {stock_code}: {e}")

        if all_stock_data:
            self.stock_data_df = pd.concat(all_stock_data, ignore_index=True)
        else:
            print("No stock data fetched.")

    def calculate_rsi(self, data, column='Close'):
        delta = data[column].diff(1)
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        avg_gains = gains.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_losses = losses.rolling(window=self.rsi_period, min_periods=1).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_summary_statistics(self):
        if self.stock_data_df.empty:
            print("Stock data is empty. Please fetch data first.")
            return
        ticker_grouped = self.stock_data_df.groupby('Ticker')
        agg_functions = {
            'Close': ['min', 'max', 'median', 'mean', 'var', 'std']
        }
        self.summary_df = ticker_grouped.agg(agg_functions).reset_index()
        self.summary_df.columns = ['{}_{}'.format(col[0], col[1]) for col in self.summary_df.columns]
        self.summary_df = self.summary_df.rename(columns={'Ticker_': 'Ticker'})

    def merge_summary(self):
        if self.stock_data_df.empty or self.summary_df.empty:
            print("Stock data or summary is empty. Please fetch data and calculate summary first.")
            return
        merged_df = pd.merge(self.stock_data_df, self.summary_df, on='Ticker', how='left')
        merged_df['Close_z_score'] = (merged_df['Close'] - merged_df['Close_mean']) / merged_df['Close_std']
        merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
        merged_df['Date'] = pd.to_datetime(merged_df['Date'])

        conditions = [
            (merged_df['Close_z_score'] <= 1) & (merged_df['Close_z_score'] >= -1),
            (merged_df['Close_z_score'] <= 2) & (merged_df['Close_z_score'] >= -2),
            (merged_df['Close_z_score'] <= 3) & (merged_df['Close_z_score'] >= -3),
            (merged_df['Close_z_score'] > 3) | (merged_df['Close_z_score'] < -3),
        ]

        values = ['Within_1', 'Within_2', 'Within_3', 'More_3']
        merged_df['ZScore_Category'] = np.select(conditions, values, default='Other')
        merged_df['RSI'] = self.calculate_rsi(merged_df)
        self.merged_df = merged_df.sort_values(by=['Ticker', 'Date'], ascending=[True, False]).reset_index(drop=True)

    def get_latest_data(self):
        if self.merged_df.empty:
            print("Merged DataFrame does not exist. Please run the merge_summary method first.")
            return
        max_date_indices = self.merged_df.groupby('Ticker')['Date'].idxmax()
        subset_df = self.merged_df.loc[max_date_indices].reset_index(drop=True)
        return subset_df

    def analyze(self):
        self.fetch_stock_data()
        self.calculate_summary_statistics()
        self.merge_summary()
        latest_data = self.get_latest_data()
        return latest_data

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

async def greet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Welcome! Use "Update stock" to analyze stocks: EDELWEISS, JIOFIN, SYNCOMF, GREENPOWER, SALASAR, COMFINTE, MISHTANN.')

async def analyze_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # User-defined parameters
    stock_codes_list = ['EDELWEISS', 'JIOFIN', 'SYNCOMF', 'GREENPOWER', 'SALASAR', 'COMFINTE', 'MISHTANN']
    period = '1y'  # Change the duration as needed
    rsi_period = 14  # You can change this value as needed

    # Create an instance of StockDataAnalyzer
    analyzer = StockDataAnalyzer(stock_codes_list, period=period, rsi_period=rsi_period)

    # Perform analysis
    latest_stock_data = analyzer.analyze()
    latest_stock_data = pd.DataFrame(latest_stock_data[['Date', 'Ticker', 'Open', 'Close', 'Volume','Close_min', 'Close_max', 'Close_median', 'Close_mean', 'Close_var','Close_std', 'Close_z_score', 'ZScore_Category', 'RSI']])

    if latest_stock_data is not None:
        # Convert the DataFrame to an image
        plt.figure(figsize=(22, 12))
        plt.axis('tight')
        plt.axis('off')
        the_table = plt.table(cellText=latest_stock_data.values, colLabels=latest_stock_data.columns, cellLoc='center', loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(8)
        the_table.scale(1.2, 1.2)

        # Save the table as an image
        image_path = 'latest_stock_data.png'
        plt.savefig(image_path, bbox_inches='tight', dpi=300)
        plt.close()

        # Send the image to Telegram
        with open(image_path, 'rb') as img:
            await update.message.reply_photo(photo=img)

        # Clean up the image file
        os.remove(image_path)

def main():
    # Replace 'YOUR_BOT_TOKEN' with your actual bot token
    bot_token = "6434478840:AAFp5aWKlmJD_haYuXPh9s1YITCt5B8yPLg"
    application = ApplicationBuilder().token(bot_token).build()
    
    # Handlers for greeting and updating stocks
    application.add_handler(CommandHandler("start", greet))  # Use CommandHandler for /start
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_stocks))  # Handle "analyze"

    application.run_polling()

if __name__ == '__main__':
    main()
