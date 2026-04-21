import nltk
nltk.download('vader_lexicon')

import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# CONFIG

st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📊 AI Stock Market Dashboard")


# STOCK DATA

stock = st.text_input("Enter Stock Symbol", "RELIANCE.NS")

data = yf.download(stock, start="2023-01-01")

st.write(data.tail())

fig, ax = plt.subplots()
ax.plot(data['Close'])
st.pyplot(fig)


# SENTIMENT + COMPANY SETUP

sia = SentimentIntensityAnalyzer()

companies = {
    "Tesla": "TSLA",
    "Apple": "AAPL",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}


# NEWS API

import os
newsapi = NewsApiClient(api_key=os.getenv("a63dbaeb6ca94c90a01b130645109fca"))

news = newsapi.get_top_headlines(
    category="business",
    language="en",
    page_size=10
)


# NEWS + PREDICTION

st.header("📰 Market News & Impact Analysis")

predictions = []
st.write("DEBUG: news loaded")

for article in news['articles']:
    headline = article['title']
    
    sentiment = sia.polarity_scores(headline)['compound']
    detected = False
    
    for name, ticker in companies.items():
        if name.lower() in headline.lower():
            predicted_change = round(sentiment * 2, 2)

            st.write(f"🔹 {headline}")
            st.write(f"   → Stock: {ticker}")
            st.write(f"   → Sentiment: {round(sentiment,2)}")

            if predicted_change > 0:
                st.success(f"   → Predicted Change: +{predicted_change}%")
            else:
                st.error(f"   → Predicted Change: {predicted_change}%")

            st.write("---")

            predictions.append((ticker, predicted_change))
            detected = True

    if not detected:
        st.write(f"🔹 {headline}")
        st.write("   → No major stock detected")
        st.write("---")


# PREDICTION TABLE

st.header("📊 Prediction Summary")

if predictions:
    df_pred = pd.DataFrame(predictions, columns=["Stock", "Predicted Change (%)"])
    st.dataframe(df_pred)
else:
    st.write("No predictions available.")


# TRENDING STOCKS

st.header("🔥 Trending Stocks Today")

stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","AAPL","TSLA"]

changes = []

for stock in stocks:
    df = yf.download(stock, period="1d", interval="1m")

    try:
        if not df.empty and 'Open' in df.columns and 'Close' in df.columns:
            
            open_price = float(df['Open'].iloc[0])
            close_price = float(df['Close'].iloc[-1])

            change = ((close_price - open_price) / open_price) * 100

            changes.append({
                "Stock": stock,
                "Change (%)": round(change, 2)
            })

    except Exception as e:
        st.write(f"Error in {stock}: {e}")

# Convert to DataFrame safely
df_change = pd.DataFrame(changes)

# 🔥 IMPORTANT: Check before sorting
if not df_change.empty:
    df_change["Change (%)"] = pd.to_numeric(df_change["Change (%)"], errors='coerce')

    gainers = df_change.sort_values(by="Change (%)", ascending=False).head(3)
    losers = df_change.sort_values(by="Change (%)", ascending=True).head(3)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Top Gainers")
        st.dataframe(gainers)

    with col2:
        st.subheader("📉 Top Losers")
        st.dataframe(losers)

else:
    st.write("No stock data available.")

for stock, change in predictions:
    if change > 0:
        st.success(f"{stock}: +{change}%")
    else:
        st.error(f"{stock}: {change}%")
