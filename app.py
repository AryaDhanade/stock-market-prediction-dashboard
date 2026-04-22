import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# -----------------------------
# DOWNLOAD VADER (FIRST TIME ONLY)
# -----------------------------
nltk.download('vader_lexicon')

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📊 AI Stock Market Dashboard")

# -----------------------------
# INIT SENTIMENT
# -----------------------------
sia = SentimentIntensityAnalyzer()

# -----------------------------
# ADD YOUR NEWS API KEY HERE
# -----------------------------
newsapi = NewsApiClient(api_key="a63dbaeb6ca94c90a01b130645109fca")

# -----------------------------
# COMPANY MAPPING
# -----------------------------
companies = {
    "Tesla": "TSLA",
    "Apple": "AAPL",
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}

# -----------------------------
# STOCK INPUT
# -----------------------------
stock = st.text_input("Enter Stock Symbol", "RELIANCE.NS")

# -----------------------------
# STOCK DATA
# -----------------------------
data = yf.download(stock, start="2023-01-01")

if not data.empty:
    st.subheader("📈 Stock Data")
    st.dataframe(data.tail())

    fig, ax = plt.subplots()
    ax.plot(data['Close'])
    ax.set_title(f"{stock} Closing Price")
    st.pyplot(fig)
else:
    st.warning("No stock data found.")

# -----------------------------
# NEWS SECTION
# -----------------------------
st.header("📰 Market News & Sentiment Analysis")

try:
    news = newsapi.get_top_headlines(
        category="business",
        language="en",
        page_size=10
    )
except:
    st.error("Error fetching news.")
    news = {"articles": []}

predictions = []

for article in news.get("articles", []):
    headline = article.get("title", "")

    if not headline:
        continue

    # -----------------------------
    # VADER SENTIMENT
    # -----------------------------
    sentiment = sia.polarity_scores(headline)['compound']

    detected = False

    for name, ticker in companies.items():
        if name.lower() in headline.lower():

            predicted_change = round(sentiment * 2, 2)

            st.write(f"🔹 {headline}")
            st.write(f"→ Stock: {ticker}")
            st.write(f"→ Sentiment: {round(sentiment, 2)}")
            st.write(f"→ Predicted Change: {predicted_change}%")
            st.write("---")

            predictions.append({
                "Stock": ticker,
                "Predicted Change (%)": predicted_change
            })

            detected = True

    if not detected:
        st.write(f"🔹 {headline}")
        st.write("→ No major stock detected")
        st.write("---")

# -----------------------------
# PREDICTION SUMMARY
# -----------------------------
st.header("📊 Prediction Summary")

if predictions:
    df_pred = pd.DataFrame(predictions)
    st.dataframe(df_pred)
else:
    st.write("No predictions available.")

# -----------------------------
# TRENDING STOCKS
# -----------------------------
st.header("🔥 Trending Stocks Today")

stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "AAPL", "TSLA"]

changes = []

for s in stocks:
    df = yf.download(s, period="1d", interval="1m")

    if not df.empty:
        open_price = float(df['Open'].iloc[0])
        close_price = float(df['Close'].iloc[-1])

        change = ((close_price - open_price) / open_price) * 100

        changes.append({
            "Stock": s,
            "Change (%)": round(change, 2)
        })

if changes:
    df_change = pd.DataFrame(changes)

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
    st.write("No trending data available.")
