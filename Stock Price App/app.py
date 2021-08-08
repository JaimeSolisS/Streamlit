import yfinance as yf
import streamlit as st
import pandas as pd
import datetime



st.write("""
# Simple Stock Price App

Get the stock closing price and volume in 3 simple steps

""")

st.write("""
### 1) Write the ticker symbol you want info from. Here are some examples: 
""")
st.image("https://i.pinimg.com/originals/33/00/6c/33006c981df81e4d61bd8b14536c1b15.jpg")

#define the ticker symbol
#tickerSymbol = 'GOOGL'
tickerSymbol= st.text_input("Ticker Symbol")

#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)

st.write("""
### 2) Select Dates: 
""")
col1, mid, col2 = st.columns([10,1,10])
with col1:
    start = st.date_input("Start Date", datetime.date(2010, 1, 1))
with col2:
    end = st.date_input("End Date", datetime.date.today())

#get the historical prices for this ticker
#tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
tickerDf = tickerData.history(period='1d', start=start, end=end)

st.write("""
### 3) Click Submit: 
""")
submit = st.button("Submit")

if submit:
    st.write("""## Closing Price""")
    st.line_chart(tickerDf.Close)
    st.write("""## Volume Price""")
    st.line_chart(tickerDf.Volume)