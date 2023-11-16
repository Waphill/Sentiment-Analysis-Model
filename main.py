from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['MSFT', 'AMZN', 'NVDA']

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
    

parsed_data = []

for ticker, news_table in news_tables.items():
    
    for row in news_table.findAll('tr'):
        
        title = row.a.text
        date_data = row.td.text.strip().split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        
        parsed_data.append([ticker, date, time, title])


# DATA FRAME
df = pd.DataFrame(parsed_data, columns=['Ticker' , 'Date', 'Time', 'Title'])

#df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].apply(lambda x: datetime.now().date() if x == 'Today' else x)

df['Time'] = pd.to_datetime(df['Time'], format='%I:%M%p').dt.time

# Applying Sentiment Analysis
sia = SentimentIntensityAnalyzer()

df['Sentiment'] = df['Title'].apply(lambda x: sia.polarity_scores(x)['compound'])

#Visualization of Sentiment Analysis
plt.figure(figsize=(10,6))
mean_df = df.groupby(['Ticker', 'Date'])['Sentiment'].mean().reset_index()
mean_df = mean_df.pivot(index='Date', columns='Ticker', values='Sentiment')
mean_df.plot(kind='bar')
plt.title('Average Sentiment Score by Ticker')
plt.ylabel('Sentiment Score')
plt.show()

