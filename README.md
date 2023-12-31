﻿# Sentiment-Analysis-Model
This Python script performs sentiment analysis on financial news headlines for a given set of stock tickers. It utilizes web scraping to extract news data from Finviz, a financial news website, and then applies sentiment analysis using the VADER sentiment intensity analyzer from the Natural Language Toolkit (NLTK). The results are visualized using Matplotlib.

**Prerequisites**

Ensure you have the following libraries installed:

- BeautifulSoup (bs4)
- NLTK (Natural Language Toolkit)
- Pandas
- Matplotlib

You can install these dependencies using the following:
    pip install beautifulsoup4 nltk pandas matplotlib

**Usage**
1. Specify the stock tickers of interest in the tickers list.
2. Run the script.

**Code Overview**
- The script starts by defining the Finviz URL and the list of stock tickers.
- It then fetches and parses the HTML content of the Finviz page for each stock ticker to extract relevant news data.
- The extracted news data is organized into a Pandas DataFrame, including columns for Ticker, Date, Time, and Title.
- Date and Time columns are formatted appropriately.
- Sentiment analysis is applied to the news headlines using the VADER sentiment intensity analyzer.
- Finally, the average sentiment scores for each stock ticker are visualized using Matplotlib in a bar chart.

**Note:**

The script assumes that the financial news headlines are structured in a specific way on the Finviz website. Any changes to the website structure may require modifications to the code.

Feel free to customize the script for additional features or adapt it to different financial news websites.
