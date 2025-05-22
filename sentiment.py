import pandas as pd
from transformers import pipeline

news_data = pd.read_csv("appleeee.csv")
news_data['date'] = pd.to_datetime(news_data['date']).dt.date

news_data = news_data[news_data['symbols'].str.contains('AAPL', na=False)]

adj_close = pd.read_csv("aapl_adj_close_3y.csv")
adj_close['Date'] = pd.to_datetime(adj_close['Date'], utc=True).dt.date

merged_data = pd.merge(news_data, adj_close, left_on='date', right_on='Date', how='inner')

merged_data = merged_data.dropna(subset=['content', 'title'])

grouped_data = merged_data.groupby('date').agg({
    'title': list,
    'content': list,
    'Close': 'first'
}).reset_index()

sentiment_pipe = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    framework="pt"
)


def get_sentiment_score(text: str) -> float:
    res = sentiment_pipe(text[:512])[0]
    if res["label"] == "positive":
        return res["score"]
    elif res["label"] == "negative":
        return -res["score"]
    return None



def calculate_average_sentiment(content_list):
    sentiment_scores = []
    for text in content_list:
        score = get_sentiment_score(text)
        if score is not None:
            sentiment_scores.append(score)
    if not sentiment_scores:
        return None
    return sum(sentiment_scores) / len(sentiment_scores)


grouped_data['average_sentiment'] = grouped_data['content'].apply(calculate_average_sentiment)

grouped_data = grouped_data.dropna(subset=['average_sentiment'])

grouped_data.to_csv("output.csv", index=False)


