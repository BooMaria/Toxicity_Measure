import dash
from dash import dcc, html, Input, Output, State
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
import re 
import nltk
nltk.download('popular')
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('floresta')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
import joblib
import sys  
import demoji
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import csv

demoji.download_codes()

# Define external CSS styles for better appearance
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Twitter Tweet Scraper", style={'textAlign': 'center'}),
    html.Label("Enter Nitter Username:", style={'font-size': '18px', 'font-weight': 'bold', 'margin-bottom': '5px'}),
    dcc.Input(id='username-input', type='text', value='eldiariodedross', style={'width': '100%', 'margin-bottom': '15px', 'padding': '10px', 'border-radius': '10px'}),
    html.Label("Enter Number of Tweets:", style={'font-size': '18px', 'font-weight': 'bold', 'margin-bottom': '5px'}),
    dcc.Input(id='num-tweets-input', type='number', value=100, style={'width': '100%', 'margin-bottom': '15px', 'padding': '10px', 'border-radius': '10px'}),
    html.Button('Scrape Tweets', id='scrape-button', style={'width': '100%', 'font-size': '16px', 'font-weight': 'bold', 'padding': '10px', 'border-radius': '10px'}),
    html.Br(),
    html.Div(id='output-container', style={'margin-top': '20px'}),
    dcc.Interval(id='update-tweets', interval=5000, n_intervals=0),  # Update tweets every 5 seconds
    html.Div(
        id='tweet-scroll',
        style={
            'width': '45%',
            'height': '400px',
            'overflowY': 'scroll',
            'border': '1px solid #ccc',
            'border-radius': '10px',
            'padding': '10px',
            'margin-top': '20px',
            'float': 'left'  # Add this line to float the scrollable div to the left
        },
    ),
    html.Div(id='total-toxicity', style={'font-size': '20px', 'font-weight': 'bold', 'padding': '20px'}),
    dcc.Graph(id='label-counts-graph', style={'width': '45%', 'float': 'left', 'padding': '10px'}),
    
    html.Div(id='results-table', style={'width': '45%', 'float': 'left', 'padding': '10px'})
    
])




def scrape_tweets(url, num_tweets, username):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': 'https://nitter.net/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    all_tweets = []

    while url and len(all_tweets) < num_tweets:
        try:
            response = requests.get(url, headers=headers, timeout=20)  # Wait up to 10 seconds for a response

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                tweet_elements = soup.find_all("div", class_="tweet-content media-body")

                for tweet_element in tweet_elements:
                    tweet_text = tweet_element.get_text(strip=True)

                    # Skip empty tweets
                    if tweet_text:
                        all_tweets.append(tweet_text)

                        if len(all_tweets) >= num_tweets:
                            print(all)
                            break

                # Find the URL for the next page based on the "Load more" button
                load_more_link = soup.select_one("div.show-more a[href^='?cursor=']")
                if load_more_link:
                    next_page_url = load_more_link["href"]
                    url = f"https://nitter.net/{username}{next_page_url}"
                else:
                    url = None  # No more pages

                # Add a delay to avoid rate limiting
                time.sleep(1)
        except Exception as e:
            print(f"Error: {str(e)}")

    return all_tweets

def scrape_tweets_from_file(num_tweets):
    with open('tweets.txt', 'r', encoding='utf-8') as file:
        all_tweets = [line.strip() for line in file.readlines()][:num_tweets]
    return all_tweets

def remove_emojis(text):
    demoji.download_codes()  # Download the emoji codes (run this once)

    text_without_emojis = demoji.replace(text, '')
    return text_without_emojis


def clean_text(text):
  text = ' '.join([ word for word in text.split(' ') if not word.startswith('@') ])
  text = re.sub(r"[^A-Za-z ]+", '', text) # keep only letters and spaces
  text = text.strip()
  return text

def remove_stop_words(text):
  stopwords_pt = stopwords.words('spanish')
  text_without_sw = [word for word in text.split(' ') if not word in stopwords_pt]
  return (" ").join(text_without_sw)

def lemmatization_nltk(text):
  lemmatizer = WordNetLemmatizer()
  words = nltk.word_tokenize(text, language='spanish')
  lemmas = [lemmatizer.lemmatize(p).lower() for p in words]
  return (" ").join(lemmas)


def remove_proper_nouns(text):
    words = []
    for word, tag in pos_tag(word_tokenize(text)):
        if tag != 'NNP' and tag != 'NNPS':  # Check for proper nouns tags
            words.append(word)
    return ' '.join(words)


def remove_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = r'https?://\S+|www\.\S+'

    # Remove URLs from the text using the pattern
    text_without_urls = re.sub(url_pattern, '', text)

    return text_without_urls

def normalize_text(text):
  text = remove_urls(text)
  text = remove_emojis(text)
  text = clean_text(text)
  text = remove_proper_nouns(text)
  text = text.lower() # outside clean_text because capitalization influences remove_proper_nouns function
  text = remove_stop_words(text)
  text = lemmatization_nltk(text)

  return text
def process_tweets_with_bert(tweets):

    tokenizer = Tokenizer()
    # cálculo de toxicidade 
    tweets_normalized = [normalize_text(tweet) for tweet in tweets]
    tweets_tokens = pad_sequences(tokenizer.texts_to_sequences(tweets_normalized), maxlen = 30)  
    #print(tweets_normalized)
    model_path = "bert_spanish_5/model.pth"
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    tweets_normalized = [normalize_text(tweet) for tweet in tweets]
    encoded_inputs = tokenizer(tweets_normalized, padding='longest', truncation=True, return_tensors='pt')
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    probs = torch.sigmoid(logits).cpu().numpy()

    toxicity_ranges = {
        0: (0, 25),    # label 0: Non-Offensive
        1: (25, 50),   # label 1: Non-Offensive but with Expletive Language
        2: (50, 100),  # label 2: Offensive Targeting a Group
        3: (50, 100)   # label 3: Offensive Targeting an Individual
    }

    toxicity_percentages = []

    for prob in probs:
        top_indices = np.argsort(prob)[-2:]
        label = np.argmax(prob)
        toxicity_range = toxicity_ranges[label]
        mapped_probability = np.interp(prob[label], [0, 1], toxicity_range)
        mapped_probability_next = np.interp(prob[top_indices[0]], [0, 1], (0,10))
        toxicity_percentage = (mapped_probability - mapped_probability_next)
        toxicity_percentages.append(toxicity_percentage)

    total_toxicity_score = np.mean(toxicity_percentages)
    print("Total Toxicity Percentage BERT multi: ", total_toxicity_score)
    print("----------------------------------------------------\n")

    results = []
    for i, tweet in enumerate(tweets):
        label = np.argmax(probs[i])
        result = {
            'tweet': tweet,
            'toxicity_percentage': toxicity_percentages[i],
            'label': label
        }
        results.append(result)

        print(f"Tweet: {tweet}")
        print(f"Predicted Label: {label}")
        print(f"Toxicity Percentages: {toxicity_percentages[i]}%\n")
        print("----------------------------------------------------")

    return results

def create_scrollable_div(content):
    return html.Div(
        style={
            'maxWidth': '500px',  # Set a maximum width for the scrollable div
            'overflowX': 'auto',  # Enable horizontal scrolling
            'whiteSpace': 'nowrap',  # Prevent line breaks
            'border': '1px solid #ccc',
            'padding': '10px',
            'border-radius': '10px'
        },
        children=content
    )

def calculate_total_toxicity(results):
    total_percentage = sum(result['toxicity_percentage'] for result in results) / len(results)
    return total_percentage

@app.callback(
    [Output('tweet-scroll', 'children'), 
     Output('results-table', 'children'),
     Output('label-counts-graph', 'figure'),
     Output('total-toxicity', 'children')],
    Input('scrape-button', 'n_clicks'),
    State('username-input', 'value'),
    State('num-tweets-input', 'value')
)
def update_tweet_scroller(n_clicks, username, num_tweets):
    if n_clicks:
        try:
            # Scrape tweets from Nitter
            url = f"https://nitter.net/{username}"
            print(url)
            tweets = scrape_tweets(url, num_tweets, username)
            # Load tweets from file
            #tweets = scrape_tweets_from_file(num_tweets)
            #print(tweets)
            if tweets: 
                results = process_tweets_with_bert(tweets)
                total_percentage = calculate_total_toxicity(results)


                tweet_display = html.Div([
                    html.H3(f"Tweets from {username}"),
                    *[html.P(tweet) for tweet in tweets]
                ])
                table = html.Table([
                    html.Tr([html.Th('Index'), html.Th('Tweet'), html.Th('Toxicity Percentage'), html.Th('Label')]),
                    *[html.Tr([
                        html.Td(i+1), 
                        html.Td(create_scrollable_div(result['tweet'])), 
                        html.Td(f'{result["toxicity_percentage"]:.2f}%'), 
                        html.Td(result['label'])]) for i, result in enumerate(results)]
                ])

                # Calculate label counts
                label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
                for result in results:
                    label_counts[result['label']] += 1

                # Create a bar chart
                fig = {
                    'data': [
                        {
                            'x': list(label_counts.keys()),
                            'y': list(label_counts.values()),
                            'type': 'bar',
                            'name': 'Label Counts',
                            'marker': {
                                'color': ['blue', 'orange', 'green', 'red']  # Set your desired colors here
                            }
                        }
                    ],
                    'layout': {
                        'title': 'Label Counts',
                        'xaxis': {'title': 'Label'},
                        'yaxis': {'title': 'Count'}
                    }
                }

                return tweet_display, table, fig, f'Total Toxicity: {total_percentage:.2f}%'
            else:
                return None, None, {}
        except Exception as e:
            return html.P(f"Error: {str(e)}")

if __name__ == '__main__':
    app.run_server(debug=True)
