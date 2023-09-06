import dash
from dash import dcc, html, Input, Output, State
import requests
from bs4 import BeautifulSoup
import time

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
    html.Div(id='output-container', style={'margin-top': '20px'})
])


def scrape_tweets(url, num_tweets, username):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Referer': 'https://nitter.net/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    all_tweets = []

    while url and len(all_tweets) < num_tweets:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            tweet_elements = soup.find_all("div", class_="tweet-content media-body")

            for tweet_element in tweet_elements:
                tweet_text = tweet_element.get_text(strip=True)

                # Skip empty tweets
                if tweet_text:
                    all_tweets.append(tweet_text)

                    if len(all_tweets) >= num_tweets:
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

    return all_tweets

@app.callback(
    Output('output-container', 'children'),
    Input('scrape-button', 'n_clicks'),
    State('username-input', 'value'),
    State('num-tweets-input', 'value')
)
def scrape_tweets_and_display_data(n_clicks, username, num_tweets):
    if n_clicks:
        try:
            # Scrape tweets from Nitter
            url = f"https://nitter.net/{username}"
            tweets = scrape_tweets(url, num_tweets, username)

            # Display the first 10 tweets
            if tweets:
                first_10_tweets = tweets[:10]
                return html.Div([
                    html.H3(f"First 10 Tweets from @{username}:"),
                    html.Ul([html.Li(tweet) for tweet in first_10_tweets])
                ])
            else:
                return html.Div("No tweets found for this user.")
        except Exception as e:
            return html.Div(f"Error: {str(e)}")

if __name__ == '__main__':
    app.run_server(debug=True)
