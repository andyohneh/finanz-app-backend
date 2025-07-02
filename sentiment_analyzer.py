# sentiment_analyzer.py (Platin-Standard: Analysiert die Stimmung von Nachrichten)
import os
import requests
from dotenv import load_dotenv
from transformers import pipeline # Das Herzstück unserer neuen KI

# --- KONFIGURATION ---
load_dotenv()
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')

# Lade das vortrainierte deutsche Sentiment-Modell
# Das Modell wird beim ersten Mal automatisch heruntergeladen.
print("Lade Sentiment-Analyse-Modell (kann beim ersten Mal dauern)...")
sentiment_pipeline = pipeline("sentiment-analysis", model="oliverguhr/german-sentiment-bert")
print("Modell geladen.")


def fetch_news(keywords: str):
    """Holt Nachrichten-Schlagzeilen zu bestimmten Schlüsselwörtern."""
    print(f"\n--- Suche nach Nachrichten für: {keywords} ---")
    url = (f"https://newsapi.org/v2/everything?"
           f"q=({keywords})&"
           f"sortBy=popularity&"
           f"language=de&"
           f"pageSize=20&" # Wir limitieren auf die 20 relevantesten Nachrichten
           f"apiKey={NEWSAPI_KEY}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        headlines = [article['title'] for article in data.get('articles', [])]
        print(f"{len(headlines)} Schlagzeilen gefunden.")
        return headlines
    except Exception as e:
        print(f"Fehler beim Abrufen der Nachrichten: {e}")
        return []

def analyze_sentiment(headlines: list):
    """Analysiert eine Liste von Schlagzeilen und gibt die Stimmung zurück."""
    if not headlines:
        return None
    
    print("Analysiere Stimmung der Schlagzeilen...")
    sentiments = sentiment_pipeline(headlines)
    
    # Wir berechnen einen durchschnittlichen "Sentiment Score"
    # Positive Scores sind gut, negative schlecht.
    total_score = 0
    for s in sentiments:
        if s['label'] == 'positive':
            total_score += s['score']
        elif s['label'] == 'negative':
            total_score -= s['score']
    
    # Durchschnitt berechnen und auf 4 Nachkommastellen runden
    average_score = round(total_score / len(sentiments), 4) if headlines else 0
    
    print(f"Durchschnittlicher Sentiment-Score: {average_score}")
    return average_score

if __name__ == '__main__':
    # Bitcoin-Analyse
    btc_headlines = fetch_news("Bitcoin OR BTC")
    btc_sentiment = analyze_sentiment(btc_headlines)
    
    # Gold-Analyse
    gold_headlines = fetch_news("Goldpreis OR XAU")
    gold_sentiment = analyze_sentiment(gold_headlines)
    
    print("\n======================================")
    print("      FINALE STIMMUNGS-ANALYSE      ")
    print("======================================")
    print(f"Aktueller Sentiment-Score für Bitcoin: {btc_sentiment}")
    print(f"Aktueller Sentiment-Score für Gold:    {gold_sentiment}")
    print("======================================")