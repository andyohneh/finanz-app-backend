# sentiment_analyzer.py (Finale, korrigierte Version)
import finnhub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# KORREKTUR: Wir importieren direkt das 'engine' und die 'daily_sentiment' Tabelle
from database import engine, daily_sentiment
from sqlalchemy.dialects.postgresql import insert

# --- KONFIGURATION ---
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
if not FINNHUB_API_KEY:
    raise ValueError("Bitte trage deinen FINNHUB_API_KEY in die .env Datei ein!")

# --- INITIALISIERUNG ---
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
analyzer = SentimentIntensityAnalyzer()

def analyze_and_store_sentiment(asset: str):
    """
    Holt Nachrichten, analysiert das Sentiment und speichert den Score des letzten Tages
    konsistent mit SQLAlchemy in der Datenbank.
    """
    yesterday = datetime.now() - timedelta(1)
    date_str = yesterday.strftime('%Y-%m-%d')
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    print(f"Sammle Nachrichten für {asset} vom {date_str}...")

    try:
        # Kategorie für Finnhub-API zuordnen
        category = 'crypto' if asset == 'BTC/USD' else 'forex'
        news = finnhub_client.general_news(category, min_id=0)
        
        if not news:
            print(f"Keine Nachrichten für die Kategorie '{category}' gefunden.")
            return

        sentiment_scores = []
        for article in news:
            article_date = datetime.fromtimestamp(article['datetime'])
            if article_date.strftime('%Y-%m-%d') == date_str:
                headline = article['headline']
                score = analyzer.polarity_scores(headline)['compound']
                sentiment_scores.append(score)

        if not sentiment_scores:
            print(f"Keine relevanten Schlagzeilen für {asset} am {date_str} gefunden.")
            avg_score = 0.0
        else:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            print(f"Durchschnittlicher Sentiment-Score für {asset}: {avg_score:.4f}")

        # Score mit SQLAlchemy in der Datenbank speichern/aktualisieren
        with engine.connect() as conn:
            stmt = insert(daily_sentiment).values(
                asset=asset,
                date=date_obj,
                sentiment_score=avg_score
            )
            # Falls für den Tag schon ein Eintrag existiert, wird er aktualisiert.
            stmt = stmt.on_conflict_do_update(
                index_elements=['asset', 'date'],
                set_={'sentiment_score': stmt.excluded.sentiment_score}
            )
            conn.execute(stmt)
            conn.commit()
            print(f"Sentiment-Score für {asset} am {date_str} erfolgreich gespeichert.")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten bei {asset}: {e}")

if __name__ == '__main__':
    # GELÖSCHT: Der Aufruf von create_sentiment_table() ist nicht mehr nötig.
    
    print("=== Starte tägliche Sentiment-Analyse ===")
    
    # KORREKTUR: Wir benutzen die richtigen Symbole mit Schrägstrich
    analyze_and_store_sentiment('BTC/USD')
    analyze_and_store_sentiment('XAU/USD')

    print("=== Sentiment-Analyse abgeschlossen ===")