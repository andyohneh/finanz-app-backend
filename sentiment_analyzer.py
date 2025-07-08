# backend/sentiment_analyzer.py (Finale Version mit Finnhub & Vader)
import finnhub
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert

# Eigene Module importieren
from database import engine, daily_sentiment

# --- KONFIGURATION ---
load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Sicherheits-Check, ob der API-Schlüssel vorhanden ist
if not FINNHUB_API_KEY:
    raise ValueError("Bitte trage deinen FINNHUB_API_KEY in die .env Datei ein!")

# Die Assets, für die wir das Sentiment analysieren wollen
ASSETS_TO_ANALYZE = ['BTC/USD', 'XAU/USD']

# --- INITIALISIERUNG ---
# Wir erstellen die "Clients" nur einmal, das ist effizienter
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
analyzer = SentimentIntensityAnalyzer()

def analyze_and_store_sentiment(asset: str):
    """
    Holt Nachrichten für ein Asset, analysiert das Sentiment der Schlagzeilen
    und speichert den Durchschnitts-Score des letzten Tages in der Datenbank.
    """
    # Wir nehmen immer die Nachrichten von gestern, um einen vollen Tag zu haben
    yesterday = datetime.now() - timedelta(1)
    date_str = yesterday.strftime('%Y-%m-%d')
    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    print(f"Sammle Nachrichten für {asset} vom {date_str}...")

    try:
        # Kategorie für Finnhub-API zuordnen
        # Für BTC nehmen wir 'crypto', für Gold ('forex') nehmen wir allgemeine Nachrichten
        category = 'crypto' if 'BTC' in asset else 'general'
        # Wir holen die 50 neuesten Schlagzeilen aus der Kategorie
        news = finnhub_client.general_news(category, min_id=0)[:50] 
        
        if not news:
            print(f"Keine relevanten Schlagzeilen für {asset} am {date_str} gefunden.")
            # Wenn keine Nachrichten da sind, nehmen wir einen neutralen Score von 0.0
            avg_score = 0.0
        else:
            # Berechne den Sentiment-Score ('compound') für jede Schlagzeile
            scores = [analyzer.polarity_scores(article['headline'])['compound'] for article in news]
            # Bilde den Durchschnitt aller Scores
            avg_score = sum(scores) / len(scores)
            print(f"Durchschnittlicher Sentiment-Score für {asset}: {avg_score:.4f}")

        # Score mit SQLAlchemy in der Datenbank speichern/aktualisieren (Upsert-Logik)
        with engine.connect() as conn:
            stmt = insert(daily_sentiment).values(
                asset=asset,
                date=date_obj,
                sentiment_score=avg_score
            )
            # Falls für den Tag und das Asset schon ein Eintrag existiert, wird er aktualisiert.
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
    print("=== Starte tägliche Sentiment-Analyse für alle Assets ===")
    for asset in ASSETS_TO_ANALYZE:
        analyze_and_store_sentiment(asset)
    print("\n=== Sentiment-Analyse abgeschlossen ===")