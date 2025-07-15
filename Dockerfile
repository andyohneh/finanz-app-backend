# Schritt 1: Wir starten mit einem sauberen, offiziellen Python-Image
FROM python:3.11-slim

# Schritt 2: DER ENTSCHEIDENDE BEFEHL
# Wir installieren die fehlende System-Bibliothek, bevor wir irgendetwas anderes tun.
RUN apt-get update && apt-get install -y libsqlite3-dev gcc

# Schritt 3: Den Arbeitsbereich im Container einrichten
WORKDIR /app

# Schritt 4: Die Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Schritt 5: Den gesamten Rest deines Codes kopieren
COPY . .

# Schritt 6: Der Standard-Befehl, der ausgeführt wird (kann überschrieben werden)
CMD ["python", "master_controller.py", "predict"]