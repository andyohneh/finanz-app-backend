// static/sw.js

// Name unserer "Vorratskammer" (Cache)
const CACHE_NAME = 'ki-finanz-app-cache-v1';

// Die Dateien, die der Butler beim ersten Mal in die Vorratskammer legen soll
const urlsToCache = [
  '/', // Die Hauptseite
  '/manifest.json', // Der "Personalausweis"
  '/static/icon-192.png', // Das kleine Icon
  '/static/icon-512.png', // Das große Icon
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css', // Die Icons für Pfeile etc.
  'https://cdn.jsdelivr.net/npm/chart.js' // Die Chart-Bibliothek
];

// Event 1: Die "Einstellung" des Butlers (Installation)
// Wird nur einmal ausgeführt, wenn die App das erste Mal gestartet wird.
self.addEventListener('install', event => {
  console.log('Service Worker wird installiert.');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Vorratskammer (Cache) wird gefüllt...');
        return cache.addAll(urlsToCache);
      })
  );
});

// Event 2: Die tägliche Arbeit des Butlers (Fetch)
// Wird jedes Mal ausgeführt, wenn die App eine Datei oder Daten anfordert.
self.addEventListener('fetch', event => {
  event.respondWith(
    // Der Butler schaut zuerst in der Vorratskammer nach...
    caches.match(event.request)
      .then(response => {
        // ... wenn er etwas findet, gibt er es sofort zurück.
        if (response) {
          return response;
        }
        // ... wenn nicht, holt er es aus dem Internet.
        return fetch(event.request);
      }
    )
  );
});