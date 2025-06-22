// static/sw.js (Version mit "Stale-While-Revalidate" Caching-Strategie)

const CACHE_NAME = 'ki-finanz-app-cache-v2'; // WICHTIG: Neue Versionsnummer!
const urlsToCache = [
  '/',
  '/dashboard',
  '/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

// Installation: Die statischen Haupt-Assets werden in den Cache gelegt.
self.addEventListener('install', event => {
  console.log('Service Worker: Installiere neue Version...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('Cache geöffnet, füge Kern-Assets hinzu.');
        return cache.addAll(urlsToCache);
      })
  );
  self.skipWaiting();
});

// Aktivierung: Alte Caches werden aufgeräumt.
self.addEventListener('activate', event => {
  console.log('Service Worker: Aktiviere neue Version...');
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            console.log('Service Worker: Lösche alten Cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  return self.clients.claim();
});


// Fetch: Hier passiert die "Stale-While-Revalidate"-Magie.
self.addEventListener('fetch', event => {
  // Wir wenden diese Strategie nur auf GET-Requests an.
  if (event.request.method !== 'GET') {
    return;
  }

  // Für API-Daten und Chart-Daten wollen wir immer die neuesten Daten (Network First).
  if (event.request.url.includes('/api/') || event.request.url.includes('/historical-data/')) {
    event.respondWith(
      fetch(event.request).catch(() => {
        // Optional: Hier könnte man eine Offline-JSON-Antwort zurückgeben
        console.log('API-Anfrage fehlgeschlagen (offline).');
      })
    );
    return;
  }
  
  // Für alle anderen Anfragen (Seiten, CSS, JS, Bilder) nutzen wir Stale-While-Revalidate.
  event.respondWith(
    caches.open(CACHE_NAME).then(cache => {
      return cache.match(event.request).then(response => {
        // 1. Gib die gecachte Antwort SOFORT zurück, falls vorhanden.
        const fetchPromise = fetch(event.request).then(networkResponse => {
          // 2. Gleichzeitig: Hole die neue Version vom Server.
          // 3. Aktualisiere den Cache mit der neuen Version für das nächste Mal.
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });

        // Gib die gecachte Version zurück, während im Hintergrund das Update läuft.
        return response || fetchPromise;
      });
    })
  );
});