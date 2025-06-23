// static/sw.js (Minimalistischer Postbote - NUR für Push-Nachrichten)

console.log('Service Worker (Postbote) Skript geladen.');

// Bei der Installation warten wir nicht, wir sind sofort bereit.
self.addEventListener('install', event => {
  console.log('Postbote: Beginne meinen Dienst.');
  self.skipWaiting();
});

// Bei der Aktivierung übernehmen wir die Kontrolle über die Seite.
self.addEventListener('activate', event => {
  console.log('Postbote: Bin jetzt aktiv und bereit für Post.');
  event.waitUntil(self.clients.claim());
});

// WICHTIG: Der Fetch-Handler ist leer. Er greift NICHT in das Caching ein.
// Deine Seite wird immer direkt vom Server geladen.
self.addEventListener('fetch', event => {
  return; 
});

// HIER kommt die Logik für den Empfang von Push-Nachrichten von Firebase.
self.addEventListener('push', event => {
  console.log('Postbote: Push-Nachricht empfangen!', event.data.text());
  
  const pushData = JSON.parse(event.data.text());

  const title = pushData.title || 'Neues KI-Signal!';
  const options = {
    body: pushData.body,
    icon: '/static/icon-192.png',
    badge: '/static/icon-192.png' // Icon für die Benachrichtigungsleiste
  };

  event.waitUntil(self.registration.showNotification(title, options));
});