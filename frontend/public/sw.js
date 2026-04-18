// Bump CACHE_NAME on every rebuild that changes app shell behavior so the
// activate handler purges prior caches. Content-hashed /assets/* are safe
// across versions — this only matters for unhashed files (index.html, sw.js,
// static icons) and for invalidating stale HTML that pins old bundle hashes.
const CACHE_NAME = 'sakhi-v2'
const STATIC_ASSETS = [
  '/manifest.json',
  '/favicon.svg',
]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  )
  self.skipWaiting()
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n)))
    )
  )
  self.clients.claim()
})

self.addEventListener('fetch', (event) => {
  const { request } = event
  if (request.method !== 'GET') return
  if (request.url.includes('/api/')) return

  const url = new URL(request.url)
  // Network-first for HTML navigations so a fresh index.html references the
  // current hashed bundles. Cache-first for everything else (hashed assets,
  // icons) for offline resilience.
  const isNav = request.mode === 'navigate' || url.pathname === '/' || url.pathname.endsWith('.html')

  if (isNav) {
    event.respondWith(
      fetch(request).then((response) => {
        if (response.ok) {
          const clone = response.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone))
        }
        return response
      }).catch(() => caches.match(request))
    )
    return
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      const fetched = fetch(request).then((response) => {
        if (response.ok) {
          const clone = response.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone))
        }
        return response
      }).catch(() => cached)
      return cached || fetched
    })
  )
})
