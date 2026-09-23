const CACHE="odyssey-shell-v2";
const ASSETS=["./","./index.html","./runtime_bootstrap.json"];
self.addEventListener("install",event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(ASSETS)).then(()=>self.skipWaiting())));
self.addEventListener("activate",event=>event.waitUntil(self.clients.claim()));
self.addEventListener("fetch",event=>{if(new URL(event.request.url).origin===location.origin)event.respondWith(caches.match(event.request).then(hit=>hit||fetch(event.request)));});
