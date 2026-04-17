/**
 * Offline audio queue using IndexedDB.
 * Stores recorded audio blobs for later processing when connectivity is available.
 */

const DB_NAME = 'sakhi_offline'
const DB_VERSION = 2
const STORE_NAME = 'recordings'
const CHUNKS_STORE = 'chunks'

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' })
      }
      if (!db.objectStoreNames.contains(CHUNKS_STORE)) {
        const store = db.createObjectStore(CHUNKS_STORE, { keyPath: 'id', autoIncrement: true })
        store.createIndex('sessionId', 'sessionId', { unique: false })
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export async function saveRecording(audioBlob, visitType = 'auto', label = '') {
  const db = await openDB()
  const entry = {
    id: Date.now(),
    date: new Date().toLocaleString('en-IN'),
    audioBlob,
    audioType: audioBlob.type,
    size: audioBlob.size,
    visitType,
    label: label || `Recording ${new Date().toLocaleTimeString('en-IN')}`,
    status: 'pending',
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).put(entry)
    tx.oncomplete = () => resolve(entry)
    tx.onerror = () => reject(tx.error)
  })
}

export async function getQueue() {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const req = tx.objectStore(STORE_NAME).getAll()
    req.onsuccess = () => resolve(req.result.sort((a, b) => b.id - a.id))
    req.onerror = () => reject(req.error)
  })
}

export async function getRecording(id) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const req = tx.objectStore(STORE_NAME).get(id)
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

export async function updateRecordingStatus(id, status) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    const req = store.get(id)
    req.onsuccess = () => {
      const entry = req.result
      if (entry) {
        entry.status = status
        store.put(entry)
      }
      tx.oncomplete = () => resolve(entry)
    }
    req.onerror = () => reject(req.error)
  })
}

export async function removeRecording(id) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).delete(id)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function clearQueue() {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    tx.objectStore(STORE_NAME).clear()
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}

export async function appendChunk(sessionId, chunk, visitType = 'auto') {
  const db = await openDB()
  const entry = {
    sessionId,
    blob: chunk,
    blobType: chunk.type,
    size: chunk.size,
    visitType,
    createdAt: Date.now(),
  }
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS_STORE, 'readwrite')
    const req = tx.objectStore(CHUNKS_STORE).add(entry)
    req.onsuccess = () => resolve(req.result)
    tx.onerror = () => reject(tx.error)
  })
}

export async function assembleChunks(sessionId) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS_STORE, 'readonly')
    const index = tx.objectStore(CHUNKS_STORE).index('sessionId')
    const req = index.getAll(sessionId)
    req.onsuccess = () => {
      const rows = req.result || []
      if (!rows.length) { resolve(null); return }
      rows.sort((a, b) => a.id - b.id)
      const type = rows[0].blobType || 'audio/webm'
      const blob = new Blob(rows.map((r) => r.blob), { type })
      resolve({ blob, visitType: rows[0].visitType, chunkCount: rows.length, firstSeen: rows[0].createdAt })
    }
    req.onerror = () => reject(req.error)
  })
}

export async function listOrphanedSessions() {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS_STORE, 'readonly')
    const req = tx.objectStore(CHUNKS_STORE).getAll()
    req.onsuccess = () => {
      const rows = req.result || []
      const bySession = new Map()
      for (const r of rows) {
        const cur = bySession.get(r.sessionId)
        if (!cur) {
          bySession.set(r.sessionId, {
            sessionId: r.sessionId,
            visitType: r.visitType,
            chunkCount: 1,
            totalSize: r.size || 0,
            firstSeen: r.createdAt,
            lastSeen: r.createdAt,
          })
        } else {
          cur.chunkCount += 1
          cur.totalSize += r.size || 0
          if (r.createdAt < cur.firstSeen) cur.firstSeen = r.createdAt
          if (r.createdAt > cur.lastSeen) cur.lastSeen = r.createdAt
        }
      }
      resolve(Array.from(bySession.values()).sort((a, b) => b.firstSeen - a.firstSeen))
    }
    req.onerror = () => reject(req.error)
  })
}

export async function clearChunks(sessionId) {
  const db = await openDB()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(CHUNKS_STORE, 'readwrite')
    const store = tx.objectStore(CHUNKS_STORE)
    const index = store.index('sessionId')
    const req = index.openCursor(IDBKeyRange.only(sessionId))
    req.onsuccess = () => {
      const cursor = req.result
      if (cursor) {
        cursor.delete()
        cursor.continue()
      }
    }
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
  })
}
