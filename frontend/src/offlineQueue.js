/**
 * Offline audio queue using IndexedDB.
 * Stores recorded audio blobs for later processing when connectivity is available.
 */

const DB_NAME = 'sakhi_offline'
const DB_VERSION = 1
const STORE_NAME = 'recordings'

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' })
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
