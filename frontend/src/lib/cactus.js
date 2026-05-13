// Capacitor plugin facade for the Cactus on-device inference SDK.
// The Kotlin-side plugin (CactusPlugin.kt) is wired in Saturday H4 per the plan.
// This JS side can be imported safely in a browser / PWA build — it just
// returns { available: false } when the plugin isn't registered.

import { Capacitor, registerPlugin } from '@capacitor/core'

// registerPlugin returns a proxy that forwards method calls to the native
// implementation if available. On web, all methods reject with UNIMPLEMENTED.
const CactusNative = registerPlugin('Cactus')

let _handle = null
let _initPromise = null

// Browser-mode simulator state. None of these are touched on Android — the
// native plugin owns model state there.
const isBrowserSim = () => Capacitor.getPlatform() !== 'android'
let _simHasModel = false
let _simLoaded = false

/**
 * Quick availability check. Returns immediately without touching native code
 * on platforms where the plugin isn't registered.
 */
export async function isAvailable() {
  if (isBrowserSim()) {
    return {
      available: true,
      handle: _simLoaded ? 9999 : 0,
      modelPath: _simHasModel ? '/sim/files/models/gemma-4-e2b-it-int4' : '',
      modelPresent: _simHasModel,
      modelFound: _simHasModel ? '/sim/files/models/gemma-4-e2b-it-int4' : undefined,
      loaded: _simLoaded,
      simulated: true,
    }
  }
  try {
    const res = await CactusNative.isAvailable()
    return { available: true, ...res }
  } catch (err) {
    return { available: false, reason: 'plugin-not-registered', error: String(err) }
  }
}

/**
 * Lazy init — reuses handle across calls.
 * @param {{ modelPath?: string; contextSize?: number }} opts
 */
export async function init(opts = {}) {
  if (isBrowserSim()) {
    if (!_simHasModel) throw new Error('No model file found. Run Import model first (simulator).')
    if (_simLoaded) return { handle: 9999, cached: true, modelPath: '/sim/files/models/gemma-4-e2b-it-int4' }
    await sleep(900) // pretend Cactus loaded ~1 s
    _simLoaded = true
    return { handle: 9999, cached: false, modelPath: '/sim/files/models/gemma-4-e2b-it-int4', initMs: 900 }
  }
  if (_handle != null) return { handle: _handle, cached: true }
  if (_initPromise) return _initPromise
  _initPromise = CactusNative.init(opts).then(
    (res) => {
      _handle = res.handle
      _initPromise = null
      return res
    },
    (err) => {
      _initPromise = null
      throw err
    }
  )
  return _initPromise
}

/**
 * Run text completion. All Cactus I/O is JSON strings at the C level;
 * the Kotlin plugin takes structured inputs and serializes them before
 * calling the native bridge.
 *
 * @param {{
 *   messages: Array<{role: string, content: string}>,
 *   tools?: object[],
 *   options?: { max_tokens?: number, temperature?: number, top_p?: number }
 * }} req
 * @returns {Promise<{ text: string, toolCalls?: object[], tokensPerSec?: number, elapsedMs?: number }>}
 */
export async function complete(req) {
  if (isBrowserSim()) {
    if (!_simLoaded) throw new Error('model not initialized — call init() first')
    await sleep(600)
    // Echo a canned Hindi response so Test Hindi shows something visible.
    const userMsg = (req?.messages || []).filter((m) => m.role === 'user').slice(-1)[0]?.content || ''
    const reply = `[simulator] नमस्ते! आप कैसे हैं? (echo of: ${userMsg.slice(0, 40)}${userMsg.length > 40 ? '…' : ''})`
    return {
      text: reply,
      raw: JSON.stringify({ response: reply, success: true, decode_tps: 4.7, prefill_tps: 12.0 }),
      elapsedMs: 600,
      decodeTps: 4.7,
      prefillTps: 12.0,
      success: true,
    }
  }
  if (_handle == null) {
    await init()
  }
  return CactusNative.complete(req)
}

/**
 * Free the loaded model. Call on app pause to release phone RAM.
 */
export async function destroy() {
  if (isBrowserSim()) {
    _simLoaded = false
    return
  }
  if (_handle == null) return
  try {
    await CactusNative.destroy()
  } finally {
    _handle = null
  }
}

/**
 * Launch the system file picker (SAF) so the user can choose a locally
 * downloaded Cactus model zip (Downloads folder, USB OTG, etc).
 * The plugin extracts the zip into app-private storage; afterwards
 * init() will see the new model folder. The zip should be on local
 * storage, not streamed from a cloud content provider — a 4 GB+ stream
 * over LTE is fragile.
 *
 * Progress callback fires at scan-complete, every 10% bucket during
 * extraction, and at done. Event shape:
 *   { phase: 'scanning_done', totalEntries }
 *   { phase: 'extracting', entries, totalEntries, bytes, pct }
 *   { phase: 'done', entries, totalEntries, bytes, pct: 100 }
 *
 * @param {(evt: object) => void} [onProgress]
 * @returns {Promise<{
 *   cancelled?: true,
 *   modelName?: string,
 *   modelPath?: string,
 *   entries?: number,
 *   bytes?: number
 * }>}
 */
export async function importModelFromZip(onProgress) {
  // Browser simulator: when there's no native plugin (Vite dev, desktop browser),
  // fake the SAF picker + extraction so the UI wiring (progress bar, log card,
  // listener subscribe/unsubscribe) can be exercised end-to-end without an APK
  // rebuild. Set localStorage.sakhi_sim_cancel = '1' to test the cancel path.
  if (Capacitor.getPlatform() !== 'android') {
    return simulateImport(onProgress)
  }
  let listener = null
  if (typeof onProgress === 'function') {
    listener = await CactusNative.addListener('importProgress', onProgress)
  }
  try {
    return await CactusNative.importModelFromZip()
  } finally {
    try { listener?.remove?.() } catch (_) {}
  }
}

/**
 * Pretend we picked a 4.68 GB zip and extracted 1963 files over ~5 s
 * (compressed from ~5 min on real hardware). Lets the desktop browser
 * exercise the full UI without an APK round-trip.
 */
async function simulateImport(onProgress) {
  const cancelled = typeof localStorage !== 'undefined' && localStorage.getItem('sakhi_sim_cancel') === '1'
  if (cancelled) return { cancelled: true }

  const TOTAL_ENTRIES = 1963
  const TOTAL_BYTES = 4679429616
  await sleep(150) // SAF picker open
  if (typeof onProgress === 'function') {
    onProgress({ phase: 'scanning_done', totalEntries: TOTAL_ENTRIES })
  }

  for (let bucket = 1; bucket <= 10; bucket++) {
    await sleep(500) // 5 s total for all 10 buckets
    const pct = bucket * 10
    const entries = Math.round((TOTAL_ENTRIES * pct) / 100)
    const bytes = Math.round((TOTAL_BYTES * pct) / 100)
    if (typeof onProgress === 'function') {
      onProgress({ phase: 'extracting', entries, totalEntries: TOTAL_ENTRIES, bytes, pct })
    }
  }

  if (typeof onProgress === 'function') {
    onProgress({ phase: 'done', entries: TOTAL_ENTRIES, totalEntries: TOTAL_ENTRIES, bytes: TOTAL_BYTES, pct: 100 })
  }
  _simHasModel = true
  return {
    modelName: 'gemma-4-e2b-it-int4',
    modelPath: '/sim/files/models/gemma-4-e2b-it-int4',
    entries: TOTAL_ENTRIES,
    bytes: TOTAL_BYTES,
  }
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)) }

export const Cactus = { isAvailable, init, complete, destroy, importModelFromZip }
export default Cactus
