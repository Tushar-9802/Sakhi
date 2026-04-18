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

/**
 * Quick availability check. Returns immediately without touching native code
 * on platforms where the plugin isn't registered.
 */
export async function isAvailable() {
  if (Capacitor.getPlatform() !== 'android') {
    return { available: false, reason: 'not-android' }
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
  if (_handle == null) {
    await init()
  }
  return CactusNative.complete(req)
}

/**
 * Free the loaded model. Call on app pause to release phone RAM.
 */
export async function destroy() {
  if (_handle == null) return
  try {
    await CactusNative.destroy()
  } finally {
    _handle = null
  }
}

export const Cactus = { isAvailable, init, complete, destroy }
export default Cactus
