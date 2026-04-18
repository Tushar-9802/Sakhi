package com.sakhi.app

import com.cactus.cactusComplete
import com.cactus.cactusDestroy
import com.cactus.cactusGetLastError
import com.cactus.cactusInit
import com.cactus.cactusReset
import com.getcapacitor.JSArray
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.File

@CapacitorPlugin(name = "Cactus")
class CactusPlugin : Plugin() {

    @Volatile private var handle: Long = 0L
    @Volatile private var modelPath: String? = null
    private val scope = CoroutineScope(Dispatchers.IO)

    /**
     * Candidate model paths. Cactus models are directories containing config.txt
     * plus per-layer .weights files. We look for such directories in:
     *   1. explicit override
     *   2. <filesDir>/models/<any subdir with config.txt>
     *   3. /sdcard/Download/<any subdir with config.txt>
     *   4. /sdcard/Download/sakhi-models/<any subdir>
     */
    private fun candidateModelPaths(explicit: String?): List<String> {
        val ctx = context
        val candidates = mutableListOf<String>()
        if (!explicit.isNullOrBlank()) candidates.add(explicit)

        fun scanDir(dir: File) {
            if (!dir.exists() || !dir.isDirectory) return
            // Is dir itself a Cactus model? (contains config.txt)
            if (File(dir, "config.txt").exists()) {
                candidates.add(dir.absolutePath)
                return
            }
            // Otherwise, scan subdirectories one level deep
            dir.listFiles()?.forEach { child ->
                if (child.isDirectory && File(child, "config.txt").exists()) {
                    candidates.add(child.absolutePath)
                }
            }
        }

        scanDir(File(ctx.filesDir, "models"))
        scanDir(File("/sdcard/Download"))
        scanDir(File("/sdcard/Download/sakhi-models"))

        return candidates.distinct()
    }

    @PluginMethod
    fun isAvailable(call: PluginCall) {
        scope.launch {
            val ret = JSObject()
            ret.put("available", true)          // plugin compiled & loaded
            ret.put("handle", handle)
            ret.put("modelPath", modelPath ?: "")
            val found = candidateModelPaths(null).firstOrNull { File(it).exists() }
            ret.put("modelPresent", found != null)
            if (found != null) ret.put("modelFound", found)
            call.resolve(ret)
        }
    }

    @PluginMethod
    fun init(call: PluginCall) {
        val explicit = call.getString("modelPath")
        scope.launch {
            try {
                if (handle != 0L) {
                    val r = JSObject()
                    r.put("handle", handle)
                    r.put("cached", true)
                    r.put("modelPath", modelPath ?: "")
                    call.resolve(r)
                    return@launch
                }

                val candidates = candidateModelPaths(explicit)
                val chosen = candidates.firstOrNull { File(it).exists() }
                    ?: run {
                        call.reject(
                            "No model file found. Tried: ${candidates.joinToString(", ")}"
                        )
                        return@launch
                    }

                // Copy directory from /sdcard/Download to app-private if needed.
                // Cactus models are folders, not single files.
                val finalPath = if (chosen.startsWith("/sdcard/")) {
                    val appModelDir = File(context.filesDir, "models").apply { mkdirs() }
                    val src = File(chosen)
                    val dest = File(appModelDir, src.name)
                    if (!File(dest, "config.txt").exists()) {
                        if (dest.exists()) dest.deleteRecursively()
                        src.copyRecursively(dest, overwrite = true)
                    }
                    dest.absolutePath
                } else chosen

                val t0 = System.currentTimeMillis()
                val h = cactusInit(finalPath, null, false)
                val elapsed = System.currentTimeMillis() - t0
                handle = h
                modelPath = finalPath

                val r = JSObject()
                r.put("handle", h)
                r.put("cached", false)
                r.put("modelPath", finalPath)
                r.put("initMs", elapsed)
                call.resolve(r)
            } catch (e: Exception) {
                call.reject("init failed: ${e.message} :: ${cactusGetLastError()}", e)
            }
        }
    }

    @PluginMethod
    fun complete(call: PluginCall) {
        val messagesJs: JSArray? = call.getArray("messages")
        val toolsJs: JSArray? = call.getArray("tools")
        val optsJs: JSObject? = call.getObject("options")
        if (messagesJs == null) {
            call.reject("messages is required")
            return
        }
        scope.launch {
            try {
                if (handle == 0L) {
                    call.reject("model not initialized — call init() first")
                    return@launch
                }

                val messagesJson = messagesJs.toString()
                val toolsJson = toolsJs?.toString()
                val optionsJson = optsJs?.toString()

                val t0 = System.currentTimeMillis()
                val resultJson = cactusComplete(handle, messagesJson, optionsJson, toolsJson, null)
                val elapsed = System.currentTimeMillis() - t0

                val parsed = try { JSONObject(resultJson) } catch (_: JSONException) { null }
                val ret = JSObject()
                ret.put("raw", resultJson)
                ret.put("elapsedMs", elapsed)
                if (parsed != null) {
                    ret.put("text", parsed.optString("response", parsed.optString("text", "")))
                    ret.put("success", parsed.optBoolean("success", true))
                    if (parsed.has("prefill_tps")) ret.put("prefillTps", parsed.optDouble("prefill_tps"))
                    if (parsed.has("decode_tps")) ret.put("decodeTps", parsed.optDouble("decode_tps"))
                    if (parsed.has("total_time_ms")) ret.put("totalTimeMs", parsed.optDouble("total_time_ms"))
                    val toolCallsArr = parsed.optJSONArray("tool_calls")
                    if (toolCallsArr != null) ret.put("toolCalls", toolCallsArr)
                } else {
                    ret.put("text", resultJson)
                }
                call.resolve(ret)
            } catch (e: Exception) {
                call.reject("complete failed: ${e.message} :: ${cactusGetLastError()}", e)
            }
        }
    }

    @PluginMethod
    fun reset(call: PluginCall) {
        scope.launch {
            try {
                if (handle != 0L) cactusReset(handle)
                call.resolve()
            } catch (e: Exception) {
                call.reject("reset failed: ${e.message}", e)
            }
        }
    }

    @PluginMethod
    fun destroy(call: PluginCall) {
        scope.launch {
            try {
                val h = handle
                if (h != 0L) {
                    handle = 0L
                    modelPath = null
                    cactusDestroy(h)
                }
                call.resolve()
            } catch (e: Exception) {
                call.reject("destroy failed: ${e.message}", e)
            }
        }
    }
}
