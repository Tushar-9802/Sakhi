package com.sakhi.app

import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
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
import java.io.FileOutputStream
import java.util.zip.ZipInputStream

private const val TAG = "CactusPlugin"
private const val RC_PICK_ZIP = 0xCACA

@CapacitorPlugin(name = "Cactus", requestCodes = [RC_PICK_ZIP])
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

    /**
     * SAF zip-import. Lets the user pick a Cactus model zip (e.g. the
     * gemma-4-e2b-it-int4.zip from huggingface.co/Cactus-Compute) from any
     * SAF-accessible source (Downloads, USB OTG, locally-downloaded Drive
     * file, …) and extracts it into app-private files dir. After import,
     * candidateModelPaths() will see the new folder and init() can load it.
     *
     * Caller is expected to land the zip in local storage first — streaming
     * a multi-GB file directly from a cloud content provider over LTE is
     * fragile (Drive may stream lazily and disconnect mid-extract).
     *
     * Replaces the adb-push + run-as flow in scripts/setup_cactus_model.sh
     * for non-developer setup paths.
     *
     * Uses the legacy startActivityForResult(call, intent, int) path with
     * handleOnActivityResult, not @ActivityCallback. The annotation-based
     * Activity Result API has been unreliable for our setup (private fun
     * + Kotlin reflection); the legacy path is plumbed explicitly through
     * BridgeActivity.onActivityResult → bridge.onActivityResult →
     * Plugin.handleOnActivityResult, no reflection involved.
     *
     * Resolves with { modelName, modelPath, entries, bytes } on success or
     * { cancelled: true } if the user backs out of the picker.
     */
    @PluginMethod
    fun importModelFromZip(call: PluginCall) {
        Log.d(TAG, "importModelFromZip: launching SAF picker")
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            // application/zip is the canonical MIME; some pickers tag zips as
            // octet-stream or x-zip-compressed. Allow all three.
            type = "*/*"
            putExtra(
                Intent.EXTRA_MIME_TYPES,
                arrayOf("application/zip", "application/x-zip-compressed", "application/octet-stream"),
            )
        }
        @Suppress("DEPRECATION")
        startActivityForResult(call, intent, RC_PICK_ZIP)
    }

    override fun handleOnActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        Log.d(TAG, "handleOnActivityResult: rc=$requestCode result=$resultCode data=$data")
        if (requestCode != RC_PICK_ZIP) {
            super.handleOnActivityResult(requestCode, resultCode, data)
            return
        }
        val call = savedCall ?: run {
            Log.e(TAG, "handleOnActivityResult: no saved call")
            return
        }
        // The call has been consumed; drop the saved reference so future
        // invocations don't reuse it.
        freeSavedCall()
        Log.d(TAG, "handleOnActivityResult: resolved saved call callbackId=${call.callbackId}")

        if (resultCode != android.app.Activity.RESULT_OK) {
            Log.d(TAG, "handleOnActivityResult: user cancelled (resultCode=$resultCode)")
            val r = JSObject()
            r.put("cancelled", true)
            call.resolve(r)
            return
        }
        val uri = data?.data ?: run {
            Log.e(TAG, "handleOnActivityResult: RESULT_OK but no URI in data")
            call.reject("picker returned no URI")
            return
        }
        val displayName = queryDisplayName(uri) ?: "model.zip"
        Log.d(TAG, "handleOnActivityResult: extracting uri=$uri displayName=$displayName")
        scope.launch {
            try {
                val report = extractZipToModels(uri, displayName)
                Log.d(TAG, "extract success: ${report.entries} files, ${report.bytes} bytes → ${report.modelPath}")
                val r = JSObject()
                r.put("modelName", report.modelName)
                r.put("modelPath", report.modelPath)
                r.put("entries", report.entries)
                r.put("bytes", report.bytes)
                call.resolve(r)
            } catch (e: SecurityException) {
                Log.e(TAG, "extract blocked: ${e.message}", e)
                call.reject("blocked unsafe zip entry: ${e.message}", e)
            } catch (e: Exception) {
                Log.e(TAG, "extract failed: ${e.message}", e)
                call.reject("extract failed: ${e.message}", e)
            }
        }
    }

    private fun queryDisplayName(uri: Uri): String? {
        return try {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun queryZipSize(uri: Uri): Long {
        return try {
            context.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
                if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else 0L
            } ?: 0L
        } catch (_: Exception) { 0L }
    }

    private data class ExtractReport(val modelName: String, val modelPath: String, val entries: Int, val bytes: Long)

    /**
     * Single-pass extraction. The earlier two-pass version had to inflate the
     * entire compressed zip just to count entries / detect wrapper folder
     * before any progress event could fire — a 4-minute silent prefix for a
     * 4.68 GB zip, because ZipInputStream.nextEntry() inflates entry data to
     * find the next header (no random access without ZipFile, which would
     * need a local temp copy).
     *
     * Now: derive the model dir name from the zip filename, skip wrapper
     * detection (candidateModelPaths() scans one level of nesting, so a
     * wrapped zip still loads — files just end up one dir deeper), and use
     * the content URI's SIZE column (compressed bytes) as the progress
     * denominator. Binary weight files compress at ~1:1 so
     * uncompressed-bytes-written tracks compressed total closely enough for
     * a smooth bar.
     *
     * Progress events fire on every 1% bucket crossover with a 500 ms
     * heartbeat fallback so a slow single-file write (the big tail weights)
     * still shows liveness via the file/byte counters.
     */
    private fun extractZipToModels(uri: Uri, displayName: String): ExtractReport {
        val cr = context.contentResolver

        val totalCompressedBytes = queryZipSize(uri).coerceAtLeast(1L)
        val modelName = displayName.removeSuffix(".zip").removeSuffix(".ZIP").ifBlank { "imported-model" }
            // Sanitize: Cactus paths are passed to native code — keep ASCII-safe.
            .replace(Regex("[^A-Za-z0-9._-]"), "-")
        val modelsDir = File(context.filesDir, "models").apply { mkdirs() }

        // One-model-at-a-time eviction: before extracting a new model, remove
        // any sibling subdirs in files/models/. A model is ~6 GB; phones don't
        // have headroom for accumulation. Same-named import (re-extract) hits
        // the deleteRecursively below, not this loop.
        modelsDir.listFiles()?.forEach { sibling ->
            if (sibling.isDirectory && sibling.name != modelName) {
                val freed = sibling.walkBottomUp().sumOf { if (it.isFile) it.length() else 0L }
                Log.d(TAG, "evicting stale model: ${sibling.name} (${freed / (1024 * 1024)} MB)")
                sibling.deleteRecursively()
            }
        }

        val target = File(modelsDir, modelName)
        if (target.exists()) target.deleteRecursively()
        target.mkdirs()
        val targetCanonical = target.canonicalPath + File.separator

        notifyListeners(
            "importProgress",
            JSObject()
                .put("phase", "scanning_done")
                .put("totalBytes", totalCompressedBytes),
        )

        var entries = 0
        var totalBytes = 0L
        var lastPct = -1
        var lastEmitMs = 0L
        val emit: () -> Unit = {
            val pct = ((totalBytes * 100) / totalCompressedBytes).toInt().coerceIn(0, 99)
            val now = System.currentTimeMillis()
            if (pct > lastPct || (now - lastEmitMs) >= 500L) {
                lastPct = pct
                lastEmitMs = now
                notifyListeners(
                    "importProgress",
                    JSObject()
                        .put("phase", "extracting")
                        .put("entries", entries)
                        .put("bytes", totalBytes)
                        .put("totalBytes", totalCompressedBytes)
                        .put("pct", pct),
                )
            }
        }

        cr.openInputStream(uri)?.use { input ->
            ZipInputStream(input).use { zis ->
                val buf = ByteArray(64 * 1024)
                var entry = zis.nextEntry
                while (entry != null) {
                    val rawName = entry.name
                    if (rawName.isNotEmpty() && !entry.isDirectory) {
                        val outFile = File(target, rawName).canonicalFile
                        // Zip-slip guard.
                        if (!outFile.path.startsWith(targetCanonical)) {
                            throw SecurityException("entry tries to escape target: $rawName")
                        }
                        outFile.parentFile?.mkdirs()
                        FileOutputStream(outFile).use { fos ->
                            var n = zis.read(buf)
                            while (n > 0) {
                                fos.write(buf, 0, n)
                                totalBytes += n.toLong()
                                n = zis.read(buf)
                                // Heartbeat inside a long-running entry so the
                                // bar keeps moving when one big weight file
                                // dominates an extract window.
                                emit()
                            }
                        }
                        entries++
                        emit()
                    }
                    entry = zis.nextEntry
                }
            }
        } ?: throw RuntimeException("cannot open zip input stream")

        if (entries == 0) throw RuntimeException("zip extracted no files")

        notifyListeners(
            "importProgress",
            JSObject()
                .put("phase", "done")
                .put("entries", entries)
                .put("bytes", totalBytes)
                .put("totalBytes", totalCompressedBytes)
                .put("pct", 100),
        )

        if (!File(target, "config.txt").exists()) {
            // Tolerate: maybe the model dir uses a different layout. Don't
            // hard-fail extraction, but signal to the caller via entries=0
            // if literally nothing extracted.
        }

        return ExtractReport(modelName, target.absolutePath, entries, totalBytes)
    }
}
