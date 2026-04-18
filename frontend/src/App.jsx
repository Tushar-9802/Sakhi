import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { saveRecording, getQueue, getRecording, removeRecording, clearQueue, updateRecordingStatus, appendChunk, assembleChunks, listOrphanedSessions, clearChunks } from './offlineQueue'
import Cactus from './lib/cactus'
import { runPipeline } from './lib/pipeline'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:8000`
const VISIT_OPTIONS = [
  { label: 'Auto-detect', value: 'auto' },
  { label: 'ANC Visit', value: 'anc_visit' },
  { label: 'PNC Visit', value: 'pnc_visit' },
  { label: 'Delivery', value: 'delivery' },
  { label: 'Child Health', value: 'child_health' },
]

const VOICE_STAGE_META = {
  asr: 'Transcribing audio...',
  normalize: 'Normalizing Hindi numbers...',
  detect: 'Detecting visit type...',
  form: 'Extracting structured form...',
  danger: 'Detecting danger signs...',
}
const TEXT_STAGE_META = {
  detect: 'Detecting visit type...',
  form: 'Extracting structured form...',
  danger: 'Detecting danger signs...',
}

function PipelineProgress({ stages }) {
  return (
    <div className="pipeline-progress">
      {stages.map((stage) => (
        <div className={`progress-step ${stage.status}`} key={stage.key}>
          <span className="step-icon">
            {stage.status === 'done' ? '\u2713' : stage.status === 'running' ? '\u25CF' : '\u25CB'}
          </span>
          <span className="step-label">
            {stage.label}
            {stage.status === 'done' && stage.time != null && <span className="step-time"> ({stage.time}s)</span>}
          </span>
        </div>
      ))}
    </div>
  )
}

function prettyLabel(text) {
  return String(text || '')
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function keyValueRows(data, prefix = '') {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return []
  const rows = []
  Object.entries(data).forEach(([key, value]) => {
    const fullKey = prefix ? `${prefix} > ${prettyLabel(key)}` : prettyLabel(key)
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      rows.push(...keyValueRows(value, fullKey))
      return
    }
    if (Array.isArray(value)) {
      rows.push({
        key: fullKey,
        value: value.length ? value.map((v) => (typeof v === 'object' ? JSON.stringify(v) : String(v))).join(', ') : '—',
      })
      return
    }
    rows.push({ key: fullKey, value: value ?? '—' })
  })
  return rows
}

function App() {
  const [activeTab, setActiveTab] = useState('voice')
  const [health, setHealth] = useState('Checking backend...')
  const [examples, setExamples] = useState([])
  const [history, setHistory] = useState(() => {
    try { return JSON.parse(localStorage.getItem('sakhi_history') || '[]') } catch { return [] }
  })
  const [viewingHistory, setViewingHistory] = useState(null)

  const [voiceVisitType, setVoiceVisitType] = useState('auto')
  const [textVisitType, setTextVisitType] = useState('auto')
  const [textInput, setTextInput] = useState('')
  const [selectedExample, setSelectedExample] = useState('')

  const [audioFile, setAudioFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState('')
  const [isRecording, setIsRecording] = useState(false)

  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const chunksRef = useRef([])

  const [voiceState, setVoiceState] = useState({
    loading: false,
    error: '',
    transcript: '',
    visitType: '',
    form: null,
    danger: null,
    timing: null,
  })

  const [textState, setTextState] = useState({
    loading: false,
    error: '',
    visitType: '',
    form: null,
    danger: null,
    timing: null,
  })

  const [pipelineStages, setPipelineStages] = useState([])

  // Field Mode state
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  const [offlineQueue, setOfflineQueue] = useState([])
  const [fieldRecording, setFieldRecording] = useState(false)
  const [syncingId, setSyncingId] = useState(null)
  const fieldRecorderRef = useRef(null)
  const fieldStreamRef = useRef(null)
  const fieldSessionIdRef = useRef(null)
  const [fieldVisitType, setFieldVisitType] = useState('auto')
  const [fieldError, setFieldError] = useState('')
  const [playingId, setPlayingId] = useState(null)
  const playAudioRef = useRef(null)
  const [orphanedSessions, setOrphanedSessions] = useState([])

  // On-device Field text-in extraction (Cactus + pipeline.js)
  const [fieldOnDeviceText, setFieldOnDeviceText] = useState('')
  const [fieldOnDeviceVisitType, setFieldOnDeviceVisitType] = useState('auto')
  const [fieldOnDeviceState, setFieldOnDeviceState] = useState({
    loading: false, error: '', transcript: '', visitType: '', form: null, danger: null, timing: null, _raw: null,
  })
  const [devViewEnabled, setDevViewEnabled] = useState(false)

  // Cactus on-device probe state
  const [cactusStatus, setCactusStatus] = useState(null)
  const [cactusBusy, setCactusBusy] = useState(false)
  const [cactusLog, setCactusLog] = useState([])
  const pushLog = (msg) => setCactusLog((prev) => [...prev.slice(-30), `[${new Date().toLocaleTimeString('en-IN')}] ${msg}`])

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then((r) => r.json())
      .then((d) => setHealth(`API: ${d.status} · Model: ${d.model}`))
      .catch(() => setHealth('API not reachable. Start FastAPI on :8000'))

    fetch(`${API_BASE}/api/examples`)
      .then((r) => r.json())
      .then((data) => {
        setExamples(data || [])
        const defaultEx = (data || []).find((e) => e.default) || data?.[0]
        if (defaultEx) {
          setSelectedExample(defaultEx.label)
          setTextInput(defaultEx.transcript || '')
        }
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }
    }
  }, [audioUrl])

  // Online/offline detection + queue loading
  useEffect(() => {
    const goOnline = () => setIsOnline(true)
    const goOffline = () => setIsOnline(false)
    window.addEventListener('online', goOnline)
    window.addEventListener('offline', goOffline)
    loadQueue()
    return () => {
      window.removeEventListener('online', goOnline)
      window.removeEventListener('offline', goOffline)
    }
  }, [])

  async function loadQueue() {
    const q = await getQueue()
    setOfflineQueue(q)
  }

  async function loadOrphaned() {
    try {
      const list = await listOrphanedSessions()
      setOrphanedSessions(list)
    } catch {
      setOrphanedSessions([])
    }
  }

  useEffect(() => {
    if (activeTab === 'field') loadOrphaned()
  }, [activeTab])

  async function recoverOrphan(sessionId, visitType) {
    try {
      const result = await assembleChunks(sessionId)
      if (result && result.blob && result.blob.size > 0) {
        await saveRecording(result.blob, visitType || 'auto', `Recovered ${new Date().toLocaleTimeString('en-IN')}`)
      }
      await clearChunks(sessionId)
      await loadOrphaned()
      await loadQueue()
    } catch (err) {
      setFieldError(`Recovery failed: ${err.message}`)
    }
  }

  async function discardOrphan(sessionId) {
    try {
      await clearChunks(sessionId)
      await loadOrphaned()
    } catch (err) {
      setFieldError(`Discard failed: ${err.message}`)
    }
  }

  async function cactusCheck() {
    setCactusBusy(true)
    try {
      const s = await Cactus.isAvailable()
      setCactusStatus(s)
      pushLog(`status: available=${s.available} modelPresent=${s.modelPresent ?? false}${s.modelFound ? ` @ ${s.modelFound}` : ''}`)
    } catch (err) {
      pushLog(`status check failed: ${err.message || err}`)
      setCactusStatus({ available: false, error: String(err) })
    } finally {
      setCactusBusy(false)
    }
  }

  async function cactusLoad() {
    setCactusBusy(true)
    try {
      pushLog('loading model...')
      const r = await Cactus.init()
      pushLog(`model loaded in ${r.initMs || '?'}ms from ${r.modelPath}`)
      setCactusStatus((s) => ({ ...(s || {}), ...r, loaded: true }))
    } catch (err) {
      pushLog(`init failed: ${err.message || err}`)
    } finally {
      setCactusBusy(false)
    }
  }

  async function cactusTest() {
    setCactusBusy(true)
    try {
      pushLog('running test completion...')
      const t0 = Date.now()
      const r = await Cactus.complete({
        messages: [
          { role: 'user', content: 'नमस्ते, आप कैसे हैं?' },
        ],
        options: { max_tokens: 64, temperature: 0.3 },
      })
      const elapsed = Date.now() - t0
      pushLog(`got ${r.text?.length || 0} chars in ${elapsed}ms (decode ${r.decodeTps?.toFixed?.(1) || '?'} tps)`)
      pushLog(`text: ${(r.text || r.raw || '').slice(0, 200)}`)
    } catch (err) {
      pushLog(`complete failed: ${err.message || err}`)
    } finally {
      setCactusBusy(false)
    }
  }

  async function cactusUnload() {
    setCactusBusy(true)
    try {
      await Cactus.destroy()
      pushLog('model unloaded')
      setCactusStatus((s) => ({ ...(s || {}), loaded: false, handle: 0 }))
    } catch (err) {
      pushLog(`destroy failed: ${err.message || err}`)
    } finally {
      setCactusBusy(false)
    }
  }

  async function processFieldOnDevice() {
    const text = fieldOnDeviceText.trim()
    if (!text) {
      setFieldOnDeviceState((s) => ({ ...s, error: 'Type a Hindi note first.' }))
      return
    }
    setFieldOnDeviceState({ loading: true, error: '', transcript: '', visitType: '', form: null, danger: null, timing: null, _raw: null })
    try {
      const result = await runPipeline({
        engine: Cactus,
        transcript: text,
        visitType: fieldOnDeviceVisitType === 'auto' ? null : fieldOnDeviceVisitType,
      })
      setFieldOnDeviceState({
        loading: false,
        error: '',
        transcript: result.transcript,
        visitType: result.visitType,
        form: result.form,
        danger: result.danger,
        timing: result.timing,
        _raw: result._raw || null,
      })
      saveToHistory('field', result.visitType, result.form, result.danger, result.transcript, result.timing)
    } catch (err) {
      setFieldOnDeviceState((s) => ({ ...s, loading: false, error: `On-device extraction failed: ${err.message || err}` }))
    }
  }

  async function startFieldRecording() {
    setFieldError('')
    // Stop any active recorders first
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
    if (fieldRecorderRef.current && fieldRecorderRef.current.state !== 'inactive') {
      fieldRecorderRef.current.stop()
    }
    // Release all mic streams
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (fieldStreamRef.current) {
      fieldStreamRef.current.getTracks().forEach((t) => t.stop())
      fieldStreamRef.current = null
    }
    // Small delay to let the OS release the device
    await new Promise((r) => setTimeout(r, 300))
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: true }
      })
      fieldStreamRef.current = stream
      const recorder = new MediaRecorder(stream)
      const sessionId = (crypto.randomUUID && crypto.randomUUID()) || `s-${Date.now()}-${Math.random().toString(36).slice(2)}`
      fieldSessionIdRef.current = sessionId
      const capturedVisitType = fieldVisitType
      recorder.ondataavailable = async (e) => {
        if (e.data && e.data.size > 0) {
          try { await appendChunk(sessionId, e.data, capturedVisitType) } catch (err) { console.error('appendChunk failed', err) }
        }
      }
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        fieldStreamRef.current = null
        try {
          const result = await assembleChunks(sessionId)
          if (result && result.blob && result.blob.size > 0) {
            await saveRecording(result.blob, capturedVisitType)
          }
          await clearChunks(sessionId)
        } catch (err) {
          setFieldError(`Save failed: ${err.message}`)
        }
        fieldSessionIdRef.current = null
        await loadQueue()
        await loadOrphaned()
      }
      fieldRecorderRef.current = recorder
      recorder.start(5000)
      setFieldRecording(true)
    } catch (err) {
      setFieldError(`Microphone error: ${err.name}: ${err.message}`)
    }
  }

  function stopFieldRecording() {
    if (!fieldRecorderRef.current) return
    fieldRecorderRef.current.stop()
    setFieldRecording(false)
  }

  async function syncRecording(id) {
    setSyncingId(id)
    setPipelineStages([])
    setVoiceState({ loading: true, error: '', transcript: '', visitType: '', form: null, danger: null, timing: null })
    setActiveTab('voice')

    const entry = await getRecording(id)
    if (!entry) { setSyncingId(null); return }

    await updateRecordingStatus(id, 'processing')
    await loadQueue()

    const file = new File([entry.audioBlob], `field-${entry.id}.webm`, { type: entry.audioType })
    const formData = new FormData()
    formData.append('audio', file)
    formData.append('visit_type', entry.visitType)

    try {
      const res = await fetch(`${API_BASE}/api/process-audio-stream`, { method: 'POST', body: formData })
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      await new Promise((resolve, reject) => {
        function read() {
          reader.read().then(({ done, value }) => {
            if (done) { resolve(); return }
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''
            for (const line of lines) {
              if (!line.startsWith('data: ')) continue
              const evt = JSON.parse(line.slice(6))
              handleSSE(evt, 'voice', VOICE_STAGE_META)
            }
            read()
          }).catch(reject)
        }
        read()
      })

      await removeRecording(id)
      await loadQueue()
    } catch (err) {
      await updateRecordingStatus(id, 'pending')
      await loadQueue()
      setVoiceState((s) => ({ ...s, loading: false, error: `Sync failed: ${err.message}` }))
    }
    setSyncingId(null)
  }

  async function syncAll() {
    const pending = offlineQueue.filter((e) => e.status === 'pending')
    for (const entry of pending) {
      await syncRecording(entry.id)
    }
  }

  async function removeFromQueue(id) {
    await removeRecording(id)
    await loadQueue()
  }

  async function clearAllQueue() {
    await clearQueue()
    await loadQueue()
  }

  async function playRecording(id) {
    if (playingId === id) {
      if (playAudioRef.current) { playAudioRef.current.pause(); playAudioRef.current = null }
      setPlayingId(null)
      return
    }
    if (playAudioRef.current) { playAudioRef.current.pause(); playAudioRef.current = null }
    const entry = await getRecording(id)
    if (!entry) return
    const url = URL.createObjectURL(entry.audioBlob)
    const audio = new Audio(url)
    audio.onended = () => { URL.revokeObjectURL(url); setPlayingId(null); playAudioRef.current = null }
    playAudioRef.current = audio
    setPlayingId(id)
    audio.play()
  }

  const dangerSigns = useMemo(
    () => textState.danger?.danger_signs || voiceState.danger?.danger_signs || fieldOnDeviceState.danger?.danger_signs || [],
    [textState.danger, voiceState.danger, fieldOnDeviceState.danger]
  )

  async function startRecording() {
    // Release any existing mic streams first
    if (fieldStreamRef.current) {
      fieldStreamRef.current.getTracks().forEach((t) => t.stop())
      fieldStreamRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        const file = new File([blob], `recording-${Date.now()}.webm`, { type: 'audio/webm' })
        if (audioUrl) URL.revokeObjectURL(audioUrl)
        setAudioFile(file)
        setAudioUrl(URL.createObjectURL(blob))
        stream.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
      mediaRecorderRef.current = recorder
      recorder.start(5000)
      setIsRecording(true)
    } catch {
      setVoiceState((s) => ({ ...s, error: 'Microphone permission denied or unavailable.' }))
    }
  }

  function stopRecording() {
    if (!mediaRecorderRef.current) return
    mediaRecorderRef.current.stop()
    setIsRecording(false)
  }

  function onUploadAudio(event) {
    const file = event.target.files?.[0]
    if (!file) return
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioFile(file)
    setAudioUrl(URL.createObjectURL(file))
    setVoiceState((s) => ({ ...s, error: '' }))
  }

  function handleSSE(evt, source, stageMeta) {
    if (evt.error) {
      const setter = source === 'voice' ? setVoiceState : setTextState
      setter((s) => ({ ...s, loading: false, error: evt.error }))
      return
    }

    if (evt.stage === 'complete') {
      const setter = source === 'voice' ? setVoiceState : setTextState
      setter({
        loading: false,
        error: '',
        transcript: evt.transcript || '',
        visitType: evt.visit_type || '',
        form: evt.form || {},
        danger: evt.danger || {},
        timing: evt.timing || {},
      })
      setPipelineStages((prev) => prev.map((s) => ({ ...s, status: 'done' })))
      saveToHistory(source, evt.visit_type, evt.form, evt.danger, evt.transcript || null, evt.timing)
      return
    }

    if (evt.status === 'running') {
      const label = stageMeta[evt.stage] || evt.stage
      setPipelineStages((prev) => {
        const exists = prev.find((s) => s.key === evt.stage)
        if (exists) return prev.map((s) => s.key === evt.stage ? { ...s, status: 'running' } : s)
        return [...prev, { key: evt.stage, label, status: 'running', time: null }]
      })
    }

    if (evt.status === 'done') {
      setPipelineStages((prev) =>
        prev.map((s) => s.key === evt.stage ? { ...s, status: 'done', time: evt.time ?? null } : s)
      )
      if (evt.transcript) {
        setVoiceState((s) => ({ ...s, transcript: evt.transcript }))
      }
    }
  }

  function processVoice() {
    if (!audioFile) {
      setVoiceState((s) => ({ ...s, error: 'Upload or record audio first.' }))
      return
    }
    setVoiceState({ loading: true, error: '', transcript: '', visitType: '', form: null, danger: null, timing: null })
    setPipelineStages([])

    const formData = new FormData()
    formData.append('audio', audioFile)
    formData.append('visit_type', voiceVisitType)

    fetch(`${API_BASE}/api/process-audio-stream`, { method: 'POST', body: formData })
      .then((res) => {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) return
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''
            for (const line of lines) {
              if (!line.startsWith('data: ')) continue
              const evt = JSON.parse(line.slice(6))
              handleSSE(evt, 'voice', VOICE_STAGE_META)
            }
            read()
          })
        }
        read()
      })
      .catch((err) => {
        setVoiceState((s) => ({ ...s, loading: false, error: err.message }))
      })
  }

  function processText() {
    if (!textInput.trim()) {
      setTextState((s) => ({ ...s, error: 'Transcript is empty.' }))
      return
    }
    setTextState({ loading: true, error: '', visitType: '', form: null, danger: null, timing: null })
    setPipelineStages([])

    fetch(`${API_BASE}/api/process-text-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript: textInput, visit_type: textVisitType }),
    })
      .then((res) => {
        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''

        function read() {
          reader.read().then(({ done, value }) => {
            if (done) return
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() || ''
            for (const line of lines) {
              if (!line.startsWith('data: ')) continue
              const evt = JSON.parse(line.slice(6))
              handleSSE(evt, 'text', TEXT_STAGE_META)
            }
            read()
          })
        }
        read()
      })
      .catch((err) => {
        setTextState((s) => ({ ...s, loading: false, error: err.message }))
      })
  }

  function onSelectExample(label) {
    setSelectedExample(label)
    const ex = examples.find((e) => e.label === label)
    if (ex) setTextInput(ex.transcript || '')
  }

  function downloadJSON() {
    const data = activeState.form
    if (!data) return
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `sakhi-${activeState.visitType || 'form'}-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function downloadCSV() {
    const rows = keyValueRows(activeState.form)
    if (!rows.length) return
    const csv = 'Field,Value\n' + rows.map((r) => `"${r.key}","${String(r.value).replace(/"/g, '""')}"`).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `sakhi-${activeState.visitType || 'form'}-${Date.now()}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const saveToHistory = useCallback((source, visitType, form, danger, transcript, timing) => {
    const entry = {
      id: Date.now(),
      date: new Date().toLocaleString('en-IN'),
      source,
      visitType,
      form,
      danger,
      transcript: transcript || null,
      timing,
    }
    setHistory((prev) => {
      const updated = [entry, ...prev].slice(0, 50)
      localStorage.setItem('sakhi_history', JSON.stringify(updated))
      return updated
    })
  }, [])

  const activeState = activeTab === 'voice'
    ? voiceState
    : activeTab === 'field'
      ? fieldOnDeviceState
      : textState

  return (
    <div className="app-shell">
      <header className="hero">
        <h1>Sakhi (सखी)</h1>
        <p>AI companion for India&apos;s ASHA health workers</p>
        <div className="badge-row">
          <span className="badge">Gemma 4 E4B</span>
          <span className="badge">Offline-First</span>
          <span className="badge">Hindi Voice</span>
        </div>
      </header>

      <div className="status-line">{health}</div>

      <div className="tabs">
        <button className={activeTab === 'voice' ? 'active' : ''} onClick={() => setActiveTab('voice')}>
          Voice to Form
        </button>
        <button className={activeTab === 'text' ? 'active' : ''} onClick={() => setActiveTab('text')}>
          Text to Form
        </button>
        <button className={activeTab === 'field' ? 'active' : ''} onClick={() => setActiveTab('field')}>
          Field Mode {offlineQueue.length > 0 ? `(${offlineQueue.length})` : ''}
        </button>
        <button className={activeTab === 'about' ? 'active' : ''} onClick={() => setActiveTab('about')}>
          About &amp; Impact
        </button>
        {history.length > 0 && (
          <button className={activeTab === 'history' ? 'active' : ''} onClick={() => { setActiveTab('history'); setViewingHistory(null) }}>
            History ({history.length})
          </button>
        )}
      </div>

      {activeTab === 'voice' && (
        <section className="panel">
          <h2>Record or upload Hindi ASHA conversation</h2>
          <div className="card">
            <div className="audio-tools">
              <button className={`btn ${isRecording ? 'danger' : ''}`} onClick={isRecording ? stopRecording : startRecording}>
                {isRecording ? 'Stop Recording' : 'Start Recording'}
              </button>
              <label className="btn secondary">
                Upload Audio File
                <input type="file" accept="audio/*" onChange={onUploadAudio} hidden />
              </label>
              <select value={voiceVisitType} onChange={(e) => setVoiceVisitType(e.target.value)}>
                {VISIT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <button className="btn primary" onClick={processVoice} disabled={voiceState.loading}>
                {voiceState.loading ? 'Processing...' : 'Process Audio'}
              </button>
            </div>
            <audio className="audio-player" controls src={audioUrl || undefined} />
            {audioFile && <p className="file-name">{audioFile.name}</p>}
          </div>
          <div className="card">
            <h3>Transcript</h3>
            <pre className="transcript">{voiceState.transcript || 'Transcript will appear here after processing audio.'}</pre>
          </div>
        </section>
      )}

      {activeTab === 'text' && (
        <section className="panel">
          <h2>Paste transcript and extract structured form</h2>
          <div className="card">
            <div className="text-tools">
              <select value={selectedExample} onChange={(e) => onSelectExample(e.target.value)}>
                <option value="">Load example...</option>
                {examples.map((ex) => (
                  <option key={ex.label} value={ex.label}>
                    {ex.label}
                  </option>
                ))}
              </select>
              <select value={textVisitType} onChange={(e) => setTextVisitType(e.target.value)}>
                {VISIT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
              <button className="btn primary" onClick={processText} disabled={textState.loading}>
                {textState.loading ? 'Extracting...' : 'Extract Structured Form'}
              </button>
            </div>
            <textarea
              className="text-input"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="Paste Hindi conversation transcript here..."
            />
          </div>
        </section>
      )}

      {activeTab === 'field' && (
        <section className="panel">
          <h2>Field Mode — Record Now, Process Later</h2>
          {orphanedSessions.length > 0 && (
            <div className="card" style={{ borderLeft: '4px solid #d97706', background: '#fffbeb' }}>
              <h3 style={{ marginTop: 0 }}>Unfinished recordings detected</h3>
              <p style={{ marginTop: 4 }}>
                {orphanedSessions.length} recording{orphanedSessions.length > 1 ? 's were' : ' was'} interrupted
                (tab closed, browser crashed, or phone locked). You can recover the partial audio or discard it.
              </p>
              <div className="queue-list">
                {orphanedSessions.map((o) => (
                  <div className="queue-item" key={o.sessionId}>
                    <div className="queue-meta">
                      <strong>{prettyLabel(o.visitType || 'auto')}</strong>
                      <span>{new Date(o.firstSeen).toLocaleString('en-IN')}</span>
                      <span>{o.chunkCount} chunk{o.chunkCount > 1 ? 's' : ''}</span>
                      <span>{(o.totalSize / 1024).toFixed(0)} KB</span>
                    </div>
                    <div className="queue-item-actions">
                      <button className="btn primary" onClick={() => recoverOrphan(o.sessionId, o.visitType)}>
                        Recover
                      </button>
                      <button className="btn secondary" onClick={() => discardOrphan(o.sessionId)}>
                        Discard
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="card">
            <div className={`connectivity-badge ${isOnline ? 'online' : 'offline'}`}>
              {isOnline ? 'Connected — ready to sync' : 'Offline — recordings saved locally'}
            </div>
            <p className="field-desc">
              Record ASHA conversations during home visits. Audio is saved on your device
              and processed when you return to the health center.
            </p>
            <div className="audio-tools">
              <button
                className={`btn ${fieldRecording ? 'danger' : 'primary'}`}
                onClick={fieldRecording ? stopFieldRecording : startFieldRecording}
              >
                {fieldRecording ? 'Stop & Save' : 'Record Visit'}
              </button>
              <select value={fieldVisitType} onChange={(e) => setFieldVisitType(e.target.value)}>
                {VISIT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>
            {fieldError && <div className="error-banner">{fieldError}</div>}
          </div>

          <div className="card" style={{ borderLeft: '4px solid #0f766e' }}>
            <h3 style={{ marginTop: 0 }}>On-device text → form (no network)</h3>
            <p className="field-desc">
              Type a short Hindi note below and Gemma 4 E2B runs entirely on this phone via Cactus
              to extract the structured form + danger signs. Use this when you need instant feedback
              without laptop access. Voice recordings above sync to a health-center laptop for
              full-accuracy Whisper-Large processing.
            </p>
            <div className="text-tools">
              <select value={fieldOnDeviceVisitType} onChange={(e) => setFieldOnDeviceVisitType(e.target.value)}>
                {VISIT_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
              <button
                className="btn primary"
                onClick={processFieldOnDevice}
                disabled={fieldOnDeviceState.loading || !fieldOnDeviceText.trim()}
              >
                {fieldOnDeviceState.loading ? 'Processing on device...' : 'Process on device'}
              </button>
            </div>
            <textarea
              className="text-input"
              value={fieldOnDeviceText}
              onChange={(e) => setFieldOnDeviceText(e.target.value)}
              placeholder="मरीज़ का नाम सुनीता है, 24 साल, गर्भावस्था 32 सप्ताह, रक्तचाप 120/80..."
            />
            {fieldOnDeviceState.loading && (
              <p style={{ fontSize: 13, color: '#555', marginTop: 8 }}>
                First run loads the model (~10 s). On-device extraction typically takes 3–5 min
                total on this phone (form + danger signs).
              </p>
            )}
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12, fontSize: 13, color: '#555', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={devViewEnabled}
                onChange={(e) => setDevViewEnabled(e.target.checked)}
              />
              Developer view — show raw model output per stage
            </label>
          </div>

          {devViewEnabled && fieldOnDeviceState._raw && (
            <div className="card" style={{ background: '#0f172a', color: '#e2e8f0', fontFamily: 'ui-monospace, Menlo, Consolas, monospace' }}>
              <h3 style={{ marginTop: 0, color: '#93c5fd' }}>Raw model output</h3>
              {fieldOnDeviceState._raw.formError && (
                <p style={{ color: '#fca5a5', fontSize: 12, margin: '4px 0' }}>
                  form parse: {fieldOnDeviceState._raw.formError}
                </p>
              )}
              <div style={{ marginBottom: 12 }}>
                <div style={{ color: '#93c5fd', fontSize: 12, marginBottom: 4 }}>$ form extractor →</div>
                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, background: '#020617', padding: 10, borderRadius: 4, maxHeight: 300, overflow: 'auto' }}>
{fieldOnDeviceState._raw.form || '(empty)'}
                </pre>
              </div>
              {fieldOnDeviceState._raw.dangerError && (
                <p style={{ color: '#fca5a5', fontSize: 12, margin: '4px 0' }}>
                  danger parse: {fieldOnDeviceState._raw.dangerError}
                </p>
              )}
              <div>
                <div style={{ color: '#93c5fd', fontSize: 12, marginBottom: 4 }}>$ danger extractor →</div>
                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 12, background: '#020617', padding: 10, borderRadius: 4, maxHeight: 300, overflow: 'auto' }}>
{fieldOnDeviceState._raw.danger || '(empty)'}
                </pre>
              </div>
              {fieldOnDeviceState.timing && (
                <div style={{ marginTop: 12, fontSize: 12, color: '#94a3b8' }}>
                  {Object.entries(fieldOnDeviceState.timing).map(([k, v]) => {
                    const isMs = k.endsWith('_ms')
                    const label = isMs ? k.slice(0, -3) : k
                    const display = isMs
                      ? (Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(2)}s` : `${v}ms`)
                      : `${v}s`
                    return <span key={k} style={{ marginRight: 12 }}>{label}={display}</span>
                  })}
                </div>
              )}
            </div>
          )}

          {offlineQueue.length > 0 && (
            <div className="card">
              <div className="queue-header">
                <h3>Saved Recordings ({offlineQueue.length})</h3>
                <div className="queue-actions">
                  {isOnline && (
                    <button className="btn primary" onClick={syncAll} disabled={syncingId != null}>
                      Sync All
                    </button>
                  )}
                  <button className="btn secondary" onClick={clearAllQueue}>Clear All</button>
                </div>
              </div>
              <div className="queue-list">
                {offlineQueue.map((entry) => (
                  <div className={`queue-item ${entry.status}`} key={entry.id}>
                    <div className="queue-meta">
                      <strong>{entry.label}</strong>
                      <span>{entry.date}</span>
                      <span>{prettyLabel(entry.visitType)}</span>
                      <span>{(entry.size / 1024).toFixed(0)} KB</span>
                      <span className={`queue-status ${entry.status}`}>
                        {entry.status === 'pending' ? 'Pending' : entry.status === 'processing' ? 'Processing...' : entry.status}
                      </span>
                    </div>
                    <div className="queue-item-actions">
                      <button className="btn secondary" onClick={() => playRecording(entry.id)}>
                        {playingId === entry.id ? 'Stop' : 'Play'}
                      </button>
                      {isOnline && entry.status === 'pending' && (
                        <button className="btn secondary" onClick={() => syncRecording(entry.id)} disabled={syncingId != null}>
                          Sync
                        </button>
                      )}
                      <button className="btn secondary" onClick={() => removeFromQueue(entry.id)}>Remove</button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {offlineQueue.length === 0 && (
            <div className="card">
              <p>No recordings saved. Record a visit above — it will be stored on your device for later processing.</p>
            </div>
          )}

          <div className="card" style={{ borderLeft: '4px solid #6366f1', background: '#f5f3ff' }}>
            <h3 style={{ marginTop: 0 }}>On-Device Probe (Cactus)</h3>
            <p style={{ marginTop: 4, color: '#555' }}>
              Diagnostic for Cactus SDK + Gemma on-device inference. Push a Cactus-format model folder to
              <code> /sdcard/Download/</code> on the phone (must contain <code>config.txt</code>).
            </p>
            <div className="audio-tools">
              <button className="btn secondary" onClick={cactusCheck} disabled={cactusBusy}>Check Status</button>
              <button className="btn secondary" onClick={cactusLoad} disabled={cactusBusy || !cactusStatus?.modelPresent}>Load Model</button>
              <button className="btn primary" onClick={cactusTest} disabled={cactusBusy || !cactusStatus?.loaded}>Test Hindi</button>
              <button className="btn secondary" onClick={cactusUnload} disabled={cactusBusy || !cactusStatus?.loaded}>Unload</button>
            </div>
            {cactusStatus && (
              <div style={{ fontSize: 13, color: '#374151', marginTop: 8 }}>
                <strong>Status:</strong> available={String(cactusStatus.available)} ·
                modelPresent={String(cactusStatus.modelPresent ?? false)} ·
                loaded={String(!!cactusStatus.loaded)} ·
                handle={cactusStatus.handle || 0}
                {cactusStatus.modelFound && <div style={{ fontSize: 11, color: '#555', wordBreak: 'break-all' }}>found: {cactusStatus.modelFound}</div>}
              </div>
            )}
            {cactusLog.length > 0 && (
              <pre style={{ fontSize: 12, background: '#fff', border: '1px solid #e5e7eb', padding: 8, marginTop: 8, maxHeight: 200, overflow: 'auto', whiteSpace: 'pre-wrap' }}>
                {cactusLog.join('\n')}
              </pre>
            )}
          </div>
        </section>
      )}

      {activeTab === 'about' && (
        <section className="panel">
          <div className="card about-card">
            <h2>What is Sakhi?</h2>
            <p>
              Sakhi (सखी — &quot;companion&quot;) is an AI-powered tool that converts Hindi voice conversations between
              ASHA health workers and patients into structured medical forms — instantly, offline, on a single laptop.
            </p>

            <h3>The Problem</h3>
            <p>
              India&apos;s 1 million+ ASHA workers conduct home visits for antenatal care, postnatal care, deliveries,
              and child health. After each visit, they manually fill paper forms — a process that takes 15-20 minutes,
              is error-prone, and often delayed. Many ASHA workers have limited literacy, making form-filling the
              hardest part of their job.
            </p>

            <h3>How Sakhi Works</h3>
            <div className="pipeline-steps">
              <div className="step">
                <strong>1. Hindi Voice Input</strong>
                <span>Record or upload the ASHA-patient conversation in Hindi/Hinglish</span>
              </div>
              <div className="step">
                <strong>2. Speech Recognition</strong>
                <span>Whisper Large V2 (Hindi-specialized, 3000hrs training) transcribes with 95% accuracy</span>
              </div>
              <div className="step">
                <strong>3. Number Normalization</strong>
                <span>Custom algorithm converts Hindi number words (एक सो दस = 110) to digits</span>
              </div>
              <div className="step">
                <strong>4. Structured Extraction</strong>
                <span>Gemma 4 E4B extracts vitals, patient info, and clinical data into NHM-standard forms</span>
              </div>
              <div className="step">
                <strong>5. Danger Sign Detection</strong>
                <span>Flags life-threatening conditions with evidence quotes — zero false alarms on normal visits</span>
              </div>
            </div>

            <h3>Why It Matters</h3>
            <ul>
              <li><strong>15-20 min saved per visit</strong> — ASHA workers do 5-10 visits/day, that&apos;s 1-3 hours saved daily</li>
              <li><strong>Offline-first</strong> — runs on a laptop with no internet, critical for rural India where 60% of visits happen</li>
              <li><strong>Hindi-native</strong> — first tool to handle Hindi medical speech with code-switching (Hindi + English medical terms)</li>
              <li><strong>Anti-hallucination</strong> — strict null policy for unmentioned fields, evidence-based danger signs only</li>
              <li><strong>22-second pipeline</strong> — voice to completed form in under 25 seconds</li>
            </ul>

            <h3>Technology</h3>
            <div className="tech-grid">
              <div className="tech-item">
                <strong>LLM</strong>
                <span>Google Gemma 4 E4B (8B params, Q4_K_M quantized, 5GB)</span>
              </div>
              <div className="tech-item">
                <strong>ASR</strong>
                <span>Collabora Whisper Large V2 Hindi (CTranslate2, 6GB)</span>
              </div>
              <div className="tech-item">
                <strong>Inference</strong>
                <span>Ollama with JSON mode — 146 tok/s on consumer GPU</span>
              </div>
              <div className="tech-item">
                <strong>Frontend</strong>
                <span>React + Vite (PWA-ready)</span>
              </div>
              <div className="tech-item">
                <strong>Backend</strong>
                <span>FastAPI (Python)</span>
              </div>
              <div className="tech-item">
                <strong>GPU</strong>
                <span>Runs on any 16GB+ VRAM GPU (tested: RTX 5070 Ti)</span>
              </div>
            </div>
          </div>
        </section>
      )}

      {activeTab === 'history' && (
        <section className="panel">
          {viewingHistory ? (
            <div className="card">
              <div className="history-detail-header">
                <button className="btn secondary" onClick={() => setViewingHistory(null)}>&larr; Back</button>
                <h3>{prettyLabel(viewingHistory.visitType)} — {viewingHistory.date}</h3>
              </div>
              {viewingHistory.transcript && (
                <div style={{ marginBottom: 12 }}>
                  <h4 style={{ color: '#0f766e', marginBottom: 4 }}>Transcript</h4>
                  <pre className="transcript">{viewingHistory.transcript}</pre>
                </div>
              )}
              <div className="results-grid">
                <div>
                  <h4 style={{ color: '#0f766e', marginBottom: 8 }}>Form Data</h4>
                  <div className="kv-grid">
                    {keyValueRows(viewingHistory.form).map((row) => (
                      <div className="kv-row" key={`${row.key}-${row.value}`}>
                        <span>{row.key}</span>
                        <strong>{String(row.value)}</strong>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h4 style={{ color: '#0f766e', marginBottom: 8 }}>Danger Signs</h4>
                  <p className="referral">{prettyLabel(viewingHistory.danger?.referral_decision?.decision || 'No referral')}</p>
                  {(viewingHistory.danger?.danger_signs || []).map((item, idx) => (
                    <div className="danger-item" key={`${item.sign}-${idx}`}>
                      <strong>{item.sign}</strong>
                      <span>{prettyLabel(item.category)}</span>
                    </div>
                  ))}
                  {!(viewingHistory.danger?.danger_signs || []).length && <p>No danger signs detected.</p>}
                </div>
              </div>
            </div>
          ) : (
            <>
              <div className="history-header">
                <h2>Visit History</h2>
                <button className="btn secondary" onClick={() => { setHistory([]); localStorage.removeItem('sakhi_history') }}>Clear All</button>
              </div>
              <div className="history-list">
                {history.map((entry) => (
                  <div className="card history-entry" key={entry.id} onClick={() => setViewingHistory(entry)}>
                    <div className="history-meta">
                      <strong>{prettyLabel(entry.visitType)}</strong>
                      <span>{entry.source === 'voice' ? 'Voice' : entry.source === 'field' ? 'On-device' : 'Text'}</span>
                      <span>{entry.date}</span>
                      {entry.timing?.total_s && <span>{entry.timing.total_s}s</span>}
                    </div>
                    {entry.transcript && <p className="history-preview">{entry.transcript.slice(0, 100)}...</p>}
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      )}

      {activeState.error && <div className="error-banner">{activeState.error}</div>}

      {activeState.loading && pipelineStages.length > 0 && (
        <div className="card">
          <h3>Processing Pipeline</h3>
          <PipelineProgress stages={pipelineStages} />
        </div>
      )}

      {(activeState.form || activeState.danger) && (
        <section className="results-grid">
          <div className="card">
            <h3>
              Form Extraction {activeState.visitType ? <span className="muted">({prettyLabel(activeState.visitType)})</span> : null}
            </h3>
            {activeState.loading ? (
              <div className="loader">Running extraction pipeline...</div>
            ) : (
              <>
                <div className="kv-grid">
                  {keyValueRows(activeState.form).map((row) => (
                    <div className="kv-row" key={`${row.key}-${row.value}`}>
                      <span>{row.key}</span>
                      <strong>{String(row.value)}</strong>
                    </div>
                  ))}
                </div>
                <div className="export-buttons">
                  <button className="btn secondary" onClick={downloadJSON}>Export JSON</button>
                  <button className="btn secondary" onClick={downloadCSV}>Export CSV</button>
                </div>
              </>
            )}
          </div>

          <div className="card danger">
            <h3>Danger Signs & Referral</h3>
            {activeState.loading ? (
              <div className="loader">Analyzing danger signs...</div>
            ) : (
              <>
                <p className="referral">{prettyLabel(activeState.danger?.referral_decision?.decision || 'No referral decision')}</p>
                <p className="reason">{activeState.danger?.referral_decision?.reason || 'No reason provided.'}</p>
                <div className="danger-list">
                  {(activeState.danger?.danger_signs || []).map((item, idx) => (
                    <div className="danger-item" key={`${item.sign}-${idx}`}>
                      <strong>{item.sign}</strong>
                      <span>{prettyLabel(item.category)}</span>
                      {item.clinical_value ? <em>Value: {String(item.clinical_value)}</em> : null}
                      {item.utterance_evidence ? <p>&quot;{item.utterance_evidence}&quot;</p> : null}
                    </div>
                  ))}
                  {!dangerSigns.length && <p>No danger signs detected.</p>}
                </div>
              </>
            )}
          </div>
        </section>
      )}

      {activeState.timing && (
        <div className="timing">
          {Object.entries(activeState.timing).map(([k, v]) => {
            const isMs = k.endsWith('_ms')
            const label = prettyLabel(isMs ? k.slice(0, -3) : k)
            const display = isMs
              ? (Number(v) >= 1000 ? `${(Number(v) / 1000).toFixed(1)}s` : `${v}ms`)
              : `${v}s`
            return (
              <span key={k}>
                {label}: <strong>{display}</strong>
              </span>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default App
