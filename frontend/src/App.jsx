import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
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

  const dangerSigns = useMemo(() => textState.danger?.danger_signs || voiceState.danger?.danger_signs || [], [textState.danger, voiceState.danger])

  async function startRecording() {
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
      recorder.start()
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

  const activeState = activeTab === 'voice' ? voiceState : textState

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
                      <span>{entry.source === 'voice' ? 'Voice' : 'Text'}</span>
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
          {Object.entries(activeState.timing).map(([k, v]) => (
            <span key={k}>
              {prettyLabel(k)}: <strong>{String(v)}s</strong>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export default App
