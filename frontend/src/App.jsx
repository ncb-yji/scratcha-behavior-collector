import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { ScratchaWidget } from 'scratcha-sdk'
import './App.css'

/** 세션 식별용 임의 ID */
function uuid() {
  return (crypto.randomUUID?.() || Math.random().toString(36).slice(2))
}
/** ms 단위 상대 타임스탬프 */
function nowMs() {
  return (performance?.now?.() ?? Date.now())
}

// ---- 환경변수 ----
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const SCRATCHA_API_KEY = import.meta.env.VITE_SCRATCHA_API_KEY ?? ''
const SCRATCHA_ENDPOINT = import.meta.env.VITE_SCRATCHA_ENDPOINT ?? 'https://api.scratcha.cloud'

// ---- 타이밍 ----
const MOVE_FLUSH_MS = 50
const FREE_FLUSH_MS = 120

// ---- 업로드 한도/유틸 (다운샘플링 없이, 큰 페이로드는 조각 전송) ----
const MAX_SINGLE_UPLOAD_BYTES = 900_000;     // 이 크기 이하면 /collect 단일 업로드
const TARGET_CHUNK_BYTES      = 300_000;     // 조각 하나 목표 크기(대략값)
const FETCH_TIMEOUT_MS        = 20_000;

function estimateBytes(obj) {
  try { return new TextEncoder().encode(JSON.stringify(obj)).length } catch { return (JSON.stringify(obj)||'').length }
}
function* walkDeep(root) {
  const stack = [root]
  while (stack.length) {
    const node = stack.pop()
    if (!node) continue
    if (node.nodeType === 1) {
      yield node
      const children = node.children || []
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i])
      const sr = node.shadowRoot
      if (sr) stack.push(sr)
    } else if (node instanceof ShadowRoot) {
      const children = node.children || []
      for (let i = children.length - 1; i >= 0; i--) stack.push(children[i])
    }
  }
}
function getElByRole(role) {
  for (const el of walkDeep(document)) {
    if (el?.matches && el.matches(`[data-role="${role}"]`)) return el
  }
  return null
}
function getRoleFromEl(el) {
  try {
    if (!el || !el.closest) return ''
    const hit = el.closest('[data-role]')
    return hit?.getAttribute?.('data-role') || ''
  } catch { return '' }
}
function elementFromPointDeep(x, y) {
  let el = document.elementFromPoint(x, y)
  let last = null
  while (el && el.shadowRoot && el.shadowRoot.elementFromPoint) {
    const inner = el.shadowRoot.elementFromPoint(x, y)
    if (!inner || inner === last || inner === el) break
    last = el = inner
  }
  return el
}
function getRoleAtPoint(x, y) {
  const el = elementFromPointDeep(x, y)
  return getRoleFromEl(el)
}

// ---- ROI 유틸 ----
function rectOfRole(role) {
  const el = document.querySelector(`[data-role="${role}"]`)
  if (!el) return null
  const r = el.getBoundingClientRect()
  return { left: r.left, top: r.top, w: r.width, h: r.height }
}

// 업로드 직전 슬림 스키마(원본 값 보존, 구조만 슬림화)
function pruneForUpload(fullPayload) {
  const meta = fullPayload?.meta || {};
  const evs = Array.isArray(fullPayload?.events) ? fullPayload.events : [];
  const labelIn = fullPayload?.label || undefined;

  const metaSlim = {
    session_id: meta.session_id,
    viewport: meta.viewport ? { w: Number(meta.viewport.w || 0), h: Number(meta.viewport.h || 0) } :
      { w: window.innerWidth, h: window.innerHeight },
    ts_resolution_ms: 1,
    roi_map: meta.roi_map || {},
  };
  if (meta.device) metaSlim.device = String(meta.device);
  if (meta.dpr != null) metaSlim.dpr = Number(meta.dpr);

  const slimEvents = [];
  for (const e of evs) {
    const t = e?.type;
    if (t === 'moves' || t === 'moves_free') {
      const p = e?.payload || {};
      slimEvents.push({
        type: t,
        payload: {
          base_t: Number(p.base_t || 0),
          dts: (p.dts || []).map(n => Math.max(1, Number(n))),
          xrs: (p.xrs || []).map(Number),
          yrs: (p.yrs || []).map(Number),
        }
      })
      continue
    }
    if (t === 'pointerdown' || t === 'pointerup' || t === 'click') {
      const out = {
        type: t,
        t: Number(e.t || 0),
        x_raw: Number(e.x_raw),
        y_raw: Number(e.y_raw),
      }
      if (t === 'click') {
        if (e.target_role) out.target_role = String(e.target_role)
        if (e.target_answer) out.target_answer = String(e.target_answer)
      }
      slimEvents.push(out)
      continue
    }
  }

  let label = undefined
  if (labelIn && (labelIn.passed != null || labelIn.selectedAnswer || labelIn.error)) {
    label = {}
    if (labelIn.passed != null) label.passed = labelIn.passed ? 1 : 0
    if (labelIn.selectedAnswer) label.selectedAnswer = String(labelIn.selectedAnswer)
    if (labelIn.error) label.error = String(labelIn.error)
  }

  return { meta: metaSlim, events: slimEvents, label }
}

// ---- 클릭 시 업로드(단일 or 조각) ----
function App() {
  const [runKey, setRunKey] = useState(0)
  const [finished, setFinished] = useState(false)

  const sessionId = useMemo(() => uuid(), [])
  const t0Ref = useRef(nowMs())

  const hostRef = useRef(null)
  const rootRef = useRef(null)
  const canvasRef = useRef(null)

  useLayoutEffect(() => {
    rootRef.current = hostRef.current
    canvasRef.current = getElByRole('canvas-container') || canvasRef.current
  }, [runKey])

  const eventsRef = useRef([])
  const moveBufRef = useRef([])
  const freeBufRef = useRef([])
  const moveTimerRef = useRef(null)
  const freeTimerRef = useRef(null)
  const isDraggingRef = useRef(false)
  const sentRef = useRef(false)
  const lastAnswerRef = useRef(null)

  const deviceRef = useRef('unknown')
  const roiVersionRef = useRef(1)

  const stopMoveTimer = () => { if (moveTimerRef.current) { clearInterval(moveTimerRef.current); moveTimerRef.current = null } }
  const startMoveTimer = () => { if (!moveTimerRef.current) moveTimerRef.current = setInterval(() => flushMoves(), MOVE_FLUSH_MS) }
  const stopFreeMoveTimer = () => { if (freeTimerRef.current) { clearInterval(freeTimerRef.current); freeTimerRef.current = null } }
  const startFreeMoveTimer = () => { if (!freeTimerRef.current) freeTimerRef.current = setInterval(() => flushFreeMoves(), FREE_FLUSH_MS) }

  function buildRoiMap() {
    const roles = [
      'scratcha-container',
      'canvas-container',
      'instruction-area',
      'instruction-container',
      'refresh-button',
      'answer-container',
      'answer-1', 'answer-2', 'answer-3', 'answer-4',
    ]
    const rm = {}
    for (const k of roles) {
      const rr = rectOfRole(k)
      if (rr && rr.w > 0 && rr.h > 0) rm[k] = rr
    }
    delete rm['instruction-container']
    return rm
  }

  function buildMeta(sessionId, roiMap) {
    return {
      session_id: sessionId,
      device: deviceRef.current,
      viewport: { w: window.innerWidth, h: window.innerHeight },
      dpr: window.devicePixelRatio || 1,
      ts_resolution_ms: 1,
      roi_map: roiMap || {},
    }
  }

  function toNorm(clientX, clientY) {
    const el = canvasRef.current || getElByRole('canvas-container') || hostRef.current
    if (!el) return null
    const r = el.getBoundingClientRect()
    const x_raw = clientX, y_raw = clientY
    const xr = (x_raw - r.left) / Math.max(1, r.width)
    const yr = (y_raw - r.top) / Math.max(1, r.height)
    const oob = (xr < 0 || xr > 1 || yr < 0 || yr > 1) ? 1 : 0
    const x = Math.min(1, Math.max(0, xr))
    const y = Math.min(1, Math.max(0, yr))
    const on_canvas = oob ? 0 : 1
    return { x, y, x_raw, y_raw, on_canvas, oob }
  }

  const flushMoves = () => {
    const buf = moveBufRef.current
    if (!buf.length) return
    const base_t = Math.round(buf[0].t - t0Ref.current)
    const xrs = [], yrs = [], dts = []
    for (let i = 0; i < buf.length; i++) {
      const cur = buf[i]
      const prevT = i === 0 ? buf[0].t : buf[i - 1].t
      xrs.push(cur.x_raw)
      yrs.push(cur.y_raw)
      dts.push(Math.max(1, Math.round(cur.t - prevT)))
    }
    eventsRef.current.push({ type: 'moves', payload: { base_t, dts, xrs, yrs } })
    moveBufRef.current = []
  }

  const flushFreeMoves = () => {
    const buf = freeBufRef.current
    if (!buf.length) return
    const base_t = Math.round(buf[0].t - t0Ref.current)
    const xrs = [], yrs = [], dts = []
    for (let i = 0; i < buf.length; i++) {
      const cur = buf[i]
      const prevT = i === 0 ? buf[0].t : buf[i - 1].t
      xrs.push(cur.x_raw)
      yrs.push(cur.y_raw)
      dts.push(Math.max(1, Math.round(cur.t - prevT)))
    }
    eventsRef.current.push({ type: 'moves_free', payload: { base_t, dts, xrs, yrs } })
    freeBufRef.current = []
  }

  // ---- 조각 업로드 보조 함수 ----
  async function fetchJSON(url, body, { timeoutMs = FETCH_TIMEOUT_MS } = {}) {
    const ac = new AbortController()
    const timer = setTimeout(() => ac.abort('timeout'), timeoutMs)
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        keepalive: false,
        signal: ac.signal,
      })
      clearTimeout(timer)
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status} ${txt}`)
      }
      return await res.json().catch(() => ({}))
    } finally {
      clearTimeout(timer)
    }
  }

  function sliceEventsByBytes(meta, events, targetBytes) {
    // meta는 첫 조각에만 포함, 이후 조각은 meta: {session_id} 만 최소 포함
    const chunks = []
    let cur = []
    let curBytes = estimateBytes({ meta, events: [] })
    for (const ev of events) {
      const addBytes = estimateBytes(ev)
      if (cur.length > 0 && (curBytes + addBytes) > targetBytes) {
        chunks.push(cur)
        cur = []
        curBytes = estimateBytes({ meta: { session_id: meta.session_id }, events: [] })
      }
      cur.push(ev)
      curBytes += addBytes
    }
    if (cur.length) chunks.push(cur)
    return chunks
  }

  const postCollect = async (label) => {
    if (sentRef.current) return
    try {
      stopMoveTimer(); stopFreeMoveTimer()
      flushMoves(); flushFreeMoves()
      const roiMap = buildRoiMap()
      const metaFull = buildMeta(sessionId, roiMap)
      const fullPayload = { meta: metaFull, events: eventsRef.current, label }
      const slim = pruneForUpload(fullPayload) // 구조만 슬림화(원본 값 유지)

      // 1) 작은 페이로드면 기존 /collect 한 번 호출 (원래 코드 흐름):contentReference[oaicite:1]{index=1}
      const singleBytes = estimateBytes({ meta: slim.meta, events: slim.events, label: slim.label })
      if (singleBytes <= MAX_SINGLE_UPLOAD_BYTES) {
        await fetchJSON(`${API_URL}/collect`, { meta: slim.meta, events: slim.events, label: slim.label })
      } else {
        // 2) 큰 페이로드면 조각 업로드 → 최종 한 번만 파일 생성
        const { meta, events } = slim
        const chunks = sliceEventsByBytes(meta, events, TARGET_CHUNK_BYTES)
        // 첫 조각에는 full meta, 이후에는 최소 meta(session_id)만
        for (let i = 0; i < chunks.length; i++) {
          const isFirst = (i === 0)
          const partMeta = isFirst ? meta : { session_id: meta.session_id }
          await fetchJSON(`${API_URL}/collect_chunk`, {
            meta: partMeta,
            events: chunks[i],
            label: null,
            part_index: i,
            total_parts: chunks.length,
          })
        }
        // 마지막에 한 번만 finalize: label 포함, 서버에서 단일 파일 업로드:contentReference[oaicite:2]{index=2}
        await fetchJSON(`${API_URL}/collect_finalize`, {
          meta: { session_id: meta.session_id },
          events: [],   // 이벤트는 이미 조각으로 보냄
          label: slim.label ?? label ?? undefined,
        })
      }

      sentRef.current = true
      eventsRef.current = []
      moveBufRef.current = []
      freeBufRef.current = []
    } catch (e) {
      console.warn('collect error:', e)
    }
  }

  const resetSession = () => {
    try { stopMoveTimer() } catch { }
    try { stopFreeMoveTimer() } catch { }
    eventsRef.current = []
    moveBufRef.current = []
    freeBufRef.current = []
    sentRef.current = false
    lastAnswerRef.current = null
    t0Ref.current = nowMs()
    setFinished(false)
    setRunKey(k => k + 1)
  }

  const handleSuccess = (result) => {
    const selectedAnswer = result?.selectedAnswer ?? result?.answer ?? lastAnswerRef.current ?? null
    postCollect({ passed: 1, selectedAnswer }).finally(() => setFinished(true))
  }
  const handleError = (error) => {
    const msg = (error?.message ?? error)?.toString?.() ?? 'unknown'
    postCollect({ passed: 0, error: msg }).finally(() => setFinished(true))
  }

  function toNorm(clientX, clientY) {
    const el = canvasRef.current || getElByRole('canvas-container') || hostRef.current
    if (!el) return null
    const r = el.getBoundingClientRect()
    const x_raw = clientX, y_raw = clientY
    const xr = (x_raw - r.left) / Math.max(1, r.width)
    const yr = (y_raw - r.top) / Math.max(1, r.height)
    const oob = (xr < 0 || xr > 1 || yr < 0 || yr > 1) ? 1 : 0
    const x = Math.min(1, Math.max(0, xr))
    const y = Math.min(1, Math.max(0, yr))
    const on_canvas = oob ? 0 : 1
    return { x, y, x_raw, y_raw, on_canvas, oob }
  }

  function onPointerDown(e) {
    deviceRef.current = String(e.pointerType || 'unknown')
    const n = toNorm(e.clientX, e.clientY)
    if (!n) return
    const t = nowMs()
    isDraggingRef.current = true
    startMoveTimer()
    moveBufRef.current = []
    freeBufRef.current = []
    eventsRef.current.push({ t: Math.round(t - t0Ref.current), type: 'pointerdown', x_raw: n.x_raw, y_raw: n.y_raw })
    moveBufRef.current.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
  }
  function onPointerMove(e) {
    if (e.pointerType) deviceRef.current = String(e.pointerType)
    const n = toNorm(e.clientX, e.clientY)
    if (!n) return
    const t = nowMs()
    if (isDraggingRef.current) {
      moveBufRef.current.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
    } else {
      startFreeMoveTimer()
      freeBufRef.current.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
    }
  }
  function onPointerUp(e) {
    const n = toNorm(e.clientX, e.clientY)
    if (!n) { isDraggingRef.current = false; stopMoveTimer(); flushMoves(); return }
    const t = nowMs()
    if (isDraggingRef.current) {
      moveBufRef.current.push({ t, x: n.x, y: n.y, x_raw: n.x_raw, y_raw: n.y_raw })
    }
    isDraggingRef.current = false
    stopMoveTimer()
    flushMoves()
    freeBufRef.current = []
    eventsRef.current.push({ t: Math.round(t - t0Ref.current), type: 'pointerup', x_raw: n.x_raw, y_raw: n.y_raw })
  }
  function onPointerCancel() {
    isDraggingRef.current = false
    stopMoveTimer()
    flushMoves()
  }
  function onClick(e) {
    const role = getRoleAtPoint(e.clientX, e.clientY)
    const answerText = (e.target?.getAttribute?.('data-answer') || '').trim() || null
    const n = toNorm(e.clientX, e.clientY)
    if (!n) return
    const t = Math.round(nowMs() - t0Ref.current)
    if (role?.startsWith('answer-')) {
      lastAnswerRef.current = (answerText || e.target?.textContent || '').trim()
    }
    eventsRef.current.push({
      t, type: 'click', x_raw: n.x_raw, y_raw: n.y_raw,
      target_role: String(role || ''), target_answer: String(answerText || '')
    })
    if (role === 'refresh-button') {
      e.preventDefault()
      resetSession()
    }
  }

  useEffect(() => {
    const rootEl = hostRef.current || document
    const optCap = { capture: true, passive: true }

    rootEl.addEventListener?.('click', onClick, optCap)

    const optWin = { passive: true, capture: true }
    window.addEventListener('pointerdown', onPointerDown, optWin)
    window.addEventListener('pointermove', onPointerMove, optWin)
    window.addEventListener('pointerup', onPointerUp, optWin)
    window.addEventListener('pointercancel', onPointerCancel, optWin)

    const mo = new MutationObserver(() => {
      const latest = getElByRole('canvas-container')
      if (latest && latest !== canvasRef.current) {
        canvasRef.current = latest
        roiVersionRef.current += 1
      }
    })
    mo.observe(document, { childList: true, subtree: true, attributes: true })

    return () => {
      rootEl.removeEventListener?.('click', onClick, optCap)
      window.removeEventListener('pointerdown', onPointerDown, optWin)
      window.removeEventListener('pointermove', onPointerMove, optWin)
      window.removeEventListener('pointerup', onPointerUp, optWin)
      window.removeEventListener('pointercancel', onPointerCancel, optWin)
      mo.disconnect()
      stopMoveTimer(); stopFreeMoveTimer()
    }
  }, [runKey])

  return (
    <>
      <div data-role="scratcha-container" ref={hostRef} style={{ display: 'inline-block' }}>
        <ScratchaWidget
          key={runKey}
          apiKey={SCRATCHA_API_KEY}
          endpoint={SCRATCHA_ENDPOINT}
          mode="normal"
          onSuccess={handleSuccess}
          onError={handleError}
        />
      </div>
    </>
  )
}

export default App
