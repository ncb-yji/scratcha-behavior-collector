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

// ---------- Shadow DOM 호환 유틸 ----------
function* walkDeep(root) {
  const stack = [root]
  while (stack.length) {
    const node = stack.pop()
    if (!node) continue
    if (node.nodeType === 1) { // Element
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
function getRectPx(el) {
  if (!el) return null
  const r = el.getBoundingClientRect()
  return { left: r.left, top: r.top, w: r.width, h: r.height }
}

// 업로드 직전 슬림 스키마
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

function rectOfRole(role) {
  const el = document.querySelector(`[data-role="${role}"]`)
  if (!el) return null
  const r = el.getBoundingClientRect()
  return { left: r.left, top: r.top, w: r.width, h: r.height }
}

const ROI_FALLBACK_FRAC = {
  'instruction-area': { x: 0.05, y: 0.02, w: 0.90, h: 0.10 },
  'canvas-container': { x: 0.05, y: 0.14, w: 0.90, h: 0.48 },
  'answers-area': { x: 0.05, y: 0.66, w: 0.90, h: 0.28 },
  'refresh-button': { x: 0.90, y: 0.02, w: 0.08, h: 0.08 },
}

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

  const postCollect = async (label) => {
    if (sentRef.current) return
    try {
      stopMoveTimer(); stopFreeMoveTimer()
      flushMoves(); flushFreeMoves()
      const roiMap = buildRoiMap()
      const metaFull = buildMeta(sessionId, roiMap)
      const fullPayload = { meta: metaFull, events: eventsRef.current, label }
      const payload = pruneForUpload(fullPayload)
      const res = await fetch(`${API_URL}/collect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        keepalive: true,
      })
      if (!res.ok) {
        console.warn('collect failed:', res.status, await res.text().catch(() => ''))
        return
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

  function onPointerDown(e) {
    deviceRef.current = String(e.pointerType || 'unknown')
    const n = toNorm(e.clientX, e.clientY)
    if (!n) return
    const t = nowMs()
    isDraggingRef.current = true
    startMoveTimer()
    // FREE 버퍼 클리어 → DRAG 중간에 섞이지 않도록
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
    // FREE 버퍼 초기화 → DRAG 뒤에 FREE가 붙는 것 방지
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

    // ✅ click은 root에서
    rootEl.addEventListener?.('click', onClick, optCap)

    // ✅ pointerdown은 window에서 (위젯 밖에서도 DOWN 수집)
    const optWin = { passive: true, capture: true }
    window.addEventListener('pointerdown', onPointerDown, optWin)
    window.addEventListener('pointermove', onPointerMove, optWin)
    window.addEventListener('pointerup', onPointerUp, optWin)
    window.addEventListener('pointercancel', onPointerCancel, optWin)

    // canvas-container 갱신 감지
    const mo = new MutationObserver(() => {
      const latest = getElByRole('canvas-container')
      if (latest && latest !== canvasRef.current) {
        canvasRef.current = latest
        roiVersionRef.current += 1
      }
    })
    mo.observe(document, { childList: true, subtree: true, attributes: true })

  const onBeforeUnload = () => {
    try {
      // 진행 중인 버퍼/타이머를 마무리
      stopMoveTimer(); 
      stopFreeMoveTimer();
      flushMoves(); 
      flushFreeMoves();

      // 최신 ROI/메타와 이벤트 수집
      const roiMap = buildRoiMap();
      const metaFull = buildMeta(sessionId, roiMap);
      const fullPayload = { meta: metaFull, events: eventsRef.current, label: undefined };
      const payload = pruneForUpload(fullPayload);

      // 언로드 시점에는 sendBeacon이 가장 안정적
      if (navigator.sendBeacon) {
        const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
        navigator.sendBeacon(`${API_URL}/collect`, blob);
      } else {
        // 폴백: 실패해도 무시
        fetch(`${API_URL}/collect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          keepalive: true,
        }).catch(() => {});
      }
    } catch {
      // 언로드 시점 에러는 로깅 생략/무시
    }
  };
  window.addEventListener('beforeunload', onBeforeUnload);

    return () => {
      rootEl.removeEventListener?.('click', onClick, optCap)

      window.removeEventListener('pointerdown', onPointerDown, optWin)
      window.removeEventListener('pointermove', onPointerMove, optWin)
      window.removeEventListener('pointerup', onPointerUp, optWin)
      window.removeEventListener('pointercancel', onPointerCancel, optWin)

      window.removeEventListener('beforeunload', onBeforeUnload)
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
      {/* {finished && ( */}
        {/* <div className="retry-bar" style={{ marginTop: 12 }}> */}
          {/* <button data-role="refresh-button" onClick={resetSession}>다시 풀기</button> */}
        {/* </div> */}
      {/* )} */}
    </>
  )
}

export default App
