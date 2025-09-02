#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_session_anim.py
--------------------------------------------
- 정적 리플레이(replay_session_all.py)와 동일 스키마/로직 준수
- ROI 보존: meta.roi_map 있으면 반드시 그림
  * dict({left,top,w,h}/{x,y,w,h}/{px:{...}}/{rect:{...}}), list/tuple[4], "l,t,w,h"
- 뷰포트 결정: meta.viewport → roi_map['scratcha-container'] → 이벤트 좌표 최댓값
- hover 전면 제거
- FREE ↔ DRAG 연속 궤적 (phase가 바뀌어도 선을 끊지 않음)
- 저장: --save-mp4 / --save-gif
- Matplotlib 일부 버전 blit/resize 이슈 회피 (blit=False)
"""

import os, json, gzip, argparse
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


# ========= 파일 로딩 =========
def _read_bytes_auto(path: str) -> bytes:
    with open(path, "rb") as f:
        raw = f.read()
    # gzip 헤더만 gzip 처리
    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        return gzip.decompress(raw)
    return raw

def load_session(path: str) -> Dict[str, Any]:
    txt = _read_bytes_auto(path).decode("utf-8", errors="replace").strip()
    # 단일 JSON 시도
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and ("events" in obj or "meta" in obj):
            return {
                "meta": obj.get("meta") or {},
                "events": obj.get("events") or [],
                "label": obj.get("label"),
            }
    except Exception:
        pass
    # JSONL 파싱
    meta, events, label = None, [], None
    for line in txt.splitlines():
        s = line.strip()
        if not s: continue
        rec = json.loads(s)
        t = rec.get("type")
        if t == "meta":
            meta = {k: v for k, v in rec.items() if k != "type"}
        elif t == "label":
            label = {k: v for k, v in rec.items() if k != "type"}
        elif t == "event":
            events.append({k: v for k, v in rec.items() if k != "type"})
        else:
            events.append(rec)
    return {"meta": meta or {}, "events": events, "label": label}


# ========= ROI/뷰포트 유틸 (all.py 동일 철학) =========
def _coerce_rect_any(v) -> Optional[Tuple[float,float,float,float]]:
    """ROI → (left, top, w, h) px. dict/list/tuple/str 및 중첩(px/rect) 지원."""
    if v is None:
        return None
    if isinstance(v, dict):
        if "px" in v and isinstance(v["px"], dict):
            return _coerce_rect_any(v["px"])
        if "rect" in v:
            return _coerce_rect_any(v["rect"])
        def num(*names):
            for n in names:
                if n in v and v[n] is not None:
                    try:
                        return float(v[n])
                    except Exception:
                        pass
            return None
        L = num("left","l","x"); T = num("top","t","y")
        W = num("w","width"); H = num("h","height")
        if None not in (L, T, W, H):
            return (L, T, W, H)
        # dict인데 0..3 인덱스 케이스
        try:
            return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        except Exception:
            return None
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        except Exception:
            return None
    if isinstance(v, str) and "," in v:
        parts = [p.strip() for p in v.split(",")]
        if len(parts) == 4:
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except Exception:
                return None
    return None

def _px_rect_to_norm(r, vw, vh):
    x, y, w, h = r
    vw = max(vw, 1.0); vh = max(vh, 1.0)
    return (x / vw, y / vh, w / vw, h / vh)

def _safe_viewport(meta: Dict[str,Any], events: List[Dict[str,Any]]) -> Tuple[float,float]:
    """meta.viewport → roi_map['scratcha-container'] → 이벤트 좌표 최댓값."""
    vp = (meta or {}).get("viewport") or {}
    vw = float(vp.get("w", 0) or 0)
    vh = float(vp.get("h", 0) or 0)
    if vw >= 2 and vh >= 2:
        return vw, vh
    rm = (meta or {}).get("roi_map") or {}
    sc = rm.get("scratcha-container") if isinstance(rm, dict) else None
    sc_rect = _coerce_rect_any(sc) if sc else None
    if sc_rect and sc_rect[2] > 1 and sc_rect[3] > 1:
        return sc_rect[2], sc_rect[3]
    # 이벤트 기반 추정
    mx = my = 0.0
    for ev in events:
        typ = ev.get("type")
        if typ in ("moves", "moves_free"):
            p = ev.get("payload") or ev
            xrs, yrs = p.get("xrs") or [], p.get("yrs") or []
            if xrs: mx = max(mx, float(max(xrs)))
            if yrs: my = max(my, float(max(yrs)))
        elif typ in ("pointerdown", "pointerup", "click"):
            xr, yr = ev.get("x_raw"), ev.get("y_raw")
            if xr is not None: mx = max(mx, float(xr))
            if yr is not None: my = max(my, float(yr))
    return max(1.0, mx * 1.05), max(1.0, my * 1.05)

def infer_role_boxes(meta: Dict[str,Any], vw: float, vh: float, ensure_root=True) -> Dict[str,Tuple[float,float,float,float]]:
    """roi_map → 정규 ROI. scratcha-container 없으면 (0,0,1,1) 보강."""
    out: Dict[str, Tuple[float,float,float,float]] = {}
    rm = (meta or {}).get("roi_map") or {}
    if isinstance(rm, dict) and rm:
        for role, raw in rm.items():
            fr = _coerce_rect_any(raw)
            if fr and fr[2] > 0 and fr[3] > 0:
                x,y,w,h = _px_rect_to_norm(fr, vw, vh)
                # 살짝 클리핑
                x = max(-0.1, min(1.1, x)); y = max(-0.1, min(1.1, y))
                w = max(0.0, min(1.2, w));  h = max(0.0, min(1.2, h))
                out[role] = (x,y,w,h)
    if ensure_root and "scratcha-container" not in out:
        out["scratcha-container"] = (0.0, 0.0, 1.0, 1.0)
    return out


# ========= 이벤트 전개 =========
def expand_moves(events: List[Dict[str,Any]]):
    """all.py와 동일: moves/moves_free 전개 + DRAG 식별용 drag_id 부여."""
    rows=[]; drag_id=-1; seen_down=False
    for ev in events:
        et=ev.get("type")
        if et=="pointerdown":
            drag_id+=1; seen_down=True; continue
        if et not in ("moves","moves_free"): continue
        pl=ev.get("payload") or {}
        bt=int(pl.get("base_t",0))
        dts=pl.get("dts") or []; xrs=pl.get("xrs") or []; yrs=pl.get("yrs") or []
        n=min(len(dts),len(xrs),len(yrs)); T=bt
        for i in range(n):
            T+=int(dts[i])
            rows.append({"phase":et,"drag_id":drag_id if et=="moves" and seen_down else -1,
                         "t":int(T),"x_raw":float(xrs[i]),"y_raw":float(yrs[i])})
    import pandas as pd, numpy as np
    df=pd.DataFrame(rows)
    if df.empty: return df
    df=df.sort_values(["t"]).reset_index(drop=True)
    # 뷰포트 정규화는 이후 단계에서 처리
    return df

def collect_misc(events: List[Dict[str,Any]]):
    """click/pointerdown/pointerup만 수집(hover 제거)."""
    rows=[]
    for ev in events:
        typ=ev.get("type")
        if typ in ("click","pointerdown","pointerup"):
            xr,yr=ev.get("x_raw"),ev.get("y_raw")
            rows.append({"t":ev.get("t"),
                         "kind":str(typ).upper(),
                         "x_raw":float(xr) if xr is not None else None,
                         "y_raw":float(yr) if yr is not None else None,
                         "target_role":ev.get("target_role"),
                         "target_answer":ev.get("target_answer")})
    import pandas as pd
    return pd.DataFrame(rows)


# ========= 애니메이션 =========
def animate_session(sess: Dict[str,Any], args):
    meta=sess.get("meta") or {}
    events=sess.get("events") or []

    # 1) 뷰포트 확정 → 2) ROI 정규화 → 3) 이벤트 전개 → 4) 정규화 좌표 준비
    vw,vh=_safe_viewport(meta, events)
    role_boxes=infer_role_boxes(meta, vw, vh, ensure_root=True)

    df_moves=expand_moves(events)
    df_misc=collect_misc(events)

    # 뷰포트 정규화 좌표(gx, gy)
    def _to_norm_xy(x_raw, y_raw):
        return (float(x_raw)/max(1.0,vw), float(y_raw)/max(1.0,vh))

    pts=[]
    if df_moves is not None and not df_moves.empty:
        for r in df_moves.sort_values("t").itertuples():
            gx,gy=_to_norm_xy(r.x_raw, r.y_raw)
            pts.append({"t":int(r.t), "gx":gx, "gy":gy, "phase":str(r.phase)})

    # 궤적 애니메이션용 시퀀스
    if not pts:
        print("No movement points to animate."); return
    pts.sort(key=lambda r:r["t"])

    xs=[p["gx"] for p in pts]
    ys=[p["gy"] for p in pts]
    ph=[p["phase"] for p in pts]

    # 이벤트 마커
    idx_down=idx_up=idx_click=[]
    if df_misc is not None and not df_misc.empty:
        idx_down  = df_misc.index[df_misc["kind"]=="POINTERDOWN"].tolist()
        idx_up    = df_misc.index[df_misc["kind"]=="POINTERUP"].tolist()
        idx_click = df_misc.index[df_misc["kind"]=="CLICK"].tolist()
        misc_x = (df_misc["x_raw"].astype(float)/max(1.0,vw)).tolist()
        misc_y = (df_misc["y_raw"].astype(float)/max(1.0,vh)).tolist()

    # 색상 규칙
    def color_for_phase(p):
        if p == "moves": return "#00bcd4"        # DRAG
        if p == "moves_free": return "#ff9800"   # FREE
        return "black"

    # Figure
    fig,ax=plt.subplots(figsize=(9,6))
    ax.set_title("Scratcha Replay Animation — Continuous Trajectory")
    ax.set_xlabel("gx (0..1)"); ax.set_ylabel("gy (0..1)")
    ax.set_xlim(-0.05,1.05); ax.set_ylim(1.05,-0.05)  # y축 반전

    # ROI 박스 (항상 유지)
    for role, rect in role_boxes.items():
        x,y,w,h = rect
        xs_r=[x, x+w, x+w, x, x]
        ys_r=[y, y, y+h, y+h, y]
        ax.plot(xs_r, ys_r, lw=1.2, color="black", alpha=0.9, zorder=1, label=role)
        if getattr(args, "show_labels", False):
            ax.text(x+0.005, y+0.015, role, fontsize=8, alpha=0.9, zorder=2)

    # 정적 마커(있을 경우)
    if df_misc is not None and not df_misc.empty:
        if idx_down:
            ax.scatter([misc_x[i] for i in idx_down],  [misc_y[i] for i in idx_down],
                       marker="v", c="#1565c0", s=50, label="DOWN", zorder=3)
        if idx_up:
            ax.scatter([misc_x[i] for i in idx_up],    [misc_y[i] for i in idx_up],
                       marker="^", c="#c62828", s=50, label="UP", zorder=3)
        if idx_click:
            ax.scatter([misc_x[i] for i in idx_click], [misc_y[i] for i in idx_click],
                       marker="*", c="#e91e63", s=60, label="CLICK", zorder=3)

    # 움직이는 포인터
    pointer = ax.scatter([], [], s=30, zorder=4)

    # 연속 궤적: phase가 바뀌어도 선을 끊지 않음(이전 점 phase 색으로 그리기)
    def init():
        pointer.set_offsets(np.empty((0,2)))
        return (pointer,)

    def update(i):
        pointer.set_offsets([[xs[i], ys[i]]])
        if i > 0:
            x0,y0 = xs[i-1], ys[i-1]
            x1,y1 = xs[i],   ys[i]
            col = color_for_phase(ph[i-1])
            ax.plot([x0,x1], [y0,y1], lw=1.6, alpha=0.95, color=col, zorder=2)
        return (pointer,)

    ani=FuncAnimation(fig, update, frames=len(xs), init_func=init, interval=30, blit=False, repeat=False)

    if getattr(args, "show_legend", False):
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # 저장/표시
    if args.save_mp4:
        fname = args.save_mp4 if args.save_mp4.endswith(".mp4") else args.save_mp4 + ".mp4"
        try:
            ani.save(fname, writer=FFMpegWriter(fps=30))
        except Exception:
            ani.save(fname, writer=PillowWriter(fps=20))
        print("[saved]", fname)
    elif args.save_gif:
        fname = args.save_gif if args.save_gif.endswith(".gif") else args.save_gif + ".gif"
        ani.save(fname, writer=PillowWriter(fps=20))
        print("[saved]", fname)
    else:
        plt.show()


# ========= main =========
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--show-legend", action="store_true")
    ap.add_argument("--show-labels", action="store_true")
    ap.add_argument("--save-mp4")
    ap.add_argument("--save-gif")
    args=ap.parse_args()

    sess=load_session(args.file)
    animate_session(sess, args)

if __name__ == "__main__":
    main()
