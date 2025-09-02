# replay_session_all.py
# -------------------------------------------------------
# 목적
#  - 세션 파일(JSON/JSONL, .gz 자동) 로딩
#  - moves / moves_free 전개, hover/click/pointerdown/up 수집
#  - x_raw/y_raw(뷰포트 px) -> gx/gy(0..1) 정규화
#  - ROI 박스 표시 (roi_map 기반, answer-1~4 같은 색)
#  - OOB 판정은 "scratcha-container" 바깥만 OOB 로 계산
#  - DOWN/UP in·OOB 색상/마커로 명확히 구분
#  - ★ START 지점(추적 시작 위치) 표시
#
# 사용:
#   pip install pandas numpy matplotlib
#   python replay_session_all.py path/to/session.json[.gz] --show-legend --show-labels --save out
# -------------------------------------------------------

import os, sys, json, argparse
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========= 파일 로딩 =========
def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        head = f.read(2)
        rest = f.read()
    data = head + rest
    if head == b"\x1f\x8b":
        import gzip
        return gzip.decompress(data)
    return data

def load_session(path: str) -> Dict[str, Any]:
    raw = _read_bytes(path)
    txt = raw.decode("utf-8", errors="replace").strip()

    # 단일 JSON 시도
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict) and ("events" in obj or "meta" in obj):
            return obj
    except json.JSONDecodeError:
        pass

    # JSONL
    meta, events, label = None, [], None
    for line in txt.splitlines():
        if not line.strip(): continue
        rec = json.loads(line)
        t = rec.get("type")
        if t == "meta":
            meta = {k: v for k, v in rec.items() if k != "type"}
        elif t == "label":
            label = {k: v for k, v in rec.items() if k != "type"}
        elif t == "event":
            events.append({k: v for k, v in rec.items() if k != "type"})
        else:
            events.append(rec)
    return {"meta": meta, "events": events, "label": label}

# ========= ROI / 사각형 유틸 =========
def _coerce_rect_any(v):
    if v is None: return None
    if isinstance(v, dict):
        if "px" in v and isinstance(v["px"], dict):
            return _coerce_rect_any(v["px"])
        if "rect" in v:
            return _coerce_rect_any(v["rect"])
        def num(*names):
            for n in names:
                if n in v and v[n] is not None:
                    try: return float(v[n])
                    except: pass
            return None
        L = num("left","l","x")
        T = num("top","t","y")
        W = num("w","width")
        H = num("h","height")
        if None not in (L, T, W, H):
            return (float(L), float(T), float(W), float(H))
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try: return (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        except: return None
    if isinstance(v, str) and "," in v:
        parts = [p.strip() for p in v.split(",")]
        if len(parts) == 4:
            try: return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
            except: return None
    return None

def _px_rect_to_norm(r, vw: float, vh: float):
    x, y, w, h = r
    vw = max(float(vw), 1.0); vh = max(float(vh), 1.0)
    return (x/vw, y/vh, w/vw, h/vh)

def _sanitize_viewport(meta: Dict[str, Any],
                       df_moves: Optional[pd.DataFrame],
                       df_misc: Optional[pd.DataFrame]) -> Tuple[float, float]:
    vp = (meta or {}).get("viewport") or {}
    vw, vh = float(vp.get("w", 0) or 0), float(vp.get("h", 0) or 0)
    if vw >= 10 and vh >= 10:
        return vw, vh
    # 폴백: 관측된 최대 raw
    mx, my = 0.0, 0.0
    for d in [df_moves, df_misc]:
        if d is None or d.empty: continue
        if "x_raw" in d.columns: mx = max(mx, d["x_raw"].max(skipna=True))
        if "y_raw" in d.columns: my = max(my, d["y_raw"].max(skipna=True))
    if mx >= 10 and my >= 10:
        return float(mx), float(my)
    return 1920.0, 1080.0

def _scratcha_px_rect(meta: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """OOB 기준용: scratcha-container 사각형(px)"""
    rm = (meta or {}).get("roi_map") or {}
    r = _coerce_rect_any(rm.get("scratcha-container"))
    if r and r[2] > 0 and r[3] > 0:
        return r
    return None

# ========= 이벤트 전개 =========
def expand_moves(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    drag_id = -1
    seen_down = False
    for ev in events:
        et = ev.get("type")
        if et == "pointerdown":
            drag_id += 1
            seen_down = True
            continue
        if et not in ("moves", "moves_free"):
            continue

        pl = ev.get("payload") or {}
        bt  = int(pl.get("base_t", 0))
        dts = pl.get("dts") or []
        xrs = pl.get("xrs") or []
        yrs = pl.get("yrs") or []

        n = min(len(dts), len(xrs), len(yrs))
        T = bt
        for i in range(n):
            T += int(dts[i])
            rows.append({
                "phase": et,
                "drag_id": drag_id if (et == "moves" and seen_down) else -1,
                "t": int(T),
                "x_raw": float(xrs[i]),
                "y_raw": float(yrs[i]),
            })

    df = pd.DataFrame(rows)
    if df.empty: return df
    df = df.sort_values(["phase", "drag_id", "t"]).reset_index(drop=True)
    # 속도 계산(참고용)
    df["dx"] = df.groupby(["phase","drag_id"])["x_raw"].diff()
    df["dy"] = df.groupby(["phase","drag_id"])["y_raw"].diff()
    df["dt"] = df.groupby(["phase","drag_id"])["t"].diff().fillna(0).astype(int)
    dt_s = df["dt"].replace(0, 1).astype(float) / 1000.0
    df["speed"] = np.sqrt((df["dx"]**2 + df["dy"]**2)) / dt_s
    return df

def collect_misc(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for ev in events:
        typ = ev.get("type")
        if typ in ("hover","click","pointerdown","pointerup"):
            rows.append({
                "t": ev.get("t"),
                "kind": str(typ).upper(),
                "x_raw": float(ev.get("x_raw")) if ev.get("x_raw") is not None else np.nan,
                "y_raw": float(ev.get("y_raw")) if ev.get("y_raw") is not None else np.nan,
                "target_role": ev.get("target_role"),
                "target_answer": ev.get("target_answer"),
            })
    return pd.DataFrame(rows)

# ========= OOB 계산 (scratcha-container 기준) =========
def infer_oob_misc(df_misc: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """x_raw/y_raw와 scratcha-container 사각형(px)로 OOB 재계산"""
    if df_misc is None or df_misc.empty:
        return df_misc
    r = _scratcha_px_rect(meta)
    out = df_misc.copy()
    out["oob"] = 0
    out["on_canvas"] = 0
    if r:
        L, T, W, H = r
        x = out["x_raw"].astype(float)
        y = out["y_raw"].astype(float)
        inside = (x >= L) & (x <= L+W) & (y >= T) & (y <= T+H)
        out.loc[inside, "on_canvas"] = 1
        out.loc[~inside, "oob"] = 1
    return out

# ========= 좌표 정규화 (중요: 컬럼 보존) =========
def to_viewport_norm(df: pd.DataFrame, meta: Dict[str, Any],
                     vwvh: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
    if df is None or df.empty: return df
    if vwvh is None:
        vw, vh = _sanitize_viewport(meta, None, None)
    else:
        vw, vh = vwvh
    vw = max(vw, 1.0); vh = max(vh, 1.0)

    out = df.copy()
    out["gx"] = (out["x_raw"].astype(float) / vw).clip(-0.5, 1.5)
    out["gy"] = (out["y_raw"].astype(float) / vh).clip(-0.5, 1.5)
    # 중요한 컬럼 보존
    for col in ("oob","on_canvas","kind","target_role","target_answer","phase","drag_id","t"):
        if col in df.columns:
            out[col] = df[col]
    return out

# ========= ROI 변환/그리기 =========
def infer_role_boxes(meta: Dict[str, Any]) -> Dict[str, Tuple[float,float,float,float]]:
    """roi_map(px) → 뷰포트 정규좌표(0..1) dict[role] = (x,y,w,h)"""
    out: Dict[str, Tuple[float,float,float,float]] = {}
    vp = (meta or {}).get("viewport") or {}
    vw, vh = float(vp.get("w", 0) or 1), float(vp.get("h", 0) or 1)
    rm = (meta or {}).get("roi_map") or {}
    for role, raw_rect in rm.items():
        r = _coerce_rect_any(raw_rect)
        if not r or r[2] <= 0 or r[3] <= 0:
            continue
        out[role] = _px_rect_to_norm(r, vw, vh)
    return out

def _plot_rect(ax, rect: Tuple[float,float,float,float],
               label: Optional[str] = None,
               color: str = "gray",
               lw: float = 1.2,
               ls: str = "-",
               z: int = 3):
    x, y, w, h = rect
    xs = [x, x+w, x+w, x, x]
    ys = [y, y, y+h, y+h, y]
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls, label=label, zorder=z)

# ========= START 지점 추출 =========
def _find_start_point(df_misc_vp: Optional[pd.DataFrame],
                      df_moves_vp: Optional[pd.DataFrame]) -> Optional[Tuple[float,float,str,int]]:
    """
    START 우선순위:
      1) 가장 이른 POINTERDOWN
      2) 가장 이른 moves 포인트
      3) 가장 이른 moves_free 포인트
      4) 가장 이른 기타 MISC(click/up 등)
    반환: (gx, gy, source_label, t)
    """
    cand: List[Tuple[int,float,float,str]] = []

    if df_misc_vp is not None and not df_misc_vp.empty:
        downs = df_misc_vp[df_misc_vp["kind"]=="POINTERDOWN"]
        if not downs.empty:
            r = downs.sort_values("t", ascending=True).iloc[0]
            cand.append((int(r["t"]), float(r["gx"]), float(r["gy"]), "POINTERDOWN"))

    if df_moves_vp is not None and not df_moves_vp.empty:
        mv = df_moves_vp[df_moves_vp["phase"]=="moves"]
        if not mv.empty:
            r = mv.sort_values("t", ascending=True).iloc[0]
            cand.append((int(r["t"]), float(r["gx"]), float(r["gy"]), "moves"))
        mf = df_moves_vp[df_moves_vp["phase"]=="moves_free"]
        if not mf.empty:
            r = mf.sort_values("t", ascending=True).iloc[0]
            cand.append((int(r["t"]), float(r["gx"]), float(r["gy"]), "moves_free"))

    if df_misc_vp is not None and not df_misc_vp.empty:
        misc_any = df_misc_vp.sort_values("t", ascending=True).iloc[0]
        cand.append((int(misc_any["t"]), float(misc_any["gx"]), float(misc_any["gy"]), str(misc_any["kind"])))

    if not cand:
        return None

    cand.sort(key=lambda x: x[0])  # t 오름차순
    t, gx, gy, src = cand[0]
    return (gx, gy, src, t)

# ========= 리포트 =========
def basic_report(payload: Dict[str, Any], df_moves: pd.DataFrame, df_misc: pd.DataFrame):
    meta  = payload.get("meta") or {}
    evs   = payload.get("events") or []
    drags = df_moves[df_moves["phase"]=="moves"]["drag_id"].nunique() if df_moves is not None and not df_moves.empty else 0
    pts   = 0 if df_moves is None or df_moves.empty else len(df_moves)
    print("=== SESSION REPORT ===")
    print("events total:", len(evs), "| drags:", drags, "| move points(raw):", pts)
    rm = meta.get("roi_map") or {}
    if "scratcha-container" in rm:
        print("scratcha-container(px):", _coerce_rect_any(rm["scratcha-container"]))

# ========= 플롯 =========
def plot_fullscreen(df_moves_vp: pd.DataFrame,
                    df_misc_vp: pd.DataFrame,
                    meta: Dict[str,Any],
                    role_boxes: Dict[str,Tuple[float,float,float,float]],
                    save_path: Optional[str] = None,
                    show_labels: bool = False,
                    show_legend: bool = False,
                    debug: bool = False):
    fig, ax = plt.subplots()
    ax.set_title("Viewport-normalized (0..1) — ROIs + DRAG/FREE + MISC + START")
    ax.set_xlabel("gx (0..1)"); ax.set_ylabel("gy (0..1)")

    # ROI 박스 (answer-1~4: 파랑, answer-container: 연파랑, 나머지: 회색)
    if role_boxes:
        for role, rect in sorted(role_boxes.items(), key=lambda kv: (0 if kv[0]=="scratcha-container" else 1, kv[0])):
            # if role in ("answer-1","answer-2","answer-3","answer-4"):
            #     color = "blue"
            # elif role == "answer-container":
            #     color = "skyblue"
            # else:
            #     color = "gray"
            color = "gray"
            z = 1 if role == "scratcha-container" else 3
            _plot_rect(ax, rect, label=role, color=color, z=z)
            if show_labels:
                x, y, w, h = rect
                ax.text(x + 0.006, y + 0.014, role, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                        zorder=10)

    # DRAG 선
    if df_moves_vp is not None and not df_moves_vp.empty:
        mv = df_moves_vp[df_moves_vp["phase"]=="moves"]
        if not mv.empty:
            for gid, g in mv.groupby("drag_id"):
                ax.plot(g["gx"], g["gy"], linewidth=1.2, zorder=5, label=f"DRAG#{gid}")

    # FREE 선 (pre/post-drag)
    if df_moves_vp is not None and not df_moves_vp.empty:
        mf = df_moves_vp[df_moves_vp["phase"]=="moves_free"]
        if not mf.empty:
            ax.plot(mf["gx"], mf["gy"], linestyle="--", linewidth=1.0, color="#2aa876",
                    zorder=4, label="FREE (pre/post-drag)")

    # MISC (DOWN/UP in·OOB 각각 구분)
    if df_misc_vp is not None and not df_misc_vp.empty:
        down_in  = df_misc_vp[(df_misc_vp["kind"]=="POINTERDOWN") & (df_misc_vp["oob"]==0)]
        down_oob = df_misc_vp[(df_misc_vp["kind"]=="POINTERDOWN") & (df_misc_vp["oob"]==1)]
        up_in    = df_misc_vp[(df_misc_vp["kind"]=="POINTERUP")   & (df_misc_vp["oob"]==0)]
        up_oob   = df_misc_vp[(df_misc_vp["kind"]=="POINTERUP")   & (df_misc_vp["oob"]==1)]

        if not down_in.empty:
            ax.scatter(down_in["gx"], down_in["gy"], s=60, marker='v',
                       c="green", zorder=7, label="DOWN(in)")
        if not up_in.empty:
            ax.scatter(up_in["gx"], up_in["gy"], s=60, marker='^',
                       c="blue", zorder=7, label="UP(in)")
        if not down_oob.empty:
            ax.scatter(down_oob["gx"], down_oob["gy"], s=110, marker='v',
                       edgecolors="red", facecolors="none", linewidths=1.8,
                       zorder=9, label="DOWN(OOB)")
        if not up_oob.empty:
            ax.scatter(up_oob["gx"], up_oob["gy"], s=110, marker='^',
                       edgecolors="orange", facecolors="none", linewidths=1.8,
                       zorder=9, label="UP(OOB)")

    # ★ START 지점 (보라색 별, 가장 높은 zorder)
    start = _find_start_point(df_misc_vp, df_moves_vp)
    if start is not None:
        sx, sy, src, st = start
        if debug:
            print(f"[START] chosen from {src} at t={st}ms -> (gx,gy)=({sx:.3f},{sy:.3f})")
        ax.scatter([sx], [sy], s=220, marker='*', c="purple",
                   edgecolors="white", linewidths=0.8,
                   zorder=12, label="START")
        ax.text(sx+0.01, sy-0.015, "START", color="purple", fontsize=9,
                weight="bold", zorder=12,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="purple", alpha=0.7))

    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.invert_yaxis()
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=160, bbox_inches="tight"); print(f"[saved] {save_path}")
    plt.show()

# ========= main =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="session file (.json or .json.gz)")
    ap.add_argument("--save", metavar="DIR", default=None, help="PNG 저장 디렉토리")
    ap.add_argument("--show-labels", action="store_true", help="ROI 텍스트 라벨 표시")
    ap.add_argument("--show-legend", action="store_true", help="범례 표시")
    ap.add_argument("--debug", action="store_true", help="중간 데이터 출력")
    args = ap.parse_args()

    if not os.path.isfile(args.path):
        print(f"file not found: {args.path}"); sys.exit(1)

    payload = load_session(args.path)
    meta, events = payload.get("meta") or {}, payload.get("events") or []

    # 전개
    df_moves = expand_moves(events)
    df_misc  = collect_misc(events)

    # OOB 재계산 (핵심: scratcha-container 기준)
    df_misc  = infer_oob_misc(df_misc, meta)

    if args.debug:
        rm = meta.get("roi_map") or {}
        print("DEBUG: roi_map keys =", list(rm.keys()))
        print("DEBUG: scratcha-container(px) =", rm.get("scratcha-container"))
        print("DEBUG: misc kinds =", df_misc["kind"].unique() if not df_misc.empty else "EMPTY")
        if not df_misc.empty:
            print("DEBUG: OOB crosstab\n", df_misc.groupby(["kind","oob"]).size())

    # 정규화
    vwvh = _sanitize_viewport(meta, df_moves, df_misc)
    df_moves_vp = to_viewport_norm(df_moves, meta, vwvh) if df_moves is not None else df_moves
    df_misc_vp  = to_viewport_norm(df_misc,  meta, vwvh) if df_misc  is not None else df_misc

    # ROI 박스
    role_boxes = infer_role_boxes(meta)

    # 리포트
    basic_report(payload, df_moves, df_misc)

    # 저장 경로
    save1 = None
    if args.save:
        base = os.path.splitext(os.path.basename(args.path))[0]
        os.makedirs(args.save, exist_ok=True)
        save1 = os.path.join(args.save, f"{base}_fullscreen.png")

    # 플롯
    plot_fullscreen(df_moves_vp, df_misc_vp, meta, role_boxes,
                    save_path=save1,
                    show_labels=args.show_labels,
                    show_legend=args.show_legend,
                    debug=args.debug)

if __name__ == "__main__":
    main()
