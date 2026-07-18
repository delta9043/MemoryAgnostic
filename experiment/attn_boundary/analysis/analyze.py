"""
analyze.py — Phase B: 추출된 attention 값의 순수 분석 (판정·청커 연결 없음).

extract_attn.py의 충분통계에서 turn별 어텐션 값들을 파생해, 경계 근처에서
어텐션이 실제로 어떻게 행동하는지를 그림과 표로만 보여준다. 결론은 사람이 낸다.

분석 목록 (output/ 에 PNG + summary.md):
  1. profile_*.png    : 경계 정렬 평균 프로파일 (경계=0, ±k turn 곡선)
  2. decomposition.png: 어텐션 질량 분해 (직전/먼 문맥/자기/특수토큰) 경계 vs 중간
  3. rawmap_*.png     : 예시 turn의 attention map 히트맵 (turn 단위로 접음, 층별)
  4. timeline_*.png   : conv별 시계열 + 세션(빨강)/gold(회색) 경계 오버레이
  5. dist_*.png       : 값 분포 비교 (경계 vs 중간, 층별 히스토그램)
  6. bias_*.png       : 값 vs turn 길이/위치 산점도 (측정 건전성)
  7. layer_evolution.png : 층 0~11에서 두 그룹 평균의 변화 (분리가 깊이에 따라?)
  8. profile_gold_vs_sess.png : 세션 경계 vs (세션 아닌) gold 경계 프로파일 비교
  9. distance_decay.png : t−k 거리별 질량 감쇠 곡선 (경계 vs 중간)

실행 (GPU 불필요):
  python experiment/attn_boundary/analysis/analyze.py \
      --gold data/chunked_data/chunks_fable-5.json \
      --layers 8,9,10,11 \
      --map_layers 8,9,10,11 \
      --tag upper-layers

주요 옵션 (실행마다 output/타임스탬프[_tag]/ 하위에 격리 저장됨):
  --layers L,L,...     대표층 — 프로파일/분포/분해/거리감쇠/summary 표에 사용.
                       timeline·bias는 이 중 마지막 층. (기본 0,5,11)
  --map_layers L,L,... 전체 attention 지도(attnmap_*)를 그릴 층.
                       (기본: --layers의 마지막 층 1개. 8,9,10,11 = LightMem 상위층)
  --tag NAME           산출물 디렉토리명 메모 (예: upper-layers, k10) — 실행 구분용
  --out DIR            산출물 루트 변경 (기본: analysis/output)
  --k N                경계 정렬 프로파일 반경 ±N turn (기본 5)
  --gold A.json B.json 2개 주면 교집합(둘 다 동의한 경계)만 gold 라벨로 사용
  * layer_evolution.png은 옵션과 무관하게 항상 전 층(0~11)을 보여줌 —
    이걸 먼저 보고 갈라지는 층을 --layers/--map_layers에 넣어 재실행 추천.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── 파생 값 정의 ──────────────────────────────────────────────────────────
QUANTITIES = {
    "mass_prev": "직전 turn(t-1)으로 가는 질량",
    "pairmean_prev": "직전 turn 토큰쌍 평균",
    "max_prev": "직전 turn 최강 토큰 연결",
    "ratio_prev": "문맥 중 직전 turn 비중",
    "entropy_ctx": "문맥 분포 정규화 엔트로피",
    "mass_far": "먼 문맥(t-2 이하) 질량",
    "mass_self": "자기 turn 내부 질량",
    "mass_cls": "CLS 질량",
}


def load_stats(stats_dir):
    out = {}
    for f in sorted(Path(stats_dir).glob("stats_*.npz")):
        d = np.load(f, allow_pickle=True)
        out[f.stem[len("stats_"):]] = {k: d[k] for k in d.files}
    return out


def derive(d):
    """conv 1개 → {quantity: [N, L]} + 거리별 질량 {k: [N, L]} (NaN=미정의)"""
    scored, n_tokens = d["scored"], d["n_tokens"]
    owner, prev, nk = d["pair_owner"], d["pair_prev"], d["pair_nk"]
    psum, pmax = d["pair_sum"], d["pair_max"]
    N = len(scored)
    L = psum.shape[1] if len(psum) else d["cls_mass"].shape[1]

    Q = {q: np.full((N, L), np.nan, np.float32) for q in QUANTITIES}
    DIST = {k: np.full((N, L), np.nan, np.float32) for k in range(1, 9)}
    for t in np.where(scored)[0]:
        idx = np.where(owner == t)[0]
        if not len(idx):
            continue
        n_q = float(n_tokens[t])
        u = prev[idx]
        mass = psum[idx] / n_q
        ctx = u != t
        Q["mass_cls"][t] = d["cls_mass"][t]
        self_r = np.where(u == t)[0]
        if len(self_r):
            Q["mass_self"][t] = mass[self_r[0]]
        pr = np.where(u == t - 1)[0]
        if len(pr):
            r = pr[0]
            Q["mass_prev"][t] = mass[r]
            Q["pairmean_prev"][t] = psum[idx[r]] / (n_q * nk[idx[r]])
            Q["max_prev"][t] = pmax[idx[r]]
            ctx_total = mass[ctx].sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                Q["ratio_prev"][t] = np.where(ctx_total > 0, mass[r] / ctx_total, np.nan)
            Q["mass_far"][t] = ctx_total - mass[r]
        K = int(ctx.sum())
        if K > 1:
            ctx_total = mass[ctx].sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = mass[ctx] / ctx_total
                Q["entropy_ctx"][t] = -np.nansum(p * np.log(p), axis=0) / np.log(K)
        for j in np.where(ctx)[0]:                   # 거리별 질량
            k = t - int(u[j])
            if 1 <= k <= 8:
                DIST[k][t] = mass[j]
    return Q, DIST


def build_turn_matrix(d, layer):
    """충분통계 → conv 전체 turn×turn pair-mean 행렬 (윈도우 밴드 밖은 NaN).
    M[t, u] = turn t(query)가 turn u(key)에 주는 토큰쌍 평균 attention."""
    N = len(d["scored"])
    n_tokens = d["n_tokens"]
    M = np.full((N, N), np.nan, np.float32)
    owner, prev, nk = d["pair_owner"], d["pair_prev"], d["pair_nk"]
    psum = d["pair_sum"][:, layer]
    for p in range(len(owner)):
        t, u = int(owner[p]), int(prev[p])
        M[t, u] = psum[p] / (float(n_tokens[t]) * float(nk[p]))
    return M


def session_starts(session_ids):
    s = [str(x) for x in session_ids]
    return {i for i in range(1, len(s)) if s[i] != s[i - 1]}


def gold_starts(paths):
    per = []
    for p in paths:
        m = {}
        for rec in json.load(open(p, encoding="utf-8")):
            starts, acc = set(), 0
            for c in rec["chunks"][:-1]:
                acc += len(c["turns"])
                starts.add(acc)
            m[rec["sample_id"]] = starts
        per.append(m)
    out = {}
    for sid in per[0]:
        s = set(per[0][sid])
        for m in per[1:]:
            s &= m.get(sid, set())
        out[sid] = s
    return out


def spearman(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


# ── 메인 ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Phase B: attention 값 분석 (판정 없음)")
    ap.add_argument("--stats_dir", type=str,
                    default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--gold", type=str, nargs="*", default=[],
                    help="gold chunk JSON들 (2개 이상이면 교집합 경계 사용)")
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "output"),
                    help="산출물 루트. 실행마다 하위에 타임스탬프 디렉토리 생성")
    ap.add_argument("--tag", type=str, default="",
                    help="실행 디렉토리명에 붙일 메모 (예: layers-all, k10)")
    ap.add_argument("--layers", type=str, default="0,5,11", help="그림에 쓸 대표 층")
    ap.add_argument("--map_layers", type=str, default="",
                    help="전체 attention 지도(attnmap_*)를 그릴 층. 기본=--layers의 마지막")
    ap.add_argument("--k", type=int, default=5, help="경계 정렬 프로파일 반경(±k turn)")
    args = ap.parse_args()

    stats = load_stats(args.stats_dir)
    if not stats:
        sys.exit(f"[analyze] {args.stats_dir} 에 stats_*.npz 없음")
    # 실행마다 격리된 디렉토리: output/YYYYmmdd_HHMMSS[_tag]/
    run_name = time.strftime("%Y%m%d_%H%M%S") + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] 산출물 디렉토리: {out_dir}", flush=True)
    show_layers = [int(x) for x in args.layers.split(",")]
    gold = gold_starts(args.gold) if args.gold else {}

    # conv별 파생값·라벨
    conv = {}
    for sid, d in stats.items():
        Q, DIST = derive(d)
        conv[sid] = {
            "Q": Q, "DIST": DIST, "N": len(d["scored"]),
            "sess": session_starts(d["session_ids"]),
            "gold": gold.get(sid, set()),
            "n_tokens": d["n_tokens"],
        }
    L = next(iter(conv.values()))["Q"]["mass_prev"].shape[1]
    print(f"[analyze] conv {len(conv)}개, layers {L}, 대표층 {show_layers}", flush=True)

    # 풀링 (그룹 비교용): boundary = 세션 시작, mid = 세션/gold 어느 쪽도 아님
    pool = {q: [] for q in QUANTITIES}
    is_b, is_g, n_tok, rel_pos = [], [], [], []
    for sid, c in conv.items():
        N = c["N"]
        for q in QUANTITIES:
            pool[q].append(c["Q"][q])
        is_b += [i in c["sess"] for i in range(N)]
        is_g += [(i in c["gold"]) and (i not in c["sess"]) for i in range(N)]
        n_tok += list(c["n_tokens"])
        rel_pos += list(np.arange(N) / max(1, N - 1))
    P = {q: np.concatenate(pool[q]) for q in QUANTITIES}
    is_b = np.array(is_b); is_g = np.array(is_g)
    mid = ~is_b & ~is_g
    n_tok = np.array(n_tok, float); rel_pos = np.array(rel_pos)
    print(f"[analyze] turns={len(is_b)} 세션경계={is_b.sum()} "
          f"gold전용경계={is_g.sum()} 중간={mid.sum()}", flush=True)

    # ── summary.md (기술 통계 표만) ─────────────────────────────────────
    def med_iqr(x):
        x = x[~np.isnan(x)]
        if not len(x):
            return "-"
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        return f"{q2:.4f} [{q1:.4f},{q3:.4f}]"

    lines = ["# attn_boundary Phase B — 어텐션 값 분석 (기술 통계)\n",
             f"- 실행: {run_name} | stats_dir: {args.stats_dir}",
             f"- 설정: layers={args.layers}, k=±{args.k}" + (f", tag={args.tag}" if args.tag else ""),
             f"- conv {len(conv)}개 / turns {len(is_b)} / layers {L}",
             f"- 세션경계 {int(is_b.sum())}, gold전용경계 {int(is_g.sum())}, 중간 {int(mid.sum())}",
             f"- gold 소스: {args.gold if args.gold else '(없음)'}\n",
             "## 값별 중앙값 [IQR] — 경계 vs 중간 (대표층)\n"]
    for l in show_layers:
        lines.append(f"### layer {l}\n")
        lines.append("| 값 | 세션경계 | gold전용경계 | 중간 |")
        lines.append("|---|---|---|---|")
        for q in QUANTITIES:
            x = P[q][:, l]
            lines.append(f"| {q} | {med_iqr(x[is_b])} | {med_iqr(x[is_g])} | {med_iqr(x[mid])} |")
        lines.append("")
    lines.append("## 측정 건전성 (spearman, 0에 가까울수록 무편향)\n")
    lines.append("| 값 | vs turn 길이 | vs conv 내 상대위치 |")
    lines.append("|---|---|---|")
    for q in QUANTITIES:
        x = P[q][:, show_layers[-1]]
        lines.append(f"| {q} (L{show_layers[-1]}) | {spearman(x, n_tok):+.3f} | {spearman(x, rel_pos):+.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[analyze] summary.md 저장", flush=True)

    # ── 플롯 ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print("[analyze] matplotlib 없음 — 플롯 생략 (summary.md만 생성)", flush=True)
        return

    def robust_lognorm(m):
        """작은 attention 값들의 차이가 드러나도록 퍼센타일 클리핑 + 로그 스케일."""
        pos = m[np.isfinite(m) & (m > 0)]
        if not len(pos):
            return None
        return LogNorm(vmin=np.percentile(pos, 5), vmax=np.percentile(pos, 99.5))

    def aligned_profile(qname, layer, centers_key):
        """경계=0 정렬 평균 곡선. centers_key: 'sess' | 'gold_only'"""
        k = args.k
        acc = np.zeros(2 * k + 1); cnt = np.zeros(2 * k + 1)
        for c in conv.values():
            centers = c["sess"] if centers_key == "sess" else (c["gold"] - c["sess"])
            s = c["Q"][qname][:, layer]
            for b in centers:
                for off in range(-k, k + 1):
                    t = b + off
                    if 0 <= t < c["N"] and not np.isnan(s[t]):
                        acc[off + k] += s[t]; cnt[off + k] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    offsets = np.arange(-args.k, args.k + 1)

    # 1) 경계 정렬 프로파일 (값 × 대표층)
    prof_qs = ["mass_prev", "max_prev", "entropy_ctx", "mass_self", "mass_cls", "ratio_prev"]
    fig, axes = plt.subplots(len(prof_qs), len(show_layers),
                             figsize=(4 * len(show_layers), 2.4 * len(prof_qs)),
                             sharex=True)
    for i, q in enumerate(prof_qs):
        for j, l in enumerate(show_layers):
            ax = axes[i][j] if len(show_layers) > 1 else axes[i]
            ax.plot(offsets, aligned_profile(q, l, "sess"), marker="o", ms=3)
            ax.axvline(0, color="crimson", ls="--", lw=0.8)
            if j == 0:
                ax.set_ylabel(q, fontsize=8)
            if i == 0:
                ax.set_title(f"layer {l}")
    axes[-1][len(show_layers) // 2].set_xlabel("relative turn from session boundary (0 = boundary turn)")
    fig.suptitle("Boundary-locked average profile", y=1.001)
    fig.tight_layout()
    fig.savefig(out_dir / "profile_boundary_locked.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 8) 세션 vs gold전용 경계 프로파일 (mass_prev)
    fig, axes = plt.subplots(1, len(show_layers), figsize=(4 * len(show_layers), 3), sharey=True)
    for j, l in enumerate(show_layers):
        ax = axes[j] if len(show_layers) > 1 else axes
        ax.plot(offsets, aligned_profile("mass_prev", l, "sess"), marker="o", ms=3,
                label="session boundary")
        if is_g.sum() >= 10:
            ax.plot(offsets, aligned_profile("mass_prev", l, "gold_only"), marker="s", ms=3,
                    label="gold-only boundary (within session)")
        ax.axvline(0, color="crimson", ls="--", lw=0.8)
        ax.set_title(f"layer {l}")
    (axes[0] if len(show_layers) > 1 else axes).set_ylabel("mass_prev")
    (axes[0] if len(show_layers) > 1 else axes).legend(fontsize=8)
    fig.suptitle("Session vs within-session topic boundary - mass_prev profile")
    fig.tight_layout()
    fig.savefig(out_dir / "profile_gold_vs_sess.png", dpi=150)
    plt.close(fig)

    # 2) 질량 분해 (경계 vs 중간, 층별 스택 바)
    comps = ["mass_prev", "mass_far", "mass_self", "mass_cls"]
    fig, axes = plt.subplots(1, len(show_layers), figsize=(3.6 * len(show_layers), 3.6), sharey=True)
    for j, l in enumerate(show_layers):
        ax = axes[j] if len(show_layers) > 1 else axes
        groups = {"boundary": is_b, "mid-topic": mid}
        bottoms = np.zeros(2)
        for comp in comps:
            vals = [np.nanmean(P[comp][:, l][g]) for g in groups.values()]
            ax.bar(list(groups), vals, bottom=bottoms, label=comp)
            bottoms += np.array(vals)
        ax.set_title(f"layer {l}")
        if j == len(show_layers) - 1:
            ax.legend(fontsize=7)
    fig.suptitle("Attention mass decomposition (mean)")
    fig.tight_layout()
    fig.savefig(out_dir / "decomposition.png", dpi=150)
    plt.close(fig)

    # 7) 층별 진화 (mass_prev, entropy_ctx)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, q in zip(axes, ["mass_prev", "entropy_ctx"]):
        bm = [np.nanmean(P[q][:, l][is_b]) for l in range(L)]
        mm = [np.nanmean(P[q][:, l][mid]) for l in range(L)]
        ax.plot(range(L), bm, marker="o", label="session boundary")
        ax.plot(range(L), mm, marker="s", label="mid-topic")
        ax.set_xlabel("layer"); ax.set_title(q); ax.legend(fontsize=8)
    fig.suptitle("Per-layer group means")
    fig.tight_layout()
    fig.savefig(out_dir / "layer_evolution.png", dpi=150)
    plt.close(fig)

    # 9) 거리 감쇠 (t−k 질량, 경계 vs 중간)
    fig, axes = plt.subplots(1, len(show_layers), figsize=(4 * len(show_layers), 3.4), sharey=True)
    ks = range(1, 9)
    for j, l in enumerate(show_layers):
        ax = axes[j] if len(show_layers) > 1 else axes
        for gname, gmask in [("boundary", "sess"), ("mid-topic", None)]:
            ys = []
            for k in ks:
                vals = []
                for c in conv.values():
                    s = c["DIST"][k][:, l]
                    if gmask == "sess":
                        pick = [t for t in c["sess"] if t < c["N"]]
                    else:
                        pick = [t for t in range(c["N"])
                                if t not in c["sess"] and t not in c["gold"]]
                    vals += [s[t] for t in pick if not np.isnan(s[t])]
                ys.append(np.mean(vals) if vals else np.nan)
            ax.plot(list(ks), ys, marker="o", ms=3, label=gname)
        ax.set_title(f"layer {l}"); ax.set_xlabel("distance k (turn t-k)")
        if j == 0:
            ax.set_ylabel("mass"); ax.legend(fontsize=8)
    fig.suptitle("Mass decay by distance")
    fig.tight_layout()
    fig.savefig(out_dir / "distance_decay.png", dpi=150)
    plt.close(fig)

    # 5) 분포 비교 히스토그램
    for q in ["mass_prev", "max_prev", "entropy_ctx", "mass_self"]:
        fig, axes = plt.subplots(1, len(show_layers), figsize=(4 * len(show_layers), 3))
        for j, l in enumerate(show_layers):
            ax = axes[j] if len(show_layers) > 1 else axes
            x = P[q][:, l]
            v = ~np.isnan(x)
            ax.hist(x[mid & v], bins=40, density=True, alpha=0.55, label="mid-topic")
            ax.hist(x[is_b & v], bins=40, density=True, alpha=0.55, label="session boundary")
            ax.set_title(f"layer {l}")
        (axes[0] if len(show_layers) > 1 else axes).legend(fontsize=8)
        fig.suptitle(f"{q} distribution")
        fig.tight_layout()
        fig.savefig(out_dir / f"dist_{q}.png", dpi=150)
        plt.close(fig)

    # 6) 편향 산점도
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    l = show_layers[-1]
    x = P["mass_prev"][:, l]
    v = ~np.isnan(x)
    axes[0].scatter(n_tok[v], x[v], s=3, alpha=0.25)
    axes[0].set_xlabel("turn token length"); axes[0].set_ylabel(f"mass_prev L{l}")
    axes[0].set_title(f"vs length (spearman {spearman(x, n_tok):+.3f})")
    axes[1].scatter(rel_pos[v], x[v], s=3, alpha=0.25)
    axes[1].set_xlabel("relative position in conv")
    axes[1].set_title(f"vs position (spearman {spearman(x, rel_pos):+.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "bias_scatter.png", dpi=150)
    plt.close(fig)

    # 4) conv별 시계열
    for sid, c in conv.items():
        s = c["Q"]["mass_prev"][:, show_layers[-1]]
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(s, lw=0.8)
        for b in c["sess"]:
            ax.axvline(b, color="crimson", ls="--", lw=0.8, alpha=0.8)
        for b in c["gold"] - c["sess"]:
            ax.axvline(b, color="gray", ls=":", lw=0.7, alpha=0.6)
        ax.set_title(f"{sid}: mass_prev L{show_layers[-1]} (red=session, gray=gold-only)")
        fig.tight_layout()
        fig.savefig(out_dir / f"timeline_{sid}.png", dpi=150)
        plt.close(fig)

    # 10) conv 전체 turn×turn attention 지도 + 경계 오버레이 (DHSA Fig.11b 스타일)
    map_layers = ([int(x) for x in args.map_layers.split(",")]
                  if args.map_layers else [show_layers[-1]])
    for sid, c in conv.items():
        d = stats[sid]
        for l in map_layers:
            M = build_turn_matrix(d, l)
            norm = robust_lognorm(M)
            if norm is None:
                continue
            N = M.shape[0]
            fig, ax = plt.subplots(figsize=(11, 10))
            Mm = np.ma.masked_invalid(M)
            cmap = plt.cm.viridis.copy()
            cmap.set_bad("#f0f0f0")                  # 윈도우 밴드 밖 = 연회색
            im = ax.imshow(Mm, cmap=cmap, norm=norm, interpolation="nearest")
            for b in c["sess"]:                      # 세션 경계 = 빨간 실선
                ax.axhline(b - 0.5, color="red", lw=0.9, alpha=0.9)
                ax.axvline(b - 0.5, color="red", lw=0.9, alpha=0.9)
            for b in c["gold"] - c["sess"]:          # gold 전용 경계 = 빨간 점선
                ax.axhline(b - 0.5, color="red", lw=0.7, ls="--", alpha=0.6)
                ax.axvline(b - 0.5, color="red", lw=0.7, ls="--", alpha=0.6)
            ax.set_xlabel("key turn u")
            ax.set_ylabel("query turn t")
            ax.set_title(f"{sid} L{l} turn-pair attention (log scale) | "
                         f"solid red=session, dashed red=gold-only")
            fig.colorbar(im, fraction=0.046)
            fig.tight_layout()
            fig.savefig(out_dir / f"attnmap_{sid}_L{l}.png", dpi=150)
            plt.close(fig)

    # 3) 예시 raw map (turn 단위로 접은 히트맵, 층별)
    for mf in sorted(Path(args.stats_dir).glob("maps_*.npz")):
        md = np.load(mf)
        ex_layers = list(md["example_layers"]) if "example_layers" in md.files else []
        for key in [k for k in md.files if k.startswith("map_t")]:
            t_idx = key[len("map_t"):]
            t2t = md[f"tok2turn_t{t_idx}"]
            maps = md[key].astype(np.float32)        # [len(ex_layers), N, N]
            turns_u = [u for u in np.unique(t2t) if u >= 0]
            T = len(turns_u)
            if T < 2:
                continue
            fig, axes = plt.subplots(1, maps.shape[0],
                                     figsize=(4.2 * maps.shape[0], 3.8))
            for j in range(maps.shape[0]):
                fold = np.zeros((T, T))
                for a, ua in enumerate(turns_u):
                    ia = np.where(t2t == ua)[0]
                    for b2, ub in enumerate(turns_u):
                        ib = np.where(t2t == ub)[0]
                        fold[a, b2] = maps[j][np.ix_(ia, ib)].mean()
                ax = axes[j] if maps.shape[0] > 1 else axes
                im = ax.imshow(fold, cmap="viridis", norm=robust_lognorm(fold))
                lab = f"L{ex_layers[j]}" if j < len(ex_layers) else f"ex{j}"
                ax.set_title(lab, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"{mf.stem} — query turn t={t_idx} (last row block = turn t)")
            fig.tight_layout()
            fig.savefig(out_dir / f"rawmap_{mf.stem}_t{t_idx}.png", dpi=150)
            plt.close(fig)

    print(f"[analyze] 플롯 저장 완료 → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
