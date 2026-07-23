"""LoCoMo Boundary Explorer HTML에 gold 경계를 합쳐 넣는다.

원본(att/cos/cmb만 있는 LightMem 산출물)을 읽어 gold 레일/통계/필터를 추가하고
html/locomo_boundary_ALL.html 로 덮어쓴다. 원본은 건드리지 않으므로 재실행 가능.

사용: python merge_gold_html.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = Path(r"C:\Program Code\memory_agnostic_project\LightMem\boundary_results\locomo_boundary_ALL.html")
GOLD = HERE / "results" / "locomo10_gold.json"
OUT = HERE / "html" / "locomo_boundary_ALL.html"


def gold_index(sample: dict) -> dict:
    """청크(세션 순서 flat) → flat turn index 기준 경계/토픽/세션시작."""
    bounds, topics, sess_starts = [], {}, []
    pos, prev_sid = 0, None
    for c in sample["chunks"]:
        if pos > 0:
            bounds.append(pos)
        topics[pos] = c["topic"]
        if c["session_id"] != prev_sid:
            sess_starts.append(pos)
            prev_sid = c["session_id"]
        pos += len(c["turns"])
    return {"gold": bounds, "gtopic": topics, "sess": sess_starts, "n": pos}


def main():
    src = SRC.read_text(encoding="utf-8")
    m = re.search(r'(<script id="data" type="application/json">)(.*?)(</script>)', src, re.S)
    data = json.loads(m.group(2))

    gold = json.load(open(GOLD, encoding="utf-8"))
    for sample in gold["samples"]:
        sid = sample["sample_id"]
        g = gold_index(sample)
        d = data[sid]
        assert g["n"] == len(d["turns"]), f"{sid}: turn 수 불일치 {g['n']} != {len(d['turns'])}"
        d["gold"] = g["gold"]
        d["gtopic"] = {str(k): v for k, v in g["gtopic"].items()}
        d["sess"] = g["sess"]

    missing = [c for c in data if "gold" not in data[c]]
    assert not missing, f"gold 없는 대화: {missing}"

    out = src[: m.start(2)] + json.dumps(data, ensure_ascii=False) + src[m.end(2) :]
    for old, new in PATCHES:
        assert old in out, f"패치 앵커 없음: {old[:60]!r}"
        out = out.replace(old, new, 1)

    OUT.write_text(out, encoding="utf-8")
    tot_g = sum(len(data[c]["gold"]) for c in data)
    tot_s = sum(len(data[c]["sess"]) for c in data)
    print(f"[html] saved: {OUT}")
    print(f"[html] gold 경계 {tot_g}개 (세션 시작 {tot_s}개 포함), 대화 {len(data)}개")


# ── HTML/CSS/JS 패치 (앵커 → 치환) ────────────────────────────────────
PATCHES = [
    # 1) gold 색 변수 (라이트/다크/명시 테마 3곳)
    ('  --att:#d9820f; --cos:#0f92aa; --cmb:#7c53e6;\n'
     '  --att-soft:#fbe7d1; --cos-soft:#d5eef2; --cmb-soft:#e7ddfa;\n'
     '  --good:#1a936f; --warn:#c26b1a;',
     '  --att:#d9820f; --cos:#0f92aa; --cmb:#7c53e6; --gld:#c2185b;\n'
     '  --att-soft:#fbe7d1; --cos-soft:#d5eef2; --cmb-soft:#e7ddfa; --gld-soft:#fadbe6;\n'
     '  --good:#1a936f; --warn:#c26b1a;'),
    ('@media (prefers-color-scheme:dark){:root{\n'
     '  --bg:#121619; --panel:#191d21; --panel2:#20262b;\n'
     '  --ink:#e7eaec; --muted:#939ba3; --faint:#6b747c; --line:#2a3037;\n'
     '  --att:#f0a341; --cos:#3cbdd4; --cmb:#a988f5;\n'
     '  --att-soft:#3a2a12; --cos-soft:#123037; --cmb-soft:#251a3f;',
     '@media (prefers-color-scheme:dark){:root{\n'
     '  --bg:#121619; --panel:#191d21; --panel2:#20262b;\n'
     '  --ink:#e7eaec; --muted:#939ba3; --faint:#6b747c; --line:#2a3037;\n'
     '  --att:#f0a341; --cos:#3cbdd4; --cmb:#a988f5; --gld:#f2739f;\n'
     '  --att-soft:#3a2a12; --cos-soft:#123037; --cmb-soft:#251a3f; --gld-soft:#3d1327;'),
    (':root[data-theme="light"]{\n'
     '  --bg:#f7f8f9; --panel:#ffffff; --panel2:#f2f4f6;\n'
     '  --ink:#1a1d21; --muted:#6b7280; --faint:#9aa2ab; --line:#e4e7ea;\n'
     '  --att:#d9820f; --cos:#0f92aa; --cmb:#7c53e6;\n'
     '  --att-soft:#fbe7d1; --cos-soft:#d5eef2; --cmb-soft:#e7ddfa;',
     ':root[data-theme="light"]{\n'
     '  --bg:#f7f8f9; --panel:#ffffff; --panel2:#f2f4f6;\n'
     '  --ink:#1a1d21; --muted:#6b7280; --faint:#9aa2ab; --line:#e4e7ea;\n'
     '  --att:#d9820f; --cos:#0f92aa; --cmb:#7c53e6; --gld:#c2185b;\n'
     '  --att-soft:#fbe7d1; --cos-soft:#d5eef2; --cmb-soft:#e7ddfa; --gld-soft:#fadbe6;'),
    (':root[data-theme="dark"]{\n'
     '  --bg:#121619; --panel:#191d21; --panel2:#20262b;\n'
     '  --ink:#e7eaec; --muted:#939ba3; --faint:#6b747c; --line:#2a3037;\n'
     '  --att:#f0a341; --cos:#3cbdd4; --cmb:#a988f5;\n'
     '  --att-soft:#3a2a12; --cos-soft:#123037; --cmb-soft:#251a3f;',
     ':root[data-theme="dark"]{\n'
     '  --bg:#121619; --panel:#191d21; --panel2:#20262b;\n'
     '  --ink:#e7eaec; --muted:#939ba3; --faint:#6b747c; --line:#2a3037;\n'
     '  --att:#f0a341; --cos:#3cbdd4; --cmb:#a988f5; --gld:#f2739f;\n'
     '  --att-soft:#3a2a12; --cos-soft:#123037; --cmb-soft:#251a3f; --gld-soft:#3d1327;'),

    # 2) 통계 타일 2x2 + gold 타일 색
    ('.stat-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}',
     '.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}'),
    ('.tile.att .v{color:var(--att)} .tile.cos .v{color:var(--cos)} .tile.cmb .v{color:var(--cmb)}',
     '.tile.att .v{color:var(--att)} .tile.cos .v{color:var(--cos)} .tile.cmb .v{color:var(--cmb)}\n'
     '.tile.gld .v{color:var(--gld)}'),
    ('.tile.att::before{background:var(--att)} .tile.cos::before{background:var(--cos)} .tile.cmb::before{background:var(--cmb)}',
     '.tile.att::before{background:var(--att)} .tile.cos::before{background:var(--cos)} .tile.cmb::before{background:var(--cmb)}\n'
     '.tile.gld::before{background:var(--gld)}'),

    # 3) gold 대비 필터 점 + 선택 방법 버튼
    ('.filt[data-f="all3"] .dot{background:linear-gradient(90deg,var(--att) 33%,var(--cos) 33% 66%,var(--cmb) 66%)}',
     '.filt[data-f="all3"] .dot{background:linear-gradient(90deg,var(--att) 33%,var(--cos) 33% 66%,var(--cmb) 66%)}\n'
     '.filt[data-f="miss"] .dot{background:var(--gld)}\n'
     '.filt[data-f="fp"] .dot{background:var(--faint)}\n'
     '.filt[data-f="hit"] .dot{background:linear-gradient(90deg,var(--gld) 50%,var(--good) 50%)}'),
    ('.seg button.on[data-m="cmb"]{color:var(--cmb)}',
     '.seg button.on[data-m="cmb"]{color:var(--cmb)}\n'
     '.seg button.on[data-m="gold"]{color:var(--gld)}'),

    # 4) 레일/헤더에 G 칼럼 추가
    ('.legend .la{background:var(--att)} .legend .lc{background:var(--cos)} .legend .lm{background:var(--cmb)}',
     '.legend .la{background:var(--att)} .legend .lc{background:var(--cos)} .legend .lm{background:var(--cmb)}\n'
     '.legend .lg{background:var(--gld)}\n'
     '.prf{margin-left:auto;display:flex;gap:10px;font-family:var(--mono);font-size:11.5px;\n'
     '  color:var(--muted);font-variant-numeric:tabular-nums}\n'
     '.prf b{color:var(--ink);font-weight:650}'),
    ('.colhdr .cA{color:var(--att)}.colhdr .cC{color:var(--cos)}.colhdr .cM{color:var(--cmb)}\n'
     ':root{--cols:52px 84px 30px 30px 30px 1fr}',
     '.colhdr .cA{color:var(--att)}.colhdr .cC{color:var(--cos)}.colhdr .cM{color:var(--cmb)}.colhdr .cG{color:var(--gld)}\n'
     ':root{--cols:52px 84px 30px 30px 30px 30px 1fr}'),
    ('.rail.b-att .tick{background:var(--att)} .rail.b-cos .tick{background:var(--cos)} .rail.b-cmb .tick{background:var(--cmb)}',
     '.rail.b-att .tick{background:var(--att)} .rail.b-cos .tick{background:var(--cos)} .rail.b-cmb .tick{background:var(--cmb)}\n'
     '.rail.b-gld .tick{background:var(--gld)}'),
    # gold 레일에도 다른 방법과 같은 가로선 (::after 기본 규칙에 b-gld 포함)
    ('.rail.b-att::after,.rail.b-cos::after,.rail.b-cmb::after{content:"";position:absolute;left:0;right:0;top:-1px;height:2px}',
     '.rail.b-att::after,.rail.b-cos::after,.rail.b-cmb::after,.rail.b-gld::after{content:"";position:absolute;left:0;right:0;top:-1px;height:2px}'),
    ('.rail.b-att::after{background:var(--att)} .rail.b-cos::after{background:var(--cos)} .rail.b-cmb::after{background:var(--cmb)}',
     '.rail.b-att::after{background:var(--att)} .rail.b-cos::after{background:var(--cos)} .rail.b-cmb::after{background:var(--cmb)}\n'
     '.rail.b-gld::after{background:var(--gld)}\n'
     '/* gold 청크 틱: 클릭하면 토픽 팝오버 */\n'
     '.tick.clk{cursor:pointer}\n'
     '.tick.clk:hover{box-shadow:0 0 0 2px var(--bg),0 0 0 3px var(--gld)}\n'
     '.gpop{position:fixed;z-index:20;display:none;max-width:320px;\n'
     '  background:var(--panel);border:1px solid var(--gld);border-radius:8px;\n'
     '  box-shadow:var(--shadow);padding:8px 11px;font-size:12.5px;line-height:1.4;color:var(--ink)}\n'
     '.gpop.on{display:block}\n'
     '.gpop .sx{display:block;font-family:var(--mono);font-size:10px;color:var(--muted);\n'
     '  letter-spacing:.04em;margin-bottom:3px}\n'
     '.gpop .sx.new{color:var(--gld);font-weight:650}\n'
     '.gpop::after{content:"";position:absolute;left:var(--ax,50%);bottom:-6px;width:10px;height:10px;\n'
     '  background:var(--panel);border-right:1px solid var(--gld);border-bottom:1px solid var(--gld);\n'
     '  transform:translateX(-50%) rotate(45deg)}'),

    # 5) 사이드바 마크업: gold 타일 + gold 대비 필터 + blkseg gold
    ('        <div class="tile cmb"><div class="k">cmb</div><div class="v" id="s-cmb">–</div></div>\n'
     '      </div>',
     '        <div class="tile cmb"><div class="k">cmb</div><div class="v" id="s-cmb">–</div></div>\n'
     '        <div class="tile gld"><div class="k">gold</div><div class="v" id="s-gold">–</div></div>\n'
     '      </div>'),
    ('        <button class="filt" data-f="all3"><span class="dot"></span>세 방법 모두 일치<span class="ct" id="ct-all3"></span></button>\n'
     '      </div>\n'
     '    </div>',
     '        <button class="filt" data-f="all3"><span class="dot"></span>세 방법 모두 일치<span class="ct" id="ct-all3"></span></button>\n'
     '      </div>\n'
     '    </div>\n'
     '\n'
     '    <div>\n'
     '      <p class="eyebrow">gold 대비 (선택 방법 기준)</p>\n'
     '      <div class="filters" id="gfilters">\n'
     '        <button class="filt" data-f="hit"><span class="dot"></span>적중<span class="ct" id="ct-hit"></span></button>\n'
     '        <button class="filt" data-f="miss"><span class="dot"></span>놓침 (gold만)<span class="ct" id="ct-miss"></span></button>\n'
     '        <button class="filt" data-f="fp"><span class="dot"></span>오탐 (gold 아님)<span class="ct" id="ct-fp"></span></button>\n'
     '      </div>\n'
     '    </div>'),
    ('        <button class="on" data-m="cmb">cmb</button>\n'
     '      </div>',
     '        <button class="on" data-m="cmb">cmb</button>\n'
     '        <button data-m="gold">gold</button>\n'
     '      </div>'),

    # 6) 상단 범례 + P/R/F1
    ('        <span><i class="lm"></i>cmb</span>\n'
     '        <span>attn 히트바 = 직전 턴 attention (짧을수록↓ 경계 신호) · 행 클릭 → 전체 보기</span>\n'
     '      </div>\n'
     '    </div>',
     '        <span><i class="lm"></i>cmb</span>\n'
     '        <span><i class="lg"></i>gold</span>\n'
     '        <span>G 틱 클릭 → 청크 토픽 · 행 클릭 → 발화 전체</span>\n'
     '      </div>\n'
     '      <div class="prf" id="prf"></div>\n'
     '    </div>'),
    ('      <div class="h-rail cM">M</div>\n'
     '      <div class="h-utt">발화 (클릭 시 전체 보기)</div>',
     '      <div class="h-rail cM">M</div>\n'
     '      <div class="h-rail cG">G</div>\n'
     '      <div class="h-utt">발화 (클릭 시 전체 보기)</div>'),

    # gold 토픽 팝오버 컨테이너
    ('    <div class="stream" id="stream"></div>\n'
     '  </main>\n'
     '</div>',
     '    <div class="stream" id="stream"></div>\n'
     '  </main>\n'
     '</div>\n'
     '<div class="gpop" id="gpop"></div>'),
]

# 7) JS 패치 — metrics/render/필터에 gold 반영
PATCHES += [
    ('function metrics(d){\n'
     '  const A=new Set(d.att),C=new Set(d.cos),M=new Set(d.cmb);',
     'function metrics(d){\n'
     '  const A=new Set(d.att),C=new Set(d.cos),M=new Set(d.cmb),G=new Set(d.gold);'),
    ('  return {A,C,M,\n'
     '    cosonly:new Set(cosonly),attonly:new Set(attonly),\n'
     '    attcos:new Set(attcos),all3:new Set(all3)};\n'
     '}',
     '  // gold 대비: 선택 방법(blk)의 적중/놓침/오탐. blk="gold"면 자기 자신이므로 전부 적중.\n'
     '  const P=new Set(d[blk]||[]);\n'
     '  const hit=[...G].filter(i=>P.has(i));\n'
     '  const miss=[...G].filter(i=>!P.has(i));\n'
     '  const fp=[...P].filter(i=>!G.has(i));\n'
     '  return {A,C,M,G,\n'
     '    cosonly:new Set(cosonly),attonly:new Set(attonly),\n'
     '    attcos:new Set(attcos),all3:new Set(all3),\n'
     '    hit:new Set(hit),miss:new Set(miss),fp:new Set(fp)};\n'
     '}'),
    ('  document.getElementById("s-cmb").textContent=d.cmb.length;\n'
     '  document.getElementById("ct-cosonly").textContent=mm.cosonly.size;\n'
     '  document.getElementById("ct-attonly").textContent=mm.attonly.size;\n'
     '  document.getElementById("ct-attcos").textContent=mm.attcos.size;\n'
     '  document.getElementById("ct-all3").textContent=mm.all3.size;',
     '  document.getElementById("s-cmb").textContent=d.cmb.length;\n'
     '  document.getElementById("s-gold").textContent=d.gold.length;\n'
     '  document.getElementById("ct-cosonly").textContent=mm.cosonly.size;\n'
     '  document.getElementById("ct-attonly").textContent=mm.attonly.size;\n'
     '  document.getElementById("ct-attcos").textContent=mm.attcos.size;\n'
     '  document.getElementById("ct-all3").textContent=mm.all3.size;\n'
     '  document.getElementById("ct-hit").textContent=blk?mm.hit.size:"–";\n'
     '  document.getElementById("ct-miss").textContent=blk?mm.miss.size:"–";\n'
     '  document.getElementById("ct-fp").textContent=blk?mm.fp.size:"–";\n'
     '  // gold 대비 P/R/F1 (정확 일치 기준). 방법 미선택이면 비교 대상 없음.\n'
     '  const tp=mm.hit.size, pr=tp/Math.max(1,tp+mm.fp.size), rc=tp/Math.max(1,tp+mm.miss.size);\n'
     '  const f1=(pr+rc)?2*pr*rc/(pr+rc):0;\n'
     '  document.getElementById("prf").innerHTML= !blk ? \'<span>비교할 방법을 선택하세요</span>\' :\n'
     '    `<span>${blk} vs gold</span><span>P <b>${pr.toFixed(2)}</b></span>`+\n'
     '    `<span>R <b>${rc.toFixed(2)}</b></span><span>F1 <b>${f1.toFixed(2)}</b></span>`;'),
    ('  const blkset=new Set(d[blk]);\n'
     '  const blkcolMap={att:"var(--att)",cos:"var(--cos)",cmb:"var(--cmb)"};',
     '  const blkset=new Set(blk?d[blk]:[]);\n'
     '  const blkcolMap={att:"var(--att)",cos:"var(--cos)",cmb:"var(--cmb)",gold:"var(--gld)"};\n'
     '  const sess=new Set(d.sess);'),

    # 블록 구분 기준: 켜진 버튼을 다시 누르면 해제(블록 보더 없음)
    ('document.getElementById("blkseg").addEventListener("click",e=>{\n'
     '  const b=e.target.closest("button"); if(!b)return;\n'
     '  blk=b.dataset.m;\n'
     '  document.querySelectorAll("#blkseg button").forEach(x=>x.classList.toggle("on",x===b));\n'
     '  render();\n'
     '});',
     'document.getElementById("blkseg").addEventListener("click",e=>{\n'
     '  const b=e.target.closest("button"); if(!b)return;\n'
     '  const off = blk===b.dataset.m;   // 같은 버튼 재클릭 → 해제\n'
     '  blk = off ? null : b.dataset.m;\n'
     '  document.querySelectorAll("#blkseg button").forEach(x=>x.classList.toggle("on",!off&&x===b));\n'
     '  render();\n'
     '});'),
    ('    const hitbg={cosonly:"var(--cos-soft)",attonly:"var(--att-soft)",\n'
     '      attcos:"color-mix(in srgb,var(--att-soft) 55%,var(--cos-soft))",all3:"var(--cmb-soft)"};',
     '    const hitbg={cosonly:"var(--cos-soft)",attonly:"var(--att-soft)",\n'
     '      attcos:"color-mix(in srgb,var(--att-soft) 55%,var(--cos-soft))",all3:"var(--cmb-soft)",\n'
     '      hit:"var(--gld-soft)",miss:"var(--gld-soft)",fp:"var(--panel2)"};'),
    # gold 레일: 청크 시작 턴(경계 + 대화 첫 턴)에 틱, 클릭하면 토픽 팝오버
    ('    row.appendChild(mkrail(isM,"b-cmb"));',
     '    row.appendChild(mkrail(isM,"b-cmb"));\n'
     '    const grail=el("div","cell rail"+(d.gtopic[i]!=null?" b-gld":""));\n'
     '    if(d.gtopic[i]!=null){\n'
     '      const tk=el("div","tick clk");\n'
     '      tk.title="클릭 → 이 gold 청크의 토픽";\n'
     '      tk.onclick=ev=>{ev.stopPropagation();showPop(tk,d.gtopic[i],sess.has(i));};\n'
     '      grail.appendChild(tk);\n'
     '    } else grail.appendChild(el("div","off"));\n'
     '    row.appendChild(grail);'),
    # gold 토픽 팝오버: 틱 위쪽에 띄우고, 바깥 클릭/스크롤 시 닫는다
    ('function escapeHtml(s){return s.replace(/[&<>]/g,c=>({\'&\':\'&amp;\',\'<\':\'&lt;\',\'>\':\'&gt;\'}[c]));}',
     'function escapeHtml(s){return s.replace(/[&<>]/g,c=>({\'&\':\'&amp;\',\'<\':\'&lt;\',\'>\':\'&gt;\'}[c]));}\n'
     '\n'
     'const gpop=document.getElementById("gpop");\n'
     'function showPop(tick,topic,isSess){\n'
     '  gpop.innerHTML=`<span class="sx${isSess?" new":""}">'
     '${isSess?"◆ 세션 시작 · GOLD CHUNK":"GOLD CHUNK"}</span>`+escapeHtml(topic||"(토픽 없음)");\n'
     '  gpop.classList.add("on");\n'
     '  const r=tick.getBoundingClientRect(), p=gpop.getBoundingClientRect();\n'
     '  const left=Math.max(8,Math.min(r.left+r.width/2-p.width/2,innerWidth-p.width-8));\n'
     '  gpop.style.left=left+"px";\n'
     '  gpop.style.top=Math.max(8,r.top-p.height-10)+"px";\n'
     '  gpop.style.setProperty("--ax",(r.left+r.width/2-left)+"px");\n'
     '}\n'
     'document.addEventListener("click",e=>{if(!e.target.closest(".tick.clk"))gpop.classList.remove("on");});\n'
     'addEventListener("scroll",()=>gpop.classList.remove("on"),true);'),

    # 필터 클릭 핸들러: 두 그룹이 상호 배타적으로 동작하도록
    ('document.getElementById("filters").addEventListener("click",e=>{\n'
     '  const b=e.target.closest(".filt"); if(!b)return;\n'
     '  filter=b.dataset.f;\n'
     '  document.querySelectorAll(".filt").forEach(x=>x.classList.toggle("on",x===b));\n'
     '  render();\n'
     '});',
     'function onFilter(e){\n'
     '  const b=e.target.closest(".filt"); if(!b)return;\n'
     '  filter=b.dataset.f;\n'
     '  document.querySelectorAll(".filt").forEach(x=>x.classList.toggle("on",x===b));\n'
     '  render();\n'
     '}\n'
     'document.getElementById("filters").addEventListener("click",onFilter);\n'
     'document.getElementById("gfilters").addEventListener("click",onFilter);'),
]


if __name__ == "__main__":
    main()
