// SFT Forge front end. Plain ES module, no build step.
// Everything drawn comes from a pipeline run (recorded JSON or a live server run): no synthetic animation data.

const $ = (s, r = document) => r.querySelector(s);
const GATES = ['dedupe', 'quality', 'safety'];
const GATE_NAMES = [...GATES, 'balance'];
const GP = [0.28, 0.52, 0.76];          // gate positions along the flow
const TRAVEL = 2.1;                     // seconds for a sample to cross the whole flow
const RUNS = [
  { id: 'dolly', label: 'Real Dolly slice', dir: 'data/dolly' },
  { id: 'bench', label: 'Synthetic contamination', dir: 'data/contaminated' },
  { id: 'live', label: 'Your data' },
];
const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (n) => Number(n).toLocaleString('en-US');
const pct = (a, b) => (b ? ((100 * a) / b).toFixed(1) + '%' : '0%');
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const frac = (x) => x - Math.floor(x);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage can be blocked */ } },
};

const app = { run: null, recs: [], byId: new Map(), meta: {}, audit: null, filtered: [], visible: 0, sel: -1,
  filters: { view: 'all', reason: '', cat: '', truth: '' }, live: { available: false, job: null, cfg: null }, speed: 1 };

/* ---------------------------------------------------------------- stats */
function newStats(recs) {
  const cats = {};
  for (const r of recs) cats[r.c] ||= { in: 0, after: 0, fin: 0 };
  return { g: GATES.map(() => ({ in: 0, rej: 0, reasons: {} })), kept: 0, bal: 0, after: 0, spawned: 0, cats, balReasons: 0 };
}
function firstReason(rec, gate) { return (rec.rs || []).find((x) => x.g === gate)?.c || 'rejected'; }
function statsFromAll(recs) {
  const st = newStats(recs);
  for (const r of recs) {
    st.spawned++; st.cats[r.c].in++;
    let alive = true;
    GATES.forEach((g, i) => {
      if (!alive) return;
      st.g[i].in++;
      if (r.g === g) { st.g[i].rej++; const c = firstReason(r, g); st.g[i].reasons[c] = (st.g[i].reasons[c] || 0) + 1; alive = false; }
    });
    if (!alive) continue;
    st.after++; st.cats[r.c].after++;
    if (r.s === 'k') { st.kept++; st.cats[r.c].fin++; } else st.bal++;
  }
  return st;
}

/* ---------------------------------------------------------------- stage simulation */
const cv = $('#cv');
const ctx = cv.getContext('2d');
let W = 0, H = 0, DPR = 1;
const pal = {};
function readPalette() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ['bone', 'lime', 'ember', 'dim', 'line', 'line-2', 'void']) pal[k] = cs.getPropertyValue('--' + k).trim();
  pal.mono = cs.getPropertyValue('--mono').trim();
}
const A = (hex, a) => {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex);
  if (m) return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
  return hex.replace(/rgba?\(([^)]+)\)/, (_, v) => { const p = v.split(',').map((x) => x.trim()); return `rgba(${p[0]},${p[1]},${p[2]},${a})`; });
};

const LAND = 0.5;                       // seconds for a survivor to fly into the SFT set
const ease = (x) => 1 - (1 - clamp(x, 0, 1)) ** 3;
const sim = {
  recs: [], n: 0, cursor: 0, acc: 0, items: [], flash: [0, 0, 0], tick: [0, 0, 0], stats: newStats([]), playing: true, base: 20,
  pulse: {}, geo: null, order: [], maxCat: 1, disp: {},
  reset(recs) {
    this.recs = recs; this.n = recs.length; this.cursor = 0; this.acc = 0; this.items = []; this.flash = [0, 0, 0]; this.tick = [0, 0, 0];
    this.stats = newStats(recs); this.pulse = {}; this.base = clamp(this.n / 28, 6, 30); this.playing = true;
    const tot = {}; for (const r of recs) tot[r.c] = (tot[r.c] || 0) + 1;
    this.order = Object.entries(tot).sort((a, b) => b[1] - a[1]).map(([c]) => c);
    this.maxCat = Math.max(1, ...Object.values(tot));
    this.disp = {}; for (const c of this.order) this.disp[c] = { after: 0, fin: 0 };
    app.visible = 0;
  },
  get rps() { return this.base * app.speed; },
  get done() { return this.cursor >= this.n && this.items.every((i) => i.state !== 0); },
  finish() {
    this.stats = statsFromAll(this.recs); this.cursor = this.n; this.items = []; this.playing = false; app.visible = this.n;
    for (const c of this.order) this.disp[c] = { after: this.stats.cats[c].after, fin: this.stats.cats[c].fin };
  },
  spawn() {
    const i = this.cursor++, rec = this.recs[i];
    this.stats.spawned++; this.stats.cats[rec.c].in++;
    this.items.push({ rec, i, age: 0, g: 0, state: 0, lane: frac(i * 0.61803398875 + 0.13), x: 0, y: 0, fade: 1 });
  },
  step(dt) {
    if (this.playing) {
      this.acc += dt * this.rps;
      let guard = 0;
      while (this.acc >= 1 && this.cursor < this.n && guard++ < 40) { this.acc -= 1; this.spawn(); }
      if (this.cursor >= this.n) this.acc = 0;
    }
    for (let k = 0; k < 3; k++) { this.flash[k] = Math.max(0, this.flash[k] - dt * 2.6); this.tick[k] = Math.max(0, this.tick[k] - dt * 4); }
    for (const c in this.pulse) this.pulse[c] = Math.max(0, this.pulse[c] - dt * 2.2);
    const sm = 1 - Math.exp(-dt * 9);
    for (const c of this.order) { const d = this.disp[c], s = this.stats.cats[c]; d.after += (s.after - d.after) * sm; d.fin += (s.fin - d.fin) * sm; }
    const live = [];
    for (const it of this.items) {
      if (it.state === 0) {
        if (this.playing) it.age += dt;
        const p = it.age / TRAVEL;
        while (it.state === 0 && it.g < 3 && p >= GP[it.g]) {
          const gi = it.g, s = this.stats.g[gi];
          s.in++;
          if (it.rec.g === GATES[gi]) {
            s.rej++; const c = firstReason(it.rec, GATES[gi]); s.reasons[c] = (s.reasons[c] || 0) + 1;
            it.state = 1; it.kill = gi; it.code = c; it.dir = frac(it.i * 0.7548776662) < 0.5 ? -1 : 1; it.fade = 1;
            this.flash[gi] = 1; this.tick[gi] = 1;
          } else {
            it.g++;
            if (gi === 2) { this.stats.after++; this.stats.cats[it.rec.c].after++; }
          }
        }
        if (it.state === 0 && p >= 1) {
          if (it.rec.s === 'k') { it.state = 2; it.land = 0; it.fade = 1; }
          else { this.stats.bal++; it.state = 3; it.dir = it.i % 2 ? 1 : -1; it.fade = 1; }
        }
      } else if (it.state === 1 || it.state === 3) {
        it.fade -= dt * (reduced ? 4 : 1.5);
      } else if (it.state === 2) {
        it.land += dt / (reduced ? 0.05 : LAND);
        if (it.land >= 1) { this.stats.kept++; this.stats.cats[it.rec.c].fin++; this.pulse[it.rec.c] = 1; it.fade = 0; }
      }
      if (it.fade > 0) live.push(it);
    }
    this.items = live;
  },
};

function layout() {
  const r = cv.getBoundingClientRect();
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
  cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  const vertical = W < 640;
  const ncat = Math.max(1, Object.keys(sim.stats.cats).length);
  if (vertical) {
    const boxH = 34 + ncat * 19 + 10, top = 46, bot = H - boxH - 14;
    sim.geo = { vertical, f0: top, f1: bot, c0: 86, c1: W - 70, box: { x: 8, y: H - boxH - 6, w: W - 16, h: boxH }, k: 1.15 };
  } else {
    const f1 = W * 0.76, top = 40, bot = H - 20;
    sim.geo = { vertical, f0: 18, f1, c0: top + 14, c1: bot - 6, box: { x: f1 + 14, y: top - 8, w: W - f1 - 26, h: H - top - 4 }, k: clamp(W / 1000, 0.85, 1.25) };
  }
}
function at(p, lane) {
  const g = sim.geo; const f = g.f0 + p * (g.f1 - g.f0), c = g.c0 + lane * (g.c1 - g.c0);
  return g.vertical ? { x: c, y: f } : { x: f, y: c };
}
function rrect(x, y, w, h, r) {
  ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}
const mono = (px) => `500 ${px}px ${pal.mono}`;
const SHORT = { exact_duplicate: 'exact dup', near_duplicate: 'near dup', semantic_duplicate: 'semantic dup', refusal_boilerplate: 'refusal', wrong_language: 'language', off_topic: 'off topic', low_alnum_ratio: 'symbols', response_too_short: 'too short', instruction_too_short: 'too short', response_too_long: 'too long' };

function boxRows() {
  const g = sim.geo, b = g.box, n = Math.max(sim.order.length, 1);
  const rowH = Math.min(21, (b.h - 36) / n), labW = g.vertical ? 118 : Math.min(112, b.w * 0.5);
  return { b, rowH, labW, bw: b.w - 24 - labW - 34, x0: b.x + 12 + labW };
}
function draw() {
  const g = sim.geo; if (!g) return;
  ctx.clearRect(0, 0, W, H);
  const k = g.k, lab = g.vertical ? 11 : 10.5;
  // gates
  GATES.forEach((name, i) => {
    const p = GP[i], a = at(p, 0), b = at(p, 1), fl = sim.flash[i], s = sim.stats.g[i], tk = sim.tick[i];
    ctx.setLineDash([4, 5]); ctx.lineWidth = 1 + fl;
    ctx.strokeStyle = fl > 0.02 ? A(pal.ember, 0.35 + 0.65 * fl) : A(pal.bone, 0.2);
    ctx.beginPath();
    if (g.vertical) { ctx.moveTo(10, a.y); ctx.lineTo(W - 10, a.y); } else { ctx.moveTo(a.x, a.y - 14); ctx.lineTo(b.x, b.y + 6); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.font = mono(lab); ctx.fillStyle = A(pal.bone, 0.62);
    const txt = name.toUpperCase(), cnt = `${fmt(s.rej)} out`;
    const cfont = mono(lab * (1 + 0.28 * tk)), ccol = A(pal.ember, 0.9);
    if (g.vertical) { ctx.textAlign = 'left'; ctx.fillText(txt, 14, a.y - 8); ctx.textAlign = 'right'; ctx.font = cfont; ctx.fillStyle = ccol; ctx.fillText(cnt, W - 14, a.y - 8); }
    else { ctx.textAlign = 'center'; ctx.fillText(txt, a.x, a.y - 22); ctx.font = cfont; ctx.fillStyle = ccol; ctx.fillText(cnt, a.x, a.y - 8); }
    ctx.textAlign = 'left';
  });
  // source label
  ctx.font = mono(lab); ctx.fillStyle = A(pal.bone, 0.5);
  ctx.fillText(`SOURCE  ${fmt(sim.stats.spawned)} / ${fmt(sim.n)}`, g.vertical ? 14 : 18, 24);
  // SFT set box: bars grow smoothly toward the real counts
  const br = boxRows(), b = br.b;
  ctx.strokeStyle = A(pal.lime, 0.55); ctx.lineWidth = 1; rrect(b.x, b.y, b.w, b.h, 12); ctx.stroke();
  ctx.fillStyle = pal.lime; ctx.font = mono(lab); ctx.fillText('SFT SET', b.x + 12, b.y + 20);
  ctx.textAlign = 'right'; ctx.fillStyle = A(pal.bone, 0.85); ctx.fillText(`${fmt(sim.stats.kept)} kept`, b.x + b.w - 12, b.y + 20); ctx.textAlign = 'left';
  sim.order.forEach((name, i) => {
    const v = sim.disp[name], y = b.y + 32 + i * br.rowH, pulse = sim.pulse[name] || 0, rh = br.rowH * 0.42, ry = y + br.rowH * 0.2;
    const full = br.bw * (sim.stats.cats[name].in / sim.maxCat);
    ctx.font = mono(g.vertical ? 10.5 : 9.5); ctx.fillStyle = A(pal.bone, 0.58); ctx.fillText(name.replace(/_/g, ' ').slice(0, 17), b.x + 12, y + br.rowH * 0.62);
    ctx.fillStyle = A(pal.bone, 0.09); rrect(br.x0, ry, br.bw, rh, 2); ctx.fill();
    ctx.fillStyle = A(pal.bone, 0.1); rrect(br.x0, ry, Math.max(2, full), rh, 2); ctx.fill();
    ctx.fillStyle = A(pal.bone, 0.18); rrect(br.x0, ry, Math.max(2, br.bw * (v.after / sim.maxCat)), rh, 2); ctx.fill();
    ctx.fillStyle = A(pal.lime, 0.75 + 0.25 * pulse); rrect(br.x0, ry, Math.max(v.fin > 0.05 ? 2 : 0, br.bw * (v.fin / sim.maxCat)), rh + pulse * 2, 2); ctx.fill();
    ctx.textAlign = 'right'; ctx.fillStyle = A(pal.bone, 0.62); ctx.fillText(fmt(sim.stats.cats[name].fin), b.x + b.w - 10, y + br.rowH * 0.62); ctx.textAlign = 'left';
  });
  // samples
  const iw = 15 * k, ih = 5.2 * k;
  let labelled = 0;
  const crowded = sim.items.length > 120;
  for (let n = sim.items.length - 1; n >= 0; n--) {
    const it = sim.items[n];
    let x, y, sc = 1;
    if (it.state === 0) {
      const pt = at(Math.min(it.age / TRAVEL, 1), it.lane); x = pt.x; y = pt.y;
      ctx.fillStyle = A(pal.bone, 0.82);
    } else if (it.state === 1) {
      const pt = at(GP[it.kill], it.lane), t = 1 - it.fade, e = 1 - (1 - Math.min(1, t / 0.9)) ** 2;
      const side = it.dir * 84 * e, kick = -16 * Math.sin(Math.min(1, t * 1.4) * Math.PI);
      x = pt.x + (g.vertical ? side : kick); y = pt.y + (g.vertical ? kick : side);
      if (g.vertical) x = clamp(x, 6, W - 6);
      sc = 1 + 0.7 * Math.max(0, 1 - t * 7);
      const rr = Math.min(1, t * 2.2);                       // a ring flashes at the gate that caught it
      if (rr < 1) { ctx.strokeStyle = A(pal.ember, 0.85 * (1 - rr)); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(pt.x, pt.y, 6 + 26 * rr, 0, 6.2832); ctx.stroke(); }
      ctx.fillStyle = A(pal.ember, 0.95 * Math.max(0, it.fade));
    } else if (it.state === 3) {
      const pt = at(1, it.lane), t = 1 - it.fade;
      x = pt.x + (g.vertical ? it.dir * 50 * t : 0); y = pt.y + (g.vertical ? 0 : it.dir * 50 * t);
      if (g.vertical) x = clamp(x, 6, W - 6);
      ctx.fillStyle = A(pal.bone, 0.45 * Math.max(0, it.fade));
    } else {
      const pt = at(1, it.lane), e = ease(it.land), i = Math.max(0, sim.order.indexOf(it.rec.c));
      const tx = br.x0 + br.bw * (sim.disp[it.rec.c].fin / sim.maxCat), ty = b.y + 32 + i * br.rowH + br.rowH * 0.41;
      x = pt.x + (tx - pt.x) * e; y = pt.y + (ty - pt.y) * e; sc = 1 - 0.4 * e;
      ctx.fillStyle = A(pal.bone, 0.82); rrect(x - iw * sc / 2, y - ih * sc / 2, iw * sc, ih * sc, 2.6); ctx.fill();   // turns lime as it lands
      ctx.fillStyle = A(pal.lime, Math.min(1, it.land * 3));
    }
    it.x = x; it.y = y;
    if (crowded && it.state === 0) ctx.fillRect(x - iw / 2, y - ih / 2, iw, ih);   // plain rectangles when many are on screen
    else { rrect(x - iw * sc / 2, y - ih * sc / 2, iw * sc, ih * sc, 2.6); ctx.fill(); }
    if (app.sel === it.i) { ctx.strokeStyle = pal.lime; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 13, 0, 6.2832); ctx.stroke(); }
    if ((it.state === 1 || it.state === 3) && labelled < 5 && it.fade > 0.3 && sim.rps < 90) {
      labelled++; ctx.font = mono(g.vertical ? 10 : 9.5); ctx.fillStyle = A(it.state === 1 ? pal.ember : pal.bone, Math.min(1, it.fade * 1.3));
      const txt = it.state === 3 ? 'cap' : (SHORT[it.code] || it.code.replace(/_/g, ' '));
      ctx.textAlign = g.vertical ? (x < W / 2 ? 'left' : 'right') : 'center';
      ctx.fillText(txt, g.vertical ? x + (x < W / 2 ? 16 : -16) : x, g.vertical ? y + 3 : y + (it.dir < 0 ? -10 : 16)); ctx.textAlign = 'left';
    }
  }
  if (!sim.items.length && sim.cursor === 0) {
    ctx.font = mono(11); ctx.fillStyle = A(pal.bone, 0.5); ctx.textAlign = 'center'; ctx.fillText('loading run', W / 2, H / 2); ctx.textAlign = 'left';
  }
}

/* ---------------------------------------------------------------- DOM readouts */
const el = {
  stats: $('#stats'), reasons: $('#reasons'), cats: $('#cats'), prog: $('#prog'), clock: $('#clock'), play: $('#play'),
};
function topReasons(obj, n = 3) {
  return Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, n).map(([c, v]) => `${c.replace(/_/g, ' ')} ${fmt(v)}`).join(' · ') || 'nothing rejected yet';
}
function syncBars(box, rows) {
  const key = rows.map((r) => r.key).join('|');
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.innerHTML = rows.map((r) => `<div class="bar-row"><span class="nm" title="${esc(r.key)}">${esc(r.label)}</span><span class="tr">${r.ghost !== undefined ? '<i class="ghost"></i><i class="lime"></i>' : '<i></i>'}</span><span class="v"></span></div>`).join('');
  }
  rows.forEach((r, k) => {
    const row = box.children[k], bars = row.querySelectorAll('.tr i');
    if (r.ghost !== undefined) { bars[0].style.width = r.ghost.toFixed(1) + '%'; bars[1].style.width = r.w.toFixed(1) + '%'; } else bars[0].style.width = r.w.toFixed(1) + '%';
    const v = row.querySelector('.v'); if (v.textContent !== r.v) v.textContent = r.v;
  });
}
function renderStats() {
  const s = sim.stats;
  const cell = (label, i) => `<div><div class="label">${label}</div><div class="n">${fmt(s.g[i].rej)} <small>out of ${fmt(s.g[i].in)}</small></div><div class="rej">${pct(s.g[i].rej, s.g[i].in)} rejected</div><div class="why">${esc(topReasons(s.g[i].reasons))}</div></div>`;
  el.stats.innerHTML = cell('Gate 1 · Dedupe', 0) + cell('Gate 2 · Quality', 1) + cell('Gate 3 · Safety', 2) +
    `<div><div class="label">SFT set</div><div class="n ok" style="color:var(--accent-text)">${fmt(s.kept)} <small>of ${fmt(sim.n)}</small></div><div class="rej ok" style="color:var(--accent-text)">${pct(s.kept, sim.n)} kept</div><div class="why">${fmt(s.bal)} trimmed by the category cap</div></div>`;
  const reasons = {};
  s.g.forEach((x) => { for (const [c, v] of Object.entries(x.reasons)) reasons[c] = (reasons[c] || 0) + v; });
  if (s.bal) reasons['category_cap (balance)'] = s.bal;
  const rows = Object.entries(reasons).sort((a, b) => b[1] - a[1]), mx = Math.max(1, ...rows.map((r) => r[1]));
  if (rows.length) syncBars(el.reasons, rows.map(([c, v]) => ({ key: c, label: c.replace(/_/g, ' '), w: 100 * v / mx, v: fmt(v) })));
  else { el.reasons.dataset.key = ''; el.reasons.innerHTML = '<div class="note">No rejections yet.</div>'; }
  const cats = sim.order.map((c) => [c, s.cats[c]]), mi = Math.max(1, sim.maxCat);
  syncBars(el.cats, cats.map(([c, v]) => ({ key: c, label: c.replace(/_/g, ' '), ghost: 100 * v.in / mi, w: 100 * v.fin / mi, v: `${fmt(v.in)} → ${fmt(v.fin)}` })));
  el.prog.style.transform = `scaleX(${sim.n ? sim.cursor / sim.n : 0})`;
  el.clock.textContent = `${fmt(sim.cursor)} / ${fmt(sim.n)} rows` + (sim.playing && sim.cursor < sim.n ? ` · ${Math.round(sim.rps)} per second` : sim.cursor >= sim.n ? ' · complete' : ' · paused');
  el.play.textContent = sim.playing && sim.cursor < sim.n ? 'Pause' : sim.cursor >= sim.n ? 'Done' : 'Play';
  el.play.disabled = sim.cursor >= sim.n;
}

/* ---------------------------------------------------------------- frame loop */
let last = 0, lastUI = 0, visibleOnScreen = true;
function frame(ts) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.05, (ts - last) / 1000 || 0.016); last = ts;
  if (!visibleOnScreen || document.hidden) return;
  sim.step(dt);
  draw();
  if (ts - lastUI > 100) {
    lastUI = ts; renderStats();
    if (sim.playing || app.visible !== sim.cursor) { app.visible = sim.cursor; if (logFollowing()) renderLog(true); }
  }
}
new IntersectionObserver((e) => { visibleOnScreen = e[0].isIntersecting; }, { threshold: 0 }).observe($('#stage'));
new ResizeObserver(() => { layout(); }).observe(cv);

/* ---------------------------------------------------------------- run loading */
function setRun(data, id) {
  app.run = id; app.meta = data.meta; app.audit = data.audit; app.recs = data.records; app.remote = data.remote || null;
  app.byId = new Map(app.recs.map((r, i) => [r.id, i]));
  app.filters = { view: 'all', reason: '', cat: '', truth: '' };
  closeInspector();
  sim.reset(app.recs); layout();
  $('#stage-tag').textContent = `${data.meta.name} · ${fmt(app.recs.length)} rows`;
  const badge = $('#badge');
  badge.textContent = data.meta.kind === 'live' ? 'Live run' : data.meta.kind === 'synthetic_contamination' ? 'Recorded run · synthetic contamination' : 'Recorded run';
  badge.className = 'pill ' + (data.meta.kind === 'live' ? 'live' : 'rec');
  buildFilters(); buildAudit(); buildBench(); applyFilters();
  document.querySelectorAll('#runs button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === id)));
  if (reduced) sim.finish();
  renderStats();
}
// A recorded run is an index (one small array per row: enough for the stream and the statistics) plus detail
// chunks of 100 rows that are fetched when a sample is opened and kept in memory.
function expandIndex(idx) {
  const rules = idx.rules;
  const records = idx.rows.map((row, i) => {
    const rec = { i, id: row[0], c: idx.cats[row[1]], s: row[2], rs: row[4].map(([g, c]) => ({ g: GATE_NAMES[g], c: idx.codes[c] })), a: row[8] };
    if (row[3] >= 0) rec.g = GATE_NAMES[row[3]];
    if (row[5] >= 0) rec.t = idx.truths[row[5]];
    if (row[6]) rec.rdf = 1;
    if (row[7] === 1) rec.w = 1;
    if (row[7] === 2) rec.om = 1;
    return rec;
  });
  return { meta: idx.meta, audit: idx.audit, records, remote: { dir: null, chunk: idx.chunk, rules, cache: new Map() } };
}
function decodeChecks(ck, rules) {
  const out = {};
  for (const e of ck) {
    const [g, r, th] = rules[e[0]];
    const c = { r, v: e[1], t: /^-?\d+(\.\d+)?$/.test(th) ? +th : th, p: e.length < 3 || e[2] === 1 };
    if (e.length > 3) c.n = e[3];
    (out[g] ||= []).push(c);
  }
  return out;
}
function getDetail(i) {
  const rec = app.recs[i];
  if (!rec || rec.full || !app.remote) return Promise.resolve(rec);
  const rm = app.remote, k = Math.floor(i / rm.chunk);
  if (!rm.cache.has(k)) {
    rm.cache.set(k, fetch(`${rm.dir}/c${k}.json`).then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }).catch((e) => { rm.cache.delete(k); throw e; }));
  }
  return rm.cache.get(k).then((chunk) => {
    chunk.forEach((d, j) => {
      const r = app.recs[k * rm.chunk + j]; if (!r || r.full) return;
      if (d.o) { r.om = 1; r.full = true; return; }
      const { ck, ...rest } = d;
      Object.assign(r, rest);
      if (ck) r.ck = decodeChecks(ck, rm.rules);
      r.full = true;
    });
    return rec;
  });
}
async function loadRecorded(id) {
  const spec = RUNS.find((r) => r.id === id);
  try {
    const res = await fetch(spec.dir + '/index.json'); if (!res.ok) throw new Error(res.status);
    const data = expandIndex(await res.json());
    data.remote.dir = spec.dir;
    setRun(data, id);
  } catch (e) {
    $('#stage-tag').textContent = 'Could not load ' + spec.dir + '. Serve this folder over http.';
  }
}

/* ---------------------------------------------------------------- stream log (virtualised) */
const logbox = $('#logbox'), logsp = $('#logsp'), ROW = 56;
function logFollowing() { return logbox.scrollTop < 60; }
function passes(r) {
  const f = app.filters;
  if (f.view === 'kept' && r.s !== 'k') return false;
  if (f.view === 'rejected' && r.s !== 'x') return false;
  if (f.view === 'balanced' && r.s !== 'b') return false;
  if (GATES.includes(f.view) && r.g !== f.view) return false;
  if (f.view === 'redacted' && !((r.rd || r.rdf) && r.s !== 'x')) return false;
  if (f.cat && r.c !== f.cat) return false;
  if (f.reason && !(r.rs || []).some((x) => x.c === f.reason)) return false;
  if (f.truth && (r.t || 'clean') !== f.truth) return false;
  return true;
}
function applyFilters() {
  const n = Math.min(app.visible || 0, app.recs.length);
  const out = [];
  for (let i = n - 1; i >= 0; i--) if (passes(app.recs[i])) out.push(i);
  app.filtered = out;
  renderLog(true);
}
function renderLog(refilter) {
  if (refilter) {
    const n = Math.min(app.visible, app.recs.length), out = [];
    for (let i = n - 1; i >= 0; i--) if (passes(app.recs[i])) out.push(i);
    app.filtered = out;
  }
  const total = app.filtered.length;
  logsp.style.height = total * ROW + 'px';
  const top = Math.floor(logbox.scrollTop / ROW), cnt = Math.ceil(logbox.clientHeight / ROW) + 2;
  let html = '';
  for (let k = top; k < Math.min(total, top + cnt); k++) {
    const r = app.recs[app.filtered[k]], why = r.s === 'x' ? (r.rs?.[0]?.c || '').replace(/_/g, ' ') : r.s === 'b' ? 'cap' : (r.rd || r.rdf) ? 'redacted' : '';
    html += `<button class="row ${r.s}" style="top:${k * ROW}px" data-i="${r.i}"><span class="dot"></span><span class="t">${esc((r.a || '').replace(/\s+/g, ' ').slice(0, 140))}</span><span class="c">${esc(r.c)}</span><span class="r" ${r.s === 'k' ? 'style="color:var(--accent-text)"' : ''}>${esc(why)}</span></button>`;
  }
  logsp.innerHTML = html || '<div class="empty">No samples match yet.</div>';
  $('#lognote').textContent = `${fmt(total)} of ${fmt(app.visible)} streamed samples match. Tap a row to open it.`;
}
logbox.addEventListener('scroll', () => requestAnimationFrame(() => renderLog(false)), { passive: true });
logsp.addEventListener('click', (e) => { const b = e.target.closest('.row'); if (b) openInspector(+b.dataset.i); });

function buildFilters() {
  const reasons = new Set(), cats = new Set(), truths = new Set();
  for (const r of app.recs) { cats.add(r.c); (r.rs || []).forEach((x) => reasons.add(x.c)); if (r.t) truths.add(r.t); }
  const sel = (id, label, opts) => `<label class="sr" for="${id}">${label}</label><select id="${id}" aria-label="${label}">${opts.map(([v, t]) => `<option value="${esc(v)}">${esc(t)}</option>`).join('')}</select>`;
  $('#filters').innerHTML =
    sel('f-view', 'View', [['all', 'All samples'], ['kept', 'In the SFT set'], ['rejected', 'Rejected'], ['dedupe', 'Rejected: dedupe'], ['quality', 'Rejected: quality'], ['safety', 'Rejected: safety'], ['balanced', 'Trimmed by cap'], ['redacted', 'Kept with PII redacted']]) +
    sel('f-reason', 'Reason', [['', 'Any reason'], ...[...reasons].sort().map((c) => [c, c.replace(/_/g, ' ')])]) +
    sel('f-cat', 'Category', [['', 'Any category'], ...[...cats].sort().map((c) => [c, c.replace(/_/g, ' ')])]) +
    (truths.size ? sel('f-truth', 'Injected defect', [['', 'Any injected defect'], ['clean', 'untouched'], ...[...truths].filter((t) => t !== 'clean').sort().map((c) => [c, c.replace(/_/g, ' ')])]) : '');
  for (const [id, key] of [['f-view', 'view'], ['f-reason', 'reason'], ['f-cat', 'cat'], ['f-truth', 'truth']]) {
    const s = $('#' + id); if (!s) continue;
    s.addEventListener('change', () => { app.filters[key] = s.value; logbox.scrollTop = 0; renderLog(true); });
  }
}

/* ---------------------------------------------------------------- inspector */
const insp = $('#insp'), scrim = $('#scrim');
let lastFocus = null;
function highlight(text) {
  return esc(text).replace(/\[(EMAIL|PHONE|CARD|SSN|IP|IBAN)\]/g, '<mark>[$1]</mark>');
}
function checkRow(c, failed) {
  const v = c.v === null || c.v === undefined ? 'none' : typeof c.v === 'boolean' ? (c.v ? 'yes' : 'no') : c.v;
  return `<div class="chk ${failed ? 'f' : ''}"><span>${esc(c.r.replace(/_/g, ' '))}</span><span class="th">${esc(v)} vs ${esc(c.t ?? '')}</span>${c.n ? `<span class="nt">${esc(c.n)}</span>` : ''}</div>`;
}
function whyBlock(r) {
  const fails = (r.rs || []).filter((x) => (r.s === 'x' ? x.g === r.g : x.g === 'balance'));
  if (r.s === 'k') return `<div class="why-blk ok"><b>Passed all three gates</b><span>${r.rd?.length || r.rdf ? 'PII was redacted before export.' : 'Nothing was flagged.'}</span></div>`;
  return fails.map((f) => `<div class="why-blk"><b>${esc((f.c || '').replace(/_/g, ' '))}</b>${f.m ? `<span>${esc(f.m)}</span>` : ''}${f.v !== null && f.v !== undefined && f.t !== null && f.t !== undefined && f.m ? `<span class="th">${esc(f.v)} vs ${esc(f.t ?? '')}</span>` : ''}</div>`).join('');
}
function renderInspector(i, loading) {
  const r = app.recs[i];
  const verdict = { k: ['In the SFT set', 'k'], x: [`Rejected at ${r.g}`, 'x'], b: ['Trimmed by category cap', 'b'] }[r.s];
  $('#i-id').textContent = `${r.id} · ${r.c.replace(/_/g, ' ')}`;
  let h = `<span class="verdict ${verdict[1]}">${verdict[0]}</span>`;
  if (r.sp) h += ` <span class="pill" style="margin-left:6px">${r.sp === 'val' ? 'validation split' : 'train split'}</span>`;
  h += `<div class="why">${whyBlock(r)}</div>`;
  if (r.t) h += `<div class="kv"><span>Ground truth: ${r.t === 'clean' ? 'untouched row' : 'injected ' + esc(r.t.replace(/_/g, ' '))}</span></div>`;
  if (loading) { $('#i-body').innerHTML = h + '<p class="note" role="status">Loading the sample detail…</p>'; return; }
  if (r.om) { $('#i-body').innerHTML = h + '<p class="note">The text of this row is left out of the public recording because it names a specific AI product. Its place in the stream and in the statistics is unchanged.</p>'; return; }
  if (r.w) h += '<p class="note warn">The text of toxic samples is withheld in the public recording. The scores below are real.</p>';
  h += `<div class="field"><div class="label">Instruction</div><pre>${highlight(r.a)}</pre></div>`;
  if (r.x) h += `<div class="field"><div class="label">Context</div><pre>${highlight(r.x)}</pre></div>`;
  h += `<div class="field"><div class="label">Response</div><pre>${highlight(r.r)}</pre></div>`;
  if (r.n && (r.n[0] > (r.a || '').length || r.n[1] > (r.r || '').length)) h += `<p class="note">Shown truncated: ${fmt(r.n[0])} and ${fmt(r.n[1])} characters in full.</p>`;
  h += '<div class="label" style="margin-top:18px">Every check</div>';
  for (const g of GATES) {
    const checks = (r.ck || {})[g] || [];
    const status = r.g === g ? 'fail' : checks.length ? 'pass' : 'skip';
    h += `<div class="gate-blk ${status}"><h4><span>${g}</span><span>${status === 'fail' ? 'rejected' : status === 'pass' ? 'passed' : 'not reached'}</span></h4>`;
    for (const c of checks) h += checkRow(c, c.p === false);
    if (status === 'pass' && g === 'safety' && r.rd) h += '<div class="chk"><span class="nt">PII redacted before export</span></div>';
    h += '</div>';
  }
  const bal = (r.rs || []).find((x) => x.g === 'balance');
  if (r.s === 'b' || r.s === 'k') h += `<div class="gate-blk ${r.s === 'k' ? 'pass' : 'skip'}"><h4><span>balance</span><span>${r.s === 'k' ? 'kept' : 'trimmed'}</span></h4>${bal ? `<div class="chk"><span class="nt">${esc(bal.m)}</span></div>` : '<div class="chk"><span class="nt">Within the category cap.</span></div>'}</div>`;
  if (r.rd?.length) h += `<div class="field"><div class="label">Redactions</div><div class="kv">${r.rd.map((x) => `<span>${esc(x.f)}: ${esc(x.k)} ${esc(x.was)}</span>`).join('')}</div></div>`;
  if (r.ed?.length) h += `<div class="field"><div class="label">Normalisation edits</div><div class="kv">${r.ed.map((x) => `<span>${esc(x.replace(/_/g, ' '))}</span>`).join('')}</div></div>`;
  if (r.sc) h += `<div class="field"><div class="label">Scores</div><div class="kv">${Object.entries(r.sc).map(([k, v]) => `<span>${esc(k)} ${esc(v)}</span>`).join('')}</div></div>`;
  if (r.tw) { const ti = app.byId.get(r.tw); h += `<div class="field"><div class="label">Twin</div>${ti !== undefined ? `<button class="link" data-twin="${ti}">Open ${esc(r.tw)}</button>` : `<span class="note">${esc(r.tw)}</span>`}</div>`; }
  $('#i-body').innerHTML = h;
}
function openInspector(i, push = true) {
  const r = app.recs[i]; if (!r) return;
  app.sel = i;
  lastFocus = lastFocus || document.activeElement;
  const need = !r.full && app.remote;
  renderInspector(i, need);
  $('#i-body').scrollTop = 0;
  insp.classList.add('on'); scrim.classList.add('on'); insp.setAttribute('aria-hidden', 'false');
  if (push) $('#i-close').focus({ preventScroll: true });
  if (need) {
    getDetail(i).then(() => { if (app.sel === i) renderInspector(i, false); })
      .catch(() => { if (app.sel === i) $('#i-body').insertAdjacentHTML('beforeend', '<p class="note warn">Could not load the detail for this sample. Serve the folder over http and try again.</p>'); });
  }
}
function closeInspector() {
  insp.classList.remove('on'); scrim.classList.remove('on'); insp.setAttribute('aria-hidden', 'true'); app.sel = -1;
  if (lastFocus && lastFocus.focus) { try { lastFocus.focus({ preventScroll: true }); } catch { /* element may be gone */ } }
  lastFocus = null;
}
function stepInspector(d) {
  const list = app.filtered; if (!list.length) return;
  const k = list.indexOf(app.sel);
  const nxt = list[clamp((k < 0 ? 0 : k + d), 0, list.length - 1)];
  openInspector(nxt, false);
}
$('#i-close').addEventListener('click', closeInspector);
scrim.addEventListener('click', closeInspector);
$('#i-prev').addEventListener('click', () => stepInspector(-1));
$('#i-next').addEventListener('click', () => stepInspector(1));
$('#i-body').addEventListener('click', (e) => { const b = e.target.closest('[data-twin]'); if (b) openInspector(+b.dataset.twin, false); });
document.addEventListener('keydown', (e) => {
  if (!insp.classList.contains('on')) return;
  if (e.key === 'Escape') closeInspector();
  if (e.key === 'ArrowRight') stepInspector(1);
  if (e.key === 'ArrowLeft') stepInspector(-1);
});
// canvas tap: nearest sample within reach
cv.addEventListener('pointerdown', (e) => {
  const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  let best = null, bd = (e.pointerType === 'touch' ? 34 : 20) ** 2;
  for (const it of sim.items) { const d = (it.x - x) ** 2 + (it.y - y) ** 2; if (d < bd) { bd = d; best = it; } }
  if (best) openInspector(best.i);
});

/* ---------------------------------------------------------------- audit and benchmark sections */
function flat(o, p = '') {
  return Object.entries(o).flatMap(([k, v]) => (v && typeof v === 'object' && !Array.isArray(v) ? flat(v, p + k + '.') : [[p + k, Array.isArray(v) ? v.join(', ') : v]]));
}
function buildAudit() {
  const a = app.audit;
  $('#gate-table').innerHTML = '<tr><th>Gate</th><th>In</th><th>Rejected</th><th>Out</th><th>Reasons</th></tr>' +
    a.gates.map((g) => `<tr><td>${g.gate}</td><td>${fmt(g.in)}</td><td>${fmt(g.rejected)}</td><td>${fmt(g.out)}</td><td style="text-align:left;white-space:normal">${esc(Object.entries(g.reasons).map(([k, v]) => `${k} ${v}`).join(', ') || '-')}</td></tr>`).join('');
  const skip = new Set(['models.embedder', 'models.embedder_revision', 'models.toxicity', 'models.toxicity_revision', 'models.perplexity', 'models.perplexity_revision']);
  $('#cfg').innerHTML = flat(a.config).filter(([k]) => !skip.has(k)).map(([k, v]) => `<div>${esc(k)}: <b style="color:var(--bone)">${esc(v)}</b></div>`).join('');
  const m = app.meta, t = a.timings_s || {};
  $('#runmeta').innerHTML = [`${fmt(a.rows_in)} rows in, ${fmt(a.rows_kept)} kept (${(a.keep_rate * 100).toFixed(1)}%)`, `train ${fmt(a.rows_train)}, validation ${fmt(a.rows_val)}`,
    `seed ${a.seed}, sftforge ${esc(a.version)}`, `models on ${esc(a.device || 'unknown')}`, `input sha256 ${esc(a.input_sha256.slice(0, 16))}`, `category cap: ${a.balance_cap ?? 'not needed'}`,
    `time: ${t.total ?? '?'} s on the ${String(a.device || '').startsWith('cuda') ? 'GPU' : 'CPU'} (dedupe ${t.dedupe}, quality ${t.quality}, safety ${t.safety})`, `${esc(m.label || '')} · created ${esc(m.created || '')}`].map((x) => `<div>${x}</div>`).join('');
  const base = 'Code: MIT. ';
  $('#credits').innerHTML = base + (m.kind === 'live' ? 'Your pasted data stays in memory and is not stored.'
    : 'Sample text comes from databricks-dolly-15k by Databricks (CC BY-SA 3.0): records shown here are derived from it and stay under CC BY-SA 3.0. ' +
      (m.kind === 'synthetic_contamination' ? 'Foreign-language rows come from OpenAssistant oasst1 (Apache-2.0); toxic snippets from AllenAI real-toxicity-prompts (Apache-2.0) and are withheld from display. All PII and secrets here are fake documentation values. ' : '') +
      'Models: all-MiniLM-L6-v2 and toxic-bert (Apache-2.0).');
}
function buildBench() {
  const b = app.meta.benchmark, box = $('#bench');
  if (!b) { box.style.display = 'none'; return; }
  box.style.display = '';
  const u = b.untouched;
  $('#bench-note').textContent = `Defects were injected at known rates into a clean slice of Dolly rows with a seeded script, so recall is measured against ground truth. ${fmt(u.n)} of the ${fmt(app.audit.rows_in)} rows were left untouched. False rejects on the untouched rows: ${u.rejected} of ${fmt(u.n)} (${(u.false_reject_rate * 100).toFixed(2)}%), which includes real defects already present in Dolly. Synthetic contamination, not real-world data.`;
  $('#bench-table').innerHTML = '<tr><th>Injected defect</th><th>Rows</th><th>Caught</th><th>Recall</th><th>Right reason</th></tr>' +
    Object.entries(b.per_defect).map(([d, v]) => `<tr><td>${esc(d.replace(/_/g, ' '))}</td><td>${v.n}</td><td>${v.caught}</td><td>${(v.recall * 100).toFixed(0)}%</td><td>${v.with_expected_reason ?? (v.expected === 'redacted' ? 'redacted' : '-')}</td></tr>`).join('') +
    '<tr><th colspan="5" style="padding-top:16px">Gate precision on this run</th></tr>' +
    Object.entries(b.per_gate).map(([g, v]) => `<tr><td>${g}</td><td>${v.rejected} rejected</td><td>${v.injected_defects} injected</td><td>${v.precision === null ? '-' : (v.precision * 100).toFixed(1) + '%'}</td><td></td></tr>`).join('');
}
$('#dl-audit').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(app.audit, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'audit.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
});

/* ---------------------------------------------------------------- transport */
$('#play').addEventListener('click', () => { sim.playing = !sim.playing; renderStats(); });
$('#replay').addEventListener('click', () => { sim.reset(app.recs); layout(); applyFilters(); renderStats(); });
$('#skip').addEventListener('click', () => { sim.finish(); applyFilters(); renderStats(); });
$('#speeds').innerHTML = [1, 2, 4, 8].map((s) => `<button data-s="${s}" aria-pressed="${s === 1}" aria-label="${s} times speed">${s}x</button>`).join('');
$('#speeds').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return; app.speed = +b.dataset.s;
  document.querySelectorAll('#speeds button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
});
$('#runs').innerHTML = RUNS.map((r) => `<button data-id="${r.id}" aria-pressed="false">${r.label}</button>`).join('');
$('#runs').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.id === 'live') { if (app.run !== 'live' || !app.recs.length) $('#live').scrollIntoView({ behavior: reduced ? 'auto' : 'smooth' }); if (app.liveData) setRun(app.liveData, 'live'); return; }
  loadRecorded(b.dataset.id);
});

/* ---------------------------------------------------------------- live mode */
const SLIDERS = [
  ['dedupe.near.threshold', 'Near-duplicate Jaccard', 0.5, 1, 0.01],
  ['dedupe.semantic.threshold', 'Semantic duplicate cosine', 0.8, 1, 0.01],
  ['quality.relevance.min_cosine', 'Minimum relevance cosine', 0, 0.4, 0.01],
  ['safety.toxicity.threshold', 'Toxicity threshold', 0.5, 0.99, 0.01],
  ['balance.max_share', 'Max category share', 0.1, 0.8, 0.01],
];
const getPath = (o, p) => p.split('.').reduce((a, k) => a?.[k], o);
function setPath(o, p, v) { const ks = p.split('.'); let t = o; ks.slice(0, -1).forEach((k) => { t = t[k] ||= {}; }); t[ks.at(-1)] = v; }
function buildSliders(cfg) {
  $('#sliders').innerHTML = SLIDERS.map(([p, l, mn, mx, st]) => { const v = getPath(cfg, p); return `<label>${l}<input type="range" data-p="${p}" min="${mn}" max="${mx}" step="${st}" value="${v}"><output>${v}</output></label>`; }).join('') +
    `<label>Toxicity gate<select id="tox-on" class="btn" style="text-transform:none"><option value="1">On (slower: runs toxic-bert)</option><option value="0">Off</option></select></label>`;
  $('#sliders').addEventListener('input', (e) => { const o = e.target.closest('label')?.querySelector('output'); if (o && e.target.type === 'range') o.textContent = e.target.value; });
}
function overrides() {
  const o = {};
  document.querySelectorAll('#sliders input[type=range]').forEach((s) => setPath(o, s.dataset.p, +s.value));
  setPath(o, 'safety.toxicity.enabled', $('#tox-on')?.value !== '0');
  return o;
}
async function probeLive() {
  try {
    if (!document.querySelector('meta[name="sftforge-server"]')) throw new Error('static');
    const h = await fetch('health', { cache: 'no-store' }).then((r) => (r.ok ? r.json() : Promise.reject()));
    if (h.status !== 'ok') throw new Error('x');
    app.live.available = true; app.live.limits = h;
    app.live.cfg = await fetch('api/config').then((r) => r.json());
    buildSliders(app.live.cfg);
    $('#livenote').textContent = `Local server found. Limits: ${fmt(h.max_rows)} rows, ${(h.max_bytes / 1e6).toFixed(0)} MB. Input is held in memory only and never executed.`;
  } catch {
    app.live.available = false;
    for (const id of ['go', 'file', 'example', 'paste']) $('#' + id).disabled = true;
    $('#live-card').style.opacity = '.7';
    $('#livenote').innerHTML = 'Live mode needs the Python service, which a static host does not have. Clone the repo and run <b>sftforge serve</b>, then open <b>http://127.0.0.1:8104</b>. The two recorded runs above work everywhere.';
    $('#livenote').className = 'note warn';
  }
}
$('#example').addEventListener('click', async () => {
  try {
    const res = await fetch('api/example'); if (!res.ok) throw new Error(res.status);
    $('#paste').value = await res.text();
    $('#livenote').className = 'note'; $('#livenote').textContent = 'Loaded a 60-row messy example with duplicates, cut-off answers, refusals, PII and fake secrets. Press Run pipeline.';
  } catch { $('#livenote').className = 'note warn'; $('#livenote').textContent = 'Could not load the example from the server.'; }
});
$('#file').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  if (f.size > (app.live.limits?.max_bytes || 2e6)) { $('#livenote').textContent = 'That file is over the size limit.'; $('#livenote').className = 'note warn'; return; }
  $('#paste').value = await f.text(); $('#livenote').className = 'note'; $('#livenote').textContent = `${f.name} loaded (${(f.size / 1024).toFixed(0)} KB). Press Run pipeline.`;
});
$('#go').addEventListener('click', async () => {
  const text = $('#paste').value.trim(); const note = $('#livenote'), bar = $('#liveprog'), fill = bar.firstElementChild;
  if (!text) { note.textContent = 'Paste some JSONL first, or load the example.'; note.className = 'note warn'; return; }
  note.className = 'note'; bar.style.display = 'block'; fill.style.width = '3%'; $('#go').disabled = true; $('#exportzip').disabled = true;
  try {
    const res = await fetch('api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, config: overrides() }) });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const job = await res.json();
    const W8 = { dedupe: 0, quality: 0.3, safety: 0.45 };
    for (;;) {
      await new Promise((r) => setTimeout(r, 450));
      const d = await fetch('api/run/' + job.id).then((r) => r.json());
      if (d.status === 'error') throw new Error(d.error);
      const base = W8[d.stage] ?? 0, span = d.stage === 'dedupe' ? 0.3 : d.stage === 'quality' ? 0.15 : 0.55;
      fill.style.width = Math.max(4, 100 * (base + span * (d.progress || 0))) + '%';
      note.textContent = d.status === 'queued' ? 'Queued.' : `Running ${d.stage || ''} ${Math.round(100 * (d.progress || 0))}%`;
      if (d.status === 'done') {
        app.liveData = d.result; app.live.job = job.id; setRun(d.result, 'live');
        const bad = d.result.meta.load_errors?.length ? ` ${d.result.meta.load_errors.length}+ unreadable lines skipped (${d.result.meta.load_errors[0]}).` : '';
        note.textContent = `Done: ${fmt(d.result.audit.rows_kept)} of ${fmt(d.result.audit.rows_in)} rows kept.${bad}`;
        $('#exportzip').disabled = false; $('#stage').scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'center' });
        break;
      }
    }
  } catch (err) { note.textContent = 'Run failed: ' + err.message; note.className = 'note warn'; }
  finally { bar.style.display = 'none'; $('#go').disabled = false; }
});
$('#exportzip').addEventListener('click', () => { if (app.live.job) location.href = 'api/run/' + app.live.job + '/export.zip'; });

/* ---------------------------------------------------------------- theme and boot */
function applyTheme(t) { document.documentElement.dataset.theme = t; $('meta[name=theme-color]').content = t === 'light' ? '#f4f3ee' : '#07080a'; readPalette(); }
$('#theme').addEventListener('click', () => { const t = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'; store.set('forge-theme', t); applyTheme(t); });
applyTheme(store.get('forge-theme') || 'dark');
layout();
requestAnimationFrame(frame);
probeLive();
loadRecorded(new URLSearchParams(location.search).get('run') === 'bench' ? 'bench' : 'dolly');
window.__forge = { sim, app };
