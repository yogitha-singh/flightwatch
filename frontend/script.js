/* ==========================================================================
   FLIGHTWATCH · script.js
   Vanilla ES6+. Sections:
     1. Config & helpers      5. Gauge / factors / flights / weather
     2. API client            6. Charts, KPIs, best-time, airline table
     3. Radar (landing)       7. Navigation & boot
     4. Airport combobox
   ========================================================================== */
(() => {
  "use strict";

  /* ======================================================================
     1. CONFIG & HELPERS
     ====================================================================== */
  // Backend URL.
  //  - ""                     -> same origin (FastAPI serves this frontend; Render single-service deploy)
  //  - "https://xxx.onrender.com" -> set this when the frontend is hosted separately (Vercel)
  // When opened via a local static server (Live Server, `python -m http.server`) we auto-point at :8000.
  const CONFIG = { API_BASE: "" };
  const API_BASE = (CONFIG.API_BASE || (
    ["localhost", "127.0.0.1"].includes(location.hostname) && location.port !== "8000" ? "http://localhost:8000" : ""
  )).replace(/\/$/, "");

  const QUICK_ROUTES = [["DEL", "BOM"], ["JFK", "LAX"], ["LHR", "DXB"], ["SIN", "SYD"]];
  const POPULAR = ["DEL", "BOM", "BLR", "JFK", "LAX", "LHR", "DXB", "SIN"];
  const GAUGE = { r: 104, get C() { return 2 * Math.PI * this.r; }, get L() { return this.C * 0.75; } };
  const prefersReduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const pad2 = (n) => String(n).padStart(2, "0");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const riskLevel = (p) => (p < 0.30 ? "low" : p < 0.60 ? "medium" : "high");
  const levelLabel = { low: "ON TIME", medium: "AT RISK", high: "DELAYED" };
  const fmtDur = (min) => (min < 60 ? `${min}m` : `${Math.floor(min / 60)}h ${pad2(min % 60)}m`);
  const hourRange = (h) => `${pad2(h)}:00–${pad2(h)}:59`;

  const state = {
    airports: [], byCode: new Map(),
    from: null, to: null,
    data: null, stats: null, visible: [],
    selected: 0, airlineFilter: null,
    chartTab: "hour", chart: null,
    sound: false, requestId: 0, modelInfo: null,
  };

  function storageGet(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
  }
  function storageSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode / blocked */ }
  }

  function toast(msg) {
    const t = $("#toast");
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => t.classList.remove("show"), 2200);
  }

  /** Animate a number from its previous value to `to`. */
  function countUp(node, to, { dur = 1200, dec = 0, suffix = "" } = {}) {
    if (!node) return;
    cancelAnimationFrame(node._raf);
    const from = Number.isFinite(node._val) ? node._val : 0;
    node._val = to;
    const fmt = (v) => v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec }) + suffix;
    if (prefersReduced || from === to) { node.textContent = fmt(to); return; }
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      const e = 1 - Math.pow(1 - k, 3);
      node.textContent = fmt(from + (to - from) * e);
      if (k < 1) node._raf = requestAnimationFrame(step);
    };
    node._raf = requestAnimationFrame(step);
  }
  const animateCounts = (root) => $$("[data-count]", root).forEach((n) =>
    countUp(n, parseFloat(n.dataset.count), { dec: parseInt(n.dataset.dec || "0", 10) }));

  /** Local calendar helpers (flights come back as naive local strings "YYYY-MM-DDTHH:MM"). */
  function todayIn(tz) {
    try { return new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date()); }
    catch { return new Date().toISOString().slice(0, 10); }
  }
  function dayLabel(localStr, tz) {
    const day = localStr.slice(0, 10);
    const diff = Math.round((Date.parse(day + "T00:00:00Z") - Date.parse(todayIn(tz) + "T00:00:00Z")) / 86400000);
    if (diff === 0) return "Today";
    if (diff === 1) return "Tomorrow";
    return new Date(day + "T00:00:00Z").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });
  }

  /* ======================================================================
     2. API CLIENT
     ====================================================================== */
  class ApiError extends Error {
    constructor(message, { kind = "http", status = 0 } = {}) { super(message); this.kind = kind; this.status = status; }
  }

  async function api(path, params = {}, { timeout = 60000 } = {}) {
    const qs = new URLSearchParams(params).toString();
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeout);
    try {
      const res = await fetch(`${API_BASE}${path}${qs ? "?" + qs : ""}`, { signal: ctrl.signal, headers: { Accept: "application/json" } });
      if (!res.ok) {
        let detail = `Request failed (${res.status})`;
        try {
          const body = await res.json();
          if (typeof body.detail === "string") detail = body.detail;
          else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join("; ");
        } catch { /* non-JSON error body */ }
        throw new ApiError(detail, { kind: "http", status: res.status });
      }
      return await res.json();
    } catch (err) {
      if (err instanceof ApiError) throw err;
      if (err.name === "AbortError") throw new ApiError("The server took too long to respond.", { kind: "timeout" });
      throw new ApiError("Could not reach the FLIGHTWATCH API.", { kind: "network" });
    } finally {
      clearTimeout(timer);
    }
  }

  function errorBanner(err, { retry = true } = {}) {
    let title = "Request failed", body = esc(err.message);
    if (err.kind === "network") {
      title = "API unreachable";
      body = `FLIGHTWATCH can't reach the backend${API_BASE ? ` at <code>${esc(API_BASE)}</code>` : ""}. ` +
        `If you're running locally, start it with <code>uvicorn main:app --port 8000</code> from the <code>backend/</code> folder.`;
    } else if (err.kind === "timeout") {
      title = "Server timed out";
      body = "Free-tier hosts sleep when idle and can take ~30 seconds to wake up. Try again in a moment.";
    } else if (err.status === 404 || err.status === 400) {
      title = "Invalid route";
    } else if (err.status === 503) {
      title = "Model unavailable";
    }
    return `<strong>${esc(title.toUpperCase())}</strong>${body}` +
      `<div class="banner-actions">${retry ? '<button type="button" class="btn-ghost" data-act="retry">Try again</button>' : ""}` +
      `<button type="button" class="btn-ghost" data-act="home">New search</button></div>`;
  }

  const LIVE_TEXT = { live: "LIVE DATA", estimated: "WX ESTIMATED", offline: "API OFFLINE", idle: "CONNECTING" };
  function setLive(s) {
    const pill = $("#livePill");
    pill.dataset.state = s;
    $("#liveText").textContent = LIVE_TEXT[s];
  }

  /* ======================================================================
     3. RADAR (landing hero canvas)
     Airports are plotted on an azimuthal-equidistant projection, so the radar
     is literally a map of the network. Blips brighten as the sweep passes.
     ====================================================================== */
  const Radar = (() => {
    const canvas = $("#radar");
    const ctx = canvas && canvas.getContext ? canvas.getContext("2d") : null;
    let w = 0, h = 0, cx = 0, cy = 0, R = 0, pts = [], raf = 0, running = false, hover = null, route = { from: null, to: null };
    const CENTER = { lat: (25 * Math.PI) / 180, lon: (50 * Math.PI) / 180 };

    function project(latDeg, lonDeg) {
      const phi = (latDeg * Math.PI) / 180, dl = (lonDeg * Math.PI) / 180 - CENTER.lon;
      const cosc = Math.sin(CENTER.lat) * Math.sin(phi) + Math.cos(CENTER.lat) * Math.cos(phi) * Math.cos(dl);
      const c = Math.acos(Math.min(1, Math.max(-1, cosc)));
      const k = c === 0 ? 1 : c / Math.sin(c);
      return {
        x: k * Math.cos(phi) * Math.sin(dl),
        y: k * (Math.cos(CENTER.lat) * Math.sin(phi) - Math.sin(CENTER.lat) * Math.cos(phi) * Math.cos(dl)),
      };
    }
    function layout() {
      if (!ctx) return;
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = rect.width; h = rect.height;
      canvas.width = Math.max(1, Math.round(w * dpr)); canvas.height = Math.max(1, Math.round(h * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cx = w / 2; cy = h / 2; R = Math.min(w, h * 1.15) * 0.47;
      const s = R / Math.PI;
      pts = state.airports.map((a) => {
        const p = project(a.lat, a.lon);
        const x = cx + p.x * s, y = cy - p.y * s;
        return { code: a.code, x, y, ang: Math.atan2(y - cy, x - cx) };
      });
    }
    function draw(now) {
      if (!ctx || !w) return;
      ctx.clearRect(0, 0, w, h);
      const theta = prefersReduced ? 0.9 : ((now - 0) * 0.00055) % (Math.PI * 2);
      // rings + crosshair
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(0,212,255,0.13)";
      [0.25, 0.5, 0.75, 1].forEach((f) => { ctx.beginPath(); ctx.arc(cx, cy, R * f, 0, Math.PI * 2); ctx.stroke(); });
      ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy); ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
      // bearing ticks
      for (let i = 0; i < 72; i++) {
        const a = (i * Math.PI) / 36, len = i % 6 === 0 ? 12 : i % 2 === 0 ? 7 : 4;
        ctx.strokeStyle = i % 6 === 0 ? "rgba(0,212,255,0.4)" : "rgba(0,212,255,0.2)";
        ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R); ctx.lineTo(cx + Math.cos(a) * (R + len), cy + Math.sin(a) * (R + len)); ctx.stroke();
      }
      ctx.fillStyle = "rgba(122,139,168,0.75)"; ctx.font = '10px "JetBrains Mono", monospace'; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      [["000", -Math.PI / 2], ["090", 0], ["180", Math.PI / 2], ["270", Math.PI]].forEach(([t, a]) => ctx.fillText(t, cx + Math.cos(a) * (R + 26), cy + Math.sin(a) * (R + 26)));
      // sweep wedge
      for (let i = 0; i < 36; i++) {
        const a = theta - i * 0.028, k = 1 - i / 36;
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, R, a - 0.03, a); ctx.closePath();
        ctx.fillStyle = `rgba(0,212,255,${0.2 * k * k})`; ctx.fill();
      }
      ctx.strokeStyle = "rgba(0,212,255,0.65)"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + Math.cos(theta) * R, cy + Math.sin(theta) * R); ctx.stroke();
      // blips
      ctx.textAlign = "left";
      for (const p of pts) {
        const d = (((theta - p.ang) % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
        const b = prefersReduced ? 0.5 : Math.exp(-d / 1.15);
        const hot = p.code === hover || p.code === route.from || p.code === route.to;
        ctx.fillStyle = `rgba(0,212,255,${0.16 + 0.84 * b})`;
        ctx.beginPath(); ctx.arc(p.x, p.y, 2 + 2.2 * b, 0, Math.PI * 2); ctx.fill();
        if (b > 0.28 && !hot) { ctx.fillStyle = `rgba(232,244,248,${b * 0.8})`; ctx.fillText(p.code, p.x + 8, p.y); }
      }
      // selected route
      const a = pts.find((p) => p.code === route.from), b = pts.find((p) => p.code === route.to);
      if (a && b) {
        const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, dx = b.x - a.x, dy = b.y - a.y;
        const qx = mx - dy * 0.22, qy = my + dx * 0.22;
        ctx.strokeStyle = "rgba(0,212,255,0.85)"; ctx.lineWidth = 1.6; ctx.setLineDash([6, 5]); ctx.lineDashOffset = -now * 0.02;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.quadraticCurveTo(qx, qy, b.x, b.y); ctx.stroke(); ctx.setLineDash([]);
      }
      const pulse = 7 + 3 * Math.sin(now * 0.005);
      [route.from, route.to, hover].forEach((code, i) => {
        const p = pts.find((q) => q.code === code); if (!p) return;
        ctx.strokeStyle = "#00d4ff"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(p.x, p.y, pulse, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = "#00d4ff"; ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2); ctx.fill();
        ctx.font = 'bold 12px "JetBrains Mono", monospace'; ctx.fillStyle = "#e8f4f8"; ctx.fillText(code, p.x + 12, p.y - 10);
        ctx.font = '10px "JetBrains Mono", monospace';
      });
    }
    function loop(now) { draw(now); if (running && !prefersReduced) raf = requestAnimationFrame(loop); }
    return {
      init() {
        if (!ctx) return;
        layout(); draw(performance.now());
        window.addEventListener("resize", () => { layout(); draw(performance.now()); });
        document.addEventListener("visibilitychange", () => { if (document.hidden) this.stop(); else if (document.body.dataset.view === "landing") this.start(); });
      },
      refresh() { layout(); draw(performance.now()); },     // after airports load
      start() { if (!ctx || running) return; running = true; layout(); raf = requestAnimationFrame(loop); },
      stop() { running = false; cancelAnimationFrame(raf); },
      setRoute(from, to) { route = { from, to }; if (!running) draw(performance.now()); },
      setHover(code) { hover = code; if (!running) draw(performance.now()); },
    };
  })();

  /* ======================================================================
     4. AIRPORT COMBOBOX (autocomplete from /api/airports)
     ====================================================================== */
  const airportLabel = (a) => `${a.code} · ${a.city}`;

  function matchAirports(query) {
    const q = query.trim().toLowerCase();
    if (!q) return POPULAR.map((c) => state.byCode.get(c)).filter(Boolean);
    return state.airports
      .map((a) => {
        const code = a.code.toLowerCase(), city = a.city.toLowerCase(), name = a.name.toLowerCase(), country = a.country.toLowerCase();
        let s = 99;
        if (code === q) s = 0; else if (code.startsWith(q)) s = 1; else if (city.startsWith(q)) s = 2;
        else if (city.includes(q)) s = 3; else if (name.includes(q)) s = 4; else if (country.includes(q)) s = 5;
        return { a, s };
      })
      .filter((x) => x.s < 99)
      .sort((x, y) => x.s - y.s || x.a.city.localeCompare(y.a.city))
      .slice(0, 8)
      .map((x) => x.a);
  }

  function createCombo(input, { onChange } = {}) {
    const list = $(`#${input.getAttribute("aria-controls")}`);
    let items = [], active = -1;

    const isOpen = () => !list.hidden;
    function close() { list.hidden = true; input.setAttribute("aria-expanded", "false"); input.removeAttribute("aria-activedescendant"); active = -1; onChange && onChange(null, "hover"); }
    function setActive(i) {
      active = i;
      $$(".combo-opt", list).forEach((li, k) => li.setAttribute("aria-selected", String(k === i)));
      const li = $$(".combo-opt", list)[i];
      if (li) { input.setAttribute("aria-activedescendant", li.id); li.scrollIntoView({ block: "nearest" }); onChange && onChange(items[i].code, "hover"); }
    }
    function open(query) {
      items = matchAirports(query);
      list.innerHTML = items.length
        ? items.map((a, i) => `<li class="combo-opt" role="option" id="${list.id}-${i}" data-code="${a.code}" aria-selected="false">
              <span class="opt-code">${a.code}</span>
              <span><span class="opt-city">${esc(a.city)}, ${esc(a.country)}</span><br><span class="opt-name">${esc(a.name)}</span></span></li>`).join("")
        : `<li class="combo-empty" role="presentation">No airport matches “${esc(query)}”.</li>`;
      list.hidden = false; input.setAttribute("aria-expanded", "true");
      setActive(items.length ? 0 : -1);
    }
    function pick(code) {
      const a = state.byCode.get(code); if (!a) return;
      input.value = airportLabel(a); input.dataset.code = a.code; input.removeAttribute("aria-invalid");
      close(); onChange && onChange(a.code, "pick");
    }
    function resolve() {
      const a = state.byCode.get(input.dataset.code || "");
      if (a && input.value === airportLabel(a)) return a.code;
      const raw = input.value.trim(); if (!raw) return null;
      const m = raw.match(/^([A-Za-z]{3})\b/);
      if (m && state.byCode.has(m[1].toUpperCase())) return m[1].toUpperCase();
      const city = state.airports.find((x) => x.city.toLowerCase() === raw.toLowerCase());
      return city ? city.code : null;
    }

    input.addEventListener("focus", () => { input.select(); const a = state.byCode.get(input.dataset.code || ""); open(a && input.value === airportLabel(a) ? "" : input.value); });
    input.addEventListener("input", () => { delete input.dataset.code; input.removeAttribute("aria-invalid"); open(input.value); onChange && onChange(resolve(), "type"); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { e.preventDefault(); if (!isOpen()) open(input.value); else if (items.length) setActive((active + 1) % items.length); }
      else if (e.key === "ArrowUp") { e.preventDefault(); if (isOpen() && items.length) setActive((active - 1 + items.length) % items.length); }
      else if (e.key === "Enter" && isOpen() && active >= 0 && items[active]) { e.preventDefault(); pick(items[active].code); input.dispatchEvent(new CustomEvent("combo-picked", { bubbles: true })); }
      else if (e.key === "Escape") { close(); }
    });
    list.addEventListener("pointerdown", (e) => { const li = e.target.closest(".combo-opt"); if (li) { e.preventDefault(); pick(li.dataset.code); input.dispatchEvent(new CustomEvent("combo-picked", { bubbles: true })); } });
    list.addEventListener("pointerover", (e) => { const li = e.target.closest(".combo-opt"); if (li) onChange && onChange(li.dataset.code, "hover"); });
    document.addEventListener("pointerdown", (e) => { if (!list.contains(e.target) && e.target !== input) close(); });

    return {
      input, resolve,
      set(code) { const a = state.byCode.get(code); if (a) { input.value = airportLabel(a); input.dataset.code = a.code; } else { input.value = ""; delete input.dataset.code; } },
      focus() { input.focus(); },
    };
  }

  const SWAP_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 4 3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7"/></svg>';
  const searches = {};

  function buildSearch(formId, prefix, isHero) {
    const form = $(`#${formId}`);
    const field = (role, label, ph) => `
      <div class="field">
        <label class="field-label" for="${prefix}${role}">${label}</label>
        <input class="combo-input" id="${prefix}${role}" data-role="${role.toLowerCase()}" type="text" role="combobox"
               aria-label="${label}" aria-autocomplete="list" aria-expanded="false" aria-controls="${prefix}${role}List"
               placeholder="${ph}" autocomplete="off" autocapitalize="characters" spellcheck="false">
        <ul class="combo-list" id="${prefix}${role}List" role="listbox" hidden></ul>
      </div>`;
    form.innerHTML = field("From", "From (IATA)", isHero ? "DEL or Delhi" : "From · DEL") +
      `<button type="button" class="swap-btn" aria-label="Swap origin and destination">${SWAP_SVG}</button>` +
      field("To", "To (IATA)", isHero ? "BOM or Mumbai" : "To · BOM") +
      `<button class="btn-primary" type="submit">Analyze route</button><div class="form-error" role="alert"></div>`;

    const sync = (code, why) => { if (isHero && (why === "pick" || why === "type")) Radar.setRoute(searches.hero.from.resolve(), searches.hero.to.resolve()); if (isHero && why === "hover") Radar.setHover(code); };
    const fromCombo = createCombo($(`#${prefix}From`), { onChange: (c, w) => sync(c, w) });
    const toCombo = createCombo($(`#${prefix}To`), { onChange: (c, w) => sync(c, w) });
    const errEl = $(".form-error", form);

    form.addEventListener("combo-picked", (e) => {   // after picking origin, jump to destination
      if (e.target === fromCombo.input && !toCombo.resolve()) toCombo.focus();
    });
    $(".swap-btn", form).addEventListener("click", (e) => {
      const a = fromCombo.resolve(), b = toCombo.resolve();
      fromCombo.set(b); toCombo.set(a);
      e.currentTarget.classList.toggle("spin");
      if (isHero) Radar.setRoute(fromCombo.resolve(), toCombo.resolve());
    });
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      errEl.textContent = "";
      const f = fromCombo.resolve(), t = toCombo.resolve();
      fromCombo.input.removeAttribute("aria-invalid"); toCombo.input.removeAttribute("aria-invalid");
      if (!state.airports.length) { errEl.textContent = "Airport list not loaded yet — check the API connection."; return; }
      if (!f) { fromCombo.input.setAttribute("aria-invalid", "true"); errEl.textContent = "Choose a valid origin airport (e.g. DEL)."; fromCombo.focus(); return; }
      if (!t) { toCombo.input.setAttribute("aria-invalid", "true"); errEl.textContent = "Choose a valid destination airport (e.g. BOM)."; toCombo.focus(); return; }
      if (f === t) { toCombo.input.setAttribute("aria-invalid", "true"); errEl.textContent = "Origin and destination must be different."; return; }
      $$(".combo-list").forEach((l) => { l.hidden = true; });
      navigate(f, t);
    });
    searches[isHero ? "hero" : "header"] = { form, from: fromCombo, to: toCombo, errEl };
  }

  function syncSearches(from, to) {
    Object.values(searches).forEach((s) => { s.from.set(from); s.to.set(to); s.errEl.textContent = ""; });
    Radar.setRoute(from, to);
  }

  /* ======================================================================
     5. RESULTS: GAUGE · RISK FACTORS · FLIGHTS · WEATHER
     ====================================================================== */
  const ICONS = {
    time: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    calendar: '<rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/>',
    season: '<circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M3 12h2M19 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/>',
    airline: '<path d="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5z"/>',
    airport: '<path d="M3 21h18M6 21V11l6-7 6 7v10M10 21v-5h4v5"/>',
    distance: '<path d="M4 12h16M14 6l6 6-6 6"/>',
    weather: '<path d="M7 17a4 4 0 0 1-.6-7.95A6 6 0 0 1 18 9.5 4 4 0 0 1 17.5 17z"/><path d="M9 20.5l1-2M13 20.5l1-2"/>',
  };
  const icon = (k) => `<svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[k] || ICONS.time}</svg>`;

  const gaugeEl = {
    panel: $("#gaugePanel"), fill: $("#gaugeFill"), needle: $("#gaugeNeedle"), num: $("#gaugeNum"), status: $("#gaugeStatus"),
    exp: $("#expNum"), conf: $("#confNum"), confBar: $("#confBar"), sel: $("#selFlight"), avg: $("#routeAvg"),
  };

  function buildGaugeStatic() {
    const NS = "http://www.w3.org/2000/svg";
    const mk = (tag, attrs) => { const n = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); return n; };
    const arc = (cls, startPct, lenPct) => mk("circle", {
      class: cls, cx: 130, cy: 130, r: GAUGE.r, transform: "rotate(135 130 130)",
      "stroke-dasharray": `${GAUGE.L * lenPct} 999`, "stroke-dashoffset": -GAUGE.L * startPct,
    });
    const segs = $("#gaugeSegs");
    segs.appendChild(arc("gauge-track", 0, 1));
    segs.appendChild(arc("gauge-seg s-low", 0, 0.295));
    segs.appendChild(arc("gauge-seg s-medium", 0.30, 0.295));
    segs.appendChild(arc("gauge-seg s-high", 0.60, 0.40));
    const ticks = $("#gaugeTicks");
    for (let i = 0; i <= 20; i++) {
      const major = i % 2 === 0, ang = ((135 + 270 * (i / 20)) * Math.PI) / 180, r1 = major ? 114 : 117, r2 = 124;
      ticks.appendChild(mk("line", {
        class: "gauge-tick" + (major ? " major" : ""),
        x1: 130 + Math.cos(ang) * r1, y1: 130 + Math.sin(ang) * r1, x2: 130 + Math.cos(ang) * r2, y2: 130 + Math.sin(ang) * r2,
      }));
    }
    gaugeEl.fill.setAttribute("stroke-dasharray", `0.01 999`);
  }

  function setGauge(f) {
    const p = f.delay_probability;
    gaugeEl.panel.dataset.level = f.risk_level;
    gaugeEl.fill.style.strokeDasharray = `${Math.max(0.01, GAUGE.L * p)} 999`;
    gaugeEl.needle.style.transform = `rotate(${225 + 270 * p}deg)`;
    countUp(gaugeEl.num, Math.round(p * 100), { dur: 1500 });
    gaugeEl.status.textContent = levelLabel[f.risk_level];
    countUp(gaugeEl.exp, Math.round(f.predicted_delay_minutes), { dur: 1200 });
    countUp(gaugeEl.conf, Math.round(f.confidence * 100), { dur: 1200, suffix: "%" });
    gaugeEl.confBar.style.width = `${Math.round(f.confidence * 100)}%`;
    gaugeEl.sel.textContent = `${f.flight_number} · ${f.airline_name} · ${f.scheduled_departure_local.slice(11)} ${dayLabel(f.scheduled_departure_local, state.data.route.origin.timezone).toLowerCase()}`;
  }

  function setFactors(f) {
    const ul = $("#factorsList");
    ul.innerHTML = f.risk_factors.length
      ? f.risk_factors.map((r) => `
          <li class="factor" data-dir="${esc(r.direction)}" data-level="${f.risk_level}">
            <span class="factor-ico">${icon(r.category)}</span>
            <div><div class="factor-text">${esc(r.factor)}</div><div class="factor-bar"><i data-w="${Math.round(r.weight * 100)}"></i></div></div>
            <span class="factor-pts">${r.impact_pts > 0 ? "+" : ""}${r.impact_pts.toFixed(1)} pts</span>
          </li>`).join("")
      : '<li class="empty">No dominant risk factors for this flight.</li>';
    requestAnimationFrame(() => requestAnimationFrame(() => $$(".factor-bar i", ul).forEach((i) => { i.style.width = i.dataset.w + "%"; })));
  }

  /* ---- sound alert (off by default) ---- */
  let audioCtx = null;
  function beep(level) {
    if (!state.sound || level === "low") return;
    try {
      audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") audioCtx.resume();
      const seq = level === "high" ? [[880, 0], [620, 0.17], [880, 0.34]] : [[520, 0]];
      seq.forEach(([freq, at]) => {
        const o = audioCtx.createOscillator(), g = audioCtx.createGain(), t = audioCtx.currentTime + at;
        o.type = "square"; o.frequency.value = freq;
        g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.05, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.15);
        o.connect(g); g.connect(audioCtx.destination); o.start(t); o.stop(t + 0.17);
      });
    } catch { /* audio blocked */ }
  }
  function initSound() {
    state.sound = !!storageGet("fw_sound", false);
    const btn = $("#soundBtn");
    const paint = () => { btn.setAttribute("aria-pressed", String(state.sound)); btn.title = `Sound alert on elevated risk: ${state.sound ? "on" : "off"}`; };
    paint();
    btn.addEventListener("click", () => {
      state.sound = !state.sound; storageSet("fw_sound", state.sound); paint();
      toast(state.sound ? "Risk sound alerts ON" : "Risk sound alerts OFF");
      if (state.sound) beep("medium");
    });
  }

  function selectFlight(i, { announce = false } = {}) {
    const f = state.visible[i]; if (!f) return;
    state.selected = i;
    $$("#flightsList .flight").forEach((c, k) => c.setAttribute("aria-current", String(k === i)));
    setGauge(f); setFactors(f);
    if (announce) beep(f.risk_level);
  }

  function flightFront(f, i) {
    const tz = state.data.route.origin.timezone;
    return `
      <div class="fl-airline"><span class="badge" aria-hidden="true">${esc(f.airline_code)}</span></div>
      <div><div class="fl-no">${esc(f.flight_number)}</div><div class="fl-sub">${esc(f.airline_name)} · ${esc(f.aircraft)}</div></div>
      <div class="fl-time"><div class="fl-clock">${esc(f.scheduled_departure_local.slice(11))}</div>
        <div class="fl-day">${esc(dayLabel(f.scheduled_departure_local, tz))} · in ${fmtDur(f.departs_in_min)}</div></div>
      <div class="fl-delay"><span class="delay-badge">+${Math.round(f.predicted_delay_minutes)} min</span><span class="fl-prob">${(f.delay_probability * 100).toFixed(0)}% risk</span></div>
      <div class="fl-status"><span class="pill">${esc(f.status)}</span></div>
      <button type="button" class="flip-btn" aria-label="Flip ${esc(f.flight_number)} for details" title="Flip for details">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg></button>`;
  }
  function flightBack(f) {
    const r = state.data.route;
    return `
      <div class="back-col"><h4>Risk drivers</h4><ul>${f.risk_factors.map((x) =>
        `<li><span>${esc(x.factor)}</span><span>${x.impact_pts > 0 ? "+" : ""}${x.impact_pts.toFixed(1)}</span></li>`).join("")}</ul></div>
      <div class="back-col"><h4>Conditions &amp; timing</h4>
        <div class="kv"><span>${esc(r.origin.code)} dep </span>${esc(f.weather_origin_condition)}</div>
        <div class="kv"><span>${esc(r.destination.code)} arr </span>${esc(f.weather_destination_condition)}</div>
        <div class="kv"><span>Arrives </span>${esc(f.scheduled_arrival_local.slice(11))} local</div>
        <div class="kv"><span>Confidence </span>${Math.round(f.confidence * 100)}%</div></div>
      <button type="button" class="flip-btn" aria-label="Flip back" title="Flip back"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 13a8 8 0 1 0 2.3-5.7M4 4v7h7"/></svg></button>`;
  }

  function renderFlightList() {
    const ul = $("#flightsList");
    state.visible = state.data.flights.filter((f) => !state.airlineFilter || f.airline_code === state.airlineFilter);
    if (!state.visible.length) { ul.innerHTML = '<li class="empty">No departures for this airline in the next 24 hours.</li>'; return; }
    ul.innerHTML = state.visible.map((f, i) =>
      `<li><div class="flight" data-i="${i}" data-level="${f.risk_level}" tabindex="0" role="button" aria-current="false"
            style="--d:${Math.min(i, 9) * 55}ms" aria-label="${esc(f.flight_number)} ${esc(f.status)}, ${(f.delay_probability * 100).toFixed(0)} percent delay risk">${flightFront(f, i)}</div></li>`).join("");
  }

  function flipCard(card) {
    const f = state.visible[+card.dataset.i]; if (!f) return;
    const toBack = !card.classList.contains("is-back");
    const swap = () => { card.classList.toggle("is-back", toBack); card.innerHTML = toBack ? flightBack(f) : flightFront(f, +card.dataset.i); };
    if (prefersReduced || !card.animate) { swap(); return; }
    card.animate([{ transform: "rotateX(0deg)" }, { transform: "rotateX(90deg)" }], { duration: 160, easing: "ease-in" }).onfinish = () => {
      swap();
      card.animate([{ transform: "rotateX(-90deg)" }, { transform: "rotateX(0deg)" }], { duration: 220, easing: "ease-out" });
    };
  }

  function bindFlightList() {
    const ul = $("#flightsList");
    ul.addEventListener("click", (e) => {
      const card = e.target.closest(".flight"); if (!card) return;
      const i = +card.dataset.i;
      if (e.target.closest(".flip-btn")) { selectFlight(i, { announce: i !== state.selected }); flipCard(card); return; }
      if (i !== state.selected) selectFlight(i, { announce: true });
    });
    ul.addEventListener("keydown", (e) => {
      const card = e.target.closest(".flight");
      if (card && e.target === card && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); selectFlight(+card.dataset.i, { announce: true }); }
    });
  }

  function renderAirlinePills(route) {
    const box = $("#airlinePills");
    box.innerHTML = route.airlines.map((a) =>
      `<button type="button" class="air-pill" data-code="${esc(a.code)}" aria-pressed="false" title="Show only ${esc(a.name)}"><b>${esc(a.code)}</b>${esc(a.name)}</button>`).join("");
  }

  function renderRouteHead(d) {
    const { origin: o, destination: dst, distance_km: km, duration_min: dur } = d.route;
    $("#rhFrom").textContent = o.code; $("#rhTo").textContent = dst.code;
    $("#rhNames").innerHTML = `${esc(o.city)}, ${esc(o.country)} <span>to</span> ${esc(dst.city)}, ${esc(dst.country)}`;
    $("#rhFacts").textContent = `${km.toLocaleString()} km · ${fmtDur(dur)} block time · ${d.flights.length} departures in the next 24 h`;
    renderAirlinePills(d.route);
  }

  function renderWeather(d) {
    const cards = [["origin", $("#wxOrigin"), d.route.origin], ["destination", $("#wxDest"), d.route.destination]];
    cards.forEach(([role, card, ap]) => {
      const w = d.weather[role];
      const level = { LOW: "low", MODERATE: "medium", HIGH: "high" }[w.severity] || "low";
      const pct = Math.round(w.delay_risk_contribution * 100);
      const live = w.source === "open-meteo";
      card.dataset.level = level;
      $(".wx-body", card)?.remove();
      card.insertAdjacentHTML("beforeend", `
        <div class="wx-body">
          <div class="wx-head">
            <div><div class="wx-role">${role === "origin" ? "Departure" : "Arrival"} · ${esc(ap.code)}</div><div class="wx-place">${esc(ap.city)} · ${esc(ap.name)}</div></div>
            <span class="wx-src ${live ? "live" : ""}" title="${live ? "Open-Meteo observation" : "Open-Meteo unreachable: climatology estimate"}">${live ? "LIVE" : "ESTIMATED"}</span>
          </div>
          <div class="wx-main"><div class="wx-temp"><span data-count="${w.temperature_c}" data-dec="0">0</span><small>°C</small></div><div class="wx-cond">${esc(w.condition)}</div></div>
          <div class="wx-metrics">
            <div class="wx-metric"><div class="k">Wind</div><div class="v"><span data-count="${w.wind_kmh}" data-dec="0">0</span> km/h</div></div>
            <div class="wx-metric"><div class="k">Visibility</div><div class="v"><span data-count="${w.visibility_km}" data-dec="1">0</span> km</div></div>
            <div class="wx-metric"><div class="k">Precipitation</div><div class="v"><span data-count="${w.precipitation_mm}" data-dec="1">0</span> mm</div></div>
            <div class="wx-metric"><div class="k">${w.humidity_pct != null ? "Humidity" : "Gusts"}</div><div class="v">${w.humidity_pct != null ? `${w.humidity_pct}%` : (w.wind_gust_kmh != null ? `${Math.round(w.wind_gust_kmh)} km/h` : "n/a")}</div></div>
          </div>
          <div class="wx-impact">
            <div class="conf-head mono"><span>Delay contribution</span><span class="sev">${esc(w.severity)} · <span data-count="${pct}">0</span>%</span></div>
            <div class="bar"><i data-w="${pct}"></i></div>
          </div>
        </div>`);
      card.classList.remove("is-loading");
      animateCounts(card);
      requestAnimationFrame(() => requestAnimationFrame(() => $$(".bar i[data-w]", card).forEach((i) => { i.style.width = Math.max(2, +i.dataset.w) + "%"; })));
    });
  }

  function renderFlightsView(d) {
    renderRouteHead(d);
    renderFlightList();
    $("#routeAvg").innerHTML = `Route outlook <b>${(d.summary.avg_delay_probability * 100).toFixed(0)}%</b> avg delay risk · safest <b>${esc(d.summary.safest_flight)}</b> · riskiest <b>${esc(d.summary.riskiest_flight)}</b>`;
    renderWeather(d);
    ["#gaugePanel", "#factorsPanel", "#flightsPanel"].forEach((s) => $(s).classList.remove("is-loading"));
    selectFlight(0, { announce: true });
    const live = Object.values(d.weather).some((w) => w.source === "open-meteo");
    setLive(live ? "live" : "estimated");
  }

  /* ======================================================================
     6. CHARTS · KPIs · BEST TIME · AIRLINE TABLE
     ====================================================================== */
  const COLORS = { primary: "#00d4ff", danger: "#ff3b3b", success: "#00d97e", muted: "#7a8ba8" };

  function renderKpis(s) {
    const delayRate = (100 - s.on_time_percentage) / 100, lvl = riskLevel(delayRate);
    $("#kpis").innerHTML = `
      <div class="kpi" data-level="${lvl}"><div class="k">On-time</div><div class="v"><span data-count="${s.on_time_percentage}" data-dec="1">0</span><small>%</small></div></div>
      <div class="kpi"><div class="k">Avg delay</div><div class="v"><span data-count="${s.average_delay_minutes}" data-dec="1">0</span><small>min</small></div></div>
      <div class="kpi"><div class="k">Worst day</div><div class="v">${esc(s.worst_day_of_week.day)}<small>${s.worst_day_of_week.delay_rate_pct}%</small></div></div>
      <div class="kpi"><div class="k">Worst hour</div><div class="v">${pad2(s.worst_hour.hour)}:00<small>${s.worst_hour.delay_rate_pct}%</small></div></div>
      <div class="kpi-foot">${s.flights_tracked.toLocaleString()} flights tracked · simulated history</div>`;
    animateCounts($("#kpis"));
  }

  function chartConfig() {
    const s = state.stats, hourly = state.chartTab === "hour";
    const rows = hourly ? s.by_hour : s.by_day;
    const labels = hourly ? rows.map((r) => pad2(r.hour)) : rows.map((r) => r.day.slice(0, 3));
    const values = rows.map((r) => r.delay_rate_pct ?? 0);
    const worst = hourly ? s.worst_hour.hour : s.worst_day_of_week.day_of_week;
    const best = hourly ? s.best_hour.hour : s.best_day_of_week.day_of_week;
    const keyOf = (r) => (hourly ? r.hour : r.day_of_week);
    const colors = rows.map((r) => keyOf(r) === worst ? COLORS.danger : keyOf(r) === best ? COLORS.success
      : (hourly && r.flights < 20) ? "rgba(122,139,168,0.3)" : "rgba(0,212,255,0.55)");
    return { rows, labels, values, colors, hourly };
  }

  function renderChart() {
    const wrap = $("#chartWrap");
    if (!state.stats) return;
    if (typeof Chart === "undefined") {
      wrap.innerHTML = '<div class="chart-fallback">Chart library unavailable (blocked CDN?). The numbers above and below are still accurate.</div>';
      return;
    }
    if (!$("#delayChart")) wrap.innerHTML = '<canvas id="delayChart" aria-label="Delay rate bar chart" role="img"></canvas>';
    const { rows, labels, values, colors, hourly } = chartConfig();
    $("#chartNote").textContent = hourly ? "% of flights delayed >15 min, by local departure hour" : "% of flights delayed >15 min, by weekday";
    if (state.chart) {
      state.chart.data.labels = labels;
      state.chart.data.datasets[0].data = values;
      state.chart.data.datasets[0].backgroundColor = colors;
      state.chart.options.plugins.tooltip.callbacks = tooltipCallbacks(rows, hourly);
      state.chart.update();
      return;
    }
    Chart.defaults.font.family = '"JetBrains Mono", monospace';
    Chart.defaults.color = COLORS.muted;
    state.chart = new Chart($("#delayChart"), {
      type: "bar",
      data: { labels, datasets: [{ data: values, backgroundColor: colors, borderRadius: 2, borderSkipped: false, maxBarThickness: 26 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: { duration: prefersReduced ? 0 : 750, easing: "easeOutQuart" },
        scales: {
          x: { grid: { display: false }, ticks: { font: { size: 10 }, maxRotation: 0, autoSkip: true } },
          y: { beginAtZero: true, grid: { color: "rgba(100,200,255,0.08)" }, border: { display: false }, ticks: { font: { size: 10 }, callback: (v) => v + "%" } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "rgba(8,12,20,0.96)", borderColor: "rgba(0,212,255,0.4)", borderWidth: 1, padding: 10, cornerRadius: 3, displayColors: false,
            titleFont: { family: '"JetBrains Mono", monospace', size: 12 }, bodyFont: { family: '"JetBrains Mono", monospace', size: 11 },
            callbacks: tooltipCallbacks(rows, hourly),
          },
        },
      },
    });
  }
  function tooltipCallbacks(rows, hourly) {
    return {
      title: (items) => { const r = rows[items[0].dataIndex]; return hourly ? hourRange(r.hour) : r.day; },
      label: (item) => {
        const r = rows[item.dataIndex];
        if (r.delay_rate_pct == null) return "No flights in sample";
        return [`Delayed: ${r.delay_rate_pct}%`, `Avg delay: ${r.avg_delay_minutes} min`, `Sample: ${r.flights} flights`];
      },
    };
  }

  function renderBest(s) {
    const lv = (r) => riskLevel(r / 100);
    const spread = (s.worst_hour.delay_rate_pct - s.best_hour.delay_rate_pct).toFixed(1);
    $("#bestBody").innerHTML = `
      <div class="best-card" data-level="${lv(s.best_hour.delay_rate_pct)}"><div class="k">Best hour</div><div class="v">${pad2(s.best_hour.hour)}:00</div><div class="s">${s.best_hour.delay_rate_pct}% delayed · avg +${Math.round(s.best_hour.avg_delay_minutes)} min</div></div>
      <div class="best-card" data-level="${lv(s.worst_hour.delay_rate_pct)}"><div class="k">Avoid</div><div class="v">${pad2(s.worst_hour.hour)}:00</div><div class="s">${s.worst_hour.delay_rate_pct}% delayed · avg +${Math.round(s.worst_hour.avg_delay_minutes)} min</div></div>
      <div class="best-card" data-level="${lv(s.best_day_of_week.delay_rate_pct)}"><div class="k">Best day</div><div class="v">${esc(s.best_day_of_week.day)}</div><div class="s">${s.best_day_of_week.delay_rate_pct}% delayed</div></div>
      <div class="best-card" data-level="${lv(s.worst_day_of_week.delay_rate_pct)}"><div class="k">Worst day</div><div class="v">${esc(s.worst_day_of_week.day)}</div><div class="s">${s.worst_day_of_week.delay_rate_pct}% delayed</div></div>
      <p class="best-note">Departing at <b>${pad2(s.best_hour.hour)}:00</b> instead of <b>${pad2(s.worst_hour.hour)}:00</b> cuts delay odds by about <b>${spread} points</b>. Safest carrier on this route: <b>${esc(s.best_airline)}</b>.</p>`;
  }

  function renderAirlines(s) {
    $("#airlineBody").innerHTML = s.airline_comparison.map((a) => {
      const lvl = riskLevel((100 - a.on_time_pct) / 100);
      return `<tr>
        <td><div class="al-name"><span class="al-code">${esc(a.airline_code)}</span><div>${esc(a.airline)}${a.rank === 1 && s.airline_comparison.length > 1 ? '<br><span class="al-best">SAFEST BET</span>' : ""}</div></div></td>
        <td><div class="ot" data-level="${lvl}"><div class="bar"><i data-w="${a.on_time_pct}"></i></div><b>${a.on_time_pct.toFixed(1)}%</b></div></td>
        <td class="r">${a.avg_delay_minutes.toFixed(1)} min</td><td class="r">${a.flights_tracked.toLocaleString()}</td></tr>`;
    }).join("");
    requestAnimationFrame(() => requestAnimationFrame(() => $$("#airlineBody .bar i").forEach((i) => { i.style.width = i.dataset.w + "%"; })));
  }

  function renderStatsView(s) {
    state.stats = s;
    renderKpis(s); renderBest(s); renderAirlines(s); renderChart();
    ["#kpiPanel", "#chartPanel", "#bestPanel", "#airlinePanel"].forEach((sel) => $(sel).classList.remove("is-loading"));
  }

  function bindTabs() {
    const tabs = $$('.tabs [role="tab"]');
    const activate = (tab) => {
      tabs.forEach((t) => { const on = t === tab; t.setAttribute("aria-selected", String(on)); t.tabIndex = on ? 0 : -1; });
      state.chartTab = tab.dataset.tab; renderChart();
    };
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => activate(t));
      t.addEventListener("keydown", (e) => {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = tabs[(i + (e.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length]; n.focus(); activate(n); }
      });
    });
  }

  /* ======================================================================
     7. NAVIGATION · RECENTS · BOOT
     ====================================================================== */
  function setView(view) {
    document.body.dataset.view = view;
    $("#landing").hidden = view !== "landing";
    $("#results").hidden = view !== "results";
    if (view === "landing") Radar.start(); else Radar.stop();
    window.scrollTo({ top: 0, behavior: "auto" });
  }

  const getRecents = () => storageGet("fw_recent", []).filter((r) => Array.isArray(r) && r.length === 2);
  function addRecent(from, to) {
    const next = [[from, to], ...getRecents().filter(([a, b]) => !(a === from && b === to))].slice(0, 4);
    storageSet("fw_recent", next);
    renderRecents();
  }
  const ARROW = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12h15M13 6l6 6-6 6"/></svg>';
  function renderQuickChips() {
    $("#quickChips").innerHTML = '<span class="chip-label">Popular</span>' + QUICK_ROUTES.map(([a, b]) =>
      `<button type="button" class="chip" data-from="${a}" data-to="${b}">${a}${ARROW}${b}</button>`).join("");
  }
  function renderRecents() {
    const r = getRecents(), box = $("#recents");
    box.hidden = !r.length;
    box.innerHTML = r.length ? '<span class="chip-label">Recent</span>' + r.map(([a, b]) =>
      `<button type="button" class="chip" data-from="${esc(a)}" data-to="${esc(b)}">${esc(a)}${ARROW}${esc(b)}</button>`).join("") : "";
  }

  function navigate(from, to, { push = true } = {}) {
    if (push) history.pushState({ from, to }, "", `?from=${from}&to=${to}`);
    runAnalysis(from, to);
  }

  function setLoading(on) {
    ["#gaugePanel", "#factorsPanel", "#flightsPanel", "#wxOrigin", "#wxDest", "#kpiPanel", "#chartPanel", "#bestPanel", "#airlinePanel"]
      .forEach((s) => $(s).classList.toggle("is-loading", on));
    if (on) $$(".loader-text").forEach((n) => { n.textContent = n.dataset.orig; });
  }

  function resetResults() {
    $("#resultsError").hidden = true;
    $("#grid").hidden = false;
    $("#rhFrom").textContent = state.from; $("#rhTo").textContent = state.to;
    $("#rhNames").innerHTML = "&nbsp;"; $("#rhFacts").textContent = ""; $("#airlinePills").innerHTML = "";
    $("#flightsList").innerHTML = ""; $("#factorsList").innerHTML = "";
    $$(".wx-body").forEach((n) => n.remove());
    $("#kpis").innerHTML = ""; $("#bestBody").innerHTML = ""; $("#airlineBody").innerHTML = "";
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    if (!$("#delayChart")) $("#chartWrap").innerHTML = '<canvas id="delayChart" aria-label="Delay rate bar chart" role="img"></canvas>';
    state.stats = null; state.data = null; state.visible = []; state.airlineFilter = null;
  }

  async function runAnalysis(from, to) {
    const id = ++state.requestId;
    state.from = from; state.to = to;
    setView("results");
    syncSearches(from, to);
    resetResults();
    setLoading(true);
    addRecent(from, to);
    document.title = `${from} → ${to} · FLIGHTWATCH`;
    const cold = setTimeout(() => { if (id === state.requestId) $$(".loader-text").forEach((n) => { n.textContent = "Waking the server (free hosting can take ~30 s)…"; }); }, 7000);

    const current = () => id === state.requestId;
    const pFlights = api("/api/flights", { from, to }).then((d) => { if (current()) { state.data = d; renderFlightsView(d); } })
      .catch((e) => { if (!current()) return; if (!(e instanceof ApiError)) console.error(e); showFatal(e instanceof ApiError ? e : new ApiError("Unexpected error while rendering results.")); });
    const pStats = api("/api/route-stats", { from, to }).then((d) => { if (current()) renderStatsView(d); })
      .catch((e) => { if (!current()) return; if (!(e instanceof ApiError)) console.error(e); showStatsError(e instanceof ApiError ? e : new ApiError("Could not render route history.")); });
    await Promise.allSettled([pFlights, pStats]);
    clearTimeout(cold);
  }

  function showFatal(err) {
    setLoading(false);
    if (err.kind === "network" || err.kind === "timeout") setLive("offline");
    const box = $("#resultsError");
    box.innerHTML = errorBanner(err);
    box.hidden = false;
    $("#grid").hidden = true;
  }
  function showStatsError(err) {
    ["#kpiPanel", "#chartPanel", "#bestPanel", "#airlinePanel"].forEach((s) => $(s).classList.remove("is-loading"));
    $("#chartWrap").innerHTML = `<div class="chart-fallback">Route history unavailable: ${esc(err.message)}</div>`;
    $("#kpis").innerHTML = '<p class="empty">Historical stats could not be loaded.</p>';
  }

  function readUrlRoute() {
    const p = new URLSearchParams(location.search);
    const from = (p.get("from") || "").toUpperCase(), to = (p.get("to") || "").toUpperCase();
    return from && to ? { from, to } : null;
  }
  function showLanding() {
    state.requestId++;                                      // cancel in-flight renders
    setView("landing");
    document.title = "FLIGHTWATCH · Know before you board.";
    Radar.setRoute(null, null);
  }

  async function loadAirports() {
    $("#heroMeta").textContent = "Connecting to the FLIGHTWATCH API…";
    const slow = setTimeout(() => { $("#heroMeta").textContent = "Waking the server (free hosting can take ~30 s)…"; }, 6000);
    try {
      const list = await api("/api/airports");
      state.airports = list; state.byCode = new Map(list.map((a) => [a.code, a]));
      $("#heroError").hidden = true;
      setLive("live");
      Radar.refresh();
      return true;
    } catch (err) {
      setLive("offline");
      const box = $("#heroError");
      box.innerHTML = errorBanner(err).replace('<button type="button" class="btn-ghost" data-act="home">New search</button>', "");
      box.hidden = false;
      $("#heroMeta").textContent = "";
      return false;
    } finally {
      clearTimeout(slow);
    }
  }

  const FACTOR_NAMES = { weather_origin: "origin weather", weather_destination: "destination weather", distance: "distance", hour: "time of day", destination: "arrival airport", airline: "airline", origin: "departure airport", season: "season", day: "weekday", holiday: "holidays" };
  async function loadModelInfo() {
    try {
      const m = await api("/api/model-info", {}, { timeout: 30000 });
      state.modelInfo = m;
      const x = m.metrics;
      $("#heroMeta").innerHTML = `<b>${state.airports.length}</b> airports<span class="sep">/</span>RandomForest<span class="sep">/</span>ROC-AUC <b>${x.roc_auc.toFixed(2)}</b><span class="sep">/</span>trained on <b>${x.rows.toLocaleString()}</b> flights`;
      const top = Object.entries(m.feature_importance).slice(0, 3).map(([k, v]) => `${FACTOR_NAMES[k] || k} ${Math.round(v * 100)}%`).join(" · ");
      $("#footModel").textContent = `Model v${m.version} · accuracy ${(x.accuracy * 100).toFixed(1)}% · ROC-AUC ${x.roc_auc.toFixed(3)} · MAE ${x.mae_minutes} min · top drivers: ${top}`;
    } catch { $("#heroMeta").textContent = ""; }
  }

  function bindGlobalHandlers() {
    $("#brandLink").addEventListener("click", (e) => {
      if (e.metaKey || e.ctrlKey) return;
      e.preventDefault(); history.pushState({}, "", location.pathname); showLanding();
    });
    $("#quickChips").addEventListener("click", chipClick);
    $("#recents").addEventListener("click", chipClick);
    function chipClick(e) {
      const c = e.target.closest(".chip"); if (!c) return;
      if (!state.airports.length) { toast("Airport list not loaded yet"); return; }
      navigate(c.dataset.from, c.dataset.to);
    }
    $("#quickChips").addEventListener("pointerover", (e) => { const c = e.target.closest(".chip"); if (c) Radar.setRoute(c.dataset.from, c.dataset.to); });
    $("#quickChips").addEventListener("pointerleave", () => Radar.setRoute(searches.hero.from.resolve(), searches.hero.to.resolve()));

    $("#airlinePills").addEventListener("click", (e) => {
      const b = e.target.closest(".air-pill"); if (!b || !state.data) return;
      const code = b.dataset.code;
      state.airlineFilter = state.airlineFilter === code ? null : code;
      $$("#airlinePills .air-pill").forEach((p) => p.setAttribute("aria-pressed", String(p.dataset.code === state.airlineFilter)));
      renderFlightList();
      if (state.visible.length) selectFlight(0); else { setFactors({ risk_factors: [], risk_level: "low" }); }
    });
    $("#compareBtn").addEventListener("click", () => {
      const p = $("#airlinePanel"); p.scrollIntoView({ behavior: prefersReduced ? "auto" : "smooth", block: "center" });
      p.classList.remove("flash"); void p.offsetWidth; p.classList.add("flash");
    });
    $("#shareBtn").addEventListener("click", async () => {
      const url = `${location.origin}${location.pathname}?from=${state.from}&to=${state.to}`;
      try { await navigator.clipboard.writeText(url); toast("Route link copied"); }
      catch {
        const ta = document.createElement("textarea"); ta.value = url; ta.style.position = "fixed"; ta.style.opacity = "0"; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); toast("Route link copied"); } catch { toast(url); } ta.remove();
      }
    });
    $("#resultsError").addEventListener("click", (e) => {
      const act = e.target.closest("[data-act]")?.dataset.act;
      if (act === "retry") runAnalysis(state.from, state.to);
      if (act === "home") { history.pushState({}, "", location.pathname); showLanding(); }
    });
    $("#heroError").addEventListener("click", async (e) => {
      if (e.target.closest('[data-act="retry"]')) { await loadAirports() && (loadModelInfo(), handleUrl()); }
    });
    window.addEventListener("popstate", handleUrl);
  }

  function handleUrl() {
    const r = readUrlRoute();
    if (!r) { showLanding(); return; }
    if (!state.byCode.has(r.from) || !state.byCode.has(r.to) || r.from === r.to) {
      showLanding();
      const box = $("#heroError");
      box.className = "banner banner--soft";
      box.innerHTML = `<strong>ROUTE NOT FOUND</strong>${esc(r.from)} → ${esc(r.to)} isn't a valid pair in the FLIGHTWATCH directory. Pick two different airports below.`;
      box.hidden = false;
      return;
    }
    runAnalysis(r.from, r.to);
  }

  function startClock() {
    const tick = () => { $("#clock").textContent = "UTC " + new Date().toISOString().slice(11, 19); };
    tick(); setInterval(tick, 1000);
  }

  async function boot() {
    $$(".loader-text").forEach((n) => { n.dataset.orig = n.textContent; });
    setLive("idle");
    startClock(); initSound();
    buildSearch("heroSearch", "hero", true);
    buildSearch("headerSearch", "hdr", false);
    buildGaugeStatic(); bindFlightList(); bindTabs(); bindGlobalHandlers();
    renderQuickChips(); renderRecents();
    Radar.init(); Radar.start();
    if (await loadAirports()) { loadModelInfo(); handleUrl(); }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
