"use strict";

const HST = "Pacific/Honolulu";
// custom = { t0, t1 } (unix s) while the date pickers define the window; otherwise the last `days`
const state = { days: 7, custom: null, episodes: [], data: {} };

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (_) { /* storage unavailable */ } },
};
async function api(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}
async function safe(fn, label) { try { return await fn(); } catch (e) { console.warn(label, e); return null; } }

function fmtHST(unix, withYear = false) {
  if (!unix) return "—";
  return new Date(unix * 1000).toLocaleString("en-US", {
    timeZone: HST, month: "short", day: "numeric", hour: "numeric", minute: "2-digit", ...(withYear ? { year: "numeric" } : {}),
  }) + " HST";
}
function rel(unix) {
  if (!unix) return "never";
  const s = Date.now() / 1000 - unix;
  if (s < 90) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${(s / 3600).toFixed(s < 36000 ? 1 : 0)} h ago`;
  return `${(s / 86400).toFixed(1)} d ago`;
}
const dur = (h) => (h == null ? "—" : h < 48 ? `${h.toFixed(1)} h` : `${(h / 24).toFixed(1)} d`);
const pct = (p) => (p == null ? "—" : p < 0.01 ? "<1%" : p > 0.99 ? ">99%" : `${Math.round(p * 100)}%`);
const num = (x, d = 1) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
const hst = (t) => new Date((t - 36000) * 1000).toISOString().slice(0, 19).replace("T", " ");
const hstDate = (t) => hst(t).slice(0, 10);
const hstMidnight = (ymd) => Date.parse(`${ymd}T00:00:00Z`) / 1000 + 36000;

/** Break the line where samples are further apart than maxGap (or 3× typical spacing if thinned). */
function withGaps(ts, ys, maxGap) {
  if (ts.length > 2) {
    const d = [];
    for (let i = 1; i < ts.length; i++) d.push(ts[i] - ts[i - 1]);
    d.sort((a, b) => a - b);
    maxGap = Math.max(maxGap, 3 * d[Math.floor(d.length / 2)]);
  }
  const x = [], y = [];
  for (let i = 0; i < ts.length; i++) {
    if (i > 0 && ts[i] - ts[i - 1] > maxGap) { x.push(hst(ts[i - 1] + 1)); y.push(null); }
    x.push(hst(ts[i])); y.push(ys[i]);
  }
  return { x, y };
}

const SERIES_START_FALLBACK = 1734949200;
function windowRange() {
  const now = Date.now() / 1000;
  if (state.custom) return [state.custom.t0, Math.min(state.custom.t1, now)];
  if (state.days !== "all") return [now - state.days * 86400, now];
  const start = state.episodes.length ? Math.min(...state.episodes.map((e) => e.start_t)) : SERIES_START_FALLBACK;
  return [start - 86400, now];
}
function spanDays() { const [t0, t1] = windowRange(); return (t1 - t0) / 86400; }
/** Query string selecting the window: explicit bounds for a custom range, else "the last N". */
function rangeQuery(unit = "hours") {
  if (state.custom) { const [t0, t1] = windowRange(); return `start=${Math.floor(t0)}&end=${Math.ceil(t1)}`; }
  return unit === "days" ? `days=${Math.ceil(spanDays())}` : `hours=${Math.ceil(spanDays() * 24)}`;
}

// ---------- chart chrome ----------
function axisStyle() {
  return { gridcolor: css("--grid"), linecolor: css("--axis"), tickcolor: css("--axis"), zeroline: false, tickfont: { color: css("--muted"), size: 11 } };
}
function baseLayout(extra = {}) {
  const [t0, t1] = windowRange();
  const ax = axisStyle();
  const lay = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: css("--ink-2") },
    margin: { l: 52, r: 18, t: 14, b: 34 }, hovermode: "x unified",
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--hair"), font: { color: css("--ink"), size: 12 } },
    showlegend: false,
  };
  for (const [k, v] of Object.entries(extra)) if (!["xaxis", "yaxis", "shapes", "annotations"].includes(k)) lay[k] = v;
  lay.xaxis = { ...ax, type: "date", range: [hst(t0), hst(t1)], ...(extra.xaxis || {}) };
  lay.yaxis = { ...ax, ...(extra.yaxis || {}) };
  lay.shapes = [...(extra.shapes || [])];
  lay.annotations = [...(extra.annotations || [])];
  addEpisodeOverlay(lay);
  return lay;
}
const config = { displayModeBar: false, responsive: true };

function addEpisodeOverlay(lay) {
  const [t0, t1] = windowRange();
  const eps = state.episodes.filter((e) => (e.end_t || t1) >= t0 && e.start_t <= t1);
  const minW = (t1 - t0) * (spanDays() > 365 ? 0.0015 : 0.004);
  for (const e of eps) {
    lay.shapes.push({ type: "rect", xref: "x", yref: "paper", x0: hst(e.start_t), x1: hst(Math.max(e.end_t || t1, e.start_t + minW)),
      y0: 0, y1: 1, fillcolor: css(e.kind === "fountaining" ? "--episode" : "--episode-alt"), line: { width: 0 }, layer: "below" });
    if (eps.length <= 14) lay.annotations.push({ xref: "x", yref: "paper", x: hst(e.start_t), y: 1, yanchor: "bottom", xanchor: "left",
      showarrow: false, text: e.kind === "fountaining" ? `E${e.label}` : "vents", font: { size: 10, color: css("--muted") } });
  }
  lay.margin = { ...lay.margin, t: Math.max(lay.margin.t, 22) };
}

// ---------- official ----------
const CODE_COLORS = { GREEN: "#0ca30c", YELLOW: "#fab219", ORANGE: "#f7941d", RED: "#d03b3b" };
const LEVEL_COLORS = { NORMAL: "#0ca30c", ADVISORY: "#fab219", WATCH: "#f7941d", WARNING: "#d03b3b" };

function renderStatus(d) {
  const s = d.status;
  if (!s) { $("alert-level").textContent = "unavailable"; return; }
  $("alert-level").textContent = s.alert_level || "—";
  $("alert-level").parentElement.style.setProperty("--sw", LEVEL_COLORS[s.alert_level] || css("--axis"));
  $("color-code").textContent = s.color_code || "—";
  $("color-code").parentElement.style.setProperty("--sw", CODE_COLORS[s.color_code] || css("--axis"));
  const since = s.alert_date_utc ? Date.parse(s.alert_date_utc.replace(" ", "T") + "Z") / 1000 : null;
  $("status-date").textContent = `Status notice ${since ? fmtHST(since, true) : "—"} · checked ${rel(s.fetched_at)}` + (d.freshness.stale ? " · ⚠ stale" : "");
  $("status-synopsis").textContent = (s.synopsis || "").replace(/^HVO Kilauea [A-Z]+\/[A-Z]+ - /, "");
}

function renderNotices(d) {
  const n = d.notices || [];
  if (!n.length) return;
  const a = n[0];
  state.data.latestNoticeT = a.sent_unix;
  $("notice-title").textContent = a.title;
  $("notice-time").textContent = `${fmtHST(a.sent_unix, true)} · ${rel(a.sent_unix)}`;
  $("notice-synopsis").textContent = a.synopsis || "";
  $("notice-text").textContent = a.text || "";
  $("notice-link").href = a.url;
  $("notice-list").innerHTML = n.slice(1).map((x) =>
    `<li><a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.type_title)}</a> <span class="muted">· ${esc(fmtHST(x.sent_unix))}</span><br><span class="muted">${esc(x.synopsis)}</span></li>`
  ).join("") + `<li><a href="${esc(d.hans_url)}" target="_blank" rel="noopener">Search all notices in HANS ↗</a></li>`;
}

const humanTimes = (s) => String(s ?? "").replace(/\b1[6-9]\d{8}\b/g, (t) => fmtHST(Number(t)));
function renderMessages(d) {
  const m = d?.messages?.[0];
  if (!m) { $("message-title").textContent = "No messages fetched yet"; return; }
  const isNew = m.t > (state.data.latestNoticeT || 0);  // posted after the latest Daily Update / notice
  $("message-title").textContent = m.kind === "Kilauea Message" ? "Kīlauea Message" : "HVO Message";
  $("message-time").innerHTML = `${esc(fmtHST(m.t, true))} · ${esc(rel(m.t))}` + (isNew ? ` · <span class="new">newer than the notice</span>` : "");
  $("message-text").innerHTML = esc(m.text) + (m.truncated ? ` <a href="${esc(d.url)}" target="_blank" rel="noopener">read more ↗</a>` : "");
}

function healthClass(h) { return h.error ? "err" : !h.last_success ? "never" : h.stale ? "stale" : ""; }
function renderHealth(d) {
  $("health").innerHTML = d.groups.map((g) => `
    <div class="health-card"><h4>${esc(g.name)}</h4>
      ${g.sources.map((s) => `<div class="health-row ${healthClass(s)}" title="${esc(s.error || s.detail || "")}">
        <span class="dot"></span><span>${esc(s.label)}</span><span class="age">${esc(s.last_success ? rel(s.last_success) : "not used yet")}${s.stale ? " · stale" : ""}</span>
        <span class="det">${esc(humanTimes(s.error ? "error: " + s.error : s.detail || ""))}</span></div>`).join("")}
    </div>`).join("");
}

// ---------- model cards ----------
function fmtFeature(f, v) {
  if (v == null) return "missing";
  switch (f) {
    case "hours_since_end": return dur(v);
    case "recovery_ratio": return `${v.toFixed(2)}×`;
    case "tilt_rate_24h": return `${v >= 0 ? "+" : ""}${v.toFixed(3)} µrad/h`;
    case "rsam_ratio_log": return `${Math.pow(10, v).toFixed(2)}×`;
    case "precursor": return v > 0 ? `yes · weight ${v.toFixed(2)}` : "no";
    case "eq_summit_ew": return v.toFixed(1);
    default: return String(Math.round(v * 100) / 100);
  }
}

function renderModelCard(m) {
  const card = $("card-hazard");
  const tiles = card.querySelector("[data-tiles]");
  if (!m || m.p12 == null) {
    tiles.innerHTML = `<div class="tile muted">No valid prediction yet${m?.missing_now?.length ? ": missing " + esc(m.missing_now.join(", ")) : ""}</div>`;
    card.querySelector("[data-trend]").textContent = "";
    card.querySelector("[data-lag]").textContent = "";
    card.querySelector("[data-factors]").innerHTML = "";
    return;
  }
  tiles.innerHTML = `<div class="tile"><div class="h">next 12 h</div><div class="v">${pct(m.p12)}</div></div>`;
  const tr = m.trend || {};
  const arrow = { up: "▲ rising", down: "▼ falling", flat: "▬ steady" }[tr.arrow || "flat"];
  const d24 = tr.delta_24h != null ? ` · ${tr.delta_24h >= 0 ? "+" : ""}${Math.round(tr.delta_24h * 100)} pts / 24 h` : "";
  card.querySelector("[data-trend]").textContent = `${arrow}${d24}`;
  card.querySelector("[data-lag]").textContent = `As of ${fmtHST(m.t)}` + (m.missing_now?.length
    ? ` · newer slots wait for ${m.missing_now.join(", ")} (input latency)` : "");
  card.querySelector("[data-factors]").innerHTML = (m.top_factors || []).map((f) => `
    <li><span>${esc(f.label)} <span class="val">· ${esc(fmtFeature(f.feature, f.value))}</span></span>
    <span class="eff ${f.effect > 0 ? "up" : "down"}">${f.effect > 0 ? "raises" : "lowers"} ${Math.abs(f.effect).toFixed(2)}</span></li>`).join("");
}

function renderProbability(d) {
  const hz = d.model;
  $("disclaimer").textContent = d.disclaimer;
  renderModelCard(hz);
  $("episode-banner").classList.toggle("hidden", !hz?.in_episode);
  if (hz?.in_episode) $("episode-banner").innerHTML = `<b>An episode appears to be in progress.</b> ${esc(hz.in_episode_reason || "")}. Onset probabilities don't apply while an episode is under way.`;
  const warns = hz?.warnings || [];
  $("prob-warnings").classList.toggle("hidden", !warns.length);
  $("prob-warnings").innerHTML = `<div class="eyebrow">Read with caution: conditions are outside the training data</div><ul>` +
    warns.map((w) => { const i = w.search(/[.:]\s/); return `<li><b>${esc(i > 0 ? w.slice(0, i + 1) : w)}</b> ${esc(i > 0 ? w.slice(i + 2) : "")}</li>`; }).join("") + "</ul>";
  const tr = d.training || {}, m = tr.metrics;
  if (m?.closed) {
    const row = (label, x = {}) => `<tr><td>${label}</td><td class="num">${num(x.brier_skill, 2)}</td><td class="num">${num(x.auc, 2)}</td>
      <td class="num">${x.n ?? "—"}</td><td class="num">${x.positives ?? "—"}</td><td class="num">${x.base_rate != null ? (x.base_rate * 100).toFixed(1) + "%" : "—"}</td></tr>`;
    $("skill").innerHTML = `
      <table class="skill"><thead><tr><th>12 h probability, ${esc(tr.version)}</th><th class="num">BSS</th><th class="num">AUC</th>
        <th class="num">Hours scored</th><th class="num">…with an onset within 12 h</th><th class="num">Base rate</th></tr></thead>
        <tbody>${row("Completed cycles", m.closed)}${row("Including the current pause", m.all)}</tbody></table>
      <p class="muted small" style="margin-top:10px">Every scored probability is out-of-sample. The model is refitted at the start of each month
      (Hawaii time) on the data available by then, and that fit only predicts the month that follows. The first fit used December 2024 – June 2025,
      so predictions start ${fmtHST(tr.first_prediction, true)}; ${tr.refits?.length ?? "—"} monthly fits so far. Brier skill score (BSS) &gt; 0 means
      better than always predicting the base rate. Latest fit: data through ${fmtHST(tr.fit_through, true)}, ${tr.n_rows} hourly rows with complete
      inputs (${tr.n_onsets} onsets, ${tr.cycles} cycles) since episode 4. Inputs: ${esc((tr.required || []).join(", "))}.</p>`;
  }
}

const BUILT_FROM = {
  hours_since_end: "episode catalog", recovery_ratio: "stitched tilt + catalog", tilt_rate_24h: "stitched tilt",
  rsam_ratio_log: "stitched tremor", eq_summit_ew: "ComCat", precursor: "HVO notices (keywords)",
};
function renderInputs(fr) {
  if (!fr?.t?.length) return;
  const i = fr.t.length - 1;
  $("inputs-note").textContent = `Latest row of the hourly feature frame (${fmtHST(fr.t[i])}), built from the hour that ended then. The model outputs None when any input is missing.`;
  $("inputs").querySelector("tbody").innerHTML = (fr.required || []).map((f) => {
    const v = fr.columns[f]?.[i];
    return `<tr><td>${esc(fr.labels[f] || f)}</td><td class="num ${v == null ? "miss" : ""}">${esc(fmtFeature(f, v))}</td>
      <td class="${v == null ? "miss" : "req"}">${v == null ? "missing" : "present"}</td><td class="muted">${esc(BUILT_FROM[f] || "")}</td></tr>`;
  }).join("");
}

function renderKPIs(prob, tilt, eq24) {
  const f = prob?.model?.features || {};
  const inf = tilt?.inflation || {};
  const items = [
    ["Since last episode", dur(f.hours_since_end), prob?.model?.last_episode ? `episode ${prob.model.last_episode} ended` : ""],
    ["Inflation since", `${num(f.inflation_urad)} µrad`, f.recovery_ratio != null ? `${num(f.recovery_ratio, 2)}× last deflation (${num(f.last_deflation_urad)} µrad)` : ""],
    ["Tilt vs last onset", `${f.gap_to_onset_urad >= 0 ? "+" : ""}${num(f.gap_to_onset_urad)} µrad`, `episode ${inf.episode || "?"} onset level`],
    ["Tilt rate, 24 h", num(f.tilt_rate_24h, 3), `µrad/h · 6 h: ${num(f.tilt_rate_6h, 3)}`],
    ["Tremor, 1 h", f.rsam_log != null ? `${Math.pow(10, f.rsam_log).toFixed(2)} µm/s` : "—", f.rsam_ratio_log != null ? `${Math.pow(10, f.rsam_ratio_log).toFixed(2)}× 24 h median` : ""],
    ["Earthquakes, 24 h", `${eq24?.counts?.last_24h ?? "—"}`, `${eq24?.counts?.summit_last_24h ?? "—"} at the summit`],
    ["Precursory activity", f.precursor ? "Reported" : "Not reported", "latest HVO update"],
  ];
  $("kpis").innerHTML = items.map(([k, v, s]) => `<div class="kpi"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join("");
}

// ---------- charts ----------
function renderProbChart(p) {
  const traces = [], ann = [], shapes = [];
  const [t0, t1] = windowRange();
  if (p.t.length) {
    const g = withGaps(p.t, p.p12.map((v) => (v == null ? null : v * 100)), 5400);
    traces.push({ x: g.x, y: g.y, mode: "lines", name: p.version, line: { color: css("--s1"), width: 1.8 }, connectgaps: false, hovertemplate: "P(onset within 12 h): %{y:.1f}%<extra></extra>" });
    const li = p.p12.findLastIndex((v) => v != null);
    if (li >= 0) ann.push({ x: hst(p.t[li]), y: p.p12[li] * 100, xanchor: "left", xshift: 6, showarrow: false, text: `${Math.round(p.p12[li] * 100)}%`, font: { size: 11, color: css("--ink-2") } });
  } else if (p.first_prediction && t1 <= p.first_prediction) {
    ann.push({ xref: "paper", yref: "paper", x: 0.5, y: 0.5, showarrow: false, text: `No predictions before ${fmtHST(p.first_prediction, true)}: the first model needed December 2024 – June 2025 to fit`, font: { size: 12, color: css("--muted") } });
  }
  for (const c of p.refits || []) shapes.push({ type: "line", xref: "x", yref: "paper", x0: hst(c), x1: hst(c), y0: 0, y1: 1, line: { color: css("--muted"), width: 1, dash: "dot" }, opacity: 0.45, layer: "below" });
  Plotly.react("chart-prob", traces, baseLayout({ margin: { l: 44, r: 44, t: 14, b: 34 }, shapes, annotations: ann, yaxis: { ticksuffix: "%", rangemode: "tozero" } }), config);
  $("prob-note").textContent = "P(onset within 12 h), one value per hour, all out-of-sample: each month comes from a model fitted only on data before that month began"
    + ((p.refits || []).length ? " (dotted lines = monthly refits)" : "") + ". Gaps = None (episode under way or an input missing).";
}

const TILT_COLORS = { uwd_release: "--s1", uwd_plot2d: "--s2", uwd_plot3m: "--s3", sdh_release: "--s4" };
const TREMOR_COLORS = { uwe: "--s1", uwe_qc: "--s2", obl: "--s3", uwb: "--s4", rimd: "--s5" };

function sourceTraces(d, colors, unit, digits) {
  // raw sources: thin and faint underneath (visible where several overlap)
  const traces = [];
  for (const s of d.sources) {
    if (!s.t.length) continue;
    const g = withGaps(s.t, s.v, 5400);
    traces.push({ x: g.x, y: g.y, mode: "lines", name: s.label, line: { color: css(colors[s.key]), width: 1 }, opacity: 0.35,
      connectgaps: false, hoverinfo: "skip" });
  }
  // stitched series on top, coloured by the source that supplied each slot
  const st = d.stitched;
  const keys = [...new Set(st.src)].filter(Boolean);
  for (const k of keys) {
    const ys = st.v.map((v, i) => (st.src[i] === k || st.src[i + 1] === k ? v : null)); // overlap 1 point so segments join
    const g = withGaps(st.t, ys, 5400);
    const label = d.sources.find((s) => s.key === k)?.label || k;
    traces.push({ x: g.x, y: g.y, mode: "lines", name: `stitched · ${label}`, line: { color: css(colors[k]), width: 2.2 },
      connectgaps: false, hovertemplate: `stitched (${label}): %{y:.${digits}f} ${unit}<extra></extra>` });
  }
  return traces;
}

function legendHtml(d, colors) {
  return d.sources.filter((s) => s.t.length).map((s) => `<span><i class="swatch" style="background:${css(colors[s.key])}"></i>${esc(s.label)}</span>`).join("") +
    `<span class="muted">thick = stitched (colour = supplying source), thin = each source's hourly median</span>`;
}

function sourcesTable(d, colors, kind) {
  const total = Object.values(d.slots_by_source || {}).reduce((a, b) => a + b, 0) + (d.missing_slots || 0);
  const rows = d.sources.map((s) => {
    const n = d.slots_by_source?.[s.key] || 0;
    let map = "";
    if (s.mapping?.offset != null && s.key !== "uwd_release") map = `offset ${s.mapping.offset >= 0 ? "+" : ""}${num(s.mapping.offset, 3)} µrad`;
    else if (s.mapping && "offset" in s.mapping && s.mapping.offset == null) map = "not aligned (no overlap)";
    if (s.mapping?.relevels?.length) map = "authoritative; re-levelling removed: " + s.mapping.relevels.map((r) => (r.offset == null ? "unresolved" : `${r.offset >= 0 ? "+" : ""}${num(r.offset, 2)} µrad`)).join(", ");
    if (s.mapping?.fits?.length) map = s.mapping.fits.map((f) => `${f.used ? "used" : "rejected"} R²=${f.r2}`).join(", ");
    if (s.mapping?.ratios?.length) map = s.mapping.ratios.map((r) => (r.used === false ? `rejected ×${Number(r.ratio).toPrecision(2)}` : `×${num(r.ratio, 3)}`)).join(", ");
    const h = s.health || {};
    return `<tr><td><span class="sw" style="background:${css(colors[s.key])}"></span>${esc(s.label)}</td>
      <td class="num">${total ? ((100 * n) / total).toFixed(n && n / total < 0.001 ? 3 : 1) + "%" : "—"}</td>
      <td>${esc(map || (s.key === "uwd_release" || s.key === "uwe" ? "authoritative" : ""))}</td>
      <td class="muted">${esc(h.last_success ? "fetched " + rel(h.last_success) : "not fetched")}</td>
      <td class="muted">${esc(s.note)}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>Source (priority order)</th><th class="num">Share of grid</th><th>Mapping</th><th>Fetched</th><th>Notes</th></tr></thead><tbody>${rows}
    <tr><td class="muted">no source (genuine gap)</td><td class="num">${total ? ((100 * (d.missing_slots || 0)) / total).toFixed(2) + "%" : "—"}</td><td colspan="3" class="muted">the model outputs None for slots that need this ${kind}</td></tr></tbody></table>`;
}

function renderTilt(d) {
  Plotly.react("chart-tilt", sourceTraces(d, TILT_COLORS, "µrad", 2), baseLayout({ yaxis: { title: { text: "µrad", font: { size: 11 } } } }), config);
  $("tilt-legend").innerHTML = legendHtml(d, TILT_COLORS);
  $("tilt-sources").innerHTML = sourcesTable(d, TILT_COLORS, "tilt");
  const r = withGaps(d.rate.t, d.rate.v, 5400);
  Plotly.react("chart-rate", [{ x: r.x, y: r.y, mode: "lines", line: { color: css("--ink"), width: 1.3 }, connectgaps: false, hovertemplate: "%{y:.3f} µrad/h<extra></extra>" }],
    baseLayout({ margin: { l: 52, r: 18, t: 8, b: 30 }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: css("--axis"), width: 1 } }] }), config);
  $("tilt-note").textContent = "Share of grid counts every hourly slot since Dec 1, 2024. Plot sources are aligned by their median offset to the series built from higher-priority sources; SDH fills only gaps where its fit to UWD is tight.";
}

function renderTremor(d) {
  Plotly.react("chart-tremor", sourceTraces(d, TREMOR_COLORS, "µm/s", 3), baseLayout({ yaxis: { type: "log", title: { text: "µm/s (log)", font: { size: 11 } }, exponentformat: "none", nticks: 6 } }), config);
  $("tremor-legend").innerHTML = legendHtml(d, TREMOR_COLORS);
  $("tremor-sources").innerHTML = sourcesTable(d, TREMOR_COLORS, "tremor");
}

const REGIONS = [["summit", "Summit", "--s1"], ["upper_erz", "Upper East Rift", "--s2"], ["swrz", "Southwest Rift", "--s3"], ["other", "Other", "--s4"]];
function renderEq(d) {
  const days = spanDays(), hourly = days <= 1, bin = hourly ? 3600 : days > 120 ? 7 * 86400 : 86400;
  const [t0, t1] = windowRange();
  const start = Math.floor((t0 - 36000) / bin) * bin + 36000, edges = [];
  for (let t = start; t <= t1; t += bin) edges.push(t);
  const counts = Object.fromEntries(REGIONS.map(([k]) => [k, new Array(edges.length).fill(0)]));
  for (const e of d.events || []) { const i = Math.floor((e.t - start) / bin); if (i >= 0 && i < edges.length) (counts[e.region] || counts.other)[i]++; }
  const traces = REGIONS.map(([k, name, c]) => ({ type: "bar", x: edges.map((t) => hst(t + bin / 2)), y: counts[k], name, width: bin * 1000 * 0.8,
    marker: { color: css(c), line: { color: css("--surface"), width: 1 } }, hovertemplate: `${name}: %{y}<extra></extra>` }));
  Plotly.react("chart-eq", traces, baseLayout({ barmode: "stack", margin: { l: 40, r: 12, t: 14, b: 34 } }), config);
  $("eq-bin-note").textContent = `events per ${hourly ? "hour" : bin > 86400 ? "week" : "day"}, by region · USGS ComCat`;
  const c = d.counts || {};
  $("eq-summary").textContent = `${c.total ?? 0} events. By depth: ` + Object.entries(c.by_depth || {}).map(([k, v]) => `${k} ${v}`).join(", ") + (c.max_mag != null ? `. Largest M${num(c.max_mag, 1)}.` : ".");
}

function renderInflation(d) {
  const inf = d?.inflation;
  if (!inf) { $("chart-infl").innerHTML = `<div class="muted small">No completed episode to measure from.</div>`; return; }
  const days = inf.t.map((t) => (t - inf.since) / 86400);
  const shapes = [], ann = [], rt = inf.ratio_thresholds_urad;
  if (rt) {
    shapes.push({ type: "rect", xref: "paper", x0: 0, x1: 1, y0: rt.p10, y1: rt.p90, fillcolor: css("--band"), line: { width: 0 }, layer: "below" });
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: rt.median, y1: rt.median, line: { color: css("--s1"), width: 1 } });
    ann.push({ xref: "paper", x: 0.01, y: rt.p90, yanchor: "bottom", xanchor: "left", showarrow: false,
      text: `typical onset: ${num(inf.thresholds.recovery_ratio.p10, 2)}–${num(inf.thresholds.recovery_ratio.p90, 2)}× last deflation`, font: { size: 11, color: css("--ink-2") } });
  }
  const ax = axisStyle();
  Plotly.react("chart-infl", [{ x: days, y: inf.v, mode: "lines", line: { color: css("--ink"), width: 1.8 }, hovertemplate: "day %{x:.1f}: %{y:.2f} µrad<extra></extra>" }], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false, hovermode: "closest",
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: css("--ink-2") },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--hair"), font: { color: css("--ink") } },
    margin: { l: 52, r: 18, t: 14, b: 40 }, shapes, annotations: ann,
    xaxis: { ...ax, title: { text: `days since episode ${inf.episode} ended`, font: { size: 11 } } },
    yaxis: { ...ax, title: { text: "µrad regained (stitched tilt)", font: { size: 11 } } } }, config);
  const thr = inf.thresholds;
  $("infl-note").textContent = rt ? `Band: 10th–90th percentile of tilt regained at onset over the last ${thr.n} episodes, scaled to episode ${inf.episode}'s ${num(inf.last_deflation_urad)} µrad deflation. Unscaled: ${num(thr.recovery_urad.p10)}–${num(thr.recovery_urad.p90)} µrad (median ${num(thr.recovery_urad.median)}).` : "";
}

function renderWeather(w, firms) {
  const cur = w?.weather?.current;
  if (cur) {
    const vis = cur.visibility != null ? (cur.visibility >= 10000 ? ">10 km" : `${(cur.visibility / 1000).toFixed(1)} km`) : "—";
    const items = [["Temperature", `${num(cur.temperature_2m)} °C`, `RH ${cur.relative_humidity_2m ?? "—"}%`],
      ["Visibility", vis, `cloud ${cur.cloud_cover ?? "—"}%, low ${cur.cloud_cover_low ?? "—"}%`],
      ["Wind", `${num(cur.wind_speed_10m, 0)} km/h`, `from ${cur.wind_direction_10m ?? "—"}°`],
      ["Precipitation", `${num(cur.precipitation)} mm`, `updated ${rel(w.weather.fetched_at)}`]];
    $("weather").innerHTML = items.map(([k, v, s]) => `<div class="kpi"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join("");
  } else $("weather").innerHTML = `<div class="kpi muted">Weather unavailable</div>`;
  const det = firms?.detections || [];
  const byDay = {};
  det.forEach((x) => { const k = new Date(x.t * 1000).toLocaleDateString("en-US", { timeZone: HST, month: "short", day: "numeric" }); (byDay[k] = byDay[k] || []).push(x); });
  const last = det.at(-1);
  $("firms").innerHTML = `<p>${det.length} hotspot detections over the caldera in the last 7 days${last ? `; latest ${esc(fmtHST(last.t))}, FRP ${num(last.frp)} MW` : ""}.</p>
    <p class="muted" style="margin-top:4px">${Object.entries(byDay).map(([k, v]) => `${esc(k)} ${v.length}`).join(" · ")}</p>
    <p class="note">Vents stay hot between episodes; a jump in detections or radiative power (FRP) is what confirms fountaining.</p>`;
}

function renderEpisodes(d) {
  $("episodes").querySelector("tbody").innerHTML = [...d.episodes].reverse().map((e) => `
    <tr class="${e.kind !== "fountaining" ? "nonfount" : ""}"><td>${esc(e.kind === "fountaining" ? e.label : "—")}</td>
      <td>${esc(fmtHST(e.start_t, true))}</td><td class="num">${esc(dur(e.duration_h))}</td><td class="num">${esc(dur(e.repose_before_h))}</td>
      <td class="num">${esc(dur(e.onset_interval_h))}</td><td class="num">${e.deflation_urad != null ? num(e.deflation_urad) + " µrad" : "—"}</td>
      <td class="num">${e.onset_recovery_ratio != null ? num(e.onset_recovery_ratio, 2) + "×" : "—"}</td>
      <td class="num">${e.fountain_height_m != null ? Math.round(e.fountain_height_m) + " m" : "—"}</td><td class="notes">${esc(e.notes)}</td></tr>`).join("");
  const sg = d.suggestions || [];
  $("suggestions").innerHTML = sg.length ? `<div class="eyebrow">Suggested from HVO notices, pending review</div>` +
    sg.slice(0, 3).map((s) => `<div class="sugg"><b>Episode ${esc(s.episode_num)} · ${esc(s.phase)}</b> <span class="muted">· ${esc(fmtHST(s.sent_unix))}</span><br><span class="muted">${esc(s.snippet)}</span></div>`).join("") : "";
}

// ---------- loading ----------
let seriesLoad = 0;
async function loadSeries() {
  const load = ++seriesLoad;
  if (!state.custom) syncDates();
  const q = rangeQuery();
  const [tilt, tremor, eq, preds] = await Promise.all([
    safe(() => api(`/api/series/tilt?${q}`), "tilt"), safe(() => api(`/api/series/tremor?${q}`), "tremor"),
    safe(() => api(`/api/earthquakes?${rangeQuery("days")}`), "eq"), safe(() => api(`/api/predictions?${q}`), "preds"),
  ]);
  if (load !== seriesLoad) return { tilt: state.data.tilt };  // the range changed while this was loading
  Object.assign(state.data, { tilt, tremor, eq, preds });
  renderSeries();
  return { tilt };
}
function renderSeries() {
  const d = state.data;
  if (d.preds) renderProbChart(d.preds);
  if (d.tilt) { renderTilt(d.tilt); renderInflation(d.tilt); }
  if (d.tremor) renderTremor(d.tremor);
  if (d.eq) renderEq(d.eq);
}

async function refresh() {
  const [status, notices, health, prob, eps, weather, firms, eq24, fr, msgs] = await Promise.all([
    safe(() => api("/api/status"), "status"), safe(() => api("/api/notices?limit=6"), "notices"), safe(() => api("/api/health"), "health"),
    safe(() => api("/api/probability"), "prob"), safe(() => api("/api/episodes"), "episodes"), safe(() => api("/api/weather"), "weather"),
    safe(() => api("/api/firms?days=7"), "firms"), safe(() => api("/api/earthquakes?days=1"), "eq24"), safe(() => api("/api/frame?hours=3"), "frame"),
    safe(() => api("/api/messages?limit=1"), "messages"),
  ]);
  if (status) renderStatus(status);
  if (notices) renderNotices(notices);
  renderMessages(msgs);
  if (health) renderHealth(health);
  if (prob) { state.data.prob = prob; renderProbability(prob); }
  if (eps) { state.episodes = eps.episodes || []; renderEpisodes(eps); }
  if (fr) renderInputs(fr);
  renderWeather(weather, firms);
  const { tilt } = await loadSeries();
  renderKPIs(prob, tilt, eq24);
  $("refreshed").textContent = `Updated ${new Date().toLocaleTimeString("en-US", { timeZone: HST, hour: "numeric", minute: "2-digit" })} HST`;
}

// ---------- controls ----------
function segment(id, attr, onPick) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    onPick(b.dataset[attr]);
  });
}
function markOn(id, attr, value) { $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset[attr] === String(value))); }

const DATE_IDS = ["range-start", "range-end"];
/** Show the active window in the date pickers (Hawaii dates). */
function syncDates() {
  const [t0, t1] = windowRange();
  $("range-start").value = hstDate(t0);
  $("range-end").value = hstDate(t1);
  DATE_IDS.forEach((id) => { $(id).max = hstDate(Date.now() / 1000); $(id).classList.toggle("on", !!state.custom); });
}
/** Use the picked dates as the window: start of the first day to the end of the last, Hawaii time. */
function applyCustom() {
  const a = $("range-start").value, b = $("range-end").value;
  if (!a || !b || a > b) return false;
  state.custom = { t0: hstMidnight(a), t1: hstMidnight(b) + 86400 };
  store.set("custom", `${a}/${b}`);
  markOn("range", "d", "");
  DATE_IDS.forEach((id) => $(id).classList.add("on"));
  return true;
}

function initControls() {
  const savedDays = store.get("days");
  if (savedDays === "all") state.days = "all";
  else if ([1, 7, 30, 90].includes(Number(savedDays))) state.days = Number(savedDays);
  markOn("range", "d", state.days);
  const [a, b] = (store.get("custom") || "").split("/");
  if (a && b) { $("range-start").value = a; $("range-end").value = b; applyCustom(); }
  segment("range", "d", (v) => { state.custom = null; state.days = v === "all" ? "all" : Number(v); store.set("days", v); store.set("custom", ""); loadSeries(); });
  DATE_IDS.forEach((id) => $(id).addEventListener("change", () => { if (applyCustom()) loadSeries(); }));
}

function initTheme() {
  const saved = store.get("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("theme", next);
    renderSeries();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderSeries);
}

function initKofi() {
  if (!window.kofiwidget2) return;  // widget script blocked or offline
  kofiwidget2.init("Support me on Ko-fi", "#c20000", "A0R028KOSI");
  $("kofi").innerHTML = kofiwidget2.getHTML();
}

window.addEventListener("DOMContentLoaded", () => { initTheme(); initControls(); initKofi(); refresh(); setInterval(refresh, 120000); });
