"use strict";

const HST = "Pacific/Honolulu";
const state = { days: 7, horizon: "24", showEpisodes: false, episodes: [], data: {} };

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
function spanDays() {
  if (state.days !== "all") return state.days;
  const start = state.episodes.length ? Math.min(...state.episodes.map((e) => e.start_t)) : SERIES_START_FALLBACK;
  return (Date.now() / 1000 - start) / 86400 + 1;
}
function windowRange() { const now = Date.now() / 1000; return [now - spanDays() * 86400, now]; }

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
  if (state.showEpisodes) addEpisodeOverlay(lay);
  return lay;
}
const config = { displayModeBar: false, responsive: true };

function addEpisodeOverlay(lay) {
  const [t0, t1] = windowRange();
  const eps = state.episodes.filter((e) => (e.end_t || t1) >= t0 && e.start_t <= t1);
  const minW = (t1 - t0) * (state.days === "all" ? 0.0015 : 0.004);
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
  const msgs = d?.messages || [];
  if (!msgs.length) { $("messages").innerHTML = `<li class="muted small">No messages fetched yet.</li>`; return; }
  const latestNotice = state.data.latestNoticeT || 0;
  $("messages").innerHTML = msgs.map((m) => {
    const isNew = m.t > latestNotice;  // posted after the latest Daily Update / notice
    const when = new Date(m.t * 1000).toLocaleString("en-US", { timeZone: HST, month: "short", day: "numeric" });
    const time = new Date(m.t * 1000).toLocaleString("en-US", { timeZone: HST, hour: "numeric", minute: "2-digit" });
    const kind = m.kind === "Kilauea Message" ? "Kīlauea message" : "HVO message";
    return `<li class="${isNew ? "new" : ""}"><div class="when"><b>${esc(time)} HST</b>${esc(when)} · ${esc(rel(m.t))}<br><span class="kind">${esc(kind)}</span></div>
      <div class="text">${esc(m.text)}${m.truncated ? ` <a href="${esc(d.url)}" target="_blank" rel="noopener">read more ↗</a>` : ""}</div></li>`;
  }).join("");
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
    case "inflation_urad": case "gap_to_onset_urad": case "last_deflation_urad": return `${v.toFixed(2)} µrad`;
    case "tilt_rate_6h": case "tilt_rate_24h": return `${v >= 0 ? "+" : ""}${v.toFixed(3)} µrad/h`;
    case "rsam_log": return `${Math.pow(10, v).toFixed(3)} µm/s`;
    case "rsam_ratio_log": case "rsam_trend_log": return `${Math.pow(10, v).toFixed(2)}×`;
    case "precursor": return v ? "yes" : "no";
    default: return String(Math.round(v * 100) / 100);
  }
}

function renderModelCard(cardId, m, effectFmt) {
  const card = $(cardId);
  const tiles = card.querySelector("[data-tiles]");
  if (!m || !m.p) {
    tiles.innerHTML = `<div class="tile muted">No valid prediction yet${m?.missing_now?.length ? ": missing " + esc(m.missing_now.join(", ")) : ""}</div>`;
    card.querySelector("[data-trend]").textContent = "";
    card.querySelector("[data-lag]").textContent = "";
    card.querySelector("[data-factors]").innerHTML = "";
    return;
  }
  tiles.innerHTML = ["12", "24", "72"].map((H) => `<div class="tile"><div class="h">next ${H} h</div><div class="v">${pct(m.p[H])}</div></div>`).join("");
  const tr = m.trend || {};
  const arrow = { up: "▲ rising", down: "▼ falling", flat: "▬ steady" }[tr.arrow || "flat"];
  const d24 = tr.delta_24h != null ? ` · ${tr.delta_24h >= 0 ? "+" : ""}${Math.round(tr.delta_24h * 100)} pts / 24 h` : "";
  card.querySelector("[data-trend]").textContent = `${arrow}${d24}`;
  card.querySelector("[data-lag]").textContent = `As of ${fmtHST(m.t)}` + (m.missing_now?.length
    ? ` · newer slots wait for ${m.missing_now.join(", ")} (input latency)` : "");
  card.querySelector("[data-factors]").innerHTML = (m.top_factors || []).map((f) => `
    <li><span>${esc(f.label)} <span class="val">· ${esc(fmtFeature(f.feature, f.value))}</span></span>
    <span class="eff ${f.effect > 0 ? "up" : "down"}">${f.effect > 0 ? "raises" : "lowers"} ${effectFmt(Math.abs(f.effect))}</span></li>`).join("");
}

function renderProbability(d) {
  const hz = d.models.hazard, ml = d.models.ml;
  $("disclaimer").textContent = d.disclaimer;
  renderModelCard("card-hazard", hz, (x) => x.toFixed(2));
  renderModelCard("card-ml", ml, (x) => `${Math.round(x * 100)} pts`);
  $("episode-banner").classList.toggle("hidden", !hz?.in_episode);
  if (hz?.in_episode) $("episode-banner").innerHTML = `<b>An episode appears to be in progress.</b> ${esc(hz.in_episode_reason || "")}. Onset probabilities don't apply while an episode is under way.`;
  const warns = hz?.warnings || [];
  $("prob-warnings").classList.toggle("hidden", !warns.length);
  $("prob-warnings").innerHTML = `<div class="eyebrow">Read with caution: conditions are outside the training data</div><ul>` +
    warns.map((w) => { const i = w.search(/[.:]\s/); return `<li><b>${esc(i > 0 ? w.slice(0, i + 1) : w)}</b> ${esc(i > 0 ? w.slice(i + 2) : "")}</li>`; }).join("") + "</ul>";
  const tr = d.training || {}, m = tr.metrics;
  if (m && m.hazard?.closed) {
    const head = `<thead><tr><th></th><th>12 h BSS</th><th>AUC</th><th>24 h BSS</th><th>AUC</th><th>72 h BSS</th><th>AUC</th></tr></thead>`;
    const row = (k, scope) => `<tr><td>${esc(tr.versions?.[k] || k)}</td>` +
      ["12", "24", "72"].map((H) => { const x = m[k]?.[scope]?.[H] || {}; return `<td>${num(x.brier_skill, 2)}</td><td>${num(x.auc, 2)}</td>`; }).join("") + "</tr>";
    $("skill").innerHTML = `
      <div class="eyebrow" style="margin-top:12px">Completed cycles</div>
      <table class="skill">${head}<tbody>${row("hazard", "closed")}${row("ml", "closed")}</tbody></table>
      <div class="eyebrow" style="margin-top:14px">Including the current, unusually long pause</div>
      <table class="skill">${head}<tbody>${row("hazard", "all")}${row("ml", "all")}</tbody></table>
      <p class="muted small" style="margin-top:10px">Brier skill score (BSS) &gt; 0 means better than predicting the base rate. Trained on
      ${tr.n_rows} hourly frame rows with complete inputs (${tr.n_onsets} onsets, ${tr.cycles} cycles) since episode 4; trained ${rel(tr.trained_at)}.
      Required inputs: hazard-1.0 ${esc((tr.required?.hazard || []).join(", "))}; ml-1.0 all 13 features.</p>`;
  }
}

const BUILT_FROM = {
  hours_since_end: "episode catalog", recovery_ratio: "stitched tilt + catalog", inflation_urad: "stitched tilt + catalog",
  gap_to_onset_urad: "stitched tilt + catalog", last_deflation_urad: "stitched tilt + catalog",
  tilt_rate_6h: "stitched tilt", tilt_rate_24h: "stitched tilt", rsam_log: "stitched tremor", rsam_ratio_log: "stitched tremor",
  rsam_trend_log: "stitched tremor", eq_summit_24h: "ComCat", eq_all_24h: "ComCat", precursor: "HVO notices (keywords)",
};
function renderInputs(fr) {
  if (!fr?.t?.length) return;
  const i = fr.t.length - 1;
  const req = fr.required || {};
  $("inputs-note").textContent = `Latest row of the 5-minute feature frame (${fmtHST(fr.t[i])}). A model outputs None when any input it requires is missing.`;
  $("inputs").querySelector("tbody").innerHTML = Object.keys(BUILT_FROM).map((f) => {
    const v = fr.columns[f]?.[i];
    const cell = (m) => (req[m] || []).includes(f) ? `<td class="${v == null ? "miss" : "req"}">${v == null ? "missing" : "required"}</td>` : `<td class="muted">—</td>`;
    return `<tr><td>${esc(fr.labels[f] || f)}</td><td class="num ${v == null ? "miss" : ""}">${esc(fmtFeature(f, v))}</td>${cell("hazard")}${cell("ml")}<td class="muted">${esc(BUILT_FROM[f])}</td></tr>`;
  }).join("");
}

function renderKPIs(prob, tilt, eq24) {
  const f = prob?.models?.hazard?.features || {};
  const inf = tilt?.inflation || {};
  const items = [
    ["Since last episode", dur(f.hours_since_end), prob?.models?.hazard ? `episode ${prob.models.hazard.last_episode} ended` : ""],
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
  const key = `p${state.horizon}`, traces = [], ann = [], shapes = [];
  for (const [model, name, color] of [["hazard", "hazard-1.0", css("--s1")], ["ml", "ml-1.0", css("--s2")]]) {
    const d = p?.models?.[model];
    if (!d) continue;
    const g = withGaps(d.t, d[key].map((v) => (v == null ? null : v * 100)), 1800);
    traces.push({ x: g.x, y: g.y, mode: "lines", name, line: { color, width: 1.8 }, connectgaps: false, hovertemplate: `${name}: %{y:.1f}%<extra></extra>` });
    const li = d[key].map((v, i) => (v == null ? -1 : i)).filter((i) => i >= 0).at(-1);
    if (li != null) ann.push({ x: hst(d.t[li]), y: d[key][li] * 100, xanchor: "left", xshift: 6, showarrow: false, text: `${Math.round(d[key][li] * 100)}%`, font: { size: 11, color: css("--ink-2") } });
  }
  const [t0] = windowRange();
  if (p?.trained_at && p.trained_at > t0) shapes.push({ type: "line", xref: "x", yref: "paper", x0: hst(p.trained_at), x1: hst(p.trained_at), y0: 0, y1: 1, line: { color: css("--muted"), width: 1 }, opacity: 0.45, layer: "below" });
  Plotly.react("chart-prob", traces, baseLayout({ margin: { l: 44, r: 44, t: 14, b: 34 }, shapes, annotations: ann, yaxis: { ticksuffix: "%", rangemode: "tozero" } }), config);
  $("prob-note").textContent = `P(onset within ${state.horizon} h) on the 5-minute grid. Gaps = None (episode under way or an input missing). Left of the faint line (training ${p?.trained_at ? fmtHST(p.trained_at) : "—"}) is in-sample.`;
}

const TILT_COLORS = { uwd_release: "--s1", uwd_plot2d: "--s2", uwd_plot3m: "--s3", sdh_release: "--s4" };
const TREMOR_COLORS = { uwe: "--s1", uwe_qc: "--s2", obl: "--s3", uwb: "--s4", rimd: "--s5" };

function sourceTraces(d, colors, unit, digits) {
  // raw sources: thin and faint underneath (visible where several overlap)
  const traces = [];
  for (const s of d.sources) {
    if (!s.t.length) continue;
    const g = withGaps(s.t, s.v, 1800);
    traces.push({ x: g.x, y: g.y, mode: "lines", name: s.label, line: { color: css(colors[s.key]), width: 1 }, opacity: 0.35,
      connectgaps: false, hoverinfo: "skip" });
  }
  // stitched series on top, coloured by the source that supplied each slot
  const st = d.stitched;
  const keys = [...new Set(st.src)].filter(Boolean);
  for (const k of keys) {
    const ys = st.v.map((v, i) => (st.src[i] === k || st.src[i + 1] === k ? v : null)); // overlap 1 point so segments join
    const g = withGaps(st.t, ys, 1800);
    const label = d.sources.find((s) => s.key === k)?.label || k;
    traces.push({ x: g.x, y: g.y, mode: "lines", name: `stitched · ${label}`, line: { color: css(colors[k]), width: 2.2 },
      connectgaps: false, hovertemplate: `stitched (${label}): %{y:.${digits}f} ${unit}<extra></extra>` });
  }
  return traces;
}

function legendHtml(d, colors) {
  return d.sources.filter((s) => s.t.length).map((s) => `<span><i class="swatch" style="background:${css(colors[s.key])}"></i>${esc(s.label)}</span>`).join("") +
    `<span class="muted">thick = stitched (colour = supplying source), thin = raw source</span>`;
}

function sourcesTable(d, colors, kind) {
  const total = Object.values(d.slots_by_source || {}).reduce((a, b) => a + b, 0) + (d.missing_slots || 0);
  const rows = d.sources.map((s) => {
    const n = d.slots_by_source?.[s.key] || 0;
    let map = "";
    if (s.mapping?.offset != null && s.key !== "uwd_release") map = `offset ${s.mapping.offset >= 0 ? "+" : ""}${num(s.mapping.offset, 3)} µrad`;
    else if (s.mapping && "offset" in s.mapping && s.mapping.offset == null) map = "not aligned (no overlap)";
    if (s.mapping?.fits?.length) map = s.mapping.fits.map((f) => `${f.used ? "used" : "rejected"} R²=${f.r2}`).join(", ");
    if (s.mapping?.ratios?.length) map = s.mapping.ratios.map((r) => `×${num(r.ratio, 3)}`).join(", ");
    const h = s.health || {};
    return `<tr><td><span class="sw" style="background:${css(colors[s.key])}"></span>${esc(s.label)}</td>
      <td class="num">${total ? ((100 * n) / total).toFixed(n && n / total < 0.001 ? 3 : 1) + "%" : "—"}</td>
      <td>${esc(map || (s.key === "uwd_release" || s.key === "uwe" ? "authoritative" : ""))}</td>
      <td class="muted">${esc(h.last_success ? "fetched " + rel(h.last_success) : "not fetched")}</td>
      <td class="muted">${esc(s.note)}</td></tr>`;
  }).join("");
  return `<table><thead><tr><th>Source (priority order)</th><th class="num">Share of grid</th><th>Mapping</th><th>Fetched</th><th>Notes</th></tr></thead><tbody>${rows}
    <tr><td class="muted">no source (genuine gap)</td><td class="num">${total ? ((100 * (d.missing_slots || 0)) / total).toFixed(2) + "%" : "—"}</td><td colspan="3" class="muted">models output None for slots that need this ${kind}</td></tr></tbody></table>`;
}

function renderTilt(d) {
  Plotly.react("chart-tilt", sourceTraces(d, TILT_COLORS, "µrad", 2), baseLayout({ yaxis: { title: { text: "µrad", font: { size: 11 } } } }), config);
  $("tilt-legend").innerHTML = legendHtml(d, TILT_COLORS);
  $("tilt-sources").innerHTML = sourcesTable(d, TILT_COLORS, "tilt");
  const r = withGaps(d.rate.t, d.rate.v, 1800);
  Plotly.react("chart-rate", [{ x: r.x, y: r.y, mode: "lines", line: { color: css("--ink"), width: 1.3 }, connectgaps: false, hovertemplate: "%{y:.3f} µrad/h<extra></extra>" }],
    baseLayout({ margin: { l: 52, r: 18, t: 8, b: 30 }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: css("--axis"), width: 1 } }] }), config);
  $("tilt-note").textContent = "Share of grid counts every 5-minute slot since Dec 1, 2024. Plot sources are aligned by their median offset to the series built from higher-priority sources; SDH fills only gaps where its fit to UWD is tight.";
}

function renderTremor(d) {
  Plotly.react("chart-tremor", sourceTraces(d, TREMOR_COLORS, "µm/s", 3), baseLayout({ yaxis: { type: "log", title: { text: "µm/s (log)", font: { size: 11 } },
    tickvals: [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], ticktext: ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"] } }), config);
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
async function loadSeries() {
  const hours = Math.ceil(spanDays() * 24);
  const [tilt, tremor, eq, preds] = await Promise.all([
    safe(() => api(`/api/series/tilt?hours=${hours}`), "tilt"), safe(() => api(`/api/series/tremor?hours=${hours}`), "tremor"),
    safe(() => api(`/api/earthquakes?days=${Math.ceil(spanDays())}`), "eq"), safe(() => api(`/api/predictions?hours=${hours}`), "preds"),
  ]);
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
    safe(() => api("/api/firms?days=7"), "firms"), safe(() => api("/api/earthquakes?days=1"), "eq24"), safe(() => api("/api/frame?hours=1"), "frame"),
    safe(() => api("/api/messages?limit=6"), "messages"),
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

function initControls() {
  const savedDays = store.get("days");
  if (savedDays === "all") state.days = "all";
  else if ([1, 7, 30, 90].includes(Number(savedDays))) state.days = Number(savedDays);
  state.showEpisodes = store.get("showEpisodes") === "1";
  if (["12", "24", "72"].includes(store.get("horizon"))) state.horizon = store.get("horizon");
  markOn("range", "d", state.days);
  markOn("horizon", "h", state.horizon);
  $("episodes-toggle").checked = state.showEpisodes;
  segment("range", "d", (v) => { state.days = v === "all" ? "all" : Number(v); store.set("days", v); loadSeries(); });
  segment("horizon", "h", (v) => { state.horizon = v; store.set("horizon", v); if (state.data.preds) renderProbChart(state.data.preds); });
  $("episodes-toggle").addEventListener("change", (e) => { state.showEpisodes = e.target.checked; store.set("showEpisodes", e.target.checked ? "1" : "0"); renderSeries(); });
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

window.addEventListener("DOMContentLoaded", () => { initTheme(); initControls(); refresh(); setInterval(refresh, 120000); });
