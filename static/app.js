"use strict";

const HST = "Pacific/Honolulu";
const state = {
  days: 7,
  horizon: "24",
  showEpisodes: false,
  episodes: [],
  data: {},
};

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
async function safe(fn, label) {
  try { return await fn(); } catch (e) { console.warn(label, e); return null; }
}

function fmtHST(unix, withYear = false) {
  if (!unix) return "—";
  return new Date(unix * 1000).toLocaleString("en-US", {
    timeZone: HST, month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
    ...(withYear ? { year: "numeric" } : {}),
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

// Plotly treats date strings as naive wall time, so hand it HST (UTC−10, no DST) strings.
const hst = (t) => new Date((t - 36000) * 1000).toISOString().slice(0, 19).replace("T", " ");

/** Insert nulls where samples are further apart than maxGap s, or 3× the typical spacing if
 *  the series was downsampled (draws a break instead of bridging a data gap). */
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

const SERIES_START_FALLBACK = 1734949200; // 2024-12-23 02:20 HST, episode 1

/** Visible window length in days; "all" = since the current eruption series began. */
function spanDays() {
  if (state.days !== "all") return state.days;
  const start = state.episodes.length ? Math.min(...state.episodes.map((e) => e.start_t)) : SERIES_START_FALLBACK;
  return (Date.now() / 1000 - start) / 86400 + 1;
}
function windowRange() {
  const now = Date.now() / 1000;
  return [now - spanDays() * 86400, now];
}

// ---------- chart chrome ----------
function axisStyle() {
  return { gridcolor: css("--grid"), linecolor: css("--axis"), tickcolor: css("--axis"), zeroline: false,
    tickfont: { color: css("--muted"), size: 11 } };
}
function baseLayout(extra = {}) {
  const [t0, t1] = windowRange();
  const ax = axisStyle();
  const lay = {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: css("--ink-2") },
    margin: { l: 52, r: 18, t: 14, b: 34 },
    hovermode: "x unified",
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

/** Shade eruption periods in the visible window; label them when there are few enough to read. */
function addEpisodeOverlay(lay) {
  const [t0, t1] = windowRange();
  const eps = state.episodes.filter((e) => (e.end_t || t1) >= t0 && e.start_t <= t1);
  const fill = css("--episode"), fillAlt = css("--episode-alt");
  const minW = (t1 - t0) * (state.days === "all" ? 0.0015 : 0.004); // keep ~9 h episodes visible
  for (const e of eps) {
    const end = Math.max(e.end_t || t1, e.start_t + minW);
    lay.shapes.push({
      type: "rect", xref: "x", yref: "paper", x0: hst(e.start_t), x1: hst(end), y0: 0, y1: 1,
      fillcolor: e.kind === "fountaining" ? fill : fillAlt, line: { width: 0 }, layer: "below",
    });
    if (eps.length <= 14) {
      lay.annotations.push({
        xref: "x", yref: "paper", x: hst(e.start_t), y: 1, yanchor: "bottom", xanchor: "left", showarrow: false,
        text: e.kind === "fountaining" ? `E${e.label}` : "vents", font: { size: 10, color: css("--muted") },
      });
    }
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
  $("status-date").textContent = `Status notice ${since ? fmtHST(since, true) : "—"} · checked ${rel(s.fetched_at)}` +
    (d.freshness.stale ? " · ⚠ stale" : "");
  $("status-synopsis").textContent = (s.synopsis || "").replace(/^HVO Kilauea [A-Z]+\/[A-Z]+ - /, "");
}

function renderNotices(d) {
  const n = d.notices || [];
  if (!n.length) return;
  const a = n[0];
  $("notice-title").textContent = a.title;
  $("notice-time").textContent = `${fmtHST(a.sent_unix, true)} · ${rel(a.sent_unix)}`;
  $("notice-synopsis").textContent = a.synopsis || "";
  $("notice-text").textContent = a.text || "";
  $("notice-link").href = a.url;
  $("notice-list").innerHTML = n.slice(1).map((x) =>
    `<li><a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.type_title)}</a> <span class="muted">· ${esc(fmtHST(x.sent_unix))}</span><br><span class="muted">${esc(x.synopsis)}</span></li>`
  ).join("") + `<li><a href="${esc(d.hans_url)}" target="_blank" rel="noopener">Search all notices in HANS ↗</a></li>`;
}

const SOURCE_NAMES = {
  usgs_status: "USGS status", hans: "HVO notices", comcat: "Earthquakes", usgs_tilt: "Tilt (live)",
  usgs_tilt_long: "Tilt (long plots)", tilt_release: "Tilt (USGS release)", fdsn_tremor: "Tremor",
  firms: "FIRMS", weather: "Weather", usgs_episode_table: "Episode table", model_predict: "Model run",
  model_train: "Model training",
};
function renderHealth(d) {
  $("freshness").innerHTML = d.sources.map((s) => {
    const err = s.last_error_at && (!s.last_success || s.last_error_at > s.last_success);
    const cls = err ? "err" : s.stale ? "stale" : "";
    const title = err ? `Last error: ${s.last_error}` : (s.detail || "");
    return `<span class="chip ${cls}" title="${esc(title)}"><span class="dot"></span>${esc(SOURCE_NAMES[s.source] || s.source)} · ${esc(rel(s.last_success))}${cls === "stale" ? " (stale)" : ""}${err ? " (error)" : ""}</span>`;
  }).join("");
}

// ---------- model cards ----------
function fmtFeature(f, v) {
  if (v == null) return "n/a";
  switch (f) {
    case "hours_since_end": return dur(v);
    case "recovery_ratio": return `${v.toFixed(2)}×`;
    case "inflation_urad": case "gap_to_onset_urad": case "last_deflation_urad": return `${v.toFixed(1)} µrad`;
    case "tilt_rate_6h": case "tilt_rate_24h": return `${v >= 0 ? "+" : ""}${v.toFixed(3)} µrad/h`;
    case "rsam_log": return `${Math.pow(10, v).toFixed(2)} µm/s`;
    case "rsam_ratio_log": case "rsam_trend_log": return `${Math.pow(10, v).toFixed(2)}×`;
    case "precursor": return v ? "yes" : "no";
    default: return String(Math.round(v * 100) / 100);
  }
}

function renderModelCard(cardId, m, effectFmt) {
  const card = $(cardId);
  const tiles = card.querySelector("[data-tiles]");
  if (!m) { tiles.innerHTML = `<div class="tile muted">Not trained yet</div>`; return; }
  tiles.innerHTML = ["12", "24", "72"].map((H) => `
    <div class="tile"><div class="h">next ${H} h</div><div class="v">${pct(m.p[H])}</div></div>`).join("");
  const tr = m.trend || {};
  const arrow = { up: "▲ rising", down: "▼ falling", flat: "▬ steady" }[tr.arrow || "flat"];
  const d24 = tr.delta_24h != null ? ` · ${tr.delta_24h >= 0 ? "+" : ""}${Math.round(tr.delta_24h * 100)} pts / 24 h` : "";
  card.querySelector("[data-trend]").textContent = `${arrow}${d24}`;
  card.querySelector("[data-factors]").innerHTML = (m.top_factors || []).slice(0, 5).map((f) => `
    <li><span>${esc(f.label)} <span class="val">· ${esc(fmtFeature(f.feature, f.value))}</span></span>
    <span class="eff ${f.effect > 0 ? "up" : "down"}">${f.effect > 0 ? "raises" : "lowers"} ${effectFmt(Math.abs(f.effect))}</span></li>`).join("");
}

function renderProbability(d) {
  const hz = d.models.hazard, ml = d.models.ml;
  $("disclaimer").textContent = d.disclaimer;
  renderModelCard("card-hazard", hz, (x) => x.toFixed(2));
  renderModelCard("card-ml", ml, (x) => `${Math.round(x * 100)} pts`);
  if (hz?.in_episode) {
    $("episode-banner").classList.remove("hidden");
    $("episode-banner").innerHTML = `<b>An episode appears to be in progress.</b> ${esc(hz.in_episode_reason || "")}. Onset probabilities don't apply while an episode is under way.`;
  } else {
    $("episode-banner").classList.add("hidden");
  }
  const warns = hz?.warnings || [];
  $("prob-warnings").classList.toggle("hidden", !warns.length);
  $("prob-warnings").innerHTML = `<div class="eyebrow">Read with caution: conditions are outside the training data</div><ul>` +
    warns.map((w) => {
      const i = w.search(/[.:]\s/);
      const head = i > 0 ? w.slice(0, i + 1) : w, rest = i > 0 ? w.slice(i + 2) : "";
      return `<li><b>${esc(head)}</b> ${esc(rest)}</li>`;
    }).join("") + "</ul>";
  const m = d.training?.metrics;
  if (m && m.hazard?.closed) {
    const head = `<thead><tr><th></th><th>12 h BSS</th><th>AUC</th><th>24 h BSS</th><th>AUC</th><th>72 h BSS</th><th>AUC</th></tr></thead>`;
    const row = (k, scope) => `<tr><td>${k === "hazard" ? "hazard-1.0" : "ml-1.0"}</td>` +
      ["12", "24", "72"].map((H) => { const x = m[k]?.[scope]?.[H] || {}; return `<td>${num(x.brier_skill, 2)}</td><td>${num(x.auc, 2)}</td>`; }).join("") + "</tr>";
    $("skill").innerHTML = `
      <div class="eyebrow" style="margin-top:12px">Completed cycles (episodes 4–${esc(hz?.last_episode ?? "")})</div>
      <table class="skill">${head}<tbody>${row("hazard", "closed")}${row("ml", "closed")}</tbody></table>
      <div class="eyebrow" style="margin-top:14px">Including the current, unusually long pause</div>
      <table class="skill">${head}<tbody>${row("hazard", "all")}${row("ml", "all")}</tbody></table>
      <p class="muted small" style="margin-top:10px">Brier skill score (BSS) &gt; 0 means better than always predicting the historical base rate.
      ${d.training.n_rows} pause-hours and ${d.training.n_onsets} onsets; trained ${rel(d.training.trained_at)}. When the current pause is held out,
      models trained on earlier cycles expect an onset by now, so skill drops. That drop is the regime change showing up in the numbers.</p>`;
  }
}

function renderKPIs(prob, tilt, tremor, eq24) {
  const f = prob?.models?.hazard?.features || {};
  const st = tilt?.state || {};
  const items = [
    ["Since last episode", dur(f.hours_since_end), prob?.models?.hazard ? `episode ${prob.models.hazard.last_episode} ended` : ""],
    ["Inflation since", `${num(st.inflation_since_last_urad)} µrad`, st.recovery_ratio ? `${num(st.recovery_ratio, 2)}× last deflation` : ""],
    ["Tilt vs last onset", `${st.gap_to_last_onset_urad >= 0 ? "+" : ""}${num(st.gap_to_last_onset_urad)} µrad`, `episode ${st.last_episode || "?"} onset level`],
    ["Tilt rate, 24 h", `${num(st.rate_24h, 3)}`, `µrad/h · 3 h: ${num(st.rate_3h, 3)}`],
    ["Tremor, 1 h", `${num(tremor?.current?.rsam_1h_ums, 2)} µm/s`, tremor?.current?.ratio_1h_24h ? `${num(tremor.current.ratio_1h_24h, 2)}× 24 h median` : ""],
    ["Earthquakes, 24 h", `${eq24?.counts?.last_24h ?? "—"}`, `${eq24?.counts?.summit_last_24h ?? "—"} at the summit`],
    ["Precursory activity", f.precursor ? "Reported" : "Not reported", "latest HVO update"],
  ];
  $("kpis").innerHTML = items.map(([k, v, s]) => `<div class="kpi"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join("");
}

// ---------- timeseries ----------
function renderProbChart(h) {
  const rows = h?.history || [];
  const key = `p${state.horizon}`;
  const trainedAt = h?.trained_at;
  const traces = [], ann = [], shapes = [];
  for (const [model, name, color] of [["hazard", "hazard-1.0", css("--s1")], ["ml", "ml-1.0", css("--s2")]]) {
    // one continuous curve per model: everything is computed by the current model
    const r = rows.filter((x) => x.model === model && x[key] != null);
    const g = withGaps(r.map((x) => x.t), r.map((x) => x[key] * 100), 3 * 3600);
    traces.push({ x: g.x, y: g.y, mode: "lines", name, line: { color, width: 1.8 }, connectgaps: false,
      hovertemplate: `${name}: %{y:.1f}%<extra></extra>` });
    const last = r.at(-1);
    if (last) ann.push({ x: hst(last.t), y: last[key] * 100, xanchor: "left", xshift: 6, showarrow: false,
      text: `${Math.round(last[key] * 100)}%`, font: { size: 11, color: css("--ink-2") } });
  }
  const [t0] = windowRange();
  if (trainedAt && trainedAt > t0) {
    // faint divider: left = in-sample (the models were trained on it), right = out-of-sample
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: hst(trainedAt), x1: hst(trainedAt), y0: 0, y1: 1,
      line: { color: css("--muted"), width: 1 }, opacity: 0.45, layer: "below" });
  }
  Plotly.react("chart-prob", traces, baseLayout({
    margin: { l: 44, r: 44, t: 14, b: 34 }, shapes, annotations: ann,
    yaxis: { ticksuffix: "%", rangemode: "tozero" },
  }), config);
  $("prob-note").textContent = `Chance of an onset within ${state.horizon} h, computed by the current models. Left of the faint grey line (when they were trained ${trainedAt ? fmtHST(trainedAt) : ""}) is in-sample; right of it is out-of-sample.`;
}

function renderTilt(d) {
  const g = withGaps(d.t, d.v, 3600);
  Plotly.react("chart-tilt", [{ x: g.x, y: g.y, mode: "lines", name: "UWD az 300°", line: { color: css("--s1"), width: 1.8 },
    connectgaps: false, hovertemplate: "%{y:.2f} µrad<extra></extra>" }],
    baseLayout({ yaxis: { title: { text: "µrad", font: { size: 11 } } } }), config);
  const r = withGaps(d.rate_t, d.rate, 3600);
  Plotly.react("chart-rate", [{ x: r.x, y: r.y, mode: "lines", name: "tilt rate", line: { color: css("--s1"), width: 1.3 },
    connectgaps: false, hovertemplate: "%{y:.3f} µrad/h<extra></extra>" }],
    baseLayout({ margin: { l: 52, r: 18, t: 8, b: 30 },
      shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: css("--axis"), width: 1 } }] }), config);
  const st = d.stitch;
  $("tilt-note").textContent = d.note + (st ? ` USGS 1-minute data runs through ${fmtHST(st.release_end, true)}; digitized plots after that.` : "");
}

function renderTremor(d) {
  const g = withGaps(d.t, d.v, 1800);
  Plotly.react("chart-tremor", [{ x: g.x, y: g.y, mode: "lines", name: "RSAM", line: { color: css("--s1"), width: 1.3 },
    connectgaps: false, hovertemplate: "%{y:.3f} µm/s<extra></extra>" }],
    baseLayout({ yaxis: { type: "log", title: { text: "µm/s (log)", font: { size: 11 } },
      tickvals: [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20], ticktext: ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2", "5", "10", "20"] } }), config);
}

const REGIONS = [["summit", "Summit", "--s1"], ["upper_erz", "Upper East Rift", "--s2"], ["swrz", "Southwest Rift", "--s3"], ["other", "Other", "--s4"]];
function renderEq(d) {
  const days = spanDays();
  const hourly = days <= 1;
  const bin = hourly ? 3600 : days > 120 ? 7 * 86400 : 86400;
  const [t0, t1] = windowRange();
  const start = Math.floor((t0 - 36000) / bin) * bin + 36000; // bins aligned to HST midnight / hours
  const edges = [];
  for (let t = start; t <= t1; t += bin) edges.push(t);
  const counts = Object.fromEntries(REGIONS.map(([k]) => [k, new Array(edges.length).fill(0)]));
  for (const e of d.events || []) {
    const i = Math.floor((e.t - start) / bin);
    if (i >= 0 && i < edges.length) (counts[e.region] || counts.other)[i]++;
  }
  const centers = edges.map((t) => hst(t + bin / 2));
  const traces = REGIONS.map(([k, name, c]) => ({
    type: "bar", x: centers, y: counts[k], name, width: bin * 1000 * 0.8,
    marker: { color: css(c), line: { color: css("--surface"), width: 1 } }, hovertemplate: `${name}: %{y}<extra></extra>`,
  }));
  Plotly.react("chart-eq", traces, baseLayout({ barmode: "stack", margin: { l: 40, r: 12, t: 14, b: 34 } }), config);
  $("eq-bin-note").textContent = `events per ${hourly ? "hour" : bin > 86400 ? "week" : "day"}, by region`;
  const c = d.counts || {};
  $("eq-summary").textContent = `${c.total ?? 0} events in ${state.days === "all" ? "the series so far" : state.days + " d"}. By depth: ` +
    Object.entries(c.by_depth || {}).map(([k, v]) => `${k} ${v}`).join(", ") + (c.max_mag != null ? `. Largest M${num(c.max_mag, 1)}.` : ".");
}

function renderInflation(d) {
  const inf = d?.inflation;
  if (!inf) { $("chart-infl").innerHTML = `<div class="muted small">No completed episode to measure from.</div>`; return; }
  const days = inf.t.map((t) => (t - inf.since) / 86400);
  const shapes = [], ann = [];
  const rt = inf.ratio_thresholds_urad;
  if (rt) {
    shapes.push({ type: "rect", xref: "paper", x0: 0, x1: 1, y0: rt.p10, y1: rt.p90, fillcolor: css("--band"), line: { width: 0 }, layer: "below" });
    shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1, y0: rt.median, y1: rt.median, line: { color: css("--s1"), width: 1 } });
    ann.push({ xref: "paper", x: 0.01, y: rt.p90, yanchor: "bottom", xanchor: "left", showarrow: false,
      text: `typical onset: ${num(inf.thresholds.recovery_ratio.p10, 2)}–${num(inf.thresholds.recovery_ratio.p90, 2)}× last deflation`,
      font: { size: 11, color: css("--ink-2") } });
  }
  const ax = axisStyle();
  Plotly.react("chart-infl", [{ x: days, y: inf.v, mode: "lines", line: { color: css("--s1"), width: 1.8 }, name: "inflation",
    hovertemplate: "day %{x:.1f}: %{y:.2f} µrad<extra></extra>" }], {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", showlegend: false, hovermode: "closest",
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: css("--ink-2") },
    hoverlabel: { bgcolor: css("--surface"), bordercolor: css("--hair"), font: { color: css("--ink") } },
    margin: { l: 52, r: 18, t: 14, b: 40 }, shapes, annotations: ann,
    xaxis: { ...ax, title: { text: `days since episode ${inf.episode} ended`, font: { size: 11 } } },
    yaxis: { ...ax, title: { text: "µrad regained", font: { size: 11 } } },
  }, config);
  const thr = inf.thresholds;
  $("infl-note").textContent = rt
    ? `Band: 10th–90th percentile of tilt regained at onset over the last ${thr.n} episodes, scaled to episode ${inf.episode}'s ${num(inf.last_deflation_urad)} µrad deflation. Unscaled, onsets came after ${num(thr.recovery_urad.p10)}–${num(thr.recovery_urad.p90)} µrad (median ${num(thr.recovery_urad.median)}).`
    : "";
}

function renderWeather(w, firms) {
  const cur = w?.weather?.current;
  if (cur) {
    const vis = cur.visibility != null ? (cur.visibility >= 10000 ? ">10 km" : `${(cur.visibility / 1000).toFixed(1)} km`) : "—";
    const items = [
      ["Temperature", `${num(cur.temperature_2m)} °C`, `RH ${cur.relative_humidity_2m ?? "—"}%`],
      ["Visibility", vis, `cloud ${cur.cloud_cover ?? "—"}%, low ${cur.cloud_cover_low ?? "—"}%`],
      ["Wind", `${num(cur.wind_speed_10m, 0)} km/h`, `from ${cur.wind_direction_10m ?? "—"}°`],
      ["Precipitation", `${num(cur.precipitation)} mm`, `updated ${rel(w.weather.fetched_at)}`],
    ];
    $("weather").innerHTML = items.map(([k, v, s]) => `<div class="kpi"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="s">${esc(s)}</div></div>`).join("");
  } else $("weather").innerHTML = `<div class="kpi muted">Weather unavailable</div>`;
  const det = firms?.detections || [];
  if (firms?.freshness?.last_error && !firms?.freshness?.last_success) {
    $("firms").innerHTML = `<span class="muted">FIRMS disabled or failing: ${esc(firms.freshness.last_error)}</span>`;
    return;
  }
  const byDay = {};
  det.forEach((x) => { const k = new Date(x.t * 1000).toLocaleDateString("en-US", { timeZone: HST, month: "short", day: "numeric" }); (byDay[k] = byDay[k] || []).push(x); });
  const last = det.at(-1);
  $("firms").innerHTML = `<p>${det.length} hotspot detections over the caldera in the last 7 days${last ? `; latest ${esc(fmtHST(last.t))}, FRP ${num(last.frp)} MW` : ""}.</p>
    <p class="muted" style="margin-top:4px">${Object.entries(byDay).map(([k, v]) => `${esc(k)} ${v.length}`).join(" · ")}</p>
    <p class="note">Vents stay hot between episodes. A jump in detections or radiative power (FRP) is what confirms fountaining.</p>`;
}

function renderEpisodes(d) {
  const eps = [...d.episodes].reverse();
  $("episodes").querySelector("tbody").innerHTML = eps.map((e) => `
    <tr class="${e.kind !== "fountaining" ? "nonfount" : ""}">
      <td>${esc(e.kind === "fountaining" ? e.label : "—")}</td>
      <td>${esc(fmtHST(e.start_t, true))}</td>
      <td class="num">${esc(dur(e.duration_h))}</td>
      <td class="num">${esc(dur(e.repose_before_h))}</td>
      <td class="num">${esc(dur(e.onset_interval_h))}</td>
      <td class="num">${e.deflation_urad != null ? num(e.deflation_urad) + " µrad" : "—"}</td>
      <td class="num">${e.onset_recovery_ratio != null ? num(e.onset_recovery_ratio, 2) + "×" : "—"}</td>
      <td class="num">${e.fountain_height_m != null ? Math.round(e.fountain_height_m) + " m" : "—"}</td>
      <td class="notes">${esc(e.notes)}</td>
    </tr>`).join("");
  const sg = d.suggestions || [];
  $("suggestions").innerHTML = sg.length ? `<div class="eyebrow">Suggested from HVO notices, pending review</div>` +
    sg.slice(0, 3).map((s) => `<div class="sugg"><b>Episode ${esc(s.episode_num)} · ${esc(s.phase)}</b> <span class="muted">· ${esc(fmtHST(s.sent_unix))}</span><br><span class="muted">${esc(s.snippet)}</span></div>`).join("") : "";
}

// ---------- loading ----------
async function loadSeries() {
  const hours = Math.ceil(spanDays() * 24);
  const [tilt, tremor, eq, hist] = await Promise.all([
    safe(() => api(`/api/tilt?hours=${hours}`), "tilt"),
    safe(() => api(`/api/tremor?hours=${hours}`), "tremor"),
    safe(() => api(`/api/earthquakes?days=${Math.ceil(spanDays())}`), "eq"),
    safe(() => api(`/api/probability/history?hours=${hours}`), "hist"),
  ]);
  Object.assign(state.data, { tilt, tremor, eq, hist });
  renderSeries();
  return { tilt, tremor };
}

function renderSeries() {
  const d = state.data;
  if (d.hist) renderProbChart(d.hist);
  if (d.tilt) { renderTilt(d.tilt); renderInflation(d.tilt); }
  if (d.tremor) renderTremor(d.tremor);
  if (d.eq) renderEq(d.eq);
}

async function refresh() {
  const [status, notices, health, prob, eps, weather, firms, eq24] = await Promise.all([
    safe(() => api("/api/status"), "status"), safe(() => api("/api/notices?limit=6"), "notices"),
    safe(() => api("/api/health"), "health"), safe(() => api("/api/probability"), "prob"),
    safe(() => api("/api/episodes"), "episodes"), safe(() => api("/api/weather"), "weather"),
    safe(() => api("/api/firms?days=7"), "firms"), safe(() => api("/api/earthquakes?days=1"), "eq24"),
  ]);
  if (status) renderStatus(status);
  if (notices) renderNotices(notices);
  if (health) renderHealth(health);
  if (prob) { state.data.prob = prob; renderProbability(prob); }
  if (eps) { state.episodes = eps.episodes || []; renderEpisodes(eps); }
  renderWeather(weather, firms);
  const { tilt, tremor } = await loadSeries();
  renderKPIs(prob, tilt, tremor, eq24);
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
function markOn(id, attr, value) {
  $(id).querySelectorAll("button").forEach((x) => x.classList.toggle("on", x.dataset[attr] === String(value)));
}

function initControls() {
  const savedDays = store.get("days");
  if (savedDays === "all") state.days = "all";
  else if ([1, 7, 30, 90].includes(Number(savedDays))) state.days = Number(savedDays);
  state.showEpisodes = store.get("showEpisodes") === "1";
  const savedH = store.get("horizon");
  if (["12", "24", "72"].includes(savedH)) state.horizon = savedH;
  markOn("range", "d", state.days);
  markOn("horizon", "h", state.horizon);
  $("episodes-toggle").checked = state.showEpisodes;

  segment("range", "d", (v) => { state.days = v === "all" ? "all" : Number(v); store.set("days", v); loadSeries(); });
  segment("horizon", "h", (v) => { state.horizon = v; store.set("horizon", v); if (state.data.hist) renderProbChart(state.data.hist); });
  $("episodes-toggle").addEventListener("change", (e) => {
    state.showEpisodes = e.target.checked; store.set("showEpisodes", e.target.checked ? "1" : "0"); renderSeries();
  });
}

function initTheme() {
  const saved = store.get("theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("theme", next);
    renderSeries();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderSeries);
}

window.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initControls();
  refresh();
  setInterval(refresh, 120000);
});
