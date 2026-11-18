const API = (window.API_BASE || "").replace(/\/$/, "");
let state = null;

// ---------- saved chats (this device only) ----------
const CKEY = "recipeforge.chats.v1";
const MAX_CHATS = 20;
const MAX_TURNS = 40;
let chatId = null;
let busy = false;
let carried = [];  // chips remembered from an earlier chat: [{kind, value}]

function loadChats() {
  try {
    const c = JSON.parse(localStorage.getItem(CKEY) || "[]");
    return Array.isArray(c) ? c : [];
  } catch { return []; }
}
function saveChats(chats) {
  let list = chats.sort((a, b) => b.updated - a.updated).slice(0, MAX_CHATS);
  while (list.length) {
    try { localStorage.setItem(CKEY, JSON.stringify(list)); return; } catch { list = list.slice(0, -1); }
  }
  try { localStorage.removeItem(CKEY); } catch { /* storage unavailable */ }
}
function newId() {
  return (crypto.randomUUID && crypto.randomUUID()) || Date.now().toString(36) + Math.random().toString(36).slice(2);
}
function updateChat(mutate) {
  const chats = loadChats();
  let c = chats.find((x) => x.id === chatId);
  if (!c) {
    c = { id: chatId, title: "New chat", created: Date.now(), turns: [], state: null, chips: [], carried };
    chats.push(c);
  }
  mutate(c);
  c.updated = Date.now();
  c.turns = c.turns.slice(-MAX_TURNS);
  saveChats(chats);
}
function relTime(ms) {
  const m = Math.round((Date.now() - ms) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

// ---------- cooking history (this device only) ----------
const HKEY = "recipeforge.history.v1";
function loadHistory() {
  try {
    const h = JSON.parse(localStorage.getItem(HKEY) || "[]");
    return Array.isArray(h) ? h : [];
  } catch { return []; }
}
function saveHistory(h) {
  try { localStorage.setItem(HKEY, JSON.stringify(h.slice(-100))); } catch { /* storage unavailable */ }
  renderHistoryNote();
}
function recordAction(id, action) {
  const h = loadHistory().filter((e) => !(e.id === id && (action !== "cooked") && e.action !== "cooked"));
  h.push({ id, action, t: Math.floor(Date.now() / 1000) });
  saveHistory(h);
}
function renderHistoryNote() {
  const cooked = new Set(loadHistory().filter((e) => e.action === "cooked").map((e) => e.id)).size;
  const rated = loadHistory().filter((e) => e.action !== "cooked").length;
  $("#history-note").innerHTML = cooked || rated
    ? `Personalising from ${cooked} cooked and ${rated} rated dishes on this device. <button class="link" id="clear-history">Clear</button>`
    : "Mark dishes as cooked, liked or disliked to personalise results. Saved only on this device.";
}

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body) {
  const res = await fetch(API + path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined);
  if (!res.ok) {
    const err = new Error(`${res.status} ${res.statusText}`);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

// ---------- tabs ----------
function showTab(name) {
  document.querySelectorAll(".side-item[data-tab]").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  if (name !== "chat") document.querySelectorAll(".chats-list li").forEach((li) => li.classList.remove("current"));
  else renderChatList();
  closeSidebar();
}
document.querySelectorAll(".side-item[data-tab]").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------- sidebar (slides in on small screens) ----------
function openSidebar() { $("#sidebar").classList.add("open"); $("#scrim").hidden = false; }
function closeSidebar() { $("#sidebar").classList.remove("open"); $("#scrim").hidden = true; }
$("#menu-btn").addEventListener("click", openSidebar);
$("#scrim").addEventListener("click", closeSidebar);

// ---------- chat ----------
const messages = $("#messages");

function updateEmpty() {
  $("#empty").hidden = !!messages.querySelector(".msg");
}

function addUser(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  messages.appendChild(el);
  updateEmpty();
}

function checkBadge(c) {
  const mark = c.status === "pass" ? "✓" : c.status === "flag" ? "!" : "✕";
  return `<span class="check ${esc(c.status)}" title="${esc(c.detail)}"><b>${mark}</b>${esc(c.label)}</span>`;
}

function sourceName(url) {
  try {
    const host = new URL(url).hostname.replace(/^www\./, "");
    return host === "archanaskitchen.com" ? "Archana's Kitchen" : host;
  } catch { return "original recipe"; }
}

function swapList(swaps) {
  return swaps.map((s) => `<li><b>${esc(s.replace)} → ${esc(s.with.join(" + "))}</b>: ${esc(s.how)}</li>`).join("");
}

function adaptHtml(a) {
  if (!a || !a.adaptable || !a.swaps.length) return "";
  const main = a.main_swapped.length ? ` <i>(changes a named ingredient: ${esc(a.main_swapped.join(", "))})</i>` : "";
  return `<details class="adapt"><summary>✓ Can be adapted with ${a.swaps.length} swap${a.swaps.length > 1 ? "s" : ""}${main}</summary>
    <ul>${swapList(a.swaps)}${swapList(a.safer)}</ul><div class="sub">The adapted recipe was re-checked against every active constraint.</div></details>`;
}

// Outline icons (Lucide, ISC licence), inlined so the page has no extra requests.
const ICON = {
  clock: '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
  users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  ext: '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3"/>',
  cooked: '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
  up: '<path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>',
  down: '<path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/>',
  alert: '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4M12 17h.01"/>',
};
const icon = (name) => `<svg class="ico" viewBox="0 0 24 24" aria-hidden="true">${ICON[name]}</svg>`;

function cardHtml(r) {
  const flagNotes = r.checks.filter((c) => c.status === "flag").map((c) => `${esc(c.label)}: ${esc(c.detail)}`);
  // Servings (and scaling for the group size) belong in the meta line, not in the reasons.
  const serveNote = r.why.find((w) => /^serves\b/i.test(w));
  const reasons = r.why.filter((w) => w !== serveNote);
  const serves = serveNote
    ? serveNote.replace(/^serves (\d+)/i, "Serves $1").replace(" · make about ", " · ≈").replace(" · make ", " · ").replace(" people", "")
    : r.servings ? `Serves ${r.servings}` : "";
  const meta = [esc(r.cuisine), esc(r.course)].filter(Boolean).join('<span class="dot">·</span>');
  return `
    <article class="card${r.flagged ? " flagged" : ""}">
      <header>
        <h3>${r.flagged ? `<span class="warn" title="Check the note below">${icon("alert")}</span>` : ""}${esc(r.name)}</h3>
        ${r.total_min ? `<span class="time">${icon("clock")}${esc(r.total_min)} min</span>` : ""}
      </header>
      <div class="meta">${meta}${serves ? `<span class="dot">·</span><span class="serves">${icon("users")}${esc(serves)}</span>` : ""}</div>
      ${r.checks.length ? `<div class="checks">${r.checks.map(checkBadge).join("")}</div>` : ""}
      ${reasons.length ? `<p class="why">${reasons.map((w) => esc(w.charAt(0).toUpperCase() + w.slice(1))).join('<span class="dot">·</span>')}</p>` : ""}
      ${flagNotes.length ? `<div class="flagnote">${flagNotes.join("<br>")}</div>` : ""}
      ${r.safer?.length ? `<details class="adapt"><summary>Safer swap</summary><ul>${swapList(r.safer)}</ul></details>` : ""}
      <footer class="actions">
        <button class="ghost" data-recipe="${esc(r.id)}">${icon("list")}Ingredients</button>
        ${r.url ? `<a class="ghost" href="${esc(r.url)}" target="_blank" rel="noopener">${icon("ext")}Recipe</a>
          <span class="source">${esc(sourceName(r.url))}</span>` : ""}
        <span class="taste" role="group" aria-label="Your feedback">
          <button class="icon-btn" data-act="cooked" data-id="${esc(r.id)}" title="I cooked this" aria-label="I cooked this">${icon("cooked")}</button>
          <button class="icon-btn" data-act="like" data-id="${esc(r.id)}" title="More like this" aria-label="More like this">${icon("up")}</button>
          <button class="icon-btn" data-act="dislike" data-id="${esc(r.id)}" title="Less like this" aria-label="Less like this">${icon("down")}</button>
        </span>
      </footer>
    </article>`;
}

function planHtml(data) {
  const days = data.results.map((r) => `<div class="day"><div class="dayname">${esc(r.day)}</div>${cardHtml(r)}</div>`).join("");
  const s = data.shopping;
  if (!s || !s.items.length) return `<div class="plan">${days}</div>`;
  // Shopping rows name days ("Mon · Wed"), not long recipe titles; day chips filter the list.
  const dayOf = Object.fromEntries(data.results.map((r) => [r.name, r.day]));
  const short = (name) => esc(name.split(/ [-|(]/)[0].replace(/^(.{34}).+$/, "$1…"));
  const chips = data.results.map((r) => `
    <button class="shop-day on" data-day="${esc(r.day)}" aria-pressed="true"><b>${esc(r.day)}</b>${short(r.name)}</button>`).join("");
  const rows = s.items.map((x) => {
    const ds = [...new Set(x.for.map((n) => dayOf[n]).filter(Boolean))];
    return `<li data-days="${esc(ds.join(","))}"><button class="shop-item" aria-pressed="false"><span class="box" aria-hidden="true"></span>
      <span class="name">${esc(x.item.charAt(0).toUpperCase() + x.item.slice(1))}</span></button>
      <span class="days">${esc(ds.join(" · "))}</span></li>`;
  }).join("");
  const list = `
    <details class="shopping">
      <summary><span class="shop-title">Shopping list</span><span class="shop-count">${s.items.length} items for ${data.results.length} dishes</span></summary>
      <div class="shop-body">
        <div class="shop-filter"><span class="sub">Showing items for:</span>
          <button class="shop-all" data-set="all">All</button><button class="shop-all" data-set="none">None</button></div>
        <div class="shop-days">${chips}</div>
        <ul class="shop-items">${rows}</ul>
        <p class="shop-empty sub" hidden>Pick at least one dish to see its ingredients.</p>
        <p class="sub shop-staples">Not listed (assumed in your kitchen): ${esc(s.staples.join(", "))}.</p>
      </div>
    </details>`;
  return `<div class="plan">${days}</div>${list}`;
}

function addBot(data) {
  const el = document.createElement("div");
  el.className = "msg bot";
  let html = `<p>${esc(data.reply)}</p>`;
  if (data.suggestions?.length) html += `<div class="suggest">${data.suggestions.map((s) => `<button>${esc(s)}</button>`).join("")}</div>`;
  if (data.mode === "plan" && data.results.length) html += planHtml(data);
  else if (data.results.length) html += `<div class="cards">${data.results.map(cardHtml).join("")}</div>`;
  if (data.closest) html += `<div class="cards">${cardHtml(data.closest)}</div>`;
  if (data.mode === "detail" && data.detail) {
    const d = data.detail;
    if (d.ingredients.length) html += `<div class="detail"><h4>Ingredients</h4><ul>${d.ingredients.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>`;
    if (d.steps.length) html += `<div class="detail"><h4>Steps</h4><ol>${d.steps.map((x) => `<li>${esc(x)}</li>`).join("")}</ol></div>`;
  }
  const ruled = data.blocked_examples?.length
    ? `<div class="ruled"><b>Ruled out — matched your words but not your constraints:</b><ul>${data.blocked_examples
      .map((b) => `<li><button class="link" data-recipe="${esc(b.id)}">${esc(b.name)}</button> — ${esc(b.reason)}${adaptHtml(b.adaptation)}</li>`)
      .join("")}</ul></div>`
    : "";
  const t = data.trace;
  if (data.mode === "search" && t.corpus) {
    const n = (x) => Number(x).toLocaleString("en-IN");
    const understood = (data.constraints || []).map((c) => c.label).join(" · ") || "no specific constraints";
    const removedTotal = Object.values(t.removed_by_constraint || {}).reduce((a, b) => a + b, 0);
    const removedBy = Object.entries(t.removed_by_constraint || {}).map(([k, v]) => `${n(v)} for ${k}`).join(", ");
    const steps = [
      `<b>Understood</b> ${esc(understood)}${t.claude_parser ? " <span class=\"sub\">(Claude helped read this one)</span>" : ""}`,
      `<b>Checked</b> ${n(t.corpus)} recipes, ingredient by ingredient${removedTotal
        ? ` — ruled out ${n(removedTotal)}${Object.keys(t.removed_by_constraint).length > 1 ? ` (${esc(removedBy)})` : ` for ${esc(Object.keys(t.removed_by_constraint)[0])}`}`
        : ""}`,
      `<b>Narrowed</b> to ${n(t.after_soft_filters ?? t.after_hard_filter)} that match your course, cuisine and ingredients`,
      `<b>Searched</b> for “${esc(t.query_text)}” by keywords and meaning, then ranked by your preferences`,
      `<b>Picked</b> ${data.results.length} varied dishes`,
    ];
    if (t.relaxed?.length) steps.push(`<b>Note</b> ${esc(t.relaxed.join("; "))}`);
    html += `<details class="trace"><summary>How I found these</summary><ol class="trace-steps">${steps.map((s) => `<li>${s}</li>`).join("")}</ol>${ruled}
      <details class="raw"><summary>Raw details</summary><pre>${esc(JSON.stringify({ parsed: data.parsed, trace: t }, null, 2))}</pre></details></details>`;
  }
  el.innerHTML = html;
  messages.appendChild(el);
  updateEmpty();
}

function renderChips(chips) {
  const isCarried = (c) => carried.some((x) => x.kind === c.kind && x.value === c.value);
  $("#chips").innerHTML = chips
    .map((c) => `<span class="chip ${c.hard ? "hard" : ""}" ${isCarried(c) ? 'title="Remembered from an earlier chat on this device"' : ""}>${c.hard ? "🔒 " : ""}${esc(c.label)}${isCarried(c) ? ' <small class="remembered">remembered</small>' : ""}<button aria-label="Remove ${esc(c.label)}" data-kind="${esc(c.kind)}" data-value="${esc(c.value)}">×</button></span>`)
    .join("");
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  busy = true;
  $("#form button").disabled = true;
  addUser(text);
  const typing = document.createElement("div");
  typing.className = "msg bot typing";
  typing.innerHTML = '<span class="dots"><i></i><i></i><i></i></span><span class="typing-text">Checking ingredients</span>';
  messages.appendChild(typing);
  typing.scrollIntoView({ behavior: "smooth", block: "end" });
  const slow = setTimeout(() => { typing.querySelector(".typing-text").textContent = "Reading your request carefully"; }, 1200);
  try {
    const data = await api("/api/chat", { message: text, state, history: loadHistory() });
    state = data.state;
    typing.remove();
    addBot(data);
    renderChips(data.constraints);
    updateChat((c) => {
      if (!c.turns.length) c.title = text.slice(0, 60);
      c.turns.push({ role: "user", text }, { role: "bot", data });
      c.state = data.state;
      c.chips = data.constraints;
      c.carried = carried;
    });
    renderChatList();
  } catch (e) {
    typing.textContent = e.status === 429
      ? "You're sending messages very quickly — please wait a minute and try again."
      : "Couldn't reach the server — please check your connection and try again.";
  } finally {
    clearTimeout(slow);
    busy = false;
    $("#form button").disabled = false;
  }
  messages.lastElementChild.scrollIntoView({ behavior: "smooth", block: "start" });
}

$("#form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("#input");
  send(input.value);
  input.value = "";
});
// Shopping list: filter by the dishes you'll actually cook, tick items off as you buy them.
function refreshShopping(shop) {
  const on = new Set([...shop.querySelectorAll(".shop-day.on")].map((b) => b.dataset.day));
  let shown = 0;
  shop.querySelectorAll(".shop-items li").forEach((li) => {
    const show = li.dataset.days.split(",").some((d) => on.has(d));
    li.hidden = !show;
    shown += show;
  });
  shop.querySelector(".shop-empty").hidden = shown > 0;
  shop.querySelector(".shop-count").textContent = `${shown} item${shown === 1 ? "" : "s"} for ${on.size} dish${on.size === 1 ? "" : "es"}`;
}
messages.addEventListener("click", (e) => {
  const shop = e.target.closest(".shopping");
  if (shop) {
    const day = e.target.closest(".shop-day");
    const all = e.target.closest(".shop-all");
    const item = e.target.closest(".shop-item");
    if (day) { day.classList.toggle("on"); day.setAttribute("aria-pressed", day.classList.contains("on")); }
    if (all) shop.querySelectorAll(".shop-day").forEach((b) => {
      b.classList.toggle("on", all.dataset.set === "all"); b.setAttribute("aria-pressed", all.dataset.set === "all");
    });
    if (item) { item.classList.toggle("done"); item.setAttribute("aria-pressed", item.classList.contains("done")); }
    if (day || all) refreshShopping(shop);
    return;
  }
  const starter = e.target.closest(".starters button[data-q]");
  if (starter) return send(starter.dataset.q);
  const s = e.target.closest(".suggest button");
  if (s) send(s.textContent);
});
$("#chips").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-kind]");
  if (!b || !state) return;
  const data = await api("/api/constraints/remove", { state, kind: b.dataset.kind, value: b.dataset.value });
  state = data.state;
  carried = carried.filter((x) => !(x.kind === b.dataset.kind && x.value === b.dataset.value));
  renderChips(data.constraints);
  if (loadChats().some((c) => c.id === chatId)) {
    updateChat((c) => { c.state = state; c.chips = data.constraints; c.carried = carried; });
  }
});

function clearMessages() {
  messages.querySelectorAll(".msg").forEach((m) => m.remove());
  updateEmpty();
}

// Start a new chat. Diets and allergy exclusions from the most recent chat carry over (remove with ×).
async function newChat({ carry = true } = {}) {
  chatId = newId();
  state = null;
  carried = [];
  clearMessages();
  renderChips([]);
  showTab("chat");
  $("#input").focus({ preventScroll: true });
  const last = loadChats()[0];
  const seed = carry && last?.state ? { diets: last.state.diets || [], exclude: last.state.exclude || [] } : null;
  if (seed && (seed.diets.length || seed.exclude.length)) {
    try {
      const data = await api("/api/constraints", { state: seed });
      state = data.state;
      carried = data.constraints.map((c) => ({ kind: c.kind, value: c.value }));
      renderChips(data.constraints);
    } catch { /* start clean if the server can't validate it */ }
  }
}

function openChat(id) {
  const c = loadChats().find((x) => x.id === id);
  if (!c) return newChat();
  chatId = c.id;
  state = c.state;
  carried = c.carried || [];
  clearMessages();
  for (const t of c.turns) {
    if (t.role === "user") addUser(t.text);
    else addBot(t.data);
  }
  renderChips(c.chips || []);
  showTab("chat");
  messages.lastElementChild?.scrollIntoView({ block: "end" });
}

function resetChat() { return newChat({ carry: false }); }
$("#new-chat").addEventListener("click", () => newChat());
$("#top-new").addEventListener("click", () => newChat());

// ---------- chat list (sidebar) ----------
function renderChatList() {
  const chats = loadChats();
  const onChat = $("#tab-chat").classList.contains("active");
  $("#chats-list").innerHTML = chats.length
    ? chats.map((c) => {
        const locks = (c.chips || []).filter((x) => x.hard).map((x) => x.label).join(", ");
        return `<li class="${onChat && c.id === chatId ? "current" : ""}">
          <button class="open" data-open="${esc(c.id)}" title="${esc(c.title)}${locks ? ` — ${esc(locks)}` : ""} · ${esc(relTime(c.updated))}">${esc(c.title)}</button>
          <button class="del" data-del="${esc(c.id)}" aria-label="Delete chat “${esc(c.title)}”">×</button></li>`;
      }).join("")
    : `<li class="none">Your chats will appear here. They stay on this device.</li>`;
}
$("#chats-list").addEventListener("click", (e) => {
  const open = e.target.closest("[data-open]");
  if (open) return openChat(open.dataset.open);
  const del = e.target.closest("[data-del]");
  if (del) {
    saveChats(loadChats().filter((c) => c.id !== del.dataset.del));
    if (del.dataset.del === chatId) newChat({ carry: false });
    renderChatList();
  }
});
messages.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  recordAction(Number(b.dataset.id), b.dataset.act);
  const group = b.closest(".taste");
  if (b.dataset.act === "cooked") {
    b.classList.add("on");
  } else {
    // Like and dislike are exclusive; "cooked" stays marked.
    group.querySelectorAll('[data-act="like"], [data-act="dislike"]').forEach((x) => x.classList.toggle("on", x === b));
  }
  group.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x.classList.contains("on")));
});
document.addEventListener("click", (e) => {
  if (e.target.id !== "clear-history") return;
  saveHistory([]);
});
renderHistoryNote();

// ---------- recipe dialog ----------
const dialog = $("#recipe-dialog");
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-recipe]");
  if (!b) return;
  const r = await api(`/api/recipe/${encodeURIComponent(b.dataset.recipe)}`, { state });
  const verdicts = Object.values(r.verdicts)
    .map((v) => checkBadge({ status: v.status, label: v.label, detail: v.reasons.join("; ") || "pass" }))
    .join("");
  $("#recipe-body").innerHTML = `
    <h3>${esc(r.name)}</h3>
    <div class="sub">${esc(r.cuisine)} · ${esc(r.course)} · ${esc(r.total_min)} min · source label: <i>${esc(r.source_diet_label)}</i></div>
    <ul class="ing">${r.ingredients
      .map((i) => `<li class="${esc(i.status)}"><span>${esc(i.line)}</span><span class="m">${esc(i.matched.join(", ") || "—")}${i.note ? ` · ${esc(i.note)}` : ""}</span></li>`)
      .join("")}</ul>
    ${r.method_only.length ? `<p class="flagnote">⚠ The cooking steps mention ${esc(r.method_only.join(", "))}, which the ingredient list doesn't include — a source-data error, so it's treated as a risk.</p>` : ""}
    <p class="sub">Highlighted lines break a constraint from the current conversation. Verdict for every supported diet:</p>
    <div class="verdicts">${verdicts}</div>
    ${r.url ? `<p><a class="link" href="${esc(r.url)}" target="_blank" rel="noopener">Original recipe ↗</a></p>` : ""}`;
  dialog.showModal();
});
dialog.addEventListener("click", (e) => {
  if (e.target === dialog || e.target.closest(".close")) dialog.close();
});

// ---------- stress test ----------
// What the Nosh Companion app's assistant answered (hands-on test, Nov 2026) vs what RecipeForge does.
const STRESS = [
  { qs: ["no onion no garlic vegetarian"], tag: "bad",
    nosh: "Suggested No Boil Methi Dal, though its own ingredient list includes garlic paste. It judged recipes by their names, not their ingredients.",
    here: "Checks every ingredient line, so garlic paste, onion powder and hidden onion (as in ketchup) are all ruled out." },
  { qs: ["Jain food options"], tag: "bad",
    nosh: "Listed 80+ recipes in no order, including Cheese Corn Maggi, whose masala usually contains onion and garlic.",
    here: "Rules out onion, garlic and root vegetables, including inside pastes and masalas, then shows 5 ranked dishes." },
  { qs: ["I'm lactose intolerant what can I make"], tag: "gap",
    nosh: "Understood the request, then listed recipes alphabetically (Aloo…, Aloo…, Andhra…) with no reasoning.",
    here: "Removes milk, curd, paneer and cream, marks ghee and butter as low-lactose risks, and ranks what's left." },
  { qs: ["vegetarian breakfast under 20 mins"], tag: "gap",
    nosh: "The app's veg / non-veg switch didn't change the assistant's answers.",
    here: "Applies both rules, vegetarian and under 20 minutes, to every result." },
  { qs: ["something for a dinner party for 6 people"], tag: "gap",
    nosh: "Gave a single recipe and left the scaling to you.",
    here: "Gives 5 options, each showing how much to scale it for 6 people." },
  { qs: ["suggest something different from what I usually make"], tag: "gap",
    nosh: "Answered confidently without knowing anything you've cooked.",
    here: "Learns from dishes you mark as cooked on this device, and says so honestly when it has no history yet." },
  { qs: ["what did I cook last week"], tag: "gap",
    nosh: "Had no cooking history to answer from.",
    here: "Answers from the dishes you've marked as cooked on this device." },
  { qs: ["plan my week, Jain dinners"], tag: "gap",
    nosh: "Showed a meal-planning calendar, but no plan.",
    here: "Plans 7 checked Jain dinners with no main ingredient repeated on back-to-back days, plus a shopping list." },
  { qs: ["I have chicken tomatoes and rice", "something similar but spicier", "make it under 15 minutes"], tag: "good",
    nosh: "Kept the context across all three messages and honestly said nothing fitted under 15 minutes.",
    here: "Does the same, and offers the closest dish when nothing fits the time limit." },
  { qs: ["I want butter chicken but I'm vegetarian"], tag: "good",
    nosh: "Suggested Paneer Butter Masala, which is good cooking sense.",
    here: "Makes the same swap (chicken → paneer) and says it did." },
];
const GROUPS = [
  { tag: "bad", title: "Unsafe answers", note: "Recommended food that breaks the diet asked for." },
  { tag: "gap", title: "Fell short", note: "Safe, but unhelpful or incomplete." },
  { tag: "good", title: "Handled well", note: "Nosh got these right too." },
];
$("#stress-summary").innerHTML = GROUPS.map((g) => `
  <div class="tile ${g.tag}"><b>${STRESS.filter((s) => s.tag === g.tag).length}</b><span>${g.title.toLowerCase()}</span></div>`).join("");
$("#stress-list").innerHTML = GROUPS.map((g) => `
  <section class="cmp-group ${g.tag}">
    <h3>${g.title}<span>${g.note}</span></h3>
    ${STRESS.map((s, i) => [s, i]).filter(([s]) => s.tag === g.tag).map(([s, i]) => `
      <article class="cmp">
        <div class="cmp-head">
          ${s.qs.length === 1
            ? `<q>${esc(s.qs[0])}</q>`
            : `<ol class="chain">${s.qs.map((q) => `<li><q>${esc(q)}</q></li>`).join("")}</ol>`}
          <button class="try" data-stress="${i}">Try it</button>
        </div>
        <div class="cmp-cols">
          <div class="col nosh"><span class="lbl">Nosh assistant</span><p>${esc(s.nosh)}</p></div>
          <div class="col ours"><span class="lbl">RecipeForge</span><p>${esc(s.here)}</p></div>
        </div>
      </article>`).join("")}
  </section>`).join("");
$("#stress-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-stress]");
  if (!b) return;
  await resetChat();
  showTab("chat");
  for (const q of STRESS[+b.dataset.stress].qs) await send(q);
});

// ---------- start: reopen the most recent chat on this device, or start a new one ----------
// A shared link like ?q=jain+dinner opens a fresh chat and runs that request.
(async () => {
  const q = new URLSearchParams(location.search).get("q");
  if (q && q.trim()) {
    await newChat({ carry: false });
    history.replaceState(null, "", location.pathname);
    send(q.trim().slice(0, 500));
    return;
  }
  const recent = loadChats()[0];
  if (recent) openChat(recent.id);
  else await newChat();
  openPageFromHash();
})();

// Shareable page links: /#stress and /#how open those pages directly.
function openPageFromHash() {
  const page = location.hash.slice(1);
  if (page === "stress" || page === "how") showTab(page);
}
window.addEventListener("hashchange", openPageFromHash);
