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
    ? `Learning your taste from ${cooked} cooked and ${rated} rated on this device · <button class="link" id="clear-history">Clear</button>`
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
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${name}`));
  if (name === "audit") loadAudit();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------- chat ----------
const messages = $("#messages");

function addUser(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.textContent = text;
  messages.appendChild(el);
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
  const list = s ? `<details class="shopping" open><summary>Shopping list (${s.items.length} items)</summary>
      <ul>${s.items.map((x) => `<li><b>${esc(x.item)}</b> <span class="sub">— ${esc(x.for.join(", "))}</span></li>`).join("")}</ul>
      <div class="sub">Pantry staples assumed: ${esc(s.staples.join(", "))}</div></details>` : "";
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
  typing.textContent = "Checking ingredients…";
  messages.appendChild(typing);
  typing.scrollIntoView({ behavior: "smooth", block: "end" });
  const slow = setTimeout(() => { typing.textContent = "Reading your request carefully…"; }, 1200);
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
messages.addEventListener("click", (e) => {
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
  messages.querySelectorAll(".msg:not(.intro)").forEach((m) => m.remove());
}

// Start a new chat. Diets and allergy exclusions from the most recent chat carry over (remove with ×).
async function newChat({ carry = true } = {}) {
  chatId = newId();
  state = null;
  carried = [];
  clearMessages();
  renderChips([]);
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
$("#reset").addEventListener("click", () => newChat());

// ---------- chat list ----------
const chatsDialog = $("#chats-dialog");
function renderChatList() {
  const chats = loadChats();
  $("#chats-list").innerHTML = chats.length
    ? chats.map((c) => `<li class="${c.id === chatId ? "current" : ""}">
        <button class="open" data-open="${esc(c.id)}"><b>${esc(c.title)}</b><span class="sub">${esc(relTime(c.updated))} · ${Math.ceil(c.turns.length / 2)} message${c.turns.length > 2 ? "s" : ""}${(c.chips || []).filter((x) => x.hard).length ? ` · ${esc(c.chips.filter((x) => x.hard).map((x) => x.label).join(", "))}` : ""}</span></button>
        <button class="del" data-del="${esc(c.id)}" aria-label="Delete chat">×</button></li>`).join("")
    : `<li class="sub">No saved chats yet.</li>`;
}
$("#chats-btn").addEventListener("click", () => { renderChatList(); chatsDialog.showModal(); });
chatsDialog.addEventListener("click", (e) => {
  if (e.target === chatsDialog || e.target.closest(".close")) return chatsDialog.close();
  if (e.target.closest("#new-chat")) { chatsDialog.close(); showTab("chat"); return newChat(); }
  const open = e.target.closest("[data-open]");
  if (open) { chatsDialog.close(); return openChat(open.dataset.open); }
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
const STRESS = [
  { qs: ["no onion no garlic vegetarian"], tag: "bad", obs: "Returned No Boil Methi Dal, whose own ingredient list includes garlic paste. Compliance was inferred from recipe names, not ingredients." },
  { qs: ["Jain food options"], tag: "bad", obs: "Returned an unranked list of 80+ recipes, including Cheese Corn Maggi — instant-noodle tastemakers typically list onion and garlic powder." },
  { qs: ["I'm lactose intolerant what can I make"], tag: "gap", obs: "Recognised lactose intolerance, then listed recipes alphabetically (Aloo…, Aloo…, Andhra…) with no ranking or reasoning." },
  { qs: ["vegetarian breakfast under 20 mins"], tag: "gap", obs: "The in-app veg / non-veg toggle did not change the assistant's answers." },
  { qs: ["I have chicken tomatoes and rice", "something similar but spicier", "make it under 15 minutes"], tag: "good", obs: "Nosh handled this chain well: it kept context and honestly reported nothing under 15 minutes." },
  { qs: ["I want butter chicken but I'm vegetarian"], tag: "good", obs: "Nosh suggested Paneer Butter Masala — good culinary reasoning." },
  { qs: ["something for a dinner party for 6 people"], tag: "gap", obs: "Returned a single recipe and asked the user to scale it themselves." },
  { qs: ["suggest something different from what I usually make"], tag: "gap", obs: "Answered confidently with no cooking history to compare against. Here, tap “Cooked it” on a few dishes first — it then avoids your usual cuisines and ingredients, and says so honestly when it has no history." },
  { qs: ["what did I cook last week"], tag: "gap", obs: "Had no history to answer from. Here it answers from dishes you've marked as cooked on this device." },
  { qs: ["plan my week, Jain dinners"], tag: "gap", obs: "Offered a meal-planning calendar but no plan. Here: 7 validated dinners, no main ingredient repeated on consecutive days, plus a shopping list." },
];
$("#stress-list").innerHTML = STRESS.map((s, i) => `
  <div class="item">
    <div class="q">${s.qs.map((q) => `<code>${esc(q)}</code>`).join(" → ")}</div>
    <div class="obs"><span class="tag ${s.tag}">${{ bad: "failure", gap: "gap", good: "handled" }[s.tag]}</span>${esc(s.obs)}</div>
    <button class="link" data-stress="${i}">Run here →</button>
  </div>`).join("");
$("#stress-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-stress]");
  if (!b) return;
  await resetChat();
  showTab("chat");
  for (const q of STRESS[+b.dataset.stress].qs) await send(q);
});

// ---------- audit ----------
let auditLoaded = false;
async function loadAudit() {
  if (auditLoaded) return;
  auditLoaded = true;
  const a = await api("/api/audit");
  const rows = Object.entries(a.labels).map(([label, v]) => `
    <tr><td>${esc(label)}</td><td class="num">${v.violations} / ${v.labelled}</td>
    <td><ul>${v.examples.slice(0, 4).map((x) => `<li><button class="link" data-recipe="${esc(x.id)}">${esc(x.name)}</button>: ${esc(x.reasons[0])}</li>`).join("")}</ul></td></tr>`).join("");
  const total = Object.values(a.labels).reduce((s, v) => s + v.violations, 0);
  $("#audit").innerHTML = `
    <p><b>${total}</b> recipes carry a diet label that their own ingredient list contradicts.</p>
    <div class="audit-wrap"><table class="audit"><thead><tr><th>Source label</th><th>Contradicted</th><th>Examples (ingredient → reason)</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

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
  else newChat();
})();
