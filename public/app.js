const API = (window.API_BASE || "").replace(/\/$/, "");
let state = null;

// ---------- saved chats (this device only) ----------
const CKEY = "recipeforge.chats.v1";
const MAX_CHATS = 20;
const MAX_TURNS = 40;
let chatId = null;
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
    ? `Your history on this device: ${cooked} cooked, ${rated} rated · <button class="link" id="clear-history">clear</button>`
    : "Tap “Cooked it”, 👍 or 👎 on a dish and I'll learn your taste (stored only on this device).";
}

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, body) {
  const res = await fetch(API + path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
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
  const icon = c.status === "pass" ? "✓" : c.status === "flag" ? "⚠" : "✕";
  return `<span class="check ${esc(c.status)}" title="${esc(c.detail)}">${icon} ${esc(c.label)}</span>`;
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

function cardHtml(r) {
  const flagNotes = r.checks.filter((c) => c.status === "flag").map((c) => `${esc(c.label)}: ${esc(c.detail)}`);
  return `
    <article class="card">
      <header><h3>${r.flagged ? "⚠ " : ""}${esc(r.name)}</h3><span class="meta">${esc(r.total_min)} min</span></header>
      <div class="sub">${esc(r.cuisine)} · ${esc(r.course)}${r.servings ? ` · serves ${esc(r.servings)}` : ""}</div>
      ${r.checks.length ? `<div class="checks">${r.checks.map(checkBadge).join("")}</div>` : ""}
      ${r.why.length ? `<ul class="why">${r.why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>` : ""}
      ${flagNotes.length ? `<div class="flagnote">${flagNotes.join("<br>")}</div>` : ""}
      ${r.safer?.length ? `<details class="adapt"><summary>Safer swap</summary><ul>${swapList(r.safer)}</ul></details>` : ""}
      <div class="actions">
        <button class="link" data-recipe="${esc(r.id)}">Ingredient check</button>
        ${r.url ? `<a class="link" href="${esc(r.url)}" target="_blank" rel="noopener">Full recipe ↗</a>` : ""}
        <span class="taste">
          <button class="pill" data-act="cooked" data-id="${esc(r.id)}">Cooked it</button>
          <button class="pill" data-act="like" data-id="${esc(r.id)}" aria-label="Like">👍</button>
          <button class="pill" data-act="dislike" data-id="${esc(r.id)}" aria-label="Dislike">👎</button>
        </span>
      </div>
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
  if (data.mode === "plan" && data.results.length) html += planHtml(data);
  else if (data.results.length) html += `<div class="cards">${data.results.map(cardHtml).join("")}</div>`;
  if (data.closest) html += `<div class="cards">${cardHtml(data.closest)}</div>`;
  if (data.blocked_examples.length) {
    html += `<div class="ruled"><b>Ruled out — these matched your words but not your constraints:</b><ul>${data.blocked_examples
      .map((b) => `<li><button class="link" data-recipe="${esc(b.id)}">${esc(b.name)}</button> — ${esc(b.reason)}${adaptHtml(b.adaptation)}</li>`)
      .join("")}</ul></div>`;
  }
  const t = data.trace;
  const removed = Object.entries(t.removed_by_constraint || {}).map(([k, v]) => `  ${k}: ${v}`).join("\n") || "  (none)";
  if (data.mode === "search") {
    html += `<details class="trace"><summary>Pipeline trace</summary><pre>${esc(
      `parsed: ${JSON.stringify(data.parsed)}\n` +
      `corpus: ${t.corpus}\nremoved by hard constraints:\n${removed}\n` +
      `after hard filter: ${t.after_hard_filter}\nafter course/cuisine/pantry: ${t.after_soft_filters ?? "-"}\n` +
      `search text: "${t.query_text}"\nfused pool (BM25+dense, RRF): ${t.fused_pool ?? "-"}\n` +
      `cross-encoder reranked: ${t.reranked ?? "-"}${t.relaxed?.length ? `\nrelaxed: ${t.relaxed.join("; ")}` : ""}`
    )}</pre></details>`;
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
  if (!text) return;
  addUser(text);
  const typing = document.createElement("div");
  typing.className = "msg bot typing";
  typing.textContent = "Checking ingredients…";
  messages.appendChild(typing);
  typing.scrollIntoView({ behavior: "smooth", block: "end" });
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
    typing.textContent = `Something went wrong (${e.message}). Please try again.`;
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
  b.closest(".taste").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  b.textContent = b.dataset.act === "cooked" ? "Cooked ✓" : b.textContent;
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
{
  const recent = loadChats()[0];
  if (recent) openChat(recent.id);
  else newChat();
}
