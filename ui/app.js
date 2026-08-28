/* insight-rag chat client.
 *
 * Reads the SSE stream from /api/chat/stream and renders three things the
 * answer text alone cannot convey: which passages were retrieved and how
 * they scored, which passages the answer actually cited, and what the turn
 * cost in latency. Citation markers are clickable and show the source text,
 * so a reader can verify a claim without leaving the page.
 */

const el = (id) => document.getElementById(id);
const thread = el("thread");
const composer = el("composer");
const questionInput = el("question");
const sendButton = el("send");
const retrievedList = el("retrieved");
const popover = el("popover");

let history = [];
let busy = false;
let citationStore = {};

/* ---------- health ---------- */
async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const body = await response.json();
    el("h-status").textContent = body.status;
    document.querySelector("#health .dot").className = "dot ok";
    el("h-llm").textContent = body.llm_provider;
    el("h-emb").textContent = body.embedding_provider;
    el("h-store").textContent = body.vector_store;
    el("h-rerank").textContent = body.reranker;
    el("h-corpus").textContent = `${body.documents} docs · ${body.chunks} chunks`;
  } catch {
    el("h-status").textContent = "unreachable";
    document.querySelector("#health .dot").className = "dot bad";
  }
}

/* ---------- rendering ---------- */
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

// Turn [n] markers into clickable chips. Escaping happens first, so document
// text can never inject markup.
function renderAnswer(text) {
  return escapeHtml(text).replace(
    /\[(\d+)\]/g,
    (match, marker) => `<span class="cite" data-marker="${marker}">${marker}</span>`
  );
}

function renderRetrieved(items) {
  if (!items.length) {
    retrievedList.innerHTML = '<li class="empty">Nothing retrieved for this question.</li>';
    return;
  }
  const top = Math.max(...items.map((i) => i.score)) || 1;
  retrievedList.innerHTML = items
    .map(
      (item) => `
      <li>
        <div class="r-top">
          <span class="r-label">${escapeHtml(item.label)}</span>
          <span class="r-score">${item.score.toFixed(3)}</span>
        </div>
        <div class="r-bar"><span style="width:${(item.score / top) * 100}%"></span></div>
      </li>`
    )
    .join("");
}

function metaChip(label, klass = "") {
  return `<span class="${klass}">${escapeHtml(label)}</span>`;
}

/* ---------- conversation ---------- */
function addTurn(question) {
  const wrapper = document.createElement("div");
  wrapper.className = "turn";
  wrapper.innerHTML = `
    <div class="bubble-user">${escapeHtml(question)}</div>
    <div class="bubble-bot">
      <div class="answer cursor"></div>
      <div class="sources" hidden></div>
      <div class="meta"></div>
    </div>`;
  thread.appendChild(wrapper);
  thread.scrollTop = thread.scrollHeight;
  return {
    answer: wrapper.querySelector(".answer"),
    sources: wrapper.querySelector(".sources"),
    meta: wrapper.querySelector(".meta"),
  };
}

async function ask(question) {
  if (busy) return;
  busy = true;
  sendButton.disabled = true;
  questionInput.value = "";
  document.querySelector(".welcome")?.remove();

  const nodes = addTurn(question);
  let buffer = "";

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let pending = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      pending += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line; keep any partial tail.
      const frames = pending.split("\n\n");
      pending = frames.pop() ?? "";

      for (const frame of frames) {
        const line = frame.trim();
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6);
        if (payload === "[DONE]") continue;

        const event = JSON.parse(payload);

        if (event.type === "meta") {
          renderRetrieved(event.retrieved || []);
        } else if (event.type === "token") {
          buffer += event.text;
          nodes.answer.innerHTML = renderAnswer(buffer);
          thread.scrollTop = thread.scrollHeight;
        } else if (event.type === "done") {
          nodes.answer.classList.remove("cursor");
          if (!event.grounded) nodes.answer.classList.add("refused");

          for (const citation of event.citations || []) {
            citationStore[citation.marker] = citation;
          }

          if (event.citations?.length) {
            nodes.sources.hidden = false;
            nodes.sources.innerHTML =
              "<h3>Sources cited</h3>" +
              event.citations
                .map(
                  (c) => `
                  <div class="source">
                    <span class="s-marker">[${c.marker}]</span>
                    <span class="s-text"><b>${escapeHtml(c.label)}</b> — ${escapeHtml(
                      c.snippet.slice(0, 180)
                    )}…</span>
                  </div>`
                )
                .join("");
          }

          const timings = event.timings_ms || {};
          nodes.meta.innerHTML = [
            metaChip(
              event.grounded ? "grounded" : "refused — not in corpus",
              event.grounded ? "good" : "warn"
            ),
            timings.retrieve !== undefined ? metaChip(`retrieve ${timings.retrieve}ms`) : "",
            timings.generate !== undefined ? metaChip(`generate ${timings.generate}ms`) : "",
            timings.total !== undefined ? metaChip(`total ${timings.total}ms`) : "",
            event.request_id ? metaChip(event.request_id) : "",
          ]
            .filter(Boolean)
            .join("");

          history.push({ role: "user", content: question });
          history.push({ role: "assistant", content: buffer });
          history = history.slice(-10);
        }
      }
    }
  } catch (error) {
    nodes.answer.classList.remove("cursor");
    nodes.answer.classList.add("refused");
    nodes.answer.textContent = `Request failed: ${error.message}`;
  } finally {
    busy = false;
    sendButton.disabled = false;
    questionInput.focus();
  }
}

/* ---------- citation popover ---------- */
document.addEventListener("click", (event) => {
  const chip = event.target.closest(".cite");
  if (chip) {
    const citation = citationStore[chip.dataset.marker];
    if (!citation) return;
    el("pop-label").textContent = citation.label;
    el("pop-snippet").textContent = citation.snippet;
    el("pop-score").textContent = citation.score.toFixed(3);
    popover.hidden = false;

    const rect = chip.getBoundingClientRect();
    popover.style.left = `${Math.min(rect.left, window.innerWidth - 440)}px`;
    popover.style.top = `${Math.min(rect.bottom + 8, window.innerHeight - 240)}px`;
    return;
  }
  if (!event.target.closest(".citation-popover")) popover.hidden = true;
});

el("pop-close").addEventListener("click", () => {
  popover.hidden = true;
});

/* ---------- wiring ---------- */
composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) ask(question);
});

for (const button of document.querySelectorAll(".suggestion")) {
  button.addEventListener("click", () => ask(button.dataset.q));
}

loadHealth();
