const API_BASE =
  (window.location.hostname === "127.0.0.1" || window.location.hostname === "localhost") &&
  window.location.port === "5173"
    ? `http://${window.location.hostname}:8000`
    : "";

const SAMPLE_QUESTIONS = [
  "What does the corpus say about gestational diabetes screening?",
  "What is recommended for HIV pre-exposure prophylaxis?",
  "What fall prevention strategies are described for older adults?",
  "What does the sepsis early recognition bundle include?",
];

const elements = {
  assistantTitle: document.querySelector("#assistantTitle"),
  statusBadge: document.querySelector("#statusBadge"),
  statusDot: document.querySelector("#statusDot"),
  connectionStatus: document.querySelector("#connectionStatus"),
  uploadSection: document.querySelector("#uploadSection"),
  refreshButton: document.querySelector("#refreshButton"),
  fileInput: document.querySelector("#fileInput"),
  indexButton: document.querySelector("#indexButton"),
  askButton: document.querySelector("#askButton"),
  clearChatButton: document.querySelector("#clearChatButton"),
  questionInput: document.querySelector("#questionInput"),
  topKInput: document.querySelector("#topKInput"),
  poolInput: document.querySelector("#poolInput"),
  minInput: document.querySelector("#minInput"),
  retrievalModeInput: document.querySelector("#retrievalModeInput"),
  modeGrid: document.querySelector("#modeGrid"),
  toggleHybrid: document.querySelector("#toggleHybrid"),
  toggleRerank: document.querySelector("#toggleRerank"),
  toggleExpand: document.querySelector("#toggleExpand"),
  debuggerQuery: document.querySelector("#debuggerQuery"),
  debuggerMode: document.querySelector("#debuggerMode"),
  debuggerTopK: document.querySelector("#debuggerTopK"),
  debuggerPool: document.querySelector("#debuggerPool"),
  featurePills: document.querySelector("#featurePills"),
  chunkScores: document.querySelector("#chunkScores"),
  alertArea: document.querySelector("#alertArea"),
  documentCount: document.querySelector("#documentCount"),
  indexState: document.querySelector("#indexState"),
  embeddingModel: document.querySelector("#embeddingModel"),
  llmModel: document.querySelector("#llmModel"),
  traceProject: document.querySelector("#traceProject"),
  chatMessages: document.querySelector("#chatMessages"),
  suggestedQuestions: document.querySelector("#suggestedQuestions"),
  latencyLabel: document.querySelector("#latencyLabel"),
  sourceCount: document.querySelector("#sourceCount"),
  rewrittenQueryLabel: document.querySelector("#rewrittenQueryLabel"),
  sourcesList: document.querySelector("#sourcesList"),
};

let chatHistory = [];
let lastQuery = "";

function icons() {
  if (window.lucide) window.lucide.createIcons();
}

function getUiMode() {
  const active = elements.modeGrid.querySelector(".mode-chip.is-active");
  return active?.dataset.uiMode || "adaptive";
}

function getBaseRetrievalMode() {
  const active = elements.modeGrid.querySelector(".mode-chip.is-active");
  return active?.dataset.mode || "adaptive";
}

function isFeatureOn(button) {
  return button.classList.contains("is-on");
}

function resolveRetrievalMode() {
  const base = getBaseRetrievalMode();
  if (base === "adaptive" || base === "corrective" || base === "advanced") return base;
  if (isFeatureOn(elements.toggleRerank) || isFeatureOn(elements.toggleExpand)) return "advanced";
  return "naive";
}

function getRetrievalSettings() {
  const topK = Number(elements.topKInput.value || 5);
  const pool = Number(elements.poolInput.value || 40);
  const minScore = Number(elements.minInput.value || 0.1);
  const expand = isFeatureOn(elements.toggleExpand) || getUiMode() === "advanced";
  return {
    top_k: topK,
    pool_size: expand ? pool : Math.max(topK, Math.min(pool, topK * 4)),
    min_score: minScore,
    retrieval_mode: resolveRetrievalMode(),
  };
}

function setFeature(button, on) {
  button.classList.toggle("is-on", on);
}

function setRetrievalModeUi(uiMode) {
  for (const chip of elements.modeGrid.querySelectorAll(".mode-chip")) {
    chip.classList.toggle("is-active", chip.dataset.uiMode === uiMode);
  }
  if (uiMode === "advanced") {
    setFeature(elements.toggleRerank, true);
    setFeature(elements.toggleExpand, true);
  }
  elements.retrievalModeInput.value = resolveRetrievalMode();
  updateDebugger();
}

function syncPills() {
  const flags = {
    hybrid: isFeatureOn(elements.toggleHybrid),
    rerank: isFeatureOn(elements.toggleRerank),
    expand: isFeatureOn(elements.toggleExpand),
  };
  for (const pill of elements.featurePills.querySelectorAll(".status-pill")) {
    pill.classList.toggle("is-on", flags[pill.dataset.pill]);
  }
}

function updateDebugger(query = lastQuery) {
  const settings = getRetrievalSettings();
  elements.retrievalModeInput.value = settings.retrieval_mode;
  elements.debuggerMode.textContent = getUiMode();
  elements.debuggerTopK.textContent = String(settings.top_k);
  elements.debuggerPool.textContent = String(settings.pool_size);
  elements.debuggerQuery.textContent = query
    ? `Analyzing chunk scores for: "${query}"`
    : "Waiting for a query…";
  syncPills();
}

function shortSourceName(path) {
  if (!path) return "Unknown source";
  const name = String(path).split(/[/\\]/).pop() || path;
  return name.replace(/^hc_\d+_/, "").replace(/_/g, " ").replace(/\.txt$/i, "");
}

function relevancePercent(score) {
  const rel = 1 / (1 + Math.max(Number(score) || 0, 0));
  return Math.min(100, Math.round(rel * 100));
}

function setOnline(online, label) {
  elements.statusBadge.classList.toggle("is-online", online);
  elements.statusBadge.classList.toggle("is-offline", !online);
  elements.connectionStatus.textContent = label;
}

function setBusy(busy) {
  elements.askButton.disabled = busy;
}

function showToast(message, type = "info") {
  elements.alertArea.textContent = message;
  elements.alertArea.className = `toast ${type}`;
  elements.alertArea.hidden = false;
  if (type === "info") {
    setTimeout(() => {
      if (elements.alertArea.textContent === message) clearToast();
    }, 5000);
  }
}

function clearToast() {
  elements.alertArea.hidden = true;
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

async function refreshStatus() {
  try {
    const status = await requestJson("/api/status");
    elements.assistantTitle.textContent = status.assistant_title || "Healthcare RAG";
    setOnline(
      true,
      status.corpus_ready ? `${status.documents} docs · index ready` : "Indexing corpus…",
    );
    elements.documentCount.textContent = String(status.documents);
    elements.indexState.textContent = status.index_exists ? "Ready" : "Building";
    elements.indexState.classList.toggle("is-ready", status.index_exists);
    elements.uploadSection.hidden = status.allow_uploads === false;
    elements.embeddingModel.textContent = `${status.embedding_provider} / ${status.embedding_model}`;
    elements.llmModel.textContent = `${status.llm_provider} / ${status.llm_model}`;
    elements.traceProject.textContent = status.langsmith_project || "—";
    clearToast();
  } catch (error) {
    setOnline(false, "Backend offline");
    showToast(error.message, "error");
  }
  icons();
}

function renderSuggestions() {
  elements.suggestedQuestions.innerHTML = "";
  if (chatHistory.length > 0) {
    elements.suggestedQuestions.hidden = true;
    return;
  }
  elements.suggestedQuestions.hidden = false;
  for (const q of SAMPLE_QUESTIONS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion-chip";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      elements.questionInput.value = q;
      elements.questionInput.focus();
      autoResizeTextarea();
      askQuestion();
    });
    elements.suggestedQuestions.appendChild(btn);
  }
}

function renderWelcome() {
  elements.chatMessages.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon"><i data-lucide="stethoscope"></i></div>
      <h3>Healthcare knowledge assistant</h3>
      <p>Ask clinical questions. Answers are grounded in 55+ indexed reference documents with cited sources.</p>
      <div class="suggestions" id="welcomeSuggestions"></div>
    </div>`;
  const container = elements.chatMessages.querySelector("#welcomeSuggestions");
  for (const q of SAMPLE_QUESTIONS) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion-chip";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      elements.questionInput.value = q;
      askQuestion();
    });
    container.appendChild(btn);
  }
  icons();
}

function renderChat() {
  if (!chatHistory.length) {
    renderWelcome();
    renderSuggestions();
    return;
  }

  elements.chatMessages.innerHTML = "";
  for (const message of chatHistory) {
    appendMessage(message);
  }
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  renderSuggestions();
  icons();
}

function appendMessage(message) {
  const isUser = message.role === "user";
  const isThinking = message.content === "__thinking__";

  const wrap = document.createElement("div");
  wrap.className = `msg ${message.role}${isThinking ? " thinking" : ""}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.innerHTML = isUser
    ? '<i data-lucide="user"></i>'
    : '<i data-lucide="bot"></i>';

  const body = document.createElement("div");
  body.className = "msg-body";

  const label = document.createElement("span");
  label.className = "msg-label";
  label.textContent = isUser ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  if (isThinking) {
    bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
  } else {
    bubble.textContent = message.content;
  }

  body.append(label, bubble);
  wrap.append(avatar, body);
  elements.chatMessages.appendChild(wrap);
}

function addThinkingMessage() {
  appendMessage({ role: "assistant", content: "__thinking__" });
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  icons();
  return elements.chatMessages.querySelector(".msg.thinking");
}

function removeThinkingMessage(node) {
  node?.remove();
}

function formatModeLabel(response) {
  const uiMode = getUiMode();
  if (uiMode === "adaptive" && response.routed_mode) return `Adaptive → ${response.routed_mode}`;
  if (uiMode === "standard" && response.retrieval_mode === "advanced") return "Standard (enhanced)";
  return uiMode.charAt(0).toUpperCase() + uiMode.slice(1);
}

async function askQuestion() {
  const question = elements.questionInput.value.trim();
  if (!question) {
    showToast("Type a question first.", "error");
    elements.questionInput.focus();
    return;
  }

  const settings = getRetrievalSettings();
  lastQuery = question;
  updateDebugger(question);

  setBusy(true);
  clearToast();
  elements.questionInput.value = "";
  autoResizeTextarea();

  elements.rewrittenQueryLabel.hidden = true;
  elements.latencyLabel.textContent = "Retrieving…";

  chatHistory.push({ role: "user", content: question });
  renderChat();
  const thinkingNode = addThinkingMessage();

  try {
    const response = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatHistory,
        top_k: settings.top_k,
        pool_size: settings.pool_size,
        min_score: settings.min_score,
        retrieval_mode: settings.retrieval_mode,
      }),
    });

    chatHistory.push(response.message);
    removeThinkingMessage(thinkingNode);
    renderChat();

    elements.latencyLabel.textContent = `${formatModeLabel(response)} · ${Math.round(response.latency_ms)} ms`;

    if (response.rewritten_query) {
      elements.rewrittenQueryLabel.textContent = `Query rewritten: "${response.rewritten_query}"`;
      elements.rewrittenQueryLabel.hidden = false;
    }

    renderSources(response.documents || []);
    renderChunkScores(response.documents || []);
  } catch (error) {
    removeThinkingMessage(thinkingNode);
    showToast(error.message, "error");
    elements.latencyLabel.textContent = "";
  } finally {
    setBusy(false);
    elements.questionInput.focus();
  }
}

function renderChunkScores(documents) {
  elements.chunkScores.innerHTML = "";
  for (const doc of documents) {
    const row = document.createElement("div");
    row.className = "score-row";
    const rel = relevancePercent(doc.score);
    row.innerHTML = `<span class="score-name">#${doc.rank} ${escapeHtml(shortSourceName(doc.metadata?.source))}</span><span class="score-val">${rel}%</span>`;
    elements.chunkScores.appendChild(row);
  }
}

function renderSources(documents) {
  const count = documents.length;
  elements.sourceCount.textContent = count === 1 ? "1 source" : `${count} sources`;

  if (!count) {
    elements.sourcesList.innerHTML = `
      <div class="empty-sources">
        <i data-lucide="file-search"></i>
        <p>No passages matched this query. Try rephrasing or switch retrieval mode.</p>
      </div>`;
    icons();
    return;
  }

  elements.sourcesList.innerHTML = "";
  for (const doc of documents) {
    const rel = relevancePercent(doc.score);
    const card = document.createElement("article");
    card.className = "source-card";
    card.innerHTML = `
      <div class="source-top">
        <span class="source-rank">${doc.rank}</span>
        <span class="source-name">${escapeHtml(shortSourceName(doc.metadata?.source))}</span>
        <span class="source-score">${rel}% match</span>
      </div>
      <div class="score-bar"><div class="score-bar-fill" style="width:${rel}%"></div></div>
      <p class="source-excerpt"></p>`;
    card.querySelector(".source-excerpt").textContent = doc.content;
    elements.sourcesList.appendChild(card);
  }
  icons();
}

function clearChat() {
  chatHistory = [];
  lastQuery = "";
  elements.latencyLabel.textContent = "";
  elements.rewrittenQueryLabel.hidden = true;
  elements.chunkScores.innerHTML = "";
  elements.sourceCount.textContent = "0 sources";
  elements.sourcesList.innerHTML = `
    <div class="empty-sources">
      <i data-lucide="file-search"></i>
      <p>Retrieved passages will appear here after you ask a question.</p>
    </div>`;
  updateDebugger("");
  renderChat();
  icons();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function autoResizeTextarea() {
  const el = elements.questionInput;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
}

function toggleFeature(button) {
  button.classList.toggle("is-on");
  elements.retrievalModeInput.value = resolveRetrievalMode();
  updateDebugger(lastQuery);
}

elements.modeGrid.addEventListener("click", (e) => {
  const chip = e.target.closest(".mode-chip");
  if (!chip) return;
  setRetrievalModeUi(chip.dataset.uiMode);
});

elements.toggleHybrid.addEventListener("click", () => toggleFeature(elements.toggleHybrid));
elements.toggleRerank.addEventListener("click", () => toggleFeature(elements.toggleRerank));
elements.toggleExpand.addEventListener("click", () => toggleFeature(elements.toggleExpand));

for (const input of [elements.topKInput, elements.poolInput, elements.minInput]) {
  input.addEventListener("input", () => updateDebugger(lastQuery));
}

elements.refreshButton.addEventListener("click", refreshStatus);
elements.askButton.addEventListener("click", askQuestion);
elements.clearChatButton.addEventListener("click", clearChat);

elements.questionInput.addEventListener("input", autoResizeTextarea);
elements.questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});

if (elements.indexButton) {
  elements.indexButton.addEventListener("click", async () => {
    try {
      await requestJson("/api/index", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: true }),
      });
      await refreshStatus();
    } catch (err) {
      showToast(err.message, "error");
    }
  });
}

setRetrievalModeUi("adaptive");
renderChat();
refreshStatus();
