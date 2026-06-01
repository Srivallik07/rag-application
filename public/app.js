const API_BASE =
  window.location.hostname === "127.0.0.1" && window.location.port === "5173"
    ? "http://127.0.0.1:8000"
    : "";

const elements = {
  connectionStatus: document.querySelector("#connectionStatus"),
  refreshButton: document.querySelector("#refreshButton"),
  fileInput: document.querySelector("#fileInput"),
  indexButton: document.querySelector("#indexButton"),
  askButton: document.querySelector("#askButton"),
  clearChatButton: document.querySelector("#clearChatButton"),
  questionInput: document.querySelector("#questionInput"),
  topKInput: document.querySelector("#topKInput"),
  retrievalModeInput: document.querySelector("#retrievalModeInput"),
  alertArea: document.querySelector("#alertArea"),
  documentCount: document.querySelector("#documentCount"),
  indexState: document.querySelector("#indexState"),
  embeddingModel: document.querySelector("#embeddingModel"),
  llmModel: document.querySelector("#llmModel"),
  traceProject: document.querySelector("#traceProject"),
  chatMessages: document.querySelector("#chatMessages"),
  latencyLabel: document.querySelector("#latencyLabel"),
  sourceCount: document.querySelector("#sourceCount"),
  sourcesList: document.querySelector("#sourcesList"),
};

let chatHistory = [];

function setBusy(button, busy) {
  button.disabled = busy;
}

function showAlert(message, type = "info") {
  elements.alertArea.textContent = message;
  elements.alertArea.className = `alert ${type}`;
  elements.alertArea.hidden = false;
}

function clearAlert() {
  elements.alertArea.hidden = true;
  elements.alertArea.textContent = "";
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  const text = await response.text();
  const payload = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

async function refreshStatus() {
  try {
    const status = await requestJson("/api/status");
    elements.connectionStatus.textContent = "Backend online";
    elements.documentCount.textContent = String(status.documents);
    elements.indexState.textContent = status.index_exists ? "Ready" : "No";
    elements.embeddingModel.textContent = `${status.embedding_provider}: ${status.embedding_model}`;
    elements.llmModel.textContent = `${status.llm_provider}: ${status.llm_model}`;
    elements.traceProject.textContent = status.langsmith_project || "-";
    clearAlert();
  } catch (error) {
    elements.connectionStatus.textContent = "Backend offline";
    showAlert(error.message, "error");
  }
}

async function uploadFiles(files) {
  if (!files.length) return;
  const body = new FormData();
  for (const file of files) {
    body.append("files", file);
  }
  showAlert("Uploading documents...");
  await requestJson("/api/documents/upload", { method: "POST", body });
  showAlert("Documents uploaded. Build the index before asking questions.");
  await refreshStatus();
}

async function indexDocuments() {
  setBusy(elements.indexButton, true);
  showAlert("Indexing documents...");
  try {
    const result = await requestJson("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true }),
    });
    showAlert(`Indexed ${result.chunks} chunks in ${result.latency_ms} ms.`);
    await refreshStatus();
  } catch (error) {
    showAlert(error.message, "error");
  } finally {
    setBusy(elements.indexButton, false);
  }
}

async function askQuestion() {
  const question = elements.questionInput.value.trim();
  if (!question) {
    showAlert("Enter a question first.", "error");
    return;
  }

  setBusy(elements.askButton, true);
  clearAlert();
  elements.questionInput.value = "";
  elements.sourcesList.innerHTML = "";
  elements.sourceCount.textContent = "0 retrieved";
  elements.latencyLabel.textContent = "";
  chatHistory.push({ role: "user", content: question });
  renderChat();
  const thinkingIndex = addThinkingMessage();

  try {
    const response = await requestJson("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: chatHistory,
        top_k: Number(elements.topKInput.value || 5),
        retrieval_mode: elements.retrievalModeInput.value,
      }),
    });
    chatHistory.push(response.message);
    removeThinkingMessage(thinkingIndex);
    renderChat();
    elements.latencyLabel.textContent = `${response.retrieval_mode || "naive"} - ${Math.round(response.latency_ms)} ms`;
    renderSources(response.documents || []);
  } catch (error) {
    removeThinkingMessage(thinkingIndex);
    showAlert(error.message, "error");
  } finally {
    setBusy(elements.askButton, false);
  }
}

function renderChat() {
  elements.chatMessages.innerHTML = "";
  if (!chatHistory.length) {
    appendMessage({
      role: "assistant",
      content: "Upload or index your documents, then ask me questions. I will answer from the retrieved sources.",
    });
    return;
  }
  for (const message of chatHistory) {
    appendMessage(message);
  }
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function appendMessage(message) {
  const bubble = window.document.createElement("div");
  bubble.className = `chat-message ${message.role}`;

  const label = window.document.createElement("span");
  label.className = "chat-role";
  label.textContent = message.role === "user" ? "You" : "Assistant";

  const content = window.document.createElement("div");
  content.className = "chat-content";
  content.textContent = message.content;

  bubble.append(label, content);
  elements.chatMessages.appendChild(bubble);
}

function addThinkingMessage() {
  const thinking = { role: "assistant", content: "Thinking..." };
  appendMessage(thinking);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  return elements.chatMessages.children.length - 1;
}

function removeThinkingMessage(index) {
  const node = elements.chatMessages.children[index];
  if (node) {
    node.remove();
  }
}

function clearChat() {
  chatHistory = [];
  elements.latencyLabel.textContent = "";
  elements.sourcesList.innerHTML = "";
  elements.sourceCount.textContent = "0 retrieved";
  renderChat();
}

function renderSources(documents) {
  elements.sourceCount.textContent = `${documents.length} retrieved`;
  elements.sourcesList.innerHTML = "";

  for (const doc of documents) {
    const item = window.document.createElement("article");
    item.className = "source-item";

    const meta = window.document.createElement("div");
    meta.className = "source-meta";
    const source = doc.metadata?.source || "unknown";
    meta.innerHTML = `<span>#${doc.rank} ${escapeHtml(source)}</span><span>score ${Number(doc.score).toFixed(4)}</span>`;

    const content = window.document.createElement("div");
    content.className = "source-content";
    content.textContent = doc.content;

    item.append(meta, content);
    elements.sourcesList.appendChild(item);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

elements.refreshButton.addEventListener("click", refreshStatus);
elements.indexButton.addEventListener("click", indexDocuments);
elements.askButton.addEventListener("click", askQuestion);
elements.clearChatButton.addEventListener("click", clearChat);
elements.fileInput.addEventListener("change", async (event) => {
  try {
    await uploadFiles([...event.target.files]);
  } catch (error) {
    showAlert(error.message, "error");
  } finally {
    elements.fileInput.value = "";
  }
});
elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    askQuestion();
  }
});

if (window.lucide) {
  window.lucide.createIcons();
}

renderChat();
refreshStatus();
