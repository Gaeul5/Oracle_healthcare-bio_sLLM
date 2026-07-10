const API_BASE_URL = "/api";

// ============================================================
// 상태
// ============================================================
let accessToken = localStorage.getItem("access_token");
let currentUser = null;
let currentSessionId = null;

// ============================================================
// DOM
// ============================================================
const mainNav = document.getElementById("main-nav");
const userInfo = document.getElementById("user-info");
const userNameLabel = document.getElementById("user-name-label");
const logoutButton = document.getElementById("logout-button");

const authView = document.getElementById("auth-view");
const goToRegisterLink = document.getElementById("go-to-register");
const goToLoginLink = document.getElementById("go-to-login");
const loginForm = document.getElementById("login-form");
const registerForm = document.getElementById("register-form");
const authStatus = document.getElementById("auth-status");

const chatMenu = document.getElementById("chat-menu");
const documentsMenu = document.getElementById("documents-menu");
const chatView = document.getElementById("chat-view");
const documentsView = document.getElementById("documents-view");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSubmitButton = chatForm.querySelector("button[type='submit']");
const chatMessagesEl = document.getElementById("chat-messages");
const currentSessionTitleEl = document.getElementById("current-session-title");
const sessionListEl = document.getElementById("session-list");
const newSessionButton = document.getElementById("new-session-button");
const sessionSidebar = document.getElementById("session-sidebar");
const toggleSidebarButton = document.getElementById("toggle-sidebar-button");
const chatModeSelect = document.getElementById("chat-mode-select");
const explanationLevelSelect = document.getElementById("explanation-level-select");

const uploadForm = document.getElementById("upload-form");
const documentTitleInput = document.getElementById("document-title");
const documentFileInput = document.getElementById("document-file");
const documentsStatus = document.getElementById("documents-status");
const documentsTableBody = document.getElementById("documents-table-body");

// ============================================================
// 공용 유틸
// ============================================================
function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// **굵게**, *기울임*만 처리하는 가벼운 inline 변환입니다.
function formatInline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

// 챗봇 답변을 문단(<p>)/목록(<ul>,<ol>)으로 나눠서 HTML로 바꿉니다.
// 반드시 escapeHtml을 먼저 적용한 텍스트에만 사용해야 안전합니다 (XSS 방지).
function renderMarkdownLite(rawText) {
  const lines = escapeHtml(rawText).split("\n");
  const htmlParts = [];
  let paragraphLines = [];
  let listItems = [];
  let listTag = null;

  function flushParagraph() {
    if (paragraphLines.length > 0) {
      htmlParts.push(`<p>${paragraphLines.join("<br>")}</p>`);
      paragraphLines = [];
    }
  }

  function flushList() {
    if (listItems.length > 0) {
      htmlParts.push(`<${listTag}>${listItems.join("")}</${listTag}>`);
      listItems = [];
      listTag = null;
    }
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    const bulletMatch = line.match(/^[-*]\s+(.*)/);
    const numberedMatch = line.match(/^\d+\.\s+(.*)/);

    if (bulletMatch || numberedMatch) {
      flushParagraph();
      const nextTag = bulletMatch ? "ul" : "ol";
      if (listTag && listTag !== nextTag) flushList();
      listTag = nextTag;
      listItems.push(`<li>${formatInline((bulletMatch || numberedMatch)[1])}</li>`);
      return;
    }

    flushList();

    if (line === "") {
      flushParagraph();
      return;
    }

    paragraphLines.push(formatInline(line));
  });

  flushList();
  flushParagraph();

  return htmlParts.join("");
}

function getChatMode() {
  return chatModeSelect ? chatModeSelect.value : "manual";
}

function getExplanationLevel() {
  return explanationLevelSelect ? explanationLevelSelect.value : "friendly";
}

toggleSidebarButton.addEventListener("click", () => {
  sessionSidebar.classList.toggle("collapsed");
});

// 로그인 토큰을 자동으로 붙여주는 fetch 래퍼입니다.
// 401(인증 실패)이 오면 토큰을 지우고 로그인 화면으로 돌려보냅니다.
async function authFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (accessToken) {
    headers["Authorization"] = `Bearer ${accessToken}`;
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (response.status === 401) {
    logout();
    throw new Error("로그인이 만료되었습니다. 다시 로그인해주세요.");
  }

  return response;
}

// ============================================================
// 화면 전환
// ============================================================
function showLoggedOutUI() {
  authView.classList.remove("hidden");
  chatView.classList.add("hidden");
  documentsView.classList.add("hidden");
  mainNav.classList.add("hidden");
  userInfo.classList.add("hidden");
}

function showLoggedInUI() {
  authView.classList.add("hidden");
  mainNav.classList.remove("hidden");
  userInfo.classList.remove("hidden");
  userNameLabel.textContent = currentUser ? `${currentUser.name}님` : "";
  showView("chat");
}

function showView(viewName) {
  const isChat = viewName === "chat";
  chatView.classList.toggle("hidden", !isChat);
  documentsView.classList.toggle("hidden", isChat);
  chatMenu.classList.toggle("active", isChat);
  documentsMenu.classList.toggle("active", !isChat);

  if (isChat) {
    loadSessions();
  } else {
    loadDocuments();
  }
}

// ============================================================
// 인증 (Step 2/3)
// ============================================================
function switchAuthTab(target) {
  const isLogin = target === "login";
  loginForm.classList.toggle("hidden", !isLogin);
  registerForm.classList.toggle("hidden", isLogin);
  showAuthStatus("");
}

// 로그인 중.../가입 처리 중... 같은 일반 안내 문구
function showAuthStatus(message) {
  authStatus.classList.remove("error", "shake");
  authStatus.textContent = message;
}

// 로그인 실패 등 에러 문구: 빨간색 + 흔들림 효과로 강조합니다.
function showAuthError(message) {
  authStatus.textContent = message;
  authStatus.classList.remove("shake");
  void authStatus.offsetWidth; // 같은 에러가 연속으로 떠도 애니메이션이 다시 재생되도록 강제 리플로우
  authStatus.classList.add("error", "shake");
}

goToRegisterLink.addEventListener("click", (event) => {
  event.preventDefault();
  switchAuthTab("register");
});

goToLoginLink.addEventListener("click", (event) => {
  event.preventDefault();
  switchAuthTab("login");
});

async function login(email, password) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "로그인에 실패했습니다.");
  }

  return response.json();
}

async function register(email, password, name) {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "회원가입에 실패했습니다.");
  }

  return response.json();
}

async function fetchMe() {
  const response = await authFetch("/me");
  if (!response.ok) {
    throw new Error("사용자 정보를 불러오지 못했습니다.");
  }
  return response.json();
}

function logout() {
  accessToken = null;
  currentUser = null;
  currentSessionId = null;
  localStorage.removeItem("access_token");
  showLoggedOutUI();
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const email = document.getElementById("login-email").value.trim();
  const password = document.getElementById("login-password").value;

  try {
    showAuthStatus("로그인 중...");
    const data = await login(email, password);
    accessToken = data.access_token;
    currentUser = data.user;
    localStorage.setItem("access_token", accessToken);
    showAuthStatus("");
    showLoggedInUI();
  } catch (error) {
    showAuthError(error.message);
  }
});

registerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = document.getElementById("register-name").value.trim();
  const email = document.getElementById("register-email").value.trim();
  const password = document.getElementById("register-password").value;

  try {
    showAuthStatus("가입 처리 중...");
    await register(email, password, name);
    showAuthStatus("");
    registerForm.reset();
    alert("회원가입이 완료되었습니다. 로그인 창으로 돌아갑니다.");
    switchAuthTab("login");
  } catch (error) {
    showAuthError(error.message);
  }
});

logoutButton.addEventListener("click", logout);

// ============================================================
// 채팅방 목록 (Step 4)
// ============================================================
async function fetchSessions() {
  const response = await authFetch("/chat/sessions");
  if (!response.ok) {
    throw new Error("채팅방 목록을 불러오지 못했습니다.");
  }
  return response.json();
}

async function createSession() {
  const response = await authFetch("/chat/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: "" }),
  });
  if (!response.ok) {
    throw new Error("채팅방을 만들지 못했습니다.");
  }
  return response.json();
}

async function fetchSessionMessages(sessionId) {
  const response = await authFetch(`/chat/sessions/${sessionId}/messages`);
  if (!response.ok) {
    throw new Error("대화 내용을 불러오지 못했습니다.");
  }
  return response.json();
}

function renderSessionList(sessions) {
  sessionListEl.innerHTML = "";

  sessions.forEach((session) => {
    const item = document.createElement("div");
    item.className = "session-item";
    item.classList.toggle("active", session.session_id === currentSessionId);
    item.textContent = session.title;
    item.dataset.id = session.session_id;
    item.addEventListener("click", () => selectSession(session.session_id, session.title));
    sessionListEl.appendChild(item);
  });
}

async function loadSessions() {
  try {
    const data = await fetchSessions();
    renderSessionList(data.sessions || []);
  } catch (error) {
    sessionListEl.innerHTML = `<div class="status-text">${escapeHtml(error.message)}</div>`;
  }
}

async function selectSession(sessionId, title) {
  currentSessionId = sessionId;
  currentSessionTitleEl.textContent = title;
  chatInput.disabled = false;
  chatSubmitButton.disabled = false;

  document.querySelectorAll(".session-item").forEach((el) => {
    el.classList.toggle("active", Number(el.dataset.id) === sessionId);
  });

  chatMessagesEl.innerHTML = "";
  try {
    const data = await fetchSessionMessages(sessionId);
    (data.messages || []).forEach((message) => {
      appendMessage(message.role, message.content, message.tools || [], message.sources || []);
    });
  } catch (error) {
    appendMessage("assistant", error.message);
  }
}

newSessionButton.addEventListener("click", async () => {
  try {
    const session = await createSession();
    await loadSessions();
    await selectSession(session.session_id, session.title);
  } catch (error) {
    sessionListEl.insertAdjacentHTML(
      "afterbegin",
      `<div class="status-text">${escapeHtml(error.message)}</div>`
    );
  }
});

// ============================================================
// 채팅 메시지 렌더링
// ============================================================
function appendMessage(role, text, tools = [], sources = []) {
  const isUser = role === "user";

  // 말풍선은 짧은 텍스트만 담당합니다 (iMessage 스타일).
  // 사용자 입력은 그대로 텍스트로, 챗봇 답변은 마크다운(굵게/문단/목록)을 해석해서 보여줍니다.
  const bubble = document.createElement("div");
  bubble.className = `message ${isUser ? "user" : "bot"}`;
  if (isUser) {
    bubble.textContent = text;
  } else {
    bubble.innerHTML = renderMarkdownLite(text);
  }
  chatMessagesEl.appendChild(bubble);

  // Tool 사용 내역/출처는 말풍선 안에 끼워 넣지 않고 그 아래 별도 블록으로 붙입니다.
  if (!isUser && (tools.length > 0 || sources.length > 0)) {
    const extra = document.createElement("div");
    extra.className = "message-extra";

    if (tools.length > 0) {
      extra.appendChild(renderToolBadges(tools));
    }
    if (sources.length > 0) {
      extra.appendChild(renderSources(sources));
    }

    chatMessagesEl.appendChild(extra);
  }
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

function renderToolBadges(tools) {
  const wrap = document.createElement("div");
  wrap.className = "tools-used";

  tools.forEach((tool) => {
    const badge = document.createElement("span");
    badge.className = `tool-badge ${tool.success === false ? "failed" : ""}`.trim();
    badge.textContent = tool.tool_name;
    wrap.appendChild(badge);
  });

  return wrap;
}

function renderSources(sources) {
  const wrap = document.createElement("div");
  wrap.className = "sources";
  wrap.innerHTML = `<strong>출처</strong>`;

  sources.forEach((source, index) => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.innerHTML = `
      <div><strong>출처 ${index + 1}</strong></div>
      <div>문서: ${escapeHtml(source.document_title || "")}</div>
      <div>파일: ${escapeHtml(source.file_name || "")}</div>
      <div>페이지: ${source.page_number ?? "-"}</div>
      <div>내용: ${escapeHtml(source.preview || "")}</div>
    `;
    wrap.appendChild(card);
  });

  return wrap;
}

// ============================================================
// 채팅 전송 (Step 8/9)
// ============================================================
async function sendMessage(sessionId, message, mode, explanationLevel) {
  const response = await authFetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      mode,
      explanation_level: explanationLevel,
    }),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "챗봇 응답을 받지 못했습니다.");
  }

  return response.json();
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message || !currentSessionId) return;

  appendMessage("user", message);
  chatInput.value = "";

  try {
    const data = await sendMessage(currentSessionId, message, getChatMode(), getExplanationLevel());
    appendMessage("assistant", data.answer, data.tools || [], data.sources || []);
    await loadSessions();
    if (currentSessionId) {
      const activeItem = sessionListEl.querySelector(`.session-item[data-id="${currentSessionId}"]`);
      if (activeItem) {
        activeItem.classList.add("active");
        currentSessionTitleEl.textContent = activeItem.textContent;
      }
    }
  } catch (error) {
    appendMessage("assistant", error.message);
  }
});

// ============================================================
// 문서 관리 (Step 6)
// ============================================================
async function fetchDocuments() {
  const response = await authFetch("/documents");
  if (!response.ok) {
    throw new Error("문서 목록을 불러오지 못했습니다.");
  }
  return response.json();
}

async function uploadDocument(title, file) {
  const formData = new FormData();
  formData.append("title", title);
  formData.append("file", file);

  const response = await authFetch("/documents", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("문서 업로드에 실패했습니다.");
  }

  return response.json();
}

async function deleteDocument(documentId) {
  const response = await authFetch(`/documents/${documentId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("문서 삭제에 실패했습니다.");
  }

  return response.json();
}

function renderDocuments(documents) {
  documentsTableBody.innerHTML = "";

  if (!documents || documents.length === 0) {
    documentsStatus.textContent = "등록된 문서가 없습니다.";
    return;
  }

  documentsStatus.textContent = `총 ${documents.length}개 문서`;

  documents.forEach((doc) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${doc.id}</td>
      <td>${escapeHtml(doc.title || "")}</td>
      <td>${escapeHtml(doc.file_name || "")}</td>
      <td>${escapeHtml(doc.file_type || "")}</td>
      <td>${doc.chunk_count ?? 0}</td>
      <td>${escapeHtml(doc.status || "")}</td>
      <td><button class="delete-button" data-id="${doc.id}">삭제</button></td>
    `;
    documentsTableBody.appendChild(row);
  });
}

async function loadDocuments() {
  try {
    documentsStatus.textContent = "문서 목록을 불러오는 중입니다...";
    const data = await fetchDocuments();
    renderDocuments(data.documents);
  } catch (error) {
    documentsStatus.textContent = error.message;
  }
}

chatMenu.addEventListener("click", () => showView("chat"));
documentsMenu.addEventListener("click", () => showView("documents"));

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const title = documentTitleInput.value.trim();
  const file = documentFileInput.files[0];

  if (!file) {
    documentsStatus.textContent = "업로드할 파일을 선택하세요.";
    return;
  }

  try {
    documentsStatus.textContent = "문서를 업로드하는 중입니다...";
    await uploadDocument(title, file);
    documentTitleInput.value = "";
    documentFileInput.value = "";
    await loadDocuments();
  } catch (error) {
    documentsStatus.textContent = error.message;
  }
});

documentsTableBody.addEventListener("click", async (event) => {
  if (!event.target.classList.contains("delete-button")) return;
  const documentId = event.target.dataset.id;

  try {
    await deleteDocument(documentId);
    await loadDocuments();
  } catch (error) {
    documentsStatus.textContent = error.message;
  }
});

// ============================================================
// 시작
// ============================================================
async function init() {
  if (!accessToken) {
    showLoggedOutUI();
    return;
  }

  try {
    currentUser = await fetchMe();
    showLoggedInUI();
  } catch (error) {
    showLoggedOutUI();
  }
}

init();
