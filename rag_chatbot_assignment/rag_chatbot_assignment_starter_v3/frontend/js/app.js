const API_BASE_URL = "/api";

const chatMenu = document.getElementById("chat-menu");
const documentsMenu = document.getElementById("documents-menu");
const chatView = document.getElementById("chat-view");
const documentsView = document.getElementById("documents-view");
const chatForm = document.getElementById("chat-form");
const uploadForm = document.getElementById("upload-form");
const chatInput = document.getElementById("chat-input");
const documentTitleInput = document.getElementById("document-title");
const documentFileInput = document.getElementById("document-file");
const documentsStatus = document.getElementById("documents-status");
const documentsTableBody = document.getElementById("documents-table-body");

function showView(viewName) {
  const isChat = viewName === "chat";
  chatView.classList.toggle("hidden", !isChat);
  documentsView.classList.toggle("hidden", isChat);
  chatMenu.classList.toggle("active", isChat);
  documentsMenu.classList.toggle("active", !isChat);
}

function appendMessage(role, text, sources = []) {
  const chatMessages = document.getElementById("chat-messages");
  const messageEl = document.createElement("div");
  messageEl.className = `message ${role}`;

  const roleLabel = role === "user" ? "사용자" : "챗봇";
  messageEl.innerHTML = `
    <div class="message-role">${roleLabel}</div>
    <div class="message-body">${escapeHtml(text)}</div>
  `;

  if (sources.length > 0) {
    messageEl.appendChild(renderSources(sources));
  }

  chatMessages.appendChild(messageEl);
  chatMessages.scrollTop = chatMessages.scrollHeight;
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

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

// ============================================================
// TODO 1. 문서 목록 조회 fetch 구현
// ============================================================
async function fetchDocuments() {
  // 목표:
  // 1. GET /api/documents 호출
  // 2. response.ok 확인
  // 3. response.json() 반환
  // 힌트:
  // const response = await fetch(`${API_BASE_URL}/documents`);
  const response = await fetch(`${API_BASE_URL}/documents`);

  if (!response.ok) {
    throw new Error("문서 목록을 불러오지 못했습니다.");
  }

  return response.json();
}

// ============================================================
// TODO 2. 문서 업로드 fetch 구현
// ============================================================
async function uploadDocument(title, file) {
  // 목표:
  // 1. FormData 생성
  // 2. title, file 추가
  // 3. POST /api/documents 호출
  // 4. response.json() 반환
  const formData = new FormData();
  formData.append("title", title);
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("문서 업로드에 실패했습니다.");
  }

  return response.json();
}

// ============================================================
// TODO 3. 채팅 fetch 구현
// ============================================================
async function sendMessage(message) {
  // 목표:
  // 1. POST /api/chat 호출
  // 2. Content-Type: application/json 설정
  // 3. body에 message, top_k 전달
  // 4. answer, sources가 들어 있는 JSON 반환

  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, top_k: 3 }),
  });

  if (!response.ok) {
    throw new Error("챗봇 응답을 받지 못했습니다.");
  }

  return response.json();
}

// ============================================================
// TODO 4. 문서 삭제 fetch 구현
// ============================================================
async function deleteDocument(documentId) {
  // 목표:
  // DELETE /api/documents/{documentId} 호출
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("문서 삭제에 실패했습니다.");
  }

  return response.json();
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

documentsMenu.addEventListener("click", () => {
  showView("documents");
  loadDocuments();
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;

  appendMessage("user", message);
  chatInput.value = "";

  try {
    const data = await sendMessage(message);
    appendMessage("bot", data.answer, data.sources || []);
  } catch (error) {
    appendMessage("bot", error.message);
  }
});

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
