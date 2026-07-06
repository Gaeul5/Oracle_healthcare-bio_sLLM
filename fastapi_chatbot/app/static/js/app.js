const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#sendButton");
const styleSelect = document.querySelector("#styleSelect");

const chatHistory = [];

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const htmlLines = [];
  let inList = false;

  for (const line of lines) {
    const heading = line.match(/^(#{1,3})\s+(.*)$/);
    const listItem = line.match(/^[-*]\s+(.*)$/);

    if (listItem) {
      if (!inList) {
        htmlLines.push("<ul>");
        inList = true;
      }
      htmlLines.push(`<li>${inline(listItem[1])}</li>`);
      continue;
    }

    if (inList) {
      htmlLines.push("</ul>");
      inList = false;
    }

    if (heading) {
      const level = heading[1].length + 2; // ### -> h5, ## -> h4, # -> h3
      htmlLines.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else if (line.trim() === "") {
      htmlLines.push("<br>");
    } else {
      htmlLines.push(`<p>${inline(line)}</p>`);
    }
  }

  if (inList) {
    htmlLines.push("</ul>");
  }

  return htmlLines.join("");
}

function inline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}

function addMessage(role, content = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  setBubbleContent(bubble, role, content);

  article.appendChild(bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;

  return bubble;
}

function setBubbleContent(bubble, role, content) {
  if (role === "assistant") {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.textContent = content;
  }
}

async function streamChat(message) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history: chatHistory,
      style: styleSelect.value,
    }),
  });

  if (!response.ok || !response.body) {
    throw new Error("챗봇 응답을 가져오지 못했습니다.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  const assistantBubble = addMessage("assistant");
  let answer = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    const chunk = decoder.decode(value, { stream: true });
    answer += chunk;
    setBubbleContent(assistantBubble, "assistant", answer);
    messages.scrollTop = messages.scrollHeight;
  }

  return answer;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = input.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  chatHistory.push({ role: "user", content: message });

  input.value = "";
  input.focus();
  sendButton.disabled = true;

  try {
    const answer = await streamChat(message);
    chatHistory.push({ role: "assistant", content: answer });
  } catch (error) {
    const errorMessage =
      error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다.";
    addMessage("assistant", errorMessage);
  } finally {
    sendButton.disabled = false;
  }
});

