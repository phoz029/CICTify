const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const chatbox = document.getElementById("chatbox");
const chatLog = document.getElementById("chat-log");
const expandBtn = document.getElementById("expand-btn");

// Toggle chatbox visibility
function toggleChat() {
  chatbox.classList.toggle("show");
}

// Show typing indicator
function showTypingIndicator() {
  const typingIndicator = document.createElement("div");
  typingIndicator.classList.add("typing-indicator");
  typingIndicator.innerHTML = `
    <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
    <div class="typing-dots">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  chatLog.appendChild(typingIndicator);
  chatLog.scrollTop = chatLog.scrollHeight;
  return typingIndicator;
}

// Hide typing indicator
function hideTypingIndicator(typingIndicator) {
  if (typingIndicator && typingIndicator.parentNode) {
    typingIndicator.remove();
  }
}

// Function to create user and bot messages
async function sendMessage() {
  const message = userInput.value.trim();
  if (message === "") return;

  // Disable input while processing
  userInput.disabled = true;
  sendBtn.disabled = true;

  // User Message
  const userMsg = document.createElement("div");
  userMsg.classList.add("chat-message", "user-message");
  userMsg.innerHTML = `
    <img src="images/userIcon.png" alt="User Avatar">
    <div class="text">${message}</div>
  `;
  chatLog.appendChild(userMsg);
  chatLog.scrollTop = chatLog.scrollHeight;

  userInput.value = "";

  // Show typing indicator
  const typingIndicator = showTypingIndicator();

  try {
    // Send to Flask backend
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const data = await response.json();
    const botReply = data.reply || "⚠️ No response received.";

    hideTypingIndicator(typingIndicator);

const botMsg = document.createElement("div");
botMsg.classList.add("chat-message", "bot-message");
botMsg.innerHTML = `
  <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
  <div class="text">${botReply}</div>
`;

// 🔊 Add TTS “Listen” button (Browser-based)
const listenBtn = document.createElement("button");
listenBtn.textContent = "🔊 Listen";
listenBtn.className = "tts-btn";
listenBtn.style.marginTop = "5px";
listenBtn.onclick = () => {
  const utterance = new SpeechSynthesisUtterance(botReply);
  utterance.rate = 1;
  utterance.pitch = 1;
  utterance.volume = 1;
  window.speechSynthesis.speak(utterance);
};


botMsg.appendChild(listenBtn);
chatLog.appendChild(botMsg);
chatLog.scrollTop = chatLog.scrollHeight;

  } catch (error) {
    console.error("Error sending message:", error);
    hideTypingIndicator(typingIndicator);

    const botMsg = document.createElement("div");
    botMsg.classList.add("chat-message", "bot-message");
    botMsg.innerHTML = `
      <img src="images/CICTify_ChatLogo.png" alt="Bot Avatar">
      <div class="text">⚠️ Server not responding. Please check your Flask app.</div>
    `;
    chatLog.appendChild(botMsg);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  // Re-enable input
  userInput.disabled = false;
  sendBtn.disabled = false;
  userInput.focus();
}

// Send on button click
sendBtn.addEventListener("click", () => {
  sendMessage();
  sendBtn.classList.add("send-clicked");
  setTimeout(() => sendBtn.classList.remove("send-clicked"), 250);
});

// Send on Enter key
userInput.addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    event.preventDefault();
    sendMessage();
  }
});

// Expand button toggle
expandBtn.addEventListener("click", () => {
  chatbox.classList.toggle("expanded");

  // Toggle icon depending on state
  if (chatbox.classList.contains("expanded")) {
    expandBtn.innerHTML = "🗗"; // restore icon
  } else {
    expandBtn.innerHTML = "⛶"; // expand icon
  }
    
});
