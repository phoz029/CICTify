(() => {
  const chatBtn = document.createElement("img");
  const chatBox = document.createElement("div");
  const chatFrame = document.createElement("iframe");

  // Floating button
  Object.assign(chatBtn.style, {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    zIndex: "9999",
    width: "70px",
    height: "70px",
    borderRadius: "50%",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
  });
  chatBtn.src = "https://cictify.onrender.com/images/floatingCICTify.png";
  chatBtn.alt = "Open CICTify Chat";

  // Chat box
  Object.assign(chatBox.style, {
    position: "fixed",
    bottom: "100px",
    right: "20px",
    width: "380px",
    height: "500px",
    background: "#fff",
    borderRadius: "16px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
    overflow: "hidden",
    zIndex: "9999",
    display: "none",
    flexDirection: "column",
  });

  // Embedded chatbot
  chatFrame.src = "https://cictify.onrender.com";
  Object.assign(chatFrame.style, {
    width: "100%",
    height: "100%",
    border: "none",
  });

  chatBox.appendChild(chatFrame);
  document.body.appendChild(chatBox);
  document.body.appendChild(chatBtn);

  // Toggle open/close
  chatBtn.addEventListener("click", () => {
    chatBox.style.display = chatBox.style.display === "none" ? "flex" : "none";
  });
})();
