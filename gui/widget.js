(() => {
  const chatBtn = document.createElement("button");
  const chatBox = document.createElement("div");
  const chatFrame = document.createElement("iframe");


  Object.assign(chatBtn.style, {
    position: "fixed",
    bottom: "20px",
    right: "20px",
    zIndex: "9999",
    backgroundColor: "#800000",
    color: "white",
    border: "none",
    borderRadius: "50%",
    width: "60px",
    height: "60px",
    fontSize: "28px",
    cursor: "pointer",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
  });
  chatBtn.textContent = "💬";


  Object.assign(chatBox.style, {
    position: "fixed",
    bottom: "90px",
    right: "20px",
    width: "370px",
    height: "480px",
    background: "#fff",
    borderRadius: "16px",
    boxShadow: "0 4px 12px rgba(0,0,0,0.3)",
    overflow: "hidden",
    zIndex: "9999",
    display: "none",
    flexDirection: "column",
  });


  chatFrame.src = "https://cictify.onrender.com";
  Object.assign(chatFrame.style, {
    width: "100%",
    height: "100%",
    border: "none",
  });

  chatBox.appendChild(chatFrame);
  document.body.appendChild(chatBox);
  document.body.appendChild(chatBtn);


  chatBtn.addEventListener("click", () => {
    chatBox.style.display = chatBox.style.display === "none" ? "flex" : "none";
  });
})();
