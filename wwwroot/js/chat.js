const chatLog = document.querySelector(".chat-log");
const chatInput = document.querySelector(".chat-input");
const sendButton = document.querySelector(".send-button");
const clearButton = document.querySelector(".clear-button");

let chatMessageHistory = []; // Create an array that holds all messages and updates.
let refreshFullChat = null;
let saveFullChat = null;

function _initializeChatLog() {
    let historyJson = localStorage.getItem('messageHistoryLog');// Try to get saved log
    let messages = historyJson ? JSON.parse(historyJson) : []; 	// Load or create new one.
    for (let i = 0; i < messages.length; i++) { if (!messages[i] || !messages[i].content) { messages.splice(i, 1); } }
    chatMessageHistory = reactiveArray(messages, () => { refreshChat(); saveChat(); });
    refreshFullChat = refreshChat; saveFullChat = saveChat;
    refreshChat();

    function refreshChat() {
        chatLog.innerHTML = chatMessageHistory.map((msg, index) => `
        <div class="chat-entry ${msg?.sender === 'user' ? 'user-entry' : 'ai-entry'}" data-index="${index}">
            <div class="portrait" style="background-image: url('${msg.portrait}')"></div>
            <div class="message-text"><p>${makeRP(msg.content, msg.sender)}</p>${getMessageControls(msg.sender, index)}</div>
        </div>`).join('');
        chatLog.scrollTop = chatLog.scrollHeight;

        function getMessageControls(sender, messageIndex) {
            let controls = []; // Only allow editing when there's no potential HTML in the content. //TODO: Maybe remove limitation.
            if (!chatMessageHistory[messageIndex].content.includes("<")) { controls.push(getControlButton("✏️", "Edit", "edit")); }
            controls.push(getControlButton("❌", "Delete", "delete"));
            return `<div class="message-controls">${controls.join('')}</div>`;

            function getControlButton(icon, title, action) { return `<span class="control-btn" data-action="${action}" data-index="${messageIndex}" title="${title}">${icon}</span>`; }
        }
    }

    function saveChat() { localStorage.setItem('messageHistoryLog', JSON.stringify(chatMessageHistory)); }
}

function _initializeSendButton() {
    chatInput.addEventListener("keydown", function (e) {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendButton.click(); }
        else if (e.key === 'Tab') {
            e.preventDefault();

            const start = this.selectionStart; const end = this.selectionEnd;
            this.value = this.value.substring(0, start) + "\t" + this.value.substring(end);
            this.selectionStart = this.selectionEnd = start + 1;
        }
    });
    sendButton.addEventListener("click", async () => {
        if (chatInput.value) {
            chatMessageHistory.push({ sender: "user", portrait: 'img/user.png', content: chatInput.value });
            chatInput.value = '';
        }

        var newResponse = null;
        const aiResponse = getAIResponse((m) => {
            if (!aiResponse) { return; }
            if (!newResponse) {
                newResponse = { sender: "assistant", portrait: 'img/assistant.png', content: "" };
                chatMessageHistory.push(newResponse);
            }
            newResponse.content += m;
            refreshFullChat();
            saveFullChat();
        });
    });
    clearButton.addEventListener("click", async () => {
        chatMessageHistory = reactiveArray([], () => { refreshFullChat(); saveFullChat(); });
        refreshFullChat();
    });


    chatInput.addEventListener("focus", () => { sendButton.classList.add("pulsate"); });
    chatInput.addEventListener("blur", () => { sendButton.classList.remove("pulsate"); });
}

function _initializeMessageControls() {
    let editingEntry = null;

    chatLog.addEventListener('click', (e) => {
        if (e.target.classList.contains('control-btn')) {
            const action = e.target.getAttribute('data-action');
            const messageIndex = parseInt(e.target.getAttribute('data-index'));
            switch (action) {
                case "edit": { beginEditMessage(messageIndex); break; }
                case "delete": { chatMessageHistory.splice(messageIndex, 1); break; }
            }
        }

        function beginEditMessage(index) {
            const chatEntry = document.querySelector(`.chat-entry[data-index="${index}"]`);
            const messageTextElement = chatEntry.querySelector(".message-text p");

            // If another text field is already being edited, finalize editing before continuing.
            if (editingEntry) { // In case this same field is being edited, return after finalizing.
                if (editingEntry === messageTextElement) { finishEditing(index); return; }
                else { editingEntry.dispatchEvent(new KeyboardEvent('keydown', { key: "Enter" })); }
            }
            messageTextElement.contentEditable = "true";
            editingEntry = messageTextElement;

            // Switch to non-stylized format for flawless editing, and put focus on the field.
            messageTextElement.innerHTML = makeHtml(chatMessageHistory[index].content);
            messageTextElement.focus();

            // Submit with either pressing enter or the 'edit' button directly.
            messageTextElement.addEventListener("keydown", onKeyDown);
            function onKeyDown(e) { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); finishEditing(index); } }
            function finishEditing(index) {
                const updatedText = htmlToText(messageTextElement.innerHTML);
                chatMessageHistory[index].content = updatedText;
                messageTextElement.innerHTML = makeRP(unescapeHtml(updatedText), chatMessageHistory[index].sender);
                messageTextElement.contentEditable = "false";
                messageTextElement.removeEventListener("keydown", onKeyDown);
                editingEntry = null;
            }

            function htmlToText(html) {
                const tempEl = document.createElement("div"); tempEl.innerHTML = html;
                let text = '';
                tempEl.childNodes.forEach(node => { if (node.nodeType === Node.TEXT_NODE) { text += node.nodeValue; } else if (node.nodeName === "BR") { text += '\n'; } else if (node.nodeName === "DIV") { text += '\n' + (node.innerText || ''); } });
                return text.trim();
            }
        }
    });
}

async function getAIResponse(callback) {
    let jsonContent = { "messages": gatherHistoryLogs(), "stream": true, max_tokens: 1000 };
    let response = await fetch("chat/completion", { method: 'POST', headers: {}, body: JSON.stringify(jsonContent) });
    var reader = response.body.getReader();
    var decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) { break; }

        let lines = decoder.decode(value, { stream: true }).split('\n');
        for (let i = 0; i < lines.length - 1; i++) {
            if (!lines[i].startsWith('data: ')) { continue; }
            var content = JSON.parse(lines[i].substring(5));
            callback(content.delta);
        }
    }

    function gatherHistoryLogs() { return chatMessageHistory.map(msg => ({ role: msg.sender, content: msg.content })); }
}