import { openModal, closeModal } from './modal.js';

class BatchElement {
    constructor(initialContent, element) {
        this.initialContent = initialContent; // 
        this.element = element; // Links to the HTML `<p>` element related to this
        this.receivedContent = []; // Token-by-token data.
    }

    getTotalContent(showInitial) {
        return (showInitial ? this.initialContent : "") + this.receivedContent.join("");
    }
}


const responseList = document.querySelector('.response-list');
const textInput = document.querySelector('.text-input');

const batchCountSlider = document.getElementById('batch_count');
const tokenCountSlider = document.getElementById('request_tokens');
const stopTokens = document.getElementById('stop_tokens');
const showToggle = document.getElementById('show_initial');
const modeToggle = document.getElementById('completion_mode_toggle');

const downloadButton = document.getElementById('download-button');
const TPSCounter = document.getElementById('TPSCounter');


console.log(modeToggle.textContent);

var batchElements = [];

tokenCountSlider.oninput = () => { tokenCountSlider.nextElementSibling.textContent = tokenCountSlider.value; };
batchCountSlider.oninput = () => { batchCountSlider.nextElementSibling.textContent = batchCountSlider.value; };
showToggle.onchange = () => { for (var i = 0; i < batchElements.length; i++) { batchElements[i].element.textContent = batchElements[i].getTotalContent(showToggle.checked); } };
modeToggle.onclick = () => { modeToggle.textContent = (modeToggle.textContent == "Chat" ? "Completion" : "Chat"); };

downloadButton.addEventListener('click', function () {
    var dateNow = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "_");
    var filename = window.prompt("Enter filename:", `batch-download-${dateNow}`) + ".json";

    var downloadContent = "[\n";
    for (var i = 0; i < batchElements.length; i++) {
        var escapedContent = batchElements[i].element.textContent
            .replace(/\r\n/g, "\n") // Replace \r\n with \\n
            .replace(/\\/g, '\\\\') // Escape backslashes
            .replace(/\n/g, "\\n")  // Replace \n with \\n
            .replace(/"/g, '\\"');  // Escape double quotes
        downloadContent += `    { "content": "${escapedContent}" }`;
        if (i != batchElements.length - 1) { downloadContent += ",\n"; }
    }

    if (filename != "null.json") { download(filename, downloadContent + "\n]"); }
});


textInput.addEventListener("keydown", (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        responseList.replaceChildren([]);
        toggleElements([downloadButton, showToggle], false);
        e.preventDefault();

        const messages = [{ role: "assistant", content: textInput.value }];
        const max_tokens = parseInt(tokenCountSlider.value);
        const stop_tokens = stopTokens.value == "" ? null : stopTokens.value.split('|');
        let newPrompt = { "stream": true, "chatQueries": [], "completionQueries": [], "stop": stop_tokens };
        let isCompletion = modeToggle.textContent == "Completion"; // Change if needed
        var selectedText = "";
        batchElements = [];

        for (var i = 0; i < batchCountSlider.value; i++) {
            const newResponse = document.createElement('p');
            newResponse.textContent = showToggle.checked ? textInput.value : "";
            newResponse.classList.add('response-entry');

            if (!isCompletion) { newPrompt.chatQueries.push({ "messages": messages, continue: true, max_tokens: max_tokens }); }
            else    { newPrompt.completionQueries.push({ "prompt": textInput.value, continue: true, max_tokens: max_tokens }); }

            const batchElement = new BatchElement(textInput.value, newResponse);
            responseList.appendChild(newResponse);
            batchElements.push(batchElement);


            // Make clicking the element open the token inspection window.
            newResponse.mousedown += () => { selectedText = window.getSelection().toString(); };
            newResponse.onclick = () => {
                const newSelectedText = window.getSelection().toString();
                if (newSelectedText == selectedText) {
                    var textContent = `------------------------ Your prompt ------------------------\n\n`;
                    textContent += batchElement.initialContent.substring(0, 100) + (batchElement.initialContent.length > 100 ? `\n[...]\n${batchElement.initialContent.substring(batchElement.initialContent.length - 100)}` : "");
                    textContent += `\n\n----------------------- Full Response -----------------------\n\n`;
                    textContent += batchElement.getTotalContent(false);
                    textContent += `\n\n---------------------- Response Tokens ----------------------\n\n`;
                    for (var j = 0; j < batchElement.receivedContent.length; j++) {
                        if (batchElement.receivedContent[j] == "") { continue; }
                        textContent += `- token ${j}: '${batchElement.receivedContent[j].replace(/\r\n/g, "\n").replace(/\n/g, "\\n") }'\n`;
                    }
                    openModal(textContent.trim());
                }
            };
        }

        updateResponses(newPrompt, batchElements);
    }
});

async function updateResponses(prompt, batchElements) {
    let response = await fetch("batch", { method: 'POST', headers: {}, body: JSON.stringify(prompt) });
    var reader = response.body.getReader();
    var decoder = new TextDecoder();

    const startTime = Date.now();
    var totalTokens = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) { break; }

        let lines = decoder.decode(value, { stream: true }).split('\n');
        for (let i = 0; i < lines.length - 1; i++) {
            if (!lines[i].startsWith('data: ')) { continue; }
            var batchResponse = JSON.parse(lines[i].substring(5));
            for (var j = 0; j < batchResponse.responses.length; j++) {
                const content = batchResponse.responses[j];
                if (content.response.delta == "") { continue; }

                batchElements[content.id].receivedContent.push(content.response.delta);
                batchElements[content.id].element.textContent += content.response.delta;
                totalTokens++;
            }
        }

        const totalSeconds = Math.round((Date.now() - startTime) / 100) / 10;
        if (totalSeconds >= 0.1) { TPSCounter.textContent = `${totalTokens} Tokens in ${totalSeconds}s (${(totalTokens / totalSeconds).toFixed(1)} T/s)`; }
    }
    TPSCounter.textContent += ` [COMPLETED]`;
    toggleElements([downloadButton, showToggle], true);
}

function toggleElements(elements, newState) {
    elements.forEach((e) => {
        e.disabled = !newState;
        if (newState == true) { e.classList.remove('disabled'); }
        else { e.classList.add('disabled'); }
    });
}

function download(filename, text) {
    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();
    document.body.removeChild(element);
}