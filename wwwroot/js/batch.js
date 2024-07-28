const responseList = document.querySelector('.response-list');
const leftInput = document.querySelector('.left-input');

const tokenCountSlider = document.getElementById('request_tokens');
const batchCountSlider = document.getElementById('batch_count');

tokenCountSlider.addEventListener('input', function () { tokenCountSlider.nextElementSibling.textContent = this.value; });
batchCountSlider.addEventListener('input', function () { batchCountSlider.nextElementSibling.textContent = this.value; });

leftInput.addEventListener("keydown", function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        responseList.replaceChildren([]);
        e.preventDefault();

        for (var i = 0; i < batchCountSlider.value; i++) {
            const newResponse = document.createElement('p');
            newResponse.textContent = leftInput.value;
            newResponse.classList.add('response-entry');
            responseList.appendChild(newResponse);

            getAIResponse((c) => { newResponse.textContent += c; }, tokenCountSlider.value);
        }
    }
});

async function getAIResponse(callback, tokenCount) {
    let jsonContent = { "messages": [{ role: "assistant", content: leftInput.value }], "stream": true, continue: true, max_tokens: parseInt(tokenCount)};
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
}