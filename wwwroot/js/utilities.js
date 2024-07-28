// This function scans the whole window for properties that are functions and start with '_initialize', and calls them once DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => { for (const prop in window) { if (typeof window[prop] === 'function' && prop.startsWith('_initialize')) { window[prop](); } } });

// An array that introduces an onChange parameter, allowing subscriptions to its 'changed' events.
function reactiveArray(array, onChange) {
    return new Proxy(array, {
        set(target, property, value) { target[property] = value; if (property !== 'length') { onChange(); } return true; },
        deleteProperty(target, property) { delete target[property]; onChange(); return true; },
        apply(target, thisArg, argumentsList) { let result = Reflect.apply(...arguments); onChange(); return result; }
    });
}

// Loads the HTML code of a specific file and embed it into the target element. Optionally initialize the element with an initFunc
function loadHTML(filename, targetElement, initFunc) { fetch(filename).then(response => response.text()).then(html => { targetElement.innerHTML = html; initFunc?.(targetElement); }).catch(console.warn); }

// Can be awaited inside async methods to apply delay.
function delay(time) { return new Promise(x => setTimeout(x, time)); }

// Various string-manipulation utilities that help with rendering to the UI messages that may contain HTML elements or whatnot.
function unescapeHtml(message) { return message.replaceAll("&amp;", "&").replaceAll("&lt;", "<").replaceAll("&gt;", ">").replaceAll("&quot;", "\"").replaceAll("&#039;", "'"); }
function escapeHtml(message) { return message.replaceAll(/&/g, "&amp;").replaceAll(/</g, "&lt;").replaceAll(/>/g, "&gt;").replaceAll(/"/g, "&quot;").replaceAll(/'/g, "&#039;"); }
function makeHtml(message) { return message.replaceAll("\r\n", "<br>").replaceAll("\n", "<br>").replaceAll("\t", "&nbsp;&nbsp;&nbsp;&nbsp;"); }
function makeRP(message, styleType) {
    message = makeHtml(escapeHtml(message));
    message = message.replace(/\*(\S[^*]+\S)\*/g, `<span style='color: ${colorType[styleType]}; font-style: italic'>$1</span>`);  // Apply the actions style to *actions*.
    message = message.replace(/&quot;(.*?)&quot;/g, `<span style='color: rgba(200, 150, 250, 0.8); font-weight: bold'>$1</span>`); // Apply the speech style to "speech".
    return message;
}

const colorType = { "assistant": `rgba(200, 150, 255, 0.6)`, "user": `rgba(100, 150, 255, 0.6)` };