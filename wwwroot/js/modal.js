var modalHTML = `
<div id="preview-modal" class="modal">
    <div class="modal-content">
        <textarea readonly id="preview-textarea" class="modal-textarea"></textarea>
        <div class="modal-buttons">
            <button id="modal-close-btn">Close</button>
        </div>
    </div>
</div>
 `;

var modalCSS = `
/* Modal background */
.modal {
    display: none;
    position: fixed;
    z-index: 1000;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5); /* Dims the background */
    pointer-events: none; /* Disable clicks on the background */
}

/* Modal content */
.modal-content {
    background-color: #fff;
    margin: 15% auto;
    padding: 20px;
    border: 1px solid #ccc;
    width: 50%;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    pointer-events: all; /* Re-enable clicks inside the modal */
}

/* Text area inside modal */
.modal-textarea {
    width: 100%;
    height: 300px;
    resize: none;
    font-size: 16px;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
}

/* Buttons inside modal */
.modal-buttons {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
}

.modal-buttons button {
    padding: 10px 20px;
    font-size: 14px;
    cursor: pointer;
}

#modal-close-btn {
    background-color: #FA5F55;
    color: white;
    border: none;
    border-radius: 5px;
}
`;

function openModal(myText) {
    var styleElement = document.createElement('style');
    var htmlElement = document.createElement('div');
    styleElement.innerHTML = modalCSS;
    htmlElement.innerHTML = modalHTML;
    document.head.appendChild(styleElement);
    document.body.appendChild(htmlElement);

    document.getElementById('preview-textarea').value = myText;
    document.getElementById('preview-modal').style.display = 'block';

    document.getElementById('modal-close-btn').onclick = closeModal;

}

function closeModal() {
    document.getElementById('preview-modal').style.display = 'none';
}


export { openModal, closeModal }