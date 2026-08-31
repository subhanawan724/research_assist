


let currentThreadId = null;

document.getElementById('submitBtn').addEventListener('click', submitQuery);
document.getElementById('resumeBtn').addEventListener('click', resumeQuery);

async function submitQuery() {
    const query = document.getElementById('query').value;
    if (!query.trim()) return;

    showStatus();
    hideQuestion();
    hideResult();
    disableSubmit();

    const response = await fetch('/api/research', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: query})
    });
    const data = await response.json();
    handleResponse(data);
}

async function resumeQuery() {
    const userResponse = document.getElementById('userResponse').value;
    hideQuestion();
    showStatus();

    const response = await fetch('/api/resume', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_response: userResponse, thread_id: currentThreadId})
    });
    const data = await response.json();
    handleResponse(data);
}

function handleResponse(data) {
    currentThreadId = data.thread_id;

    if (data.status === "waiting_for_input") {
        hideStatus();
        document.getElementById('questionText').innerText = data.question;
        showQuestion();
    } else {
        hideStatus();
        document.getElementById('resultText').innerHTML = marked.parse(data.answer);
        
        //document.getElementById('traceLink').href = `https://smith.langchain.com/o//projects/p/?thread=${currentThreadId}`;
        showResult();
        enableSubmit();
    }
}

function showStatus() { document.getElementById('statusArea').classList.remove('hidden'); }
function hideStatus() { document.getElementById('statusArea').classList.add('hidden'); }
function showQuestion() { document.getElementById('questionBox').classList.remove('hidden'); }
function hideQuestion() { document.getElementById('questionBox').classList.add('hidden'); }
function showResult() { document.getElementById('resultCard').classList.remove('hidden'); }
function hideResult() { document.getElementById('resultCard').classList.add('hidden'); }
function disableSubmit() { document.getElementById('submitBtn').disabled = true; }
function enableSubmit() { document.getElementById('submitBtn').disabled = false; }