import os
import re
from html import unescape
from urllib.parse import quote
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "zai-org/GLM-5.3-Flash"

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)


class ChatRequest(BaseModel):
    message: str
    history: list = []
    personality: str = "Friendly"
    response_style: str = "Balanced"


PERSONALITIES = {
    "Friendly": "Be friendly, casual and approachable.",
    "Teacher": "Act like a patient teacher. Explain concepts simply and clearly.",
    "Coding Assistant": "Act like an expert coding assistant. Give accurate and practical programming help.",
    "Professional": "Use a professional, polished and formal communication style."
}

STYLES = {
    "Short": "Keep answers concise and direct.",
    "Balanced": "Give a balanced answer with enough explanation but avoid unnecessary length.",
    "Detailed": "Give detailed explanations with useful examples when appropriate."
}


def needs_web_search(message):
    keywords = [
        "search",
        "latest",
        "today",
        "current",
        "recent",
        "news",
        "now",
        "weather",
        "price",
        "who is the current",
        "what happened",
        "this week",
        "this month"
    ]

    text = message.lower()

    return any(
        keyword in text
        for keyword in keywords
    )


def web_search(query):
    try:
        url = (
            "https://html.duckduckgo.com/html/?q="
            + quote(query)
        )

        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urlopen(
            request,
            timeout=10
        ) as response:

            page = response.read().decode(
                "utf-8",
                errors="ignore"
            )

        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.IGNORECASE | re.DOTALL
        )

        results = []

        for link, title in pattern.findall(page)[:5]:

            clean_title = re.sub(
                r"<.*?>",
                "",
                title
            )

            clean_title = unescape(
                clean_title
            ).strip()

            results.append({
                "title": clean_title,
                "href": link
            })

        return results

    except Exception:
        return []


@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return """
<!DOCTYPE html>
<html>

<head>

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>RAIZEN</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #070b14;
    color: white;
}

.header {
    padding: 20px;
    text-align: center;
    background: #0d1424;
    border-bottom: 1px solid #202a40;
}

.header h1 {
    margin: 0;
    font-size: 30px;
}

.header p {
    margin: 6px 0 0;
    color: #8fa3c7;
}

.controls {
    padding: 12px;
    display: flex;
    gap: 10px;
    justify-content: center;
    flex-wrap: wrap;
    background: #0a1020;
}

select,
button,
textarea {
    font-family: Arial, sans-serif;
}

select {
    background: #111a2d;
    color: white;
    border: 1px solid #293653;
    padding: 10px;
    border-radius: 8px;
}

button {
    cursor: pointer;
}

.action {
    background: #111a2d;
    color: white;
    border: 1px solid #293653;
    padding: 9px 13px;
    border-radius: 8px;
}

.action:hover {
    background: #1b2944;
}

.chat {
    max-width: 900px;
    margin: auto;
    padding: 20px;
    min-height: 65vh;
}

.message {
    margin: 12px 0;
    padding: 12px 15px;
    border-radius: 12px;
    line-height: 1.5;
    white-space: pre-wrap;
}

.user {
    background: #18243b;
    margin-left: 15%;
}

.raizen {
    background: #101827;
    border: 1px solid #202d47;
    margin-right: 15%;
}

.search-box {
    max-width: 900px;
    margin: auto;
    padding: 15px;
    display: flex;
    gap: 10px;
}

textarea {
    flex: 1;
    resize: none;
    min-height: 50px;
    background: #101827;
    color: white;
    border: 1px solid #293653;
    border-radius: 10px;
    padding: 12px;
    outline: none;
}

.send {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0 20px;
}

@media (max-width: 600px) {

    .user,
    .raizen {
        margin-left: 0;
        margin-right: 0;
    }

    .search-box {
        padding: 10px;
    }

    .send {
        padding: 0 15px;
    }

}

</style>

</head>

<body>

<div class="header">

<h1>⚡ RAIZEN</h1>

<p>Your AI Assistant</p>

</div>

<div class="controls">

<select id="personality">

<option>Friendly</option>
<option>Teacher</option>
<option>Coding Assistant</option>
<option>Professional</option>

</select>

<select id="style">

<option>Short</option>
<option selected>Balanced</option>
<option>Detailed</option>

</select>

<button
    class="action"
    onclick="newChat()">

New Chat

</button>

<button
    class="action"
    onclick="clearChat()">

Clear Chat

</button>

</div>

<div
    id="chat"
    class="chat">
</div>

<div class="search-box">

<textarea
    id="message"
    placeholder="Ask RAIZEN anything..."
    onkeydown="handleKey(event)">
</textarea>

<button
    class="send"
    onclick="sendMessage()">

Send

</button>

</div>

<script>

let history = JSON.parse(
    localStorage.getItem(
        "raizen_history"
    ) || "[]"
);


function saveHistory() {

    localStorage.setItem(
        "raizen_history",
        JSON.stringify(history)
    );

}


function displayMessage(
    role,
    text
) {

    const chat =
        document.getElementById(
            "chat"
        );

    const div =
        document.createElement(
            "div"
        );

    div.className =
        role === "user"
        ? "message user"
        : "message raizen";

    div.textContent = text;

    chat.appendChild(div);

    window.scrollTo(
        0,
        document.body.scrollHeight
    );

}


function loadHistory() {

    document.getElementById(
        "chat"
    ).innerHTML = "";

    history.forEach(
        function(item) {

            displayMessage(
                item.role,
                item.content
            );

        }
    );

}


function newChat() {

    history = [];

    saveHistory();

    loadHistory();

}


function clearChat() {

    history = [];

    saveHistory();

    loadHistory();

}


async function sendMessage() {

    const input =
        document.getElementById(
            "message"
        );

    const message =
        input.value.trim();

    if (!message) {
        return;
    }

    displayMessage(
        "user",
        message
    );

    history.push({
        role: "user",
        content: message
    });

    saveHistory();

    input.value = "";

    displayMessage(
        "assistant",
        "🌐 RAIZEN is thinking..."
    );

    const personality =
        document.getElementById(
            "personality"
        ).value;

    const responseStyle =
        document.getElementById(
            "style"
        ).value;

    try {

        const response =
            await fetch(
                "/chat",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        message:
                            message,

                        history:
                            history.slice(-12),

                        personality:
                            personality,

                        response_style:
                            responseStyle

                    })
                }
            );

        const data =
            await response.json();

        const chat =
            document.getElementById(
                "chat"
            );

        if (chat.lastChild) {

            chat.removeChild(
                chat.lastChild
            );

        }

        if (!response.ok) {

            displayMessage(
                "assistant",
                data.detail ||
                "Something went wrong."
            );

            return;

        }

        displayMessage(
            "assistant",
            data.reply
        );

        history.push({

            role: "assistant",

            content: data.reply

        });

        saveHistory();

    } catch (error) {

        const chat =
            document.getElementById(
                "chat"
            );

        if (chat.lastChild) {

            chat.removeChild(
                chat.lastChild
            );

        }

        displayMessage(
            "assistant",
            "Connection error. Please try again."
        );

    }

}


function handleKey(event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        sendMessage();

    }

}


loadHistory();

</script>

</body>

</html>
"""


@app.post("/chat")
def chat(request: ChatRequest):

    if not HF_TOKEN:

        raise HTTPException(
            status_code=500,
            detail=(
                "HF_TOKEN is not configured "
                "on the server."
            )
        )

    personality = PERSONALITIES.get(
        request.personality,
        PERSONALITIES["Friendly"]
    )

    response_style = STYLES.get(
        request.response_style,
        STYLES["Balanced"]
    )

    search_context = ""

    if needs_web_search(
        request.message
    ):

        results = web_search(
            request.message
        )

        if results:

            search_context = (
                "\n\nWEB SEARCH RESULTS:\n"
                "Use these results as "
                "supporting information:\n"
            )

            for index, result in enumerate(
                results,
                start=1
            ):

                search_context += (
                    f"\n{index}. "
                    f"{result['title']}\n"
                    f"Source: "
                    f"{result['href']}\n"
                )

    system_prompt = f"""
You are RAIZEN, an advanced futuristic AI assistant.

You were created and developed by Raihan Kausar.

If someone asks who invented, created, developed,
or made you, say that Raihan Kausar created and
developed you.

Personality:
{personality}

Response style:
{response_style}

Rules:

- Be helpful and accurate.
- Use conversation history when useful.
- If web search results are provided, use them.
- Do not claim that you searched the web if no
  search results were provided.
- Do not invent facts or sources.
- Explain things clearly.
"""

    if search_context:

        system_prompt += search_context

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for item in request.history[-12:]:

        role = item.get(
            "role",
            "user"
        )

        content = item.get(
            "content",
            ""
        )

        if (
            role in ["user", "assistant"]
            and content
        ):

            messages.append({

                "role": role,

                "content": content

            })

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=500
        )

        reply = response.choices[0].message.content

        return {
            "reply": reply
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"AI error: {str(error)}"
        )


@app.get("/health")
def health():

    return {
        "status": "RAIZEN online",
        "token_loaded": bool(HF_TOKEN),
        "model": MODEL,
        "web_search": True
    }
