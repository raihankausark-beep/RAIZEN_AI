import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient


app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)


class ChatRequest(BaseModel):
    message: str
    history: list = []
    personality: str = "Friendly"
    response_style: str = "Balanced"


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>RAIZEN</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #080b14;
    color: white;
    font-family: Arial, sans-serif;
    height: 100vh;
    overflow: hidden;
}

button {
    font-family: inherit;
}

.sidebar {
    position: fixed;
    left: 0;
    top: 0;
    bottom: 0;
    width: 260px;
    background: #0d1220;
    border-right: 1px solid #242b3d;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

.logo {
    font-size: 25px;
    font-weight: bold;
    margin-bottom: 25px;
}

.logo span {
    color: #9b5cff;
}

.new-chat {
    border: 1px solid #30394f;
    background: #171e30;
    color: white;
    padding: 13px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 15px;
}

.new-chat:hover {
    background: #202941;
}

.sidebar-title {
    color: #7f8aa3;
    font-size: 12px;
    margin-top: 28px;
    margin-bottom: 10px;
}

.history {
    overflow-y: auto;
    flex: 1;
}

.chat-item {
    padding: 11px;
    border-radius: 9px;
    margin-bottom: 6px;
    color: #cbd3e2;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.chat-item:hover {
    background: #181f31;
}

.bottom-buttons {
    display: flex;
    gap: 8px;
    margin-top: 10px;
}

.small-button {
    flex: 1;
    border: 1px solid #30394f;
    background: transparent;
    color: #aeb8ca;
    padding: 9px;
    border-radius: 9px;
    cursor: pointer;
}

.small-button:hover {
    background: #181f31;
}

.main {
    margin-left: 260px;
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.header {
    height: 68px;
    border-bottom: 1px solid #242b3d;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 25px;
    font-size: 21px;
    font-weight: bold;
}

.status {
    font-size: 12px;
    color: #7ee787;
    font-weight: normal;
}

.chat {
    flex: 1;
    overflow-y: auto;
    padding: 30px;
    width: 100%;
    max-width: 1000px;
    margin: auto;
}

.welcome {
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #8e99af;
}

.welcome h1 {
    font-size: 46px;
    color: white;
    margin-bottom: 10px;
}

.message-row {
    display: flex;
    margin-bottom: 18px;
}

.message-row.user {
    justify-content: flex-end;
}

.message-row.ai {
    justify-content: flex-start;
}

.message {
    max-width: 78%;
    padding: 14px 18px;
    border-radius: 17px;
    line-height: 1.55;
    white-space: pre-wrap;
    word-wrap: break-word;
}

.message-row.user .message {
    background: #7040d8;
    border-bottom-right-radius: 5px;
}

.message-row.ai .message {
    background: #171f31;
    border-bottom-left-radius: 5px;
}

.input-area {
    border-top: 1px solid #242b3d;
    background: #0d1220;
    padding: 17px;
}

.input-box {
    max-width: 1000px;
    margin: auto;
    display: flex;
    background: #171f31;
    border: 1px solid #293249;
    border-radius: 17px;
    padding: 7px;
}

.input-box input {
    flex: 1;
    border: none;
    outline: none;
    background: transparent;
    color: white;
    padding: 13px;
    font-size: 16px;
}

.send-button {
    border: none;
    background: #7040d8;
    color: white;
    border-radius: 12px;
    padding: 0 20px;
    cursor: pointer;
    font-size: 15px;
}

.send-button:hover {
    background: #8050e8;
}

/* SETTINGS */

.settings-panel {
    position: fixed;
    right: 20px;
    top: 78px;
    width: 300px;
    background: #101728;
    border: 1px solid #30394f;
    border-radius: 15px;
    padding: 20px;
    display: none;
    z-index: 10;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
}

.settings-panel.show {
    display: block;
}

.settings-panel h2 {
    margin-top: 0;
}

.setting-label {
    display: block;
    color: #8e99af;
    font-size: 13px;
    margin-top: 18px;
    margin-bottom: 7px;
}

select {
    width: 100%;
    padding: 11px;
    border-radius: 9px;
    border: 1px solid #30394f;
    background: #171f31;
    color: white;
    outline: none;
}

.close-settings {
    width: 100%;
    margin-top: 20px;
    padding: 10px;
    border-radius: 9px;
    border: 1px solid #30394f;
    background: transparent;
    color: white;
    cursor: pointer;
}

@media(max-width: 700px) {

    .sidebar {
        display: none;
    }

    .main {
        margin-left: 0;
    }

    .chat {
        padding: 15px;
    }

    .message {
        max-width: 90%;
    }

    .welcome h1 {
        font-size: 34px;
    }

    .settings-panel {
        right: 10px;
        left: 10px;
        width: auto;
    }
}

</style>
</head>


<body>


<div class="sidebar">

    <div class="logo">
        ⚡ <span>RAIZEN</span>
    </div>

    <button class="new-chat" onclick="newChat()">
        + New Chat
    </button>

    <div class="sidebar-title">
        CHAT HISTORY
    </div>

    <div id="history" class="history"></div>

    <div class="bottom-buttons">

        <button class="small-button" onclick="clearAll()">
            Clear
        </button>

        <button class="small-button" onclick="toggleSettings()">
            Settings
        </button>

    </div>

</div>


<div class="main">

    <div class="header">

        <span>⚡ RAIZEN</span>

        <span class="status">
            ● Online
        </span>

    </div>


    <div id="chat" class="chat">

        <div id="welcome" class="welcome">

            <h1>RAIZEN</h1>

            <p>Your intelligent AI assistant</p>

        </div>

    </div>


    <div class="input-area">

        <div class="input-box">

            <input
                id="message"
                type="text"
                placeholder="Ask RAIZEN anything..."
                autocomplete="off"
            >

            <button
                class="send-button"
                onclick="sendMessage()"
            >
                Send
            </button>

        </div>

    </div>

</div>


<div id="settingsPanel" class="settings-panel">

    <h2>⚙️ Settings</h2>

    <label class="setting-label">
        Personality
    </label>

    <select id="personality">

        <option value="Friendly">
            Friendly
        </option>

        <option value="Teacher">
            Teacher
        </option>

        <option value="Coding Assistant">
            Coding Assistant
        </option>

        <option value="Professional">
            Professional
        </option>

    </select>


    <label class="setting-label">
        Response Style
    </label>

    <select id="responseStyle">

        <option value="Short">
            Short
        </option>

        <option value="Balanced" selected>
            Balanced
        </option>

        <option value="Detailed">
            Detailed
        </option>

    </select>


    <button
        class="close-settings"
        onclick="saveSettings()"
    >
        Save Settings
    </button>

</div>


<script>

let conversation = [];

let savedChats = JSON.parse(
    localStorage.getItem("raizen_chats") || "{}"
);

let currentChatId = null;

let settings = JSON.parse(
    localStorage.getItem("raizen_settings") || "{}"
);


const personalitySelect =
    document.getElementById("personality");

const responseStyleSelect =
    document.getElementById("responseStyle");


if (settings.personality) {
    personalitySelect.value =
        settings.personality;
}

if (settings.responseStyle) {
    responseStyleSelect.value =
        settings.responseStyle;
}


function saveSettings() {

    settings = {
        personality:
            personalitySelect.value,

        responseStyle:
            responseStyleSelect.value
    };

    localStorage.setItem(
        "raizen_settings",
        JSON.stringify(settings)
    );

    toggleSettings();
}


function toggleSettings() {

    const panel =
        document.getElementById("settingsPanel");

    panel.classList.toggle("show");
}


function saveChats() {

    localStorage.setItem(
        "raizen_chats",
        JSON.stringify(savedChats)
    );
}


function renderHistory() {

    const history =
        document.getElementById("history");

    history.innerHTML = "";

    Object.keys(savedChats)
        .reverse()
        .forEach(function(id) {

            const item =
                document.createElement("div");

            item.className = "chat-item";

            item.textContent =
                savedChats[id].title || "New Chat";

            item.onclick = function() {
                loadChat(id);
            };

            history.appendChild(item);
        });
}


function saveCurrentChat() {

    if (conversation.length === 0) {
        return;
    }

    const firstUser =
        conversation.find(function(item) {
            return item.type === "user";
        });

    const title = firstUser
        ? firstUser.text.substring(0, 30)
        : "New Chat";

    if (!currentChatId) {
        currentChatId = crypto.randomUUID();
    }

    savedChats[currentChatId] = {
        title: title,
        messages: conversation
    };

    saveChats();
    renderHistory();
}


function loadChat(id) {

    const chatData =
        savedChats[id];

    if (!chatData) {
        return;
    }

    currentChatId = id;

    conversation =
        chatData.messages || [];

    const chatBox =
        document.getElementById("chat");

    chatBox.innerHTML = "";

    if (conversation.length === 0) {
        showWelcome();
        return;
    }

    conversation.forEach(function(item) {

        addMessage(
            item.text,
            item.type,
            false
        );

    });
}


function showWelcome() {

    document.getElementById("chat").innerHTML = `
        <div id="welcome" class="welcome">
            <h1>RAIZEN</h1>
            <p>Your intelligent AI assistant</p>
        </div>
    `;
}


function addMessage(
    text,
    type,
    save = true
) {

    const welcome =
        document.getElementById("welcome");

    if (welcome) {
        welcome.remove();
    }

    const row =
        document.createElement("div");

    row.className =
        "message-row " + type;

    const message =
        document.createElement("div");

    message.className =
        "message";

    message.textContent =
        text;

    row.appendChild(message);

    document
        .getElementById("chat")
        .appendChild(row);

    const chatBox =
        document.getElementById("chat");

    chatBox.scrollTop =
        chatBox.scrollHeight;

    if (save) {

        conversation.push({
            text: text,
            type: type
        });

        saveCurrentChat();
    }

    return row;
}


async function sendMessage() {

    const input =
        document.getElementById("message");

    const text =
        input.value.trim();

    if (!text) {
        return;
    }

    addMessage(
        text,
        "user"
    );

    input.value = "";

    input.disabled = true;

    const typing =
        addMessage(
            "⚡ RAIZEN is thinking...",
            "ai"
        );

    try {

        const response =
            await fetch("/chat", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    message: text,

                    history:
                        conversation,

                    personality:
                        personalitySelect.value,

                    response_style:
                        responseStyleSelect.value

                })

            });


        const data =
            await response.json();


        typing.remove();


        if (data.reply) {

            addMessage(
                "⚡ RAIZEN: " + data.reply,
                "ai"
            );

        } else {

            addMessage(
                "⚠️ " +
                (
                    data.detail ||
                    "Something went wrong."
                ),
                "ai"
            );
        }

    } catch (error) {

        typing.remove();

        addMessage(
            "⚠️ Unable to connect to RAIZEN.",
            "ai"
        );
    }

    input.disabled = false;

    input.focus();
}


document
    .getElementById("message")
    .addEventListener(
        "keydown",
        function(event) {

            if (event.key === "Enter") {
                sendMessage();
            }

        }
    );


function newChat() {

    conversation = [];

    currentChatId = null;

    showWelcome();

}


function clearAll() {

    conversation = [];

    if (currentChatId) {

        delete savedChats[
            currentChatId
        ];

        saveChats();
    }

    currentChatId = null;

    showWelcome();

    renderHistory();
}


renderHistory();

</script>

</body>
</html>
""")


@app.post("/chat")
def chat_ai(request: ChatRequest):

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured."
        )

    try:

        personality_prompts = {

            "Friendly":
                "Be friendly, casual and approachable.",

            "Teacher":
                "Act like a patient teacher. Explain concepts simply and clearly.",

            "Coding Assistant":
                "Act like an expert coding assistant. Give accurate and practical programming help.",

            "Professional":
                "Use a professional, polished and formal communication style."
        }


        style_prompts = {

            "Short":
                "Keep answers concise and direct.",

            "Balanced":
                "Give a balanced answer with enough explanation but avoid unnecessary length.",

            "Detailed":
                "Give detailed explanations with useful examples when appropriate."
        }


        personality_instruction = personality_prompts.get(
            request.personality,
            personality_prompts["Friendly"]
        )


        style_instruction = style_prompts.get(
            request.response_style,
            style_prompts["Balanced"]
        )


        system_prompt = (
            "You are RAIZEN, an advanced futuristic AI assistant. "
            "You are intelligent, helpful, friendly and confident. "
            "Give clear and useful answers. "
            "You were created and developed by Raihan Kausar. "
            "If someone asks who invented, created, developed, "
            "or made you, say that Raihan Kausar created and "
            "developed you. "
            + personality_instruction
            + " "
            + style_instruction
        )


        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]


        for item in request.history[-12:]:

            role_type = item.get("type")

            text = item.get("text", "")

            if not text:
                continue


            if role_type == "user":

                messages.append({
                    "role": "user",
                    "content": text
                })


            elif role_type == "ai":

                if text.startswith("⚡ RAIZEN: "):

                    text = text.replace(
                        "⚡ RAIZEN: ",
                        "",
                        1
                    )

                messages.append({
                    "role": "assistant",
                    "content": text
                })


        if (
            not messages
            or messages[-1].get("content")
            != request.message
        ):

            messages.append({
                "role": "user",
                "content": request.message
            })


        response = client.chat.completions.create(
            model="zai-org/GLM-5.3-Flash",
            messages=messages,
            max_tokens=500
        )


        reply = response.choices[0].message.content


        return {
            "reply": reply
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/health")
def health():

    return {
        "status": "RAIZEN is online",
        "token_loaded": bool(HF_TOKEN)
    }
