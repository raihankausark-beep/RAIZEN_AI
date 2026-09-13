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


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>RAIZEN AI 2.0</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #080d1a;
    color: white;
    height: 100vh;
    display: flex;
    overflow: hidden;
}

/* SIDEBAR */

.sidebar {
    width: 250px;
    background: #0b1120;
    border-right: 1px solid #20283a;
    padding: 20px;
    display: flex;
    flex-direction: column;
}

.logo {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 25px;
}

.logo span {
    color: #9b5cff;
}

.new-chat {
    width: 100%;
    padding: 12px;
    border: 1px solid #303b55;
    background: #172033;
    color: white;
    border-radius: 12px;
    cursor: pointer;
    font-size: 15px;
}

.new-chat:hover {
    background: #202b43;
}

.history-title {
    margin-top: 30px;
    color: #8995ad;
    font-size: 13px;
}

.history {
    margin-top: 10px;
    color: #cbd3e1;
    font-size: 14px;
}

/* MAIN */

.main {
    flex: 1;
    display: flex;
    flex-direction: column;
}

/* HEADER */

.header {
    height: 70px;
    border-bottom: 1px solid #20283a;
    display: flex;
    align-items: center;
    padding: 0 25px;
    font-size: 20px;
    font-weight: bold;
}

/* CHAT */

#chat {
    flex: 1;
    overflow-y: auto;
    padding: 30px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    max-width: 950px;
    width: 100%;
    margin: auto;
}

.welcome {
    margin: auto;
    text-align: center;
    color: #9ca7bd;
}

.welcome h1 {
    color: white;
    font-size: 42px;
}

.message {
    max-width: 75%;
    padding: 14px 18px;
    border-radius: 18px;
    line-height: 1.5;
    white-space: pre-wrap;
}

.user {
    align-self: flex-end;
    background: #7c3aed;
    border-bottom-right-radius: 5px;
}

.ai {
    align-self: flex-start;
    background: #172033;
    border-bottom-left-radius: 5px;
}

/* INPUT */

.input-area {
    padding: 18px;
    border-top: 1px solid #20283a;
    background: #0b1120;
}

.input-box {
    max-width: 950px;
    margin: auto;
    display: flex;
    background: #172033;
    border-radius: 18px;
    padding: 8px;
}

input {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: white;
    padding: 14px;
    font-size: 16px;
}

.send {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    cursor: pointer;
    font-size: 16px;
}

.clear {
    margin-top: 15px;
    background: transparent;
    border: 1px solid #303b55;
    color: #9ca7bd;
    padding: 8px;
    border-radius: 8px;
    cursor: pointer;
}

@media(max-width:700px) {

    .sidebar {
        display: none;
    }

    #chat {
        padding: 15px;
    }

    .message {
        max-width: 90%;
    }

    .welcome h1 {
        font-size: 30px;
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

    <div class="history-title">
        CONVERSATION
    </div>

    <div class="history" id="history">
        Current Chat
    </div>

    <button class="clear" onclick="clearChat()">
        Clear Chat
    </button>

</div>


<div class="main">

    <div class="header">
        RAIZEN AI 2.0
    </div>


    <div id="chat">

        <div class="welcome" id="welcome">

            <h1>Welcome to RAIZEN ⚡</h1>

            <p>
                Your Advanced AI Assistant
            </p>

        </div>

    </div>


    <div class="input-area">

        <div class="input-box">

            <input
                id="message"
                type="text"
                placeholder="Ask RAIZEN anything..."
                onkeydown="handleKey(event)"
            >

            <button
                class="send"
                onclick="sendMessage()"
            >
                Send
            </button>

        </div>

    </div>

</div>


<script>

let conversation = [];


/* LOAD SAVED CHAT */

window.onload = function() {

    const saved =
        localStorage.getItem("raizen_chat");

    if (saved) {

        conversation =
            JSON.parse(saved);

        conversation.forEach(item => {

            addMessage(
                item.text,
                item.type,
                false
            );

        });

    }

};


/* ADD MESSAGE */

function addMessage(text, type, save = true) {

    const welcome =
        document.getElementById("welcome");

    if (welcome) {
        welcome.remove();
    }

    const message =
        document.createElement("div");

    message.className =
        "message " + type;

    message.textContent = text;

    document
        .getElementById("chat")
        .appendChild(message);

    document
        .getElementById("chat")
        .scrollTop =
        document
        .getElementById("chat")
        .scrollHeight;


    if (save) {

        conversation.push({
            text: text,
            type: type
        });

        localStorage.setItem(
            "raizen_chat",
            JSON.stringify(conversation)
        );

    }

    return message;
}


/* SEND */

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

        const result =
            await fetch("/chat", {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    message: text,

                    history: conversation

                })

            });


        const data =
            await result.json();


        typing.remove();


        if (data.reply) {

            addMessage(
                "⚡ RAIZEN: " +
                data.reply,
                "ai"
            );

        } else {

            addMessage(
                "⚠️ Error: " +
                (data.detail ||
                "Unknown error"),
                "ai"
            );

        }

    }

    catch (error) {

        typing.remove();

        addMessage(
            "⚠️ Could not connect to RAIZEN.",
            "ai"
        );

    }


    input.disabled = false;

    input.focus();

}


/* ENTER */

function handleKey(event) {

    if (event.key === "Enter") {

        sendMessage();

    }

}


/* NEW CHAT */

function newChat() {

    conversation = [];

    localStorage.removeItem(
        "raizen_chat"
    );

    document.getElementById(
        "chat"
    ).innerHTML = `

        <div class="welcome" id="welcome">

            <h1>
                Welcome to RAIZEN ⚡
            </h1>

            <p>
                Your Advanced AI Assistant
            </p>

        </div>

    `;

}


/* CLEAR */

function clearChat() {

    newChat();

}

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

        messages = [

            {
                "role": "system",
                "content": (
                    "You are RAIZEN, an advanced "
                    "futuristic AI assistant. "
                    "You are intelligent, helpful, "
                    "friendly and confident. "
                    "Give clear and useful answers."
                )
            }

        ]


        # ADD PREVIOUS CONVERSATION

        for item in request.history[-10:]:

            if item.get("type") == "user":

                messages.append({
                    "role": "user",
                    "content": item.get("text", "")
                })

            elif item.get("type") == "ai":

                text = item.get("text", "")

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


        messages.append({

            "role": "user",
            "content": request.message

        })


        response = client.chat.completions.create(

            model="zai-org/GLM-5.3-Flash",

            messages=messages,

            max_tokens=500

        )


        return {

            "reply":
                response.choices[0]
                .message.content

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


@app.get("/health")
def health():

    return {

        "status":
            "RAIZEN AI 2.0 is online",

        "token_loaded":
            bool(HF_TOKEN)

    }
