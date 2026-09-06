import os

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


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>RAIZEN AI</title>

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
    flex-direction: column;
}

/* HEADER */

header {
    height: 70px;
    display: flex;
    align-items: center;
    padding: 0 30px;
    border-bottom: 1px solid #20283a;
    background: #0b1120;
}

.logo {
    font-size: 25px;
    font-weight: bold;
}

.logo span {
    color: #9b5cff;
}

/* CHAT AREA */

#chat {
    flex: 1;
    overflow-y: auto;
    padding: 30px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    max-width: 900px;
    width: 100%;
    margin: auto;
}

/* MESSAGE BUBBLES */

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

/* WELCOME */

.welcome {
    text-align: center;
    margin: auto;
    color: #9ca7bd;
}

.welcome h1 {
    color: white;
    font-size: 42px;
    margin-bottom: 10px;
}

/* INPUT */

.input-area {
    padding: 18px;
    border-top: 1px solid #20283a;
    background: #0b1120;
}

.input-box {
    max-width: 900px;
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

button {
    background: #7c3aed;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    cursor: pointer;
    font-size: 16px;
}

button:hover {
    opacity: 0.9;
}

.typing {
    opacity: 0.7;
    font-style: italic;
}

@media (max-width: 600px) {

    #chat {
        padding: 15px;
    }

    .message {
        max-width: 90%;
    }

    header {
        padding: 0 20px;
    }

    .welcome h1 {
        font-size: 30px;
    }

}

</style>
</head>


<body>

<header>
    <div class="logo">
        ⚡ <span>RAIZEN</span> AI
    </div>
</header>


<div id="chat">

    <div class="welcome" id="welcome">
        <h1>Welcome to RAIZEN ⚡</h1>
        <p>Your Advanced AI Assistant</p>
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

        <button onclick="sendMessage()">
            Send
        </button>

    </div>

</div>


<script>

const chat = document.getElementById("chat");


function handleKey(event) {

    if (event.key === "Enter") {
        sendMessage();
    }

}


function addMessage(text, type) {

    const welcome = document.getElementById("welcome");

    if (welcome) {
        welcome.remove();
    }

    const message = document.createElement("div");

    message.className = "message " + type;

    message.textContent = text;

    chat.appendChild(message);

    chat.scrollTop = chat.scrollHeight;

    return message;
}


async function sendMessage() {

    const input = document.getElementById("message");

    const text = input.value.trim();

    if (!text) {
        return;
    }


    addMessage(text, "user");

    input.value = "";

    input.disabled = true;


    const typing = addMessage(
        "⚡ RAIZEN is thinking...",
        "ai typing"
    );


    try {

        const result = await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                message: text
            })

        });


        const data = await result.json();


        typing.remove();


        if (data.reply) {

            addMessage(
                "⚡ RAIZEN: " + data.reply,
                "ai"
            );

        }

        else {

            addMessage(
                "⚠️ Error: " +
                (data.detail || "Unknown error"),
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

        response = client.chat.completions.create(

            model="zai-org/GLM-5.3-Flash",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are RAIZEN, an advanced futuristic AI assistant. "
                        "You are intelligent, helpful, friendly and confident. "
                        "Give clear and useful answers. "
                        "Your name is RAIZEN."
                    )
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],

            max_tokens=500

        )


        return {
            "reply": response.choices[0].message.content
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/health")
def health():

    return {
        "status": "RAIZEN AI is online",
        "token_loaded": bool(HF_TOKEN)
    }
