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
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            height: 70px;
            display: flex;
            align-items: center;
            padding: 0 30px;
            border-bottom: 1px solid #20283a;
        }

        .logo {
            font-size: 26px;
            font-weight: bold;
        }

        .logo span {
            color: #9b5cff;
        }

        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        h1 {
            font-size: 42px;
            margin-bottom: 10px;
        }

        p {
            color: #9ca7bd;
            text-align: center;
        }

        .chat-box {
            width: 100%;
            max-width: 750px;
            display: flex;
            background: #172033;
            padding: 8px;
            border-radius: 18px;
            margin-top: 30px;
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
            padding: 10px 18px;
            cursor: pointer;
            font-size: 16px;
        }

        #response {
            margin-top: 25px;
            max-width: 750px;
            width: 100%;
            background: #172033;
            padding: 18px;
            border-radius: 15px;
            display: none;
            white-space: pre-wrap;
        }

        @media (max-width: 600px) {

            h1 {
                font-size: 30px;
            }

            header {
                padding: 0 20px;
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


<main>

    <h1>Welcome to RAIZEN ⚡</h1>

    <p>Your Advanced AI Assistant</p>


    <div class="chat-box">

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


    <div id="response"></div>

</main>


<script>

function handleKey(event) {

    if (event.key === "Enter") {
        sendMessage();
    }

}


async function sendMessage() {

    const input = document.getElementById("message");

    const response = document.getElementById("response");

    const text = input.value.trim();


    if (!text) {
        return;
    }


    response.style.display = "block";

    response.textContent = "⚡ RAIZEN is thinking...";

    input.value = "";


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


        if (data.reply) {

            response.textContent =
                "⚡ RAIZEN: " + data.reply;

        }

        else {

            response.textContent =
                "⚠️ Error: " +
                (data.detail || "Unknown error");

        }

    }

    catch (error) {

        response.textContent =
            "⚠️ Could not connect to RAIZEN.";

    }

}

</script>


</body>
</html>
""")


@app.post("/chat")
def chat(request: ChatRequest):

    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="HF_TOKEN is not configured."
        )

    try:

        response = client.chat.completions.create(

            model="zai-org/GLM-5.3:novita",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are RAIZEN, an advanced AI assistant. "
                        "You are helpful, intelligent, friendly, and clear. "
                        "Give useful answers to the user."
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
        "token_loaded": bool(HF_TOKEN),
        "token_length": len(HF_TOKEN) if HF_TOKEN else 0
    }
