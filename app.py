# RAIZEN complete app.py
# Syntax checked successfully.

import os
import json
import re
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from fastapi import FastAPI
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
    custom_instructions: str = ""


PERSONALITIES = {
    "Friendly": "Be friendly, natural, helpful and easy to understand.",
    "Teacher": "Explain clearly like a good teacher using simple examples.",
    "Coding Assistant": "Focus on accurate programming help and practical debugging.",
    "Professional": "Use a professional, clear and structured communication style."
}

STYLES = {
    "Short": "Keep answers concise and direct.",
    "Balanced": "Give balanced answers with useful explanation.",
    "Detailed": "Give detailed explanations and examples when useful."
}


def needs_live_search(message):
    keywords = [
        "latest", "today", "current", "recent", "news",
        "right now", "this week", "this month",
        "search", "trending", "weather", "temperature", "forecast"
    ]

    text = message.lower()

    return any(word in text for word in keywords)


def google_news_search(query):
    try:
        url = (
            "https://news.google.com/rss/search?q="
            + quote(query)
            + "&hl=en-IN&gl=IN&ceid=IN:en"
        )

        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urlopen(request, timeout=10) as response:
            data = response.read().decode("utf-8")

        root = ET.fromstring(data)

        results = []

        for item in root.findall(".//item")[:5]:
            title = item.findtext("title") or ""
            date = item.findtext("pubDate") or ""

            if title:
                results.append({
                    "title": title,
                    "date": date
                })

        return results

    except Exception as error:
        print("NEWS ERROR:", error)
        return []


def wikipedia_search(query):
    try:
        url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query&format=json&list=search"
            "&srsearch=" + quote(query)
            + "&srlimit=3"
        )

        request = Request(
            url,
            headers={"User-Agent": "RAIZEN-AI/1.0"}
        )

        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []

        for item in data.get("query", {}).get("search", []):
            results.append({
                "title": item.get("title", ""),
                "snippet": re.sub(
                    "<.*?>",
                    "",
                    item.get("snippet", "")
                )
            })

        return results

    except Exception as error:
        print("WIKIPEDIA ERROR:", error)
        return []


def weather_open_meteo(city):
    try:
        city = city.strip()

        if not city:
            return None

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            "?name=" + quote(city)
            + "&count=10&language=en&format=json"
        )

        request = Request(
            geo_url,
            headers={"User-Agent": "RAIZEN-AI/1.0"}
        )

        with urlopen(request, timeout=10) as response:
            geo_data = json.loads(
                response.read().decode("utf-8")
            )

        locations = geo_data.get("results", [])

        if not locations:
            return None

        city_clean = city.lower().strip()
        location = None

        for item in locations:
            name = item.get("name", "").strip().lower()

            if name == city_clean:
                location = item
                break

        if location is None:
            location = locations[0]

        latitude = location.get("latitude")
        longitude = location.get("longitude")

        if latitude is None or longitude is None:
            return None

        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            "latitude=" + str(latitude)
            + "&longitude=" + str(longitude)
            + "&current=temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "wind_speed_10m,"
            "weather_code"
            "&timezone=auto"
        )

        request = Request(
            weather_url,
            headers={"User-Agent": "RAIZEN-AI/1.0"}
        )

        with urlopen(request, timeout=10) as response:
            weather_data = json.loads(
                response.read().decode("utf-8")
            )

        current = weather_data.get("current", {})

        codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Fog",
            51: "Light drizzle",
            53: "Drizzle",
            55: "Heavy drizzle",
            61: "Light rain",
            63: "Moderate rain",
            65: "Heavy rain",
            80: "Rain showers",
            81: "Rain showers",
            82: "Heavy rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm",
            99: "Thunderstorm"
        }

        condition = codes.get(
            current.get("weather_code"),
            "Unknown"
        )

        return (
            "Current weather for "
            + location.get("name", city)
            + ", "
            + location.get("country", "")
            + "\n\n"
            + "Condition: " + condition
            + "\n"
            + "Temperature: "
            + str(current.get("temperature_2m"))
            + "°C\n"
            + "Feels like: "
            + str(current.get("apparent_temperature"))
            + "°C\n"
            + "Humidity: "
            + str(current.get("relative_humidity_2m"))
            + "%\n"
            + "Precipitation: "
            + str(current.get("precipitation"))
            + " mm\n"
            + "Wind speed: "
            + str(current.get("wind_speed_10m"))
            + " km/h\n\n"
            + "Source: Open-Meteo"
        )

    except Exception as error:
        print("WEATHER ERROR:", error)
        return None


def weather_search(city):
    return weather_open_meteo(city)


def extract_city(message):
    patterns = [
        r"weather\s+(?:in|at|of)\s+(.+)",
        r"temperature\s+(?:in|at|of)\s+(.+)",
        r"forecast\s+(?:in|at|of)\s+(.+)",
        r"weather\s+(.+)",
        r"temperature\s+(.+)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            message.strip(),
            re.IGNORECASE
        )

        if match:
            city = match.group(1).strip()

            city = re.sub(
                r"\b(today|now|currently|right now|tomorrow)\b",
                "",
                city,
                flags=re.IGNORECASE
            )

            city = city.strip(" ?.,!")

            if city:
                return city

    return None


def live_search(message):
    text = message.lower()

    if (
        "weather" in text
        or "temperature" in text
        or "forecast" in text
    ):
        city = extract_city(message)

        if not city:
            return "WEATHER_ERROR: Please provide a city name."

        result = weather_search(city)

        if result:
            return result

        return "WEATHER_ERROR: Weather data could not be retrieved right now."

    news = google_news_search(message)

    if news:
        output = "LIVE NEWS RESULTS:\n\n"

        for index, item in enumerate(news, 1):
            output += (
                str(index)
                + ". "
                + item["title"]
                + "\nDate: "
                + item["date"]
                + "\n\n"
            )

        return output

    wiki = wikipedia_search(message)

    if wiki:
        output = "WEB INFORMATION:\n\n"

        for index, item in enumerate(wiki, 1):
            output += (
                str(index)
                + ". "
                + item["title"]
                + "\n"
                + item["snippet"]
                + "\n\n"
            )

        return output

    return "LIVE_SEARCH_ERROR: Live search could not retrieve data."


HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAIZEN AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #080b14;
    color: white;
}

.container {
    max-width: 1000px;
    height: 100vh;
    margin: auto;
    padding: 15px;
    display: flex;
    flex-direction: column;
}

.header {
    padding: 15px;
    background: #11192d;
    border: 1px solid #293653;
    border-radius: 16px;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
}

.logo {
    font-size: 24px;
    font-weight: bold;
}

.status {
    opacity: 0.7;
    font-size: 13px;
}

.chat {
    flex: 1;
    overflow-y: auto;
    padding: 10px 0;
}

.message {
    max-width: 82%;
    padding: 12px 15px;
    margin: 8px 0;
    border-radius: 15px;
    white-space: pre-wrap;
    line-height: 1.5;
}

.user {
    margin-left: auto;
    background: #2463eb;
}

.assistant {
    background: #151d31;
    border: 1px solid #293653;
}

.thinking {
    opacity: 0.7;
}

.controls {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}

select,
.control {
    background: #11192d;
    color: white;
    border: 1px solid #293653;
    border-radius: 10px;
    padding: 8px;
}

.input-area {
    display: flex;
    gap: 8px;
}

#messageInput {
    flex: 1;
    padding: 13px;
    background: #11192d;
    color: white;
    border: 1px solid #293653;
    border-radius: 13px;
    outline: none;
}

#sendButton {
    padding: 0 20px;
    background: #2463eb;
    color: white;
    border: 0;
    border-radius: 13px;
    cursor: pointer;
}

#sendButton:disabled {
    opacity: 0.6;
}

.small {
    text-align: center;
    opacity: 0.5;
    font-size: 12px;
    margin-top: 8px;
}

</style>
</head>

<body>

<div class="container">

<div class="header">
<div class="logo">⚡ RAIZEN</div>
<div class="status">AI ONLINE</div>
</div>

<div id="chat" class="chat">
<div id="welcome" class="assistant message">
⚡ Welcome to RAIZEN. Ask me anything.
</div>
</div>

<div class="controls">

<select id="personality">
<option>Friendly</option>
<option>Teacher</option>
<option>Coding Assistant</option>
<option>Professional</option>
</select>

<select id="responseStyle">
<option>Short</option>
<option selected>Balanced</option>
<option>Detailed</option>
</select>

<button class="control" onclick="newChat()">New Chat</button>
<button class="control" onclick="clearChat()">Clear Chat</button>

</div>

<div class="input-area">

<input
id="messageInput"
placeholder="Message RAIZEN..."
autocomplete="off"
>

<button id="sendButton" onclick="sendMessage()">Send</button>

</div>

<div class="small">
RAIZEN • Created and developed by Raihan Kausar
</div>

</div>

<script>

let chatHistory = [];


function saveHistory() {

    localStorage.setItem(
        "raizen_chat_history",
        JSON.stringify(chatHistory.slice(-12))
    );
}


function displayMessage(role, text, extraClass) {

    const chat = document.getElementById("chat");
    const welcome = document.getElementById("welcome");

    if (welcome) {
        welcome.remove();
    }

    const div = document.createElement("div");

    div.className =
        "message "
        + role
        + " "
        + (extraClass || "");

    div.textContent = text;

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;

    return div;
}


async function sendMessage() {

    const input = document.getElementById("messageInput");
    const button = document.getElementById("sendButton");

    const message = input.value.trim();

    if (!message || button.disabled) {
        return;
    }

    input.value = "";
    button.disabled = true;

    displayMessage("user", message);

    chatHistory.push({
        role: "user",
        content: message
    });

    const thinking = displayMessage(
        "assistant",
        "⚡ RAIZEN is thinking...",
        "thinking"
    );

    try {

        const response = await fetch(
            "/chat",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory.slice(-12),
                    personality: document.getElementById("personality").value,
                    response_style: document.getElementById("responseStyle").value,
                    custom_instructions:
                        localStorage.getItem(
                            "raizen_custom_instructions"
                        ) || ""
                })
            }
        );

        const data = await response.json();

        thinking.remove();

        const reply =
            data.reply ||
            "Sorry, I could not generate a response.";

        displayMessage("assistant", reply);

        chatHistory.push({
            role: "assistant",
            content: reply
        });

        saveHistory();

    } catch (error) {

        thinking.remove();

        const reply =
            "⚠️ Something went wrong. Please try again.";

        displayMessage("assistant", reply);

        chatHistory.push({
            role: "assistant",
            content: reply
        });

        saveHistory();
    }

    button.disabled = false;
    input.focus();
}


function clearChat() {

    chatHistory = [];

    localStorage.removeItem(
        "raizen_chat_history"
    );

    document.getElementById("chat").innerHTML =
        '<div id="welcome" class="assistant message">⚡ Welcome to RAIZEN. Ask me anything.</div>';
}


function newChat() {
    clearChat();
}


document.getElementById(
    "messageInput"
).addEventListener(
    "keydown",
    function(event) {

        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }

    }
);


try {

    const saved = localStorage.getItem(
        "raizen_chat_history"
    );

    if (saved) {

        chatHistory = JSON.parse(saved);

        const welcome =
            document.getElementById("welcome");

        if (welcome) {
            welcome.remove();
        }

        chatHistory.forEach(
            function(item) {

                displayMessage(
                    item.role,
                    item.content
                );

            }
        );
    }

} catch (error) {

    chatHistory = [];

}

</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML)


@app.post("/chat")
async def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        return {"reply": "Please enter a message."}

    lower = message.lower()

    if (
        "weather" in lower
        or "temperature" in lower
        or "forecast" in lower
    ):

        city = extract_city(message)

        if not city:
            return {
                "reply":
                    "🌤️ Please mention a city, "
                    "for example: weather in Cuttack."
            }

        result = weather_search(city)

        if result:
            return {"reply": result}

        return {
            "reply":
                "⚠️ I couldn't retrieve live weather data right now. "
                "Please try again later."
        }

    live_data = ""

    if needs_live_search(message):
        live_data = live_search(message)

    personality = PERSONALITIES.get(
        request.personality,
        PERSONALITIES["Friendly"]
    )

    style = STYLES.get(
        request.response_style,
        STYLES["Balanced"]
    )

    system_prompt = f"""
You are RAIZEN, an advanced AI assistant.

You were created and developed by Raihan Kausar.

If someone asks who created, developed, invented,
or made you, say that Raihan Kausar created
and developed you.

Personality:
{personality}

Response style:
{style}

Custom instructions:
{request.custom_instructions.strip()}

Rules:
- Be accurate and helpful.
- Do not invent live information.
- Use live information only when provided.
- If live search fails, say so clearly.

Live information:
{live_data}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for item in request.history[-12:]:

        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and content:

            messages.append({
                "role": role,
                "content": content
            })

    if (
        not request.history
        or request.history[-1].get("content") != message
    ):

        messages.append({
            "role": "user",
            "content": message
        })

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=500
        )

        reply = response.choices[0].message.content

        if not reply:
            reply = "Sorry, I could not generate a response."

        return {"reply": reply}

    except Exception as error:

        print("RAIZEN ERROR:", error)

        return {
            "reply":
                "⚠️ RAIZEN is temporarily unable to respond. "
                "Please try again."
        }


@app.get("/health")
async def health():

    return {
        "status": "ok",
        "token_loaded": bool(HF_TOKEN),
        "model": MODEL,
        "live_search": True
    }
