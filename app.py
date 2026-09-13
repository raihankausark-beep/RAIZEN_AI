import os
import json
import re
from urllib.parse import quote
from urllib.request import urlopen, Request

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient


# ============================================================
# RAIZEN
# ============================================================

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "zai-org/GLM-5.3-Flash"

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)


# ============================================================
# PERSONALITY / RESPONSE STYLE
# ============================================================

PERSONALITIES = {
    "Friendly": (
        "Be friendly, natural, helpful and easy to talk to. "
        "Use simple language."
    ),

    "Teacher": (
        "Act like a good teacher. Explain concepts clearly "
        "with simple examples and step-by-step explanations."
    ),

    "Coding Assistant": (
        "Focus on programming, debugging and technical accuracy. "
        "Give practical code and explain important parts."
    ),

    "Professional": (
        "Use a professional, clear and structured communication style."
    )
}


STYLES = {
    "Short": (
        "Keep the answer short and direct unless more detail is necessary."
    ),

    "Balanced": (
        "Give a balanced answer with enough explanation but avoid unnecessary length."
    ),

    "Detailed": (
        "Give a detailed explanation with useful examples and steps."
    )
}


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str
    history: list = []
    personality: str = "Friendly"
    response_style: str = "Balanced"
    custom_instructions: str = ""


# ============================================================
# LIVE SEARCH DETECTION
# ============================================================

def needs_live_search(message):
    text = message.lower()

    keywords = [
        "latest",
        "today",
        "current",
        "recent",
        "news",
        "right now",
        "this week",
        "this month",
        "search",
        "trending",
        "weather",
        "temperature",
        "forecast"
    ]

    return any(word in text for word in keywords)


# ============================================================
# GOOGLE NEWS SEARCH
# ============================================================

def google_news_search(query):

    try:
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={quote(query)}"
            "&hl=en-IN"
            "&gl=IN"
            "&ceid=IN:en"
        )

        request = Request(
            rss_url,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urlopen(request, timeout=10) as response:
            data = response.read().decode("utf-8")

        import xml.etree.ElementTree as ET

        root = ET.fromstring(data)

        results = []

        for item in root.findall(".//item")[:5]:

            title = item.findtext("title")
            link = item.findtext("link")
            pub_date = item.findtext("pubDate")

            if title:
                results.append({
                    "title": title,
                    "link": link,
                    "date": pub_date
                })

        return results

    except Exception:
        return []


# ============================================================
# WIKIPEDIA SEARCH
# ============================================================

def wikipedia_search(query):

    try:

        url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query"
            "&format=json"
            "&list=search"
            f"&srsearch={quote(query)}"
            "&srlimit=3"
        )

        request = Request(
            url,
            headers={
                "User-Agent": "RAIZEN-AI/1.0"
            }
        )

        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []

        for item in data.get("query", {}).get("search", []):

            results.append({
                "title": item.get("title"),
                "snippet": re.sub(
                    "<.*?>",
                    "",
                    item.get("snippet", "")
                )
            })

        return results

    except Exception:
        return []


# ============================================================
# WEATHER - OPEN METEO
# ============================================================

def weather_open_meteo(city):

    try:

        city = city.strip()

        if not city:
            return None

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={quote(city)}"
            "&count=10"
            "&language=en"
            "&format=json"
        )

        request = Request(
            geo_url,
            headers={
                "User-Agent": "RAIZEN-AI/1.0"
            }
        )

        with urlopen(request, timeout=10) as response:
            geo_data = json.loads(
                response.read().decode("utf-8")
            )

        locations = geo_data.get("results", [])

        if not locations:
            return None

        city_clean = city.lower().strip()

        exact_match = None

        for location in locations:

            name = location.get(
                "name",
                ""
            ).strip().lower()

            if name == city_clean:
                exact_match = location
                break

        if exact_match:
            location = exact_match

        else:

            starts_match = None

            for item in locations:

                name = item.get(
                    "name",
                    ""
                ).strip().lower()

                if name.startswith(city_clean):
                    starts_match = item
                    break

            location = starts_match or locations[0]

        latitude = location.get("latitude")
        longitude = location.get("longitude")

        if latitude is None or longitude is None:
            return None

        weather_url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}"
            f"&longitude={longitude}"
            "&current="
            "temperature_2m,"
            "relative_humidity_2m,"
            "apparent_temperature,"
            "precipitation,"
            "wind_speed_10m,"
            "weather_code"
            "&timezone=auto"
        )

        request = Request(
            weather_url,
            headers={
                "User-Agent": "RAIZEN-AI/1.0"
            }
        )

        with urlopen(request, timeout=10) as response:
            weather_data = json.loads(
                response.read().decode("utf-8")
            )

        current = weather_data.get("current", {})

        temperature = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        feels_like = current.get("apparent_temperature")
        precipitation = current.get("precipitation")
        wind = current.get("wind_speed_10m")
        weather_code = current.get("weather_code")

        weather_names = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

        condition = weather_names.get(
            weather_code,
            "Unknown"
        )

        actual_city = location.get(
            "name",
            city
        )

        country = location.get(
            "country",
            ""
        )

        result = (
            f"Current weather for {actual_city}, {country}\n\n"
            f"Condition: {condition}\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%\n"
            f"Precipitation: {precipitation} mm\n"
            f"Wind speed: {wind} km/h\n\n"
            f"Source: Open-Meteo"
        )

        return result

    except Exception:
        return None


# ============================================================
# WEATHER SEARCH
# ============================================================

def weather_search(city):

    result = weather_open_meteo(city)

    if result:
        return result

    return None


# ============================================================
# EXTRACT CITY FROM WEATHER QUERY
# ============================================================

def extract_city(message):

    text = message.strip()

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
            text,
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


# ============================================================
# LIVE SEARCH
# ============================================================

def live_search(message):

    text = message.lower()

    # WEATHER
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

        return (
            "WEATHER_ERROR: Weather data could not be retrieved "
            "right now. Please try again later."
        )

    # NEWS
    news = google_news_search(message)

    if news:

        output = "LIVE NEWS RESULTS:\n\n"

        for index, item in enumerate(news, 1):

            output += (
                f"{index}. {item.get('title', 'No title')}\n"
                f"Date: {item.get('date', 'Unknown')}\n\n"
            )

        return output

    # WIKIPEDIA FALLBACK
    wiki = wikipedia_search(message)

    if wiki:

        output = "WEB INFORMATION:\n\n"

        for index, item in enumerate(wiki, 1):

            output += (
                f"{index}. {item.get('title', 'No title')}\n"
                f"{item.get('snippet', '')}\n\n"
            )

        return output

    return "LIVE_SEARCH_ERROR: Live search could not retrieve data."


# ============================================================
# HTML
# ============================================================

HTML = """
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

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top,
            #17213d,
            #080b14 55%
        );

    color: white;

    min-height: 100vh;
}

.container {

    max-width: 1100px;

    margin: auto;

    height: 100vh;

    display: flex;

    flex-direction: column;

    padding: 18px;
}

.header {

    display: flex;

    align-items: center;

    justify-content: space-between;

    padding: 14px 18px;

    background: rgba(20, 28, 50, 0.85);

    border: 1px solid #273556;

    border-radius: 18px;

    margin-bottom: 14px;

    backdrop-filter: blur(12px);
}

.logo {

    font-size: 24px;

    font-weight: 800;

    letter-spacing: 1px;
}

.status {

    font-size: 12px;

    color: #8db4ff;
}

.chat {

    flex: 1;

    overflow-y: auto;

    padding: 10px 4px 20px;

    display: flex;

    flex-direction: column;

    gap: 12px;
}

.message {

    max-width: 82%;

    padding: 13px 16px;

    border-radius: 16px;

    line-height: 1.55;

    white-space: pre-wrap;

    word-wrap: break-word;
}

.user {

    align-self: flex-end;

    background: #2463eb;

    border-bottom-right-radius: 5px;
}

.assistant {

    align-self: flex-start;

    background: #151d31;

    border: 1px solid #293653;

    border-bottom-left-radius: 5px;
}

.thinking {

    opacity: 0.7;

    border: 1px solid #293653;
}

.input-area {

    display: flex;

    gap: 10px;

    padding-top: 10px;
}

.input {

    flex: 1;

    background: #10172a;

    color: white;

    border: 1px solid #2a3859;

    outline: none;

    border-radius: 15px;

    padding: 14px 16px;

    font-size: 15px;
}

.input:focus {

    border-color: #4d7cff;
}

.send {

    border: none;

    border-radius: 15px;

    padding: 0 22px;

    background: #2867f0;

    color: white;

    font-weight: 700;

    cursor: pointer;
}

.send:hover {

    background: #3b75f2;
}

.send:disabled {

    opacity: 0.6;

    cursor: not-allowed;
}

.controls {

    display: flex;

    gap: 8px;

    margin-top: 10px;

    flex-wrap: wrap;
}

select,
button.control {

    background: #11192d;

    color: white;

    border: 1px solid #2a3859;

    border-radius: 10px;

    padding: 8px 11px;
}

button.control {

    cursor: pointer;
}

.empty {

    margin: auto;

    text-align: center;

    opacity: 0.65;
}

.small {

    font-size: 12px;

    opacity: 0.6;

    text-align: center;

    padding-top: 8px;
}

@media(max-width: 600px) {

    .container {
        padding: 10px;
    }

    .message {
        max-width: 92%;
    }

    .send {
        padding: 0 16px;
    }
}

</style>

</head>

<body>

<div class="container">

    <div class="header">

        <div class="logo">
            ⚡ RAIZEN
        </div>

        <div class="status">
            AI ONLINE
        </div>

    </div>


    <div id="chat" class="chat">

        <div id="empty" class="empty">

            <h2>⚡ Welcome to RAIZEN</h2>

            <p>
                Ask me anything.
            </p>

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


        <button
            class="control"
            onclick="newChat()">
            New Chat
        </button>


        <button
            class="control"
            onclick="clearChat()">
            Clear Chat
        </button>

    </div>


    <div class="input-area">

        <input
            id="messageInput"
            class="input"
            placeholder="Message RAIZEN..."
            autocomplete="off"
        >

        <button
            id="sendButton"
            class="send"
            onclick="sendMessage()">
            Send
        </button>

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


function loadHistory() {

    try {

        const saved =
            localStorage.getItem(
                "raizen_chat_history"
            );

        if (!saved) {
            return;
        }

        chatHistory =
            JSON.parse(saved);

        for (
            const item of chatHistory
        ) {

            displayMessage(
                item.role,
                item.content
            );
        }

    } catch (error) {

        chatHistory = [];

    }
}


function displayMessage(
    role,
    text,
    extraClass = ""
) {

    const chat =
        document.getElementById("chat");

    const empty =
        document.getElementById("empty");

    if (empty) {
        empty.remove();
    }

    const div =
        document.createElement("div");

    div.className =
        "message " +
        role +
        " " +
        extraClass;

    div.textContent = text;

    chat.appendChild(div);

    chat.scrollTop =
        chat.scrollHeight;

    return div;
}


async function sendMessage() {

    const input =
        document.getElementById(
            "messageInput"
        );

    const sendButton =
        document.getElementById(
            "sendButton"
        );

    const message =
        input.value.trim();

    if (!message) {
        return;
    }

    input.value = "";

    sendButton.disabled = true;


    displayMessage(
        "user",
        message
    );


    chatHistory.push({
        role: "user",
        content: message
    });


    const thinkingMessage =
        displayMessage(
            "assistant",
            "⚡ RAIZEN is thinking...",
            "thinking"
        );


    const personality =
        document.getElementById(
            "personality"
        ).value;


    const responseStyle =
        document.getElementById(
            "responseStyle"
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

                        message: message,

                        history:
                            chatHistory.slice(-12),

                        personality:
                            personality,

                        response_style:
                            responseStyle,

                        custom_instructions:
                            localStorage.getItem(
                                "raizen_custom_instructions"
                            ) || ""

                    })

                }
            );


        const data =
            await response.json();


        if (thinkingMessage) {
            thinkingMessage.remove();
        }


        const reply =
            data.reply ||
            "Sorry, I could not generate a response.";


        displayMessage(
            "assistant",
            reply
        );


        chatHistory.push({
            role: "assistant",
            content: reply
        });


        saveHistory();


    } catch (error) {

        if (thinkingMessage) {
            thinkingMessage.remove();
        }


        const errorMessage =
            "⚠️ Something went wrong. Please try again.";


        displayMessage(
            "assistant",
            errorMessage
        );


        chatHistory.push({
            role: "assistant",
            content: errorMessage
        });


        saveHistory();

    } finally {

        sendButton.disabled = false;

        input.focus();

    }

}


function clearChat() {

    chatHistory = [];

    localStorage.removeItem(
        "raizen_chat_history"
    );

    const chat =
        document.getElementById("chat");

    chat.innerHTML = `
        <div id="empty" class="empty">
            <h2>⚡ Welcome to RAIZEN</h2>
            <p>Ask me anything.</p>
        </div>
    `;
}


function newChat() {

    clearChat();

}


document
    .getElementById("messageInput")
    .addEventListener(
        "keydown",
        function(event) {

            if (
                event.key === "Enter"
                && !event.shiftKey
            ) {

                event.preventDefault();

                sendMessage();

            }

        }
    );


loadHistory();

</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():

    return HTMLResponse(
        content=HTML
    )


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:

        return {
            "reply": "Please enter a message."
        }


    # --------------------------------------------------------
    # DIRECT WEATHER RESPONSE
    # --------------------------------------------------------

    if (
        "weather" in message.lower()
        or "temperature" in message.lower()
        or "forecast" in message.lower()
    ):

        city = extract_city(message)

        if not city:

            return {
                "reply":
                    "🌤️ Please mention a city name, "
                    "for example: weather in Cuttack."
            }


        weather_result =
            weather_search(city)


        if weather_result:

            return {
                "reply":
                    weather_result
            }


        return {
            "reply":
                "⚠️ I couldn't retrieve live weather "
                "data right now. Please try again later."
        }


    # --------------------------------------------------------
    # LIVE SEARCH
    # --------------------------------------------------------

    live_data = ""

    if needs_live_search(message):

        live_data = live_search(message)


    # --------------------------------------------------------
    # SYSTEM PROMPT
    # --------------------------------------------------------

    personality =
        PERSONALITIES.get(
            request.personality,
            PERSONALITIES["Friendly"]
        )


    style =
        STYLES.get(
            request.response_style,
            STYLES["Balanced"]
        )


    custom =
        request.custom_instructions.strip()


    system_prompt = f"""

You are RAIZEN, an advanced AI assistant.

You were created and developed by Raihan Kausar.

If someone asks who invented, created, developed,
or made you, say that Raihan Kausar created
and developed you.

PERSONALITY:
{personality}

RESPONSE STYLE:
{style}

CUSTOM USER INSTRUCTIONS:
{custom}

IMPORTANT:

- Be accurate and helpful.
- Do not pretend to know information you do not know.
- Keep explanations clear.
- If live information is provided below, use it carefully.
- If LIVE_SEARCH_ERROR appears, clearly tell the user
  that live search could not retrieve data.
- If WEATHER_ERROR appears, clearly explain that weather
  data could not be retrieved or that a city name is required.
- Do not treat an error message as real-world information.
- When live information is available, mention that it is live/current.

LIVE INFORMATION:
{live_data}

"""


    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]


    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    for item in request.history[-12:]:

        role = item.get("role")

        content = item.get("content")


        if role in ["user", "assistant"] and content:

            messages.append({
                "role": role,
                "content": content
            })


    # --------------------------------------------------------
    # CURRENT MESSAGE
    # --------------------------------------------------------

    if not request.history or (
        request.history[-1].get("content")
        != message
    ):

        messages.append({
            "role": "user",
            "content": message
        })


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    try:

        response = client.chat.completions.create(

            model=MODEL,

            messages=messages,

            max_tokens=500

        )


        reply = response.choices[0].message.content


        if not reply:

            reply = (
                "Sorry, I could not generate a response."
            )


        return {
            "reply": reply
        }


    except Exception as error:

        print(
            "RAIZEN ERROR:",
            str(error)
        )

        return {
            "reply":
                "⚠️ RAIZEN is temporarily unable "
                "to respond. Please try again."
        }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {

        "status": "ok",

        "token_loaded":
            bool(HF_TOKEN),

        "model":
            MODEL,

        "live_search":
            True

    }
