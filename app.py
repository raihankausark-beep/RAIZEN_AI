import os
import json
import re
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

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


def needs_live_search(message):
    words = [
        "latest",
        "today",
        "current",
        "recent",
        "news",
        "right now",
        "this week",
        "this month",
        "search",
        "what happened",
        "trending",
        "weather",
        "temperature",
        "forecast"
    ]

    text = message.lower()

    return any(word in text for word in words)


def google_news_search(query):
    url = (
        "https://news.google.com/rss/search?q="
        + quote(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 RAIZEN/1.0"
        }
    )

    with urlopen(request, timeout=15) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)

    results = []

    for item in root.findall(".//item")[:6]:
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        published = item.findtext("pubDate", "").strip()

        if title and link:
            results.append({
                "title": title,
                "link": link,
                "published": published
            })

    return results


def wikipedia_search(query):
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query&list=search&format=json"
        "&utf8=1&srlimit=5&srsearch="
        + quote(query)
    )

    request = Request(
        url,
        headers={
            "User-Agent": "RAIZEN/1.0"
        }
    )

    with urlopen(request, timeout=15) as response:
        data = json.loads(
            response.read().decode("utf-8")
        )

    results = []

    for item in data.get("query", {}).get("search", [])[:5]:
        title = item.get("title", "")

        snippet = re.sub(
            r"<.*?>",
            "",
            item.get("snippet", "")
        )

        if title:
            results.append({
                "title": title,
                "snippet": snippet,
                "link": (
                    "https://en.wikipedia.org/wiki/"
                    + quote(title.replace(" ", "_"))
                )
            })

    return results


def weather_search(city):
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        "?name="
        + quote(city)
        + "&count=1&language=en&format=json"
    )

    geo_request = Request(
        geo_url,
        headers={
            "User-Agent": "RAIZEN/1.0"
        }
    )

    with urlopen(geo_request, timeout=15) as response:
        geo_data = json.loads(
            response.read().decode("utf-8")
        )

    locations = geo_data.get("results", [])

    if not locations:
        return []

    location = locations[0]

    latitude = location["latitude"]
    longitude = location["longitude"]

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude="
        + str(latitude)
        + "&longitude="
        + str(longitude)
        + "&current="
        "temperature_2m,"
        "relative_humidity_2m,"
        "apparent_temperature,"
        "precipitation,"
        "weather_code,"
        "wind_speed_10m"
        "&timezone=auto"
    )

    weather_request = Request(
        weather_url,
        headers={
            "User-Agent": "RAIZEN/1.0"
        }
    )

    with urlopen(weather_request, timeout=15) as response:
        weather_data = json.loads(
            response.read().decode("utf-8")
        )

    current = weather_data.get("current", {})

    if not current:
        return []

    return [{
        "city": location.get("name", city),
        "country": location.get("country", ""),
        "temperature": current.get("temperature_2m"),
        "feels_like": current.get("apparent_temperature"),
        "humidity": current.get("relative_humidity_2m"),
        "rain": current.get("precipitation"),
        "wind": current.get("wind_speed_10m"),
        "time": current.get("time")
    }]


def extract_city(message):
    text = message.strip()

    patterns = [
        r"weather\s+(?:in|at|for)\s+(.+)",
        r"temperature\s+(?:in|at|for)\s+(.+)",
        r"forecast\s+(?:in|for)\s+(.+)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            city = match.group(1)

            city = re.split(
                r"\b(?:today|tomorrow|now|right now)\b",
                city,
                flags=re.IGNORECASE
            )[0]

            return city.strip(" ?.,!")

    return None


def live_search(query):
    text = query.lower()

    # WEATHER
    if any(
        word in text
        for word in [
            "weather",
            "temperature",
            "forecast"
        ]
    ):
        city = extract_city(query)

        if not city:
            return [
                "WEATHER_ERROR: Please specify the city name."
            ]

        try:
            weather = weather_search(city)

            if weather:
                item = weather[0]

                return [
                    "LIVE WEATHER: "
                    + item["city"]
                    + ", "
                    + item["country"]
                    + " | Temperature: "
                    + str(item["temperature"])
                    + "°C"
                    + " | Feels like: "
                    + str(item["feels_like"])
                    + "°C"
                    + " | Humidity: "
                    + str(item["humidity"])
                    + "%"
                    + " | Rain: "
                    + str(item["rain"])
                    + " mm"
                    + " | Wind: "
                    + str(item["wind"])
                    + " km/h"
                    + " | Updated: "
                    + str(item["time"])
                ]

            return [
                "WEATHER_ERROR: No weather data found."
            ]

        except Exception as error:
            return [
                "WEATHER_ERROR: "
                + str(error)
            ]

    # NEWS
    news_error = "Unknown error."

    try:
        news = google_news_search(query)

        if news:
            results = []

            for item in news:
                results.append(
                    "NEWS: "
                    + item["title"]
                    + " | "
                    + item["published"]
                    + " | "
                    + item["link"]
                )

            return results

        news_error = "No Google News results."

    except Exception as error:
        news_error = str(error)

    # WIKIPEDIA FALLBACK
    wiki_error = "Unknown error."

    try:
        wiki = wikipedia_search(query)

        if wiki:
            results = []

            for item in wiki:
                results.append(
                    "REFERENCE: "
                    + item["title"]
                    + " | "
                    + item["snippet"]
                    + " | "
                    + item["link"]
                )

            return results

        wiki_error = "No Wikipedia results."

    except Exception as error:
        wiki_error = str(error)

    return [
        "LIVE_SEARCH_ERROR: "
        "Google News failed: "
        + news_error
        + " | Wikipedia failed: "
        + wiki_error
    ]


@app.get("/", response_class=HTMLResponse)
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
    background: #070b14;
    color: white;
    font-family: Arial, sans-serif;
}

.header {
    text-align: center;
    padding: 20px;
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

select,
.action {
    background: #111a2d;
    color: white;
    border: 1px solid #293653;
    padding: 10px;
    border-radius: 8px;
}

.action {
    cursor: pointer;
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
    cursor: pointer;
}

.send:hover {
    background: #1d4ed8;
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

}

</style>

</head>

<body>

<div class="header">

<h1>⚡ RAIZEN</h1>

<p>AI Assistant • Live Information</p>

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
        "⚡ RAIZEN is thinking..."
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

    live_context = ""

    if needs_live_search(request.message):

        results = live_search(request.message)

        if results:

            live_context = (
                "\n\nLIVE INFORMATION "
                "FROM WEB SOURCES:\n"
                "Use these sources when "
                "answering. Do not invent "
                "information.\n"
            )

            for index, result in enumerate(
                results,
                1
            ):

                live_context += (
                    f"\n{index}. "
                    f"{result}\n"
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
- If live web information is provided, use it.
- Clearly say when information could not be found.
- Never invent sources.
- Never pretend you searched if no live results exist.
- Keep answers natural and easy to understand.
- If LIVE_SEARCH_ERROR appears, clearly tell the user
  that live search could not retrieve data.
- If WEATHER_ERROR appears, clearly explain that weather
  data could not be retrieved or that a city name is required.
- Do not treat an error message as real-world information.
- When live information is available, mention that the
  information is live/current.

{live_context}
"""

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
        "live_search": True
    }
