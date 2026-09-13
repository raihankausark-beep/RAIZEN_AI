import os
import json
import re
import base64
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from docx import Document

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "zai-org/GLM-5.3-Flash"

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)
VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

vision_client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)


class ChatRequest(BaseModel):
    message: str
    history: list = []
    personality: str = "Friendly"
    response_style: str = "Balanced"
    custom_instructions: str = ""
    document_text: str = ""



class FileRequest(BaseModel):
    message: str
    document_text: str
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



def extract_document_text(filename, raw_bytes):
    from io import BytesIO
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        reader = PdfReader(BytesIO(raw_bytes))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()

    if lower_name.endswith(".docx"):
        document = Document(BytesIO(raw_bytes))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(parts).strip()

    if lower_name.endswith(".txt"):
        return raw_bytes.decode("utf-8", errors="replace").strip()

    raise ValueError("Unsupported file type. Use PDF, DOCX, or TXT.")


def limit_document_text(text, max_chars=50000):
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Document truncated for processing.]"


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
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=10) as response:
            data = response.read().decode("utf-8")

        root = ET.fromstring(data)
        results = []

        for item in root.findall(".//item")[:5]:
            title = item.findtext("title") or ""
            date = item.findtext("pubDate") or ""
            link = item.findtext("link") or ""
            source = item.findtext("source") or "Google News"
            if title:
                results.append({
                    "title": title,
                    "date": date,
                    "link": link,
                    "source": source
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
            "&srsearch=" + quote(query) + "&srlimit=3"
        )
        request = Request(url, headers={"User-Agent": "RAIZEN-AI/1.0"})

        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            results.append({
                "title": title,
                "snippet": re.sub("<.*?>", "", item.get("snippet", "")),
                "link": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                "source": "Wikipedia"
            })

        return results
    except Exception as error:
        print("WIKIPEDIA ERROR:", error)
        return []


# Direct coordinates prevent Cuttack from being incorrectly resolved
# to a nearby locality such as Urali.
CITY_COORDINATES = {
    "cuttack": {
        "latitude": 20.4625,
        "longitude": 85.8830,
        "name": "Cuttack",
        "country": "India"
    },
    "bhubaneswar": {
        "latitude": 20.2961,
        "longitude": 85.8245,
        "name": "Bhubaneswar",
        "country": "India"
    },
    "delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090,
        "name": "Delhi",
        "country": "India"
    },
    "mumbai": {
        "latitude": 19.0760,
        "longitude": 72.8777,
        "name": "Mumbai",
        "country": "India"
    },
    "kolkata": {
        "latitude": 22.5726,
        "longitude": 88.3639,
        "name": "Kolkata",
        "country": "India"
    },
    "chennai": {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "name": "Chennai",
        "country": "India"
    },
    "hyderabad": {
        "latitude": 17.3850,
        "longitude": 78.4867,
        "name": "Hyderabad",
        "country": "India"
    },
    "bangalore": {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "name": "Bengaluru",
        "country": "India"
    },
    "bengaluru": {
        "latitude": 12.9716,
        "longitude": 77.5946,
        "name": "Bengaluru",
        "country": "India"
    }
}


def weather_condition(code):
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
        56: "Light freezing drizzle",
        57: "Heavy freezing drizzle",
        61: "Light rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Light snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Rain showers",
        81: "Rain showers",
        82: "Heavy rain showers",
        85: "Snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with hail",
        99: "Thunderstorm with heavy hail"
    }
    return codes.get(code, "Unknown")


def get_city_location(city):
    city = city.strip()
    if not city:
        return None

    known = CITY_COORDINATES.get(city.lower())
    if known:
        return known

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
        geo_data = json.loads(response.read().decode("utf-8"))

    locations = geo_data.get("results", [])
    if not locations:
        return None

    city_clean = city.lower()
    for item in locations:
        if item.get("name", "").strip().lower() == city_clean:
            return item

    return locations[0]


def weather_open_meteo(city):
    try:
        location = get_city_location(city)
        if not location:
            return None

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
            weather_data = json.loads(response.read().decode("utf-8"))

        current = weather_data.get("current", {})
        condition = weather_condition(current.get("weather_code"))

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
            + "Â°C\n"
            + "Feels like: "
            + str(current.get("apparent_temperature"))
            + "Â°C\n"
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
        print("OPEN-METEO WEATHER ERROR:", error)
        return None


def weather_met_norway(city):
    try:
        location = get_city_location(city)
        if not location:
            return None

        latitude = location.get("latitude")
        longitude = location.get("longitude")

        if latitude is None or longitude is None:
            return None

        url = (
            "https://api.met.no/weatherapi/locationforecast/2.0/compact?"
            "lat=" + str(round(float(latitude), 4))
            + "&lon=" + str(round(float(longitude), 4))
        )

        request = Request(
            url,
            headers={
                "User-Agent": (
                    "RAIZEN-AI/1.0 "
                    "(https://github.com/raihankausark-beep/RAIZEN_AI)"
                )
            }
        )

        with urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))

        timeseries = (
            data.get("properties", {}).get("timeseries", [])
        )
        if not timeseries:
            return None

        now = datetime.now(timezone.utc)
        selected = timeseries[0]

        for item in timeseries:
            timestamp = item.get("time")
            if not timestamp:
                continue
            try:
                item_time = datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
                if item_time >= now:
                    selected = item
                    break
            except Exception:
                continue

        data_block = selected.get("data", {})
        instant = (
            data_block.get("instant", {}).get("details", {})
        )
        next_hour = data_block.get("next_1_hours", {})
        summary = next_hour.get("summary", {})
        details = next_hour.get("details", {})

        temperature = instant.get("air_temperature")
        humidity = instant.get("relative_humidity")
        wind_speed = instant.get("wind_speed")
        precipitation = details.get("precipitation_amount")

        symbol = summary.get("symbol_code", "unknown")
        condition = (
            symbol.replace("_", " ")
            .replace("day", "")
            .replace("night", "")
            .strip()
            .title()
        )

        return (
            "Current weather for "
            + location.get("name", city)
            + ", "
            + location.get("country", "")
            + "\n\n"
            + "Condition: " + condition
            + "\n"
            + "Temperature: " + str(temperature) + "Â°C\n"
            + "Humidity: " + str(humidity) + "%\n"
            + "Precipitation: " + str(precipitation) + " mm\n"
            + "Wind speed: " + str(wind_speed) + " m/s\n\n"
            + "Source: MET Norway"
        )
    except Exception as error:
        print("MET NORWAY WEATHER ERROR:", error)
        return None


def weather_search(city):
    result = weather_open_meteo(city)
    if result:
        return result

    result = weather_met_norway(city)
    if result:
        return result

    return None


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
    text = message.lower().strip()

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
            "WEATHER_ERROR: "
            "Weather data could not be retrieved right now."
        )

    news = google_news_search(message)

    if news:
        output = "LIVE NEWS RESULTS:\n\n"
        for index, item in enumerate(news, 1):
            output += (
                str(index)
                + ". "
                + item["title"]
                + "\nSource: "
                + item.get("source", "Google News")
                + "\nDate: "
                + item.get("date", "")
                + "\nLink: "
                + item.get("link", "")
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
                + "\nSource: "
                + item.get("source", "Wikipedia")
                + "\nLink: "
                + item.get("link", "")
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
* { box-sizing: border-box; }

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

.file-panel {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
    flex-wrap: wrap;
}
.file-input {
    max-width: 100%;
    color: #cbd5e1;
    font-size: 13px;
}
.file-name {
    font-size: 12px;
    opacity: 0.7;
}

.small {
    text-align: center;
    opacity: 0.5;
    font-size: 12px;
    margin-top: 8px;
}

.message a {
    color: #7db2ff;
    text-decoration: underline;
    word-break: break-all;
}

.search-badge {
    display: inline-block;
    font-size: 11px;
    padding: 3px 7px;
    border: 1px solid #3a4d73;
    border-radius: 999px;
    opacity: 0.8;
    margin-bottom: 5px;
}
</style>
</head>

<body>
<div class="container">

<div class="header">
<div class="logo">â¡ RAIZEN</div>
<div class="status">AI ONLINE</div>
</div>

<div id="chat" class="chat">
<div id="welcome" class="assistant message">
â¡ Welcome to RAIZEN. Ask me anything.
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

<div class="file-panel">
<input id="fileInput" class="file-input" type="file" accept=".pdf,.docx,.txt">
<span id="fileName" class="file-name">No document selected</span>
<button class="control" onclick="clearDocument()">Remove document</button>
</div>

<div class="file-panel">
<input id="imageInput" class="file-input" type="file"
       accept=".png,.jpg,.jpeg,.webp">

<span id="imageName" class="file-name">No image selected</span>

<button class="control" onclick="analyzeImage()">
🖼️ Analyze Image
</button>
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
RAIZEN â¢ Created and developed by Raihan Kausar
</div>

</div>

<script>
let chatHistory = [];
let documentText = "";
let documentName = "";

document.getElementById("fileInput").addEventListener("change", async function() {
    const file = this.files[0];
    if (!file) return;

    const lower = file.name.toLowerCase();
    if (![".pdf", ".docx", ".txt"].some(ext => lower.endsWith(ext))) {
        displayMessage("assistant", "â ï¸ Please upload a PDF, DOCX, or TXT file.");
        this.value = "";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    document.getElementById("fileName").textContent = "Reading " + file.name + "...";

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Upload failed.");
        }

        documentText = data.text || "";
        documentName = data.filename || file.name;
        document.getElementById("fileName").textContent = "ð " + documentName + " ready";

        displayMessage("assistant",
            "ð " + documentName +
            " is ready. Ask me to summarize it, explain it, find important points, or create questions from it."
        );
    } catch (error) {
        documentText = "";
        documentName = "";
        document.getElementById("fileName").textContent = "Upload failed";
        displayMessage("assistant", "â ï¸ " + error.message);
    }
});

function clearDocument() {
    documentText = "";
    documentName = "";
    document.getElementById("fileInput").value = "";
    document.getElementById("fileName").textContent = "No document selected";
}

function saveHistory() {
    localStorage.setItem(
        "raizen_chat_history",
        JSON.stringify(chatHistory.slice(-12))
    );
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function linkify(html) {
    return html.replace(
        /(https?:\/\/[^\s<]+)/g,
        function(url) {
            const cleanUrl = url.replace(/[.,)]+$/, "");
            const trailing = url.slice(cleanUrl.length);
            return '<a href="' + cleanUrl + '" target="_blank" rel="noopener noreferrer">'
                + cleanUrl + '</a>' + trailing;
        }
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
        "message " + role + " " + (extraClass || "");

    div.innerHTML = linkify(escapeHtml(text));

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;

    return div;
}

async function analyzeImage() {
    const input = document.getElementById("imageInput");
    const messageInput = document.getElementById("messageInput");

    if (!input.files.length) {
        alert("Please select an image first.");
        return;
    }

    const file = input.files[0];

    if (file.size > 8 * 1024 * 1024) {
        alert("Image must be smaller than 8 MB.");
        return;
    }

    const allowed = ["image/png", "image/jpeg", "image/webp"];

    if (!allowed.includes(file.type)) {
        alert("Please upload PNG, JPG, JPEG, or WEBP.");
        return;
    }

    displayMessage("You", "🖼️ Analyzing image...");

    const formData = new FormData();
    formData.append("file", file);
    formData.append(
        "message",
        messageInput.value.trim() || "Describe this image."
    );

    try {
        const response = await fetch("/vision", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        displayMessage("RAIZEN", data.reply);

    } catch (error) {
        displayMessage(
            "RAIZEN",
            "⚠️ I couldn't analyze the image right now."
        );
    }
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
        "â¡ RAIZEN is thinking...",
        "thinking"
    );

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory.slice(-12),
                personality:
                    document.getElementById("personality").value,
                response_style:
                    document.getElementById("responseStyle").value,
                custom_instructions:
                    localStorage.getItem(
                        "raizen_custom_instructions"
                    ) || "",
                document_text: documentText
            })
        });

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
            "â ï¸ Something went wrong. Please try again.";

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
        '<div id="welcome" class="assistant message">â¡ Welcome to RAIZEN. Ask me anything.</div>';
}

function newChat() {
    clearChat();
}

document.getElementById("messageInput").addEventListener(
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

        chatHistory.forEach(function(item) {
            displayMessage(
                item.role,
                item.content
            );
        });
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
    
@app.post("/vision")
async def vision(
    file: UploadFile = File(...),
    message: str = "Describe this image."
):
    filename = file.filename or "image"

    allowed_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp"
    }

    extension = os.path.splitext(filename.lower())[1]

    if extension not in allowed_types:
        return {
            "reply": "⚠️ Please upload a PNG, JPG, JPEG, or WEBP image."
        }

    try:
        raw_bytes = await file.read()

        if len(raw_bytes) > 8 * 1024 * 1024:
            return {
                "reply": "⚠️ Image is too large. Maximum size is 8 MB."
            }

        mime_type = allowed_types[extension]

        image_base64 = base64.b64encode(raw_bytes).decode("utf-8")

        data_url = f"data:{mime_type};base64,{image_base64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    },
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
        ]

        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=700
        )

        reply = response.choices[0].message.content

        if not reply:
            reply = "Sorry, I could not understand the image."

        return {"reply": reply}

    except Exception as error:
        print("VISION ERROR:", error)

        return {
            "reply": "⚠️ I couldn't analyze this image right now."
        }



@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "document"
    if not filename.lower().endswith((".pdf", ".docx", ".txt")):
        return {"error": "Unsupported file type. Please upload PDF, DOCX, or TXT."}

    try:
        raw_bytes = await file.read()
        if len(raw_bytes) > 10 * 1024 * 1024:
            return {"error": "File is too large. Maximum size is 10 MB."}

        text = extract_document_text(filename, raw_bytes)
        if not text:
            return {"error": "I could not extract readable text from this file."}

        return {
            "filename": filename,
            "characters": len(text),
            "text": limit_document_text(text)
        }
    except Exception as error:
        print("FILE ERROR:", error)
        return {"error": "Could not read this document."}


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
                "reply": (
                    "ð¤ï¸ Please mention a city, "
                    "for example: weather in Cuttack."
                )
            }

        result = weather_search(city)

        if result:
            return {"reply": result}

        return {
            "reply": (
                "â ï¸ I couldn't retrieve live weather data right now. "
                "Please try again later."
            )
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

    document_context = "No document is currently uploaded."
    if request.document_text.strip():
        document_context = (
            "An uploaded document is available. Use its extracted text as the "
            "main source for questions specifically about that document. "
            "If the answer is not present, say so clearly.\n\nDOCUMENT TEXT:\n"
            + limit_document_text(request.document_text)
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
- For live news/web results, use the supplied source, date, and link.
- When a link is supplied, keep it in the answer so the user can open the source.
- If live search fails, say so clearly.
- If WEATHER_ERROR appears, explain that live
  weather data could not be retrieved.
- If LIVE_SEARCH_ERROR appears, explain that
  live search could not retrieve data.
- If an uploaded document is present, answer document questions from it.
- For summarize, key points, explain, quiz, questions, or important-points
  requests, use the uploaded document as the main source.
- Do not claim to have read a document if no document text was supplied.

Live information:
{live_data}

Uploaded document:
{document_context}
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
            "reply": (
                "â ï¸ RAIZEN is temporarily unable to respond. "
                "Please try again."
            )
        }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "token_loaded": bool(HF_TOKEN),
        "model": MODEL,
        "live_search": True
    }
