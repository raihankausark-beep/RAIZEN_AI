import os
import json
import re
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient


# =========================================================
# RAIZEN
# =========================================================

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")

MODEL = "zai-org/GLM-5.3-Flash"

client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN
)


# =========================================================
# REQUEST
# =========================================================

class ChatRequest(BaseModel):
    message: str
    history: list = []
    personality: str = "Friendly"
    response_style: str = "Balanced"


# =========================================================
# PERSONALITIES
# =========================================================

PERSONALITIES = {
    "Friendly":
        "Be friendly, warm, natural and helpful.",

    "Teacher":
        "Explain things like a good teacher using simple examples.",

    "Coding Assistant":
        "Focus on programming and technical accuracy.",

    "Professional":
        "Be professional, precise and well structured."
}


# =========================================================
# RESPONSE STYLES
# =========================================================

STYLES = {
    "Short":
        "Keep answers concise and direct.",

    "Balanced":
        "Give a clear answer with enough explanation.",

    "Detailed":
        "Give detailed and well-structured explanations."
}


# =========================================================
# URL FETCH
# =========================================================

def fetch_url(url, timeout=10):

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 RAIZEN-AI"
        }
    )

    with urlopen(
        request,
        timeout=timeout
    ) as response:

        return response.read().decode(
            "utf-8",
            errors="ignore"
        )


# =========================================================
# LIVE SEARCH DETECTION
# =========================================================

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
        "trending",
        "search",
        "weather",
        "temperature",
        "forecast"
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


# =========================================================
# WEATHER QUERY
# =========================================================

def is_weather_query(message):

    text = message.lower()

    return any(
        word in text
        for word in [
            "weather",
            "temperature",
            "forecast"
        ]
    )


# =========================================================
# CITY EXTRACTION
# =========================================================

def extract_city(message):

    text = message.strip()

    patterns = [
        r"weather\s+(?:in|at|for)\s+(.+)",
        r"temperature\s+(?:in|at|for)\s+(.+)",
        r"forecast\s+(?:in|at|for)\s+(.+)",
        r"weather\s+(.+)",
        r"temperature\s+(.+)",
        r"forecast\s+(.+)"
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
                r"\b(today|tomorrow|now|right now)\b",
                "",
                city,
                flags=re.IGNORECASE
            ).strip()

            if city:
                return city

    return None


# =========================================================
# OPEN-METEO WEATHER
# EXACT CITY MATCH FIX
# =========================================================

def weather_open_meteo(city):

    try:

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
            f"?name={quote(city)}"
            "&count=10"
            "&language=en"
            "&format=json"
        )

        geo_data = json.loads(
            fetch_url(geo_url)
        )

        results = geo_data.get(
            "results",
            []
        )

        if not results:
            return None


        # -------------------------------------------------
        # FIND BEST LOCATION
        # Exact city name gets highest priority
        # -------------------------------------------------

        city_clean = city.strip().lower()

        exact_match = None

        for location in results:

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

            # Prefer a result whose name starts
            # with the requested city

            starts_match = None

            for item in results:

                name = item.get(
                    "name",
                    ""
                ).strip().lower()

                if name.startswith(
                    city_clean
                ):

                    starts_match = item
                    break


            if starts_match:

                location = starts_match

            else:

                location = results[0]


        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location.get(
            "name",
            city
        )

        country = location.get(
            "country",
            ""
        )


        # -------------------------------------------------
        # WEATHER DATA
        # -------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
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

        weather_data = json.loads(
            fetch_url(weather_url)
        )

        current = weather_data.get(
            "current"
        )

        if not current:
            return None


        weather_codes = {

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

            80: "Rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",

            95: "Thunderstorm"
        }


        weather_code = current.get(
            "weather_code"
        )

        condition = weather_codes.get(
            weather_code,
            "Unknown conditions"
        )


        return {

            "city": city_name,

            "country": country,

            "temperature":
                current.get(
                    "temperature_2m"
                ),

            "feels_like":
                current.get(
                    "apparent_temperature"
                ),

            "humidity":
                current.get(
                    "relative_humidity_2m"
                ),

            "rain":
                current.get(
                    "precipitation"
                ),

            "wind":
                current.get(
                    "wind_speed_10m"
                ),

            "condition":
                condition,

            "time":
                current.get(
                    "time"
                ),

            "source":
                "Open-Meteo"
        }


    except Exception:

        return None


# =========================================================
# WTTR FALLBACK
# =========================================================

def weather_wttr(city):

    try:

        url = (
            f"https://wttr.in/{quote(city)}"
            "?format=j1"
        )

        data = json.loads(
            fetch_url(url)
        )

        current_list = data.get(
            "current_condition",
            []
        )

        if not current_list:
            return None

        current = current_list[0]


        description = current.get(
            "weatherDesc",
            [{}]
        )


        condition = (
            description[0].get(
                "value",
                "Unknown"
            )
            if description
            else "Unknown"
        )


        area = data.get(
            "nearest_area",
            [{}]
        )


        area_name = city

        country_name = ""


        if area:

            area_name = (
                area[0]
                .get(
                    "areaName",
                    [{}]
                )[0]
                .get(
                    "value",
                    city
                )
            )

            country_name = (
                area[0]
                .get(
                    "country",
                    [{}]
                )[0]
                .get(
                    "value",
                    ""
                )
            )


        return {

            "city":
                area_name,

            "country":
                country_name,

            "temperature":
                current.get(
                    "temp_C"
                ),

            "feels_like":
                current.get(
                    "FeelsLikeC"
                ),

            "humidity":
                current.get(
                    "humidity"
                ),

            "rain":
                current.get(
                    "precipMM"
                ),

            "wind":
                current.get(
                    "windspeedKmph"
                ),

            "condition":
                condition,

            "time":
                current.get(
                    "observation_time"
                ),

            "source":
                "wttr.in"
        }


    except Exception:

        return None


# =========================================================
# WEATHER SEARCH
# =========================================================

def weather_search(city):

    result = weather_open_meteo(
        city
    )

    if result:
        return result


    result = weather_wttr(
        city
    )

    if result:
        return result


    return None


# =========================================================
# GOOGLE NEWS
# =========================================================

def google_news_search(query):

    try:

        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={quote(query)}"
            "&hl=en-IN"
            "&gl=IN"
            "&ceid=IN:en"
        )

        xml_data = fetch_url(
            rss_url
        )

        root = ET.fromstring(
            xml_data
        )

        results = []


        for item in root.findall(
            ".//item"
        )[:5]:

            title = item.findtext(
                "title",
                ""
            )

            link = item.findtext(
                "link",
                ""
            )

            pub_date = item.findtext(
                "pubDate",
                ""
            )


            if title:

                results.append(
                    f"{title}\n"
                    f"Published: {pub_date}\n"
                    f"Source: {link}"
                )


        return results


    except Exception:

        return []


# =========================================================
# WIKIPEDIA
# =========================================================

def wikipedia_search(query):

    try:

        url = (
            "https://en.wikipedia.org/w/api.php?"
            "action=query"
            "&list=search"
            f"&srsearch={quote(query)}"
            "&format=json"
            "&utf8=1"
        )

        data = json.loads(
            fetch_url(url)
        )

        search_results = (
            data
            .get("query", {})
            .get("search", [])
        )

        results = []


        for item in search_results[:3]:

            title = item.get(
                "title",
                ""
            )

            snippet = re.sub(
                "<.*?>",
                "",
                item.get(
                    "snippet",
                    ""
                )
            )


            if title:

                results.append(
                    f"{title}: {snippet}"
                )


        return results


    except Exception:

        return []


# =========================================================
# LIVE SEARCH
# =========================================================

def live_search(message):

    if is_weather_query(
        message
    ):

        city = extract_city(
            message
        )


        if not city:

            return [
                "WEATHER_ERROR: "
                "A city name is required."
            ]


        weather = weather_search(
            city
        )


        if not weather:

            return [
                "WEATHER_ERROR: "
                "Live weather data could not be retrieved."
            ]


        weather_text = (

            "LIVE WEATHER:\n"

            f"Location: "
            f"{weather['city']}, "
            f"{weather['country']}\n"

            f"Temperature: "
            f"{weather['temperature']}°C\n"

            f"Feels like: "
            f"{weather['feels_like']}°C\n"

            f"Condition: "
            f"{weather['condition']}\n"

            f"Humidity: "
            f"{weather['humidity']}%\n"

            f"Rain: "
            f"{weather['rain']} mm\n"

            f"Wind: "
            f"{weather['wind']} km/h\n"

            f"Updated: "
            f"{weather['time']}\n"

            f"Source: "
            f"{weather['source']}"
        )


        return [
            weather_text
        ]


    # NEWS

    results = google_news_search(
        message
    )

    if results:
        return results


    # WIKIPEDIA FALLBACK

    results = wikipedia_search(
        message
    )

    if results:
        return results


    return [
        "LIVE_SEARCH_ERROR: "
        "Live information could not be retrieved."
    ]


# =========================================================
# CHAT
# =========================================================

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


    message_text = (
        request.message.strip()
    )


    # =====================================================
    # DIRECT WEATHER
    # =====================================================

    if is_weather_query(
        message_text
    ):

        city = extract_city(
            message_text
        )


        if not city:

            return {
                "reply":
                    "🌦️ Please tell me the city name.\n\n"
                    "Example: weather in Cuttack"
            }


        weather = weather_search(
            city
        )


        if weather:

            reply = (

                "🌦️ **Current Weather**\n\n"

                f"📍 Location: "
                f"{weather['city']}, "
                f"{weather['country']}\n\n"

                f"🌡️ Temperature: "
                f"{weather['temperature']}°C\n"

                f"🥵 Feels like: "
                f"{weather['feels_like']}°C\n"

                f"☁️ Condition: "
                f"{weather['condition']}\n"

                f"💧 Humidity: "
                f"{weather['humidity']}%\n"

                f"🌧️ Rain: "
                f"{weather['rain']} mm\n"

                f"💨 Wind: "
                f"{weather['wind']} km/h\n\n"

                f"🕒 Updated: "
                f"{weather['time']}\n"

                f"📡 Source: "
                f"{weather['source']}"
            )


            return {
                "reply": reply
            }


        return {

            "reply": (
                "🌦️ Sorry bhai, I couldn't retrieve "
                f"live weather data for {city} right now.\n\n"
                "Please try again shortly."
            )

        }


    # =====================================================
    # PERSONALITY
    # =====================================================

    personality = PERSONALITIES.get(
        request.personality,
        PERSONALITIES["Friendly"]
    )


    response_style = STYLES.get(
        request.response_style,
        STYLES["Balanced"]
    )


    # =====================================================
    # LIVE INFORMATION
    # =====================================================

    live_context = ""


    if needs_live_search(
        message_text
    ):

        results = live_search(
            message_text
        )


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


    # =====================================================
    # SYSTEM PROMPT
    # =====================================================

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
- If WEATHER_ERROR appears, explain that weather data
  could not be retrieved or a city name is required.
- Do not treat an error message as real-world information.
- When live information is available, mention that it
  is live/current.

{live_context}
"""


    # =====================================================
    # HISTORY
    # =====================================================

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
            role in [
                "user",
                "assistant"
            ]
            and content
        ):

            messages.append({

                "role": role,

                "content": content

            })


    # =====================================================
    # AI
    # =====================================================

    try:

        response = (
            client
            .chat
            .completions
            .create(

                model=MODEL,

                messages=messages,

                max_tokens=500

            )
        )


        reply = (
            response
            .choices[0]
            .message
            .content
        )


        if (
            not reply
            or not reply.strip()
        ):

            reply = (
                "Sorry bhai, I couldn't "
                "generate a response right now."
            )


        return {
            "reply": reply
        }


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"AI error: {str(error)}"
        )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "RAIZEN online",

        "token_loaded":
            bool(HF_TOKEN),

        "model":
            MODEL,

        "provider":
            "novita",

        "live_search":
            True,

        "weather":
            True

    }


# =========================================================
# FRONTEND
# =========================================================

HTML = r"""
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

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
        linear-gradient(
            135deg,
            #070b16,
            #0d1324,
            #080c17
        );

    color: white;

    min-height: 100vh;
}


.container {

    width: 100%;

    max-width: 1000px;

    margin: auto;

    min-height: 100vh;

    display: flex;

    flex-direction: column;

    padding: 20px;
}


.header {

    text-align: center;

    padding: 20px 10px;
}


.logo {

    font-size: 34px;

    font-weight: 800;

    letter-spacing: 2px;
}


.subtitle {

    color: #9ca9c7;

    margin-top: 6px;

    font-size: 14px;
}


.chat {

    flex: 1;

    overflow-y: auto;

    padding: 20px 5px;

    display: flex;

    flex-direction: column;

    gap: 12px;
}


.message {

    max-width: 82%;

    padding: 13px 16px;

    border-radius: 18px;

    line-height: 1.5;

    white-space: pre-wrap;

    word-wrap: break-word;
}


.user {

    align-self: flex-end;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    border-bottom-right-radius: 5px;
}


.assistant {

    align-self: flex-start;

    background: #151d31;

    border: 1px solid #27324c;

    border-bottom-left-radius: 5px;
}


.thinking {

    opacity: 0.7;

    border: 1px solid #293653;
}


.input-area {

    display: flex;

    gap: 10px;

    padding: 12px 0 5px;
}


.input {

    flex: 1;

    border: none;

    outline: none;

    border-radius: 16px;

    background: #121a2c;

    color: white;

    border: 1px solid #293653;

    padding: 14px 16px;

    font-size: 16px;
}


.send {

    border: none;

    border-radius: 16px;

    padding: 0 22px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #7c3aed
        );

    color: white;

    font-weight: 700;

    cursor: pointer;
}


.send:disabled {

    opacity: 0.6;

    cursor: not-allowed;
}


.options {

    display: flex;

    gap: 8px;

    flex-wrap: wrap;

    margin-bottom: 8px;
}


select {

    background: #121a2c;

    color: white;

    border: 1px solid #293653;

    border-radius: 10px;

    padding: 8px;
}


.clear {

    background: #121a2c;

    color: #aeb9d3;

    border: 1px solid #293653;

    border-radius: 10px;

    padding: 8px 12px;

    cursor: pointer;
}


@media(max-width:600px) {

    .container {
        padding: 12px;
    }

    .message {
        max-width: 90%;
    }

    .input-area {
        gap: 7px;
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

<div class="subtitle">
Advanced AI Assistant
</div>

</div>


<div class="options">

<select id="personality">

<option>Friendly</option>

<option>Teacher</option>

<option>Coding Assistant</option>

<option>Professional</option>

</select>


<select id="style">

<option>Balanced</option>

<option>Short</option>

<option>Detailed</option>

</select>


<button
    class="clear"
    onclick="clearChat()">

Clear Chat

</button>

</div>


<div
    id="chat"
    class="chat">

</div>


<div class="input-area">

<input
    id="input"
    class="input"
    placeholder="Ask RAIZEN anything..."
    autocomplete="off">


<button
    id="sendButton"
    class="send"
    onclick="sendMessage()">

Send

</button>

</div>


</div>


<script>

let history =
    JSON.parse(
        localStorage.getItem(
            "raizen_history"
        ) || "[]"
    );


const chat =
    document.getElementById(
        "chat"
    );


const input =
    document.getElementById(
        "input"
    );


const sendButton =
    document.getElementById(
        "sendButton"
    );


function displayMessage(
    role,
    text,
    extraClass = ""
) {

    const div =
        document.createElement(
            "div"
        );


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


function renderHistory() {

    chat.innerHTML = "";


    history.forEach(
        item => {

            displayMessage(
                item.role,
                item.content
            );

        }
    );

}


renderHistory();


async function sendMessage() {

    const text =
        input.value.trim();


    if (!text) {
        return;
    }


    input.value = "";


    displayMessage(
        "user",
        text
    );


    history.push({

        role: "user",

        content: text

    });


    localStorage.setItem(
        "raizen_history",
        JSON.stringify(history)
    );


    sendButton.disabled = true;


    const thinkingMessage =
        displayMessage(
            "assistant",
            "⚡ RAIZEN is thinking...",
            "thinking"
        );


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

                    body:
                        JSON.stringify({

                            message:
                                text,

                            history:
                                history.slice(
                                    -12
                                ),

                            personality:
                                document
                                .getElementById(
                                    "personality"
                                )
                                .value,

                            response_style:
                                document
                                .getElementById(
                                    "style"
                                )
                                .value

                        })

                }
            );


        const data =
            await response.json();


        if (thinkingMessage) {
            thinkingMessage.remove();
        }


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Server error"
            );

        }


        const reply =
            data.reply ||
            "Sorry, I couldn't respond.";


        displayMessage(
            "assistant",
            reply
        );


        history.push({

            role: "assistant",

            content: reply

        });


        localStorage.setItem(
            "raizen_history",
            JSON.stringify(history)
        );


    }

    catch (error) {

        if (thinkingMessage) {
            thinkingMessage.remove();
        }


        const errorMessage =
            "⚠️ Sorry bhai, something went wrong.\n\n"
            + error.message;


        displayMessage(
            "assistant",
            errorMessage
        );


        history.push({

            role: "assistant",

            content: errorMessage

        });


        localStorage.setItem(
            "raizen_history",
            JSON.stringify(history)
        );

    }


    finally {

        sendButton.disabled = false;

        input.focus();

    }

}


function clearChat() {

    history = [];

    localStorage.removeItem(
        "raizen_history"
    );

    chat.innerHTML = "";

}


input.addEventListener(
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

</script>


</body>

</html>
"""


# =========================================================
# HOME
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML
