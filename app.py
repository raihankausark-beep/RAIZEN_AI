import os
import json
import re
import base64
import hashlib
import secrets
import time
from urllib.parse import urlencode
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen
from io import BytesIO
import xml.etree.ElementTree as ET

from fastapi import FastAPI, Request, UploadFile, File, Form
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import psycopg
from psycopg.rows import dict_row
from pypdf import PdfReader
from docx import Document

app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("AUTH0_SESSION_SECRET"),
    same_site="lax",
    https_only=True,
)
oauth = OAuth()
@app.get("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth_callback")
    return await oauth.auth0.authorize_redirect(request, redirect_uri)


@app.get("/callback")
async def auth_callback(request: Request):
    token = await oauth.auth0.authorize_access_token(request)
    user = token.get("userinfo")

    request.session["user"] = {
        "sub": user.get("sub"),
        "name": user.get("name"),
        "email": user.get("email"),
        "picture": user.get("picture"),
}
    return RedirectResponse(url="/")
@app.get("/logout")
async def logout(request: Request):
    request.session.clear()

    domain = os.getenv("AUTH0_DOMAIN")
    client_id = os.getenv("AUTH0_CLIENT_ID")

    params = urlencode({
        "client_id": client_id,
        "returnTo": "https://raizen-ai.onrender.com/login"
    })

    return RedirectResponse(
        f"https://{domain}/v2/logout?{params}"
    )

    
@app.get("/auth/user")
async def auth_user(request: Request):
    user = request.session.get("user")

    if not user:
        return {
            "ok": False,
            "user": None
        }

    email = user.get("email") or user.get("sub") or "auth0_user"
    username = email.strip().lower()

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT username FROM raizen_users WHERE username_key = %s",
                    (username,)
                )
                existing = cur.fetchone()

                if not existing:
                    salt, password_hash = hash_password(
                        secrets.token_urlsafe(32)
                    )

                    cur.execute(
                        """INSERT INTO raizen_users
                           (username_key, username, salt, password_hash, created_at)
                           VALUES (%s, %s, %s, %s, %s)""",
                        (
                            username,
                            username,
                            salt,
                            password_hash,
                            datetime.now(timezone.utc)
                        )
                    )

            conn.commit()

        session_token = create_auth_token(username)

        return {
            "ok": True,
            "user": user,
            "token": session_token,
            "username": username
        }

    except Exception as error:
        print("AUTH0 USER ERROR:", error)
        return {
            "ok": False,
            "user": None
        }
oauth.register(
    name="auth0",
    client_id=os.getenv("AUTH0_CLIENT_ID"),
    client_secret=os.getenv("AUTH0_CLIENT_SECRET"),
    server_metadata_url=(
        f"https://{os.getenv('AUTH0_DOMAIN')}/.well-known/openid-configuration"
    ),
    client_kwargs={
        "scope": "openid profile email",
    },
)
from fastapi.responses import PlainTextResponse

@app.get("/robots.txt", response_class=PlainTextResponse)
def robots():
    return """User-agent: *
Allow: /

Sitemap: https://raizen-ai.onrender.com/sitemap.xml
"""

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = "zai-org/GLM-5.3-Flash"

MAX_CHAT_OUTPUT_TOKENS = 2000
MAX_HISTORY_MESSAGES = 30
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_DOCUMENT_CONTEXT_CHARS = 120000

client = InferenceClient(api_key=HF_TOKEN)

VISION_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

vision_client = InferenceClient(api_key=HF_TOKEN)

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

image_client = InferenceClient(api_key=HF_TOKEN)


DATABASE_URL = os.getenv("DATABASE_URL")
AUTH_TOKEN_TTL = 60 * 60 * 24 * 7


class AuthRequest(BaseModel):
    username: str
    password: str


def get_db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured on the server.")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=10)


def init_database():
    if not DATABASE_URL:
        print("DATABASE WARNING: DATABASE_URL is not configured.")
        return False

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS raizen_users (
                        username_key TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS raizen_sessions (
                        token_hash TEXT PRIMARY KEY,
                        username_key TEXT NOT NULL REFERENCES raizen_users(username_key) ON DELETE CASCADE,
                        expires_at TIMESTAMPTZ NOT NULL
                    )
                """)
                cur.execute("DELETE FROM raizen_sessions WHERE expires_at < NOW()")
            conn.commit()
        migrate_legacy_users()
        return True
    except Exception as error:
        print("DATABASE INIT ERROR:", error)
        return False


def migrate_legacy_users():
    legacy_file = "raizen_users.json"
    if not os.path.exists(legacy_file):
        return

    try:
        with open(legacy_file, "r", encoding="utf-8") as f:
            users = json.load(f)
        if not isinstance(users, dict) or not users:
            return

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for user in users.values():
                    username = str(user.get("username", "")).strip()
                    salt = str(user.get("salt", ""))
                    password_hash = str(user.get("password_hash", ""))
                    created_at = user.get("created_at") or datetime.now(timezone.utc).isoformat()
                    if username and salt and password_hash:
                        cur.execute(
                            """INSERT INTO raizen_users
                               (username_key, username, salt, password_hash, created_at)
                               VALUES (%s, %s, %s, %s, %s)
                               ON CONFLICT (username_key) DO NOTHING""",
                            (username.lower(), username, salt, password_hash, created_at)
                        )
            conn.commit()
        print("DATABASE: legacy users migrated successfully.")
    except Exception as error:
        print("LEGACY USER MIGRATION ERROR:", error)


@app.on_event("startup")
async def startup_database():
    init_database()


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()
    return salt, digest


def verify_password(password, salt, expected_hash):
    _, digest = hash_password(password, salt)
    return secrets.compare_digest(digest, expected_hash)


def hash_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_auth_token(username):
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc).timestamp() + AUTH_TOKEN_TTL
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO raizen_sessions (token_hash, username_key, expires_at)
                       VALUES (%s, %s, to_timestamp(%s))""",
                    (hash_token(token), username.lower(), expires_at)
                )
            conn.commit()
        return token
    except Exception as error:
        print("SESSION CREATE ERROR:", error)
        raise


def get_authenticated_username(token):
    if not token:
        return ""

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT u.username
                    FROM raizen_sessions s
                    JOIN raizen_users u
                      ON u.username_key = s.username_key
                    WHERE s.token_hash = %s
                      AND s.expires_at > NOW()
                    """,
                    (hash_token(token),)
                )

                row = cur.fetchone()

        return row["username"] if row else ""

    except Exception as error:
        print("AUTH CHECK ERROR:", error)
        return ""


class ChatRequest(BaseModel):
    token: str = ""
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


def build_document_context(text, question=""):
    text = (text or "").strip()
    if not text:
        return "No document is currently uploaded."
    if len(text) <= MAX_DOCUMENT_CONTEXT_CHARS:
        return text
    chunk_size = 7000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    q_words = set(re.findall(r"[a-zA-Z0-9]{4,}", question.lower()))
    scored = []
    for idx, chunk in enumerate(chunks):
        words = set(re.findall(r"[a-zA-Z0-9]{4,}", chunk.lower()))
        scored.append((len(q_words & words), idx, chunk))
    keep = max(2, MAX_DOCUMENT_CONTEXT_CHARS // chunk_size)
    selected = {0, len(chunks) - 1}
    for score, idx, chunk in sorted(scored, reverse=True)[:keep]:
        selected.add(idx)
    context = "\n\n--- DOCUMENT SECTION ---\n\n".join(chunks[i] for i in sorted(selected))
    return context[:MAX_DOCUMENT_CONTEXT_CHARS] + "\n\n[Large document: relevant sections selected automatically.]"

def chat_completion_with_retry(messages, max_tokens):
    last_error = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens)
        except Exception as error:
            last_error = error
            if attempt < 2:
                time.sleep(1.2 * (attempt + 1))
    raise last_error


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

        for item in root.findall(".//item")[:8]:
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
            "&srsearch=" + quote(query) + "&srlimit=5"
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
            + "Temperature: " + str(temperature) + "°C\n"
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
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
<meta name="theme-color" content="#000000">
<title>RAIZEN AI</title>
<style>
:root{--bg:#000;--panel:#171717;--panel2:#202020;--text:#f5f5f5;--muted:#9b9b9b;--line:#2b2b2b;--accent:#4d8dff}
*{box-sizing:border-box}
html,body{margin:0;padding:0;width:100%;height:100%;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
body{overflow:hidden}
button,input,select{font:inherit}
button{color:inherit}
.container{display:flex;width:100%;height:100dvh;min-height:100vh;flex-direction:column;background:#000}

/* Top bar */
.topbar{height:72px;min-height:72px;padding:12px 16px;display:flex;align-items:center;justify-content:space-between;background:#000;position:relative;z-index:20}
.top-left,.top-right{display:flex;align-items:center;gap:10px}
.icon-btn{width:48px;height:48px;border:0;border-radius:15px;background:#343434;display:grid;place-items:center;cursor:pointer;font-size:25px;transition:.18s}
.icon-btn:hover{background:#414141;transform:translateY(-1px)}
.icon-btn:active{transform:scale(.97)}
.menu-icon{font-size:25px;line-height:1}
.brand-pill{height:44px;padding:0 18px;border-radius:999px;background:#182b43;color:#54a2ff;display:flex;align-items:center;gap:7px;font-size:19px;font-weight:800;letter-spacing:.1px}
.brand-pill span{font-size:17px}
.account-pill{display:none}
.status{display:none}

/* Main chat */
.chat{flex:1;min-height:0;overflow-y:auto;padding:0 18px 180px;scroll-behavior:smooth}
.chat::-webkit-scrollbar{width:6px}.chat::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:10px}
.hero{max-width:850px;margin:8vh auto 0;text-align:center;padding:22px 10px 10px;background:transparent;border:0;box-shadow:none}
.hero-kicker{display:none}
.hero-title{font-size:42px;line-height:1.1;font-weight:800;letter-spacing:-1.5px;margin:0 0 10px}
.hero-subtitle{font-size:15px;color:#8f8f8f;line-height:1.55;margin:0 auto 30px;max-width:560px}
.prompt-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;max-width:720px;margin:0 auto}
.prompt-card{min-height:70px;padding:13px 14px;border-radius:18px;background:#111;border:1px solid #242424;color:#e8e8e8;text-align:left;cursor:pointer;transition:.18s;font-size:13px;line-height:1.35}
.prompt-card:hover{background:#191919;border-color:#3a3a3a;transform:translateY(-1px)}
.prompt-card strong{display:block;color:#fff;font-size:14px;margin-bottom:4px}
#welcome{display:none}
.message{max-width:min(780px,90%);padding:14px 17px;margin:13px auto;border-radius:20px;white-space:pre-wrap;line-height:1.58;animation:messageIn .2s ease;font-size:15px}
@keyframes messageIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.user{margin-left:auto;margin-right:auto;background:#252525;border:1px solid #303030}
.assistant{background:#111;border:1px solid #222}
.thinking{opacity:.65}
.message a{color:#79adff;text-decoration:underline;word-break:break-all}
.search-badge{display:inline-block;font-size:11px;padding:3px 7px;border:1px solid #444;border-radius:999px;opacity:.8;margin-bottom:5px}

/* Bottom composer */
.composer-wrap{position:fixed;left:0;right:0;bottom:0;z-index:30;padding:12px 18px calc(14px + env(safe-area-inset-bottom));background:linear-gradient(to top,#000 72%,rgba(0,0,0,.96) 86%,transparent)}
.composer{width:min(900px,100%);margin:auto}
.tool-strip{display:flex;gap:8px;overflow-x:auto;padding:0 2px 9px;scrollbar-width:none}
.tool-strip::-webkit-scrollbar{display:none}
.tool-chip{white-space:nowrap;border:1px solid #303030;background:#151515;border-radius:999px;padding:8px 12px;color:#d7d7d7;font-size:12px;cursor:pointer}
.tool-chip:hover{background:#202020}
.input-area{display:flex;align-items:center;gap:8px;background:#202020;border:1px solid #2b2b2b;border-radius:28px;padding:6px 7px 6px 9px;box-shadow:0 8px 35px rgba(0,0,0,.45)}
#messageInput{flex:1;min-width:0;border:0;outline:0;background:transparent;color:#fff;padding:11px 7px;font-size:16px}
#messageInput::placeholder{color:#8c8c8c}
.circle-btn{width:43px;height:43px;flex:0 0 43px;border:0;border-radius:50%;background:#363636;color:#fff;display:grid;place-items:center;cursor:pointer;font-size:21px}
.circle-btn:hover{background:#444}
#sendButton{width:43px;height:43px;flex:0 0 43px;border:0;border-radius:50%;background:#4b8df8;color:#fff;display:grid;place-items:center;cursor:pointer;font-size:21px;font-weight:700}
#sendButton:hover{filter:brightness(1.08)}
#sendButton:disabled{opacity:.45;cursor:not-allowed}
.composer-note{text-align:center;color:#6f6f6f;font-size:10px;margin-top:7px}

/* Hidden/secondary controls */
.controls{display:none}
.file-panel,.vision-panel{display:none;align-items:center;gap:8px;max-width:900px;margin:0 auto 8px;padding:8px 11px;background:#151515;border:1px solid #292929;border-radius:14px;font-size:12px}
.file-input,.vision-input{max-width:100%;color:#bbb;font-size:12px}.file-name,.vision-name{color:#aaa;font-size:12px}.control{border:1px solid #333;background:#202020;border-radius:10px;padding:7px 10px;cursor:pointer}
.small{display:none}

/* History drawer */
.history-drawer{position:fixed;top:0;left:-370px;width:350px;height:100dvh;background:#0b0b0b;border-right:1px solid #2b2b2b;z-index:1000;padding:22px;transition:left .25s ease;overflow-y:auto;box-shadow:20px 0 60px rgba(0,0,0,.5)}
.history-drawer.open{left:0}
.history-top{display:flex;align-items:center;justify-content:space-between;font-size:19px;margin-bottom:18px}.history-top button{background:transparent;border:0;color:#fff;font-size:22px;cursor:pointer}
.new-chat-history,.clear-history-btn{width:100%;padding:12px;border-radius:13px;border:1px solid #303030;background:#171717;color:#fff;cursor:pointer;margin-bottom:12px}.new-chat-history:hover,.clear-history-btn:hover{background:#222}
.history-item{position:relative;padding:13px 40px 13px 13px;margin-bottom:8px;border-radius:13px;background:#151515;border:1px solid transparent;color:#fff;cursor:pointer}.history-item:hover{background:#1e1e1e;border-color:#333}.history-title{font-size:14px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.history-date{font-size:11px;opacity:.45;margin-top:5px}.history-delete{position:absolute;right:10px;top:12px;background:transparent;border:0;color:#888;cursor:pointer}.history-empty{text-align:center;padding:30px 10px;color:#666;font-size:13px}

/* Auth/creator kept hidden for open-access mode */
.auth-screen{display:none!important}
#creatorResetPanel{display:none!important}

@media(max-width:650px){
 .topbar{height:68px;min-height:68px;padding:10px 12px}.icon-btn{width:46px;height:46px;border-radius:14px}.brand-pill{height:42px;padding:0 15px;font-size:18px}.chat{padding:0 12px 170px}.hero{margin-top:10vh;padding:10px 4px}.hero-title{font-size:34px}.hero-subtitle{font-size:14px;max-width:330px;margin-bottom:25px}.prompt-grid{grid-template-columns:1fr 1fr;gap:8px}.prompt-card{min-height:65px;padding:11px;font-size:12px;border-radius:16px}.prompt-card strong{font-size:13px}.message{max-width:94%;font-size:14px}.composer-wrap{padding:8px 10px calc(10px + env(safe-area-inset-bottom))}.tool-chip{font-size:11px;padding:7px 10px}.input-area{border-radius:25px;padding-left:8px}.circle-btn,#sendButton{width:41px;height:41px;flex-basis:41px}.history-drawer{width:88%;left:-92%}.history-drawer.open{left:0}
}
</style>
</head>
<body>

<div id="authScreen" class="auth-screen"></div>
<div id="creatorDashboard" class="auth-screen"></div>

<div class="container">
  <header class="topbar">
    <div class="top-left">
      <button class="icon-btn" aria-label="Open menu" onclick="toggleHistory()"><span class="menu-icon">☰</span></button>
      <div class="brand-pill"><span>✦</span> RAIZEN</div>
    </div>
    <div class="top-right">
      <button class="icon-btn" aria-label="New chat" onclick="newChat()">◌</button>
      <span id="accountPill" class="account-pill">⚡ Open Access</span>
      <span class="status">AI ONLINE</span>
    </div>
  </header>

  <div id="historyDrawer" class="history-drawer">
    <div class="history-top"><strong>Chat history</strong><button onclick="toggleHistory()">✕</button></div>
    <button class="new-chat-history" onclick="newChat();toggleHistory()">＋ New chat</button>
    <div id="historyList"></div>
    <button class="clear-history-btn" onclick="clearAllHistory()">🗑️ Clear history</button>
  </div>

  <main id="chat" class="chat">
    <section class="hero" id="heroPanel">
      <div class="hero-title">How can I help?</div>
      <div class="hero-subtitle">Chat with RAIZEN, create images, work with files, search the web, and more.</div>
      <div class="prompt-grid">
        <button class="prompt-card" onclick="usePrompt('Help me write or edit something')"><strong>✎ Write or edit</strong>Draft, rewrite, improve</button>
        <button class="prompt-card" onclick="generateImageFromInput()"><strong>▧ Create an image</strong>Generate from your prompt</button>
        <button class="prompt-card" onclick="usePrompt('Search the latest AI news')"><strong>◎ Search the web</strong>Find current information</button>
        <button class="prompt-card" onclick="document.getElementById('fileInput').click()"><strong>▤ Add a file</strong>PDF, DOCX or TXT</button>
        <button class="prompt-card" onclick="document.getElementById('visionInput').click()"><strong>◉ Analyze an image</strong>Ask about a picture</button>
        <button class="prompt-card" onclick="usePrompt('What is the weather in Cuttack today?')"><strong>☼ Weather</strong>Check current conditions</button>
      </div>
    </section>
    <div id="welcome" class="assistant message">⚡ Welcome to RAIZEN. Ask me anything.</div>
  </main>

  <div class="composer-wrap">
    <div class="composer">
      <div class="tool-strip">
        <button class="tool-chip" onclick="document.getElementById('fileInput').click()">＋ File</button>
        <button class="tool-chip" onclick="document.getElementById('visionInput').click()">▧ Image</button>
        <button class="tool-chip" onclick="generateImageFromInput()">✦ Create image</button>
        <button class="tool-chip" onclick="usePrompt('Search the web for ')" >◎ Web search</button>
      </div>

      <div class="file-panel">
        <input id="fileInput" class="file-input" type="file" accept=".pdf,.docx,.txt">
        <span id="fileName" class="file-name">No document selected</span>
        <button class="control" onclick="clearDocument()">Remove</button>
      </div>

      <div class="vision-panel">
        <input id="visionInput" class="vision-input" type="file" accept="image/png,image/jpeg,image/webp">
        <span id="visionName" class="vision-name">No image selected</span>
        <button class="control" onclick="clearVision()">Remove</button>
      </div>

      <div class="controls">
        <select id="personality"><option>Friendly</option><option>Teacher</option><option>Coding Assistant</option><option>Professional</option></select>
        <select id="responseStyle"><option>Short</option><option selected>Balanced</option><option>Detailed</option></select>
      </div>

      <div class="input-area">
        <button class="circle-btn" aria-label="Add" onclick="document.getElementById('fileInput').click()">＋</button>
        <input id="messageInput" placeholder="Message RAIZEN..." autocomplete="off">
        <button class="circle-btn" aria-label="Image" onclick="document.getElementById('visionInput').click()">▧</button>
        <button type="button" id="sendButton" aria-label="Send" onclick="window.sendMessage()">↑</button>
      </div>
      <div class="composer-note">RAIZEN • Created and developed by Raihan Kausar</div>
    </div>
  </div>
</div>

<script>
let authToken = "";
let loggedInUsername = "";
let authMode = "login";

function showAuthMode(mode){ return false; }
function submitAuth(event){ if(event) event.preventDefault(); return false; }
async function checkLogin(){
    try {
        const response = await fetch("/auth/user");
        const data = await response.json();

        if (data.ok && data.user) {
            authToken = 
                data.username ||
                data.user.email ||
                data.user.name
                "User";

            showAppAfterLogin();
        } else {
            window.location.href = "/login";
        }
    } catch (error) {
        window.location.href = "/login";
    }
}

function showLoginScreen(){
    window.location.href = "/login";
}
function showAppAfterLogin(){ 
     showAppDirect(); }
function showAppDirect(){
  const authScreen=document.getElementById("authScreen");
  const container=document.querySelector(".container");
  if(authScreen) authScreen.style.display="none";
  if(container) container.style.display="flex";
  const accountPill=document.getElementById("accountPill");
  if(accountPill) accountPill.textContent="👤 " + loggedInUsername;
  loadChatHistory();
}
function toggleCreatorReset(){ return false; }
function resetCreatorPassword(){ return false; }
async function logout(){ showAppDirect(); }

let chatHistory = [];
let savedChats = [];
let currentChatId = null;
let documentText = "";
let documentName = "";
let visionFile = null;

document.getElementById("fileInput").addEventListener("change", async function() {
    const file = this.files[0];
    if (!file) return;

    const lower = file.name.toLowerCase();
    if (![".pdf", ".docx", ".txt"].some(ext => lower.endsWith(ext))) {
        displayMessage("assistant", "⚠️ Please upload a PDF, DOCX, or TXT file.");
        this.value = "";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("token", authToken);
    document.getElementById("fileName").textContent = "Reading " + file.name + "...";

    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Upload failed.");
        }

        documentText = data.text || "";
        documentName = data.filename || file.name;
        document.getElementById("fileName").textContent = "📄 " + documentName + " ready";

        displayMessage("assistant",
            "📄 " + documentName +
            " is ready. Ask me to summarize it, explain it, find important points, or create questions from it."
        );
    } catch (error) {
        documentText = "";
        documentName = "";
        document.getElementById("fileName").textContent = "Upload failed";
        displayMessage("assistant", "⚠️ " + error.message);
    }
});

function clearDocument() {
    documentText = "";
    documentName = "";
    document.getElementById("fileInput").value = "";
    document.getElementById("fileName").textContent = "No document selected";
}

document.getElementById("visionInput").addEventListener("change", function() {
    const file = this.files[0];
    if (!file) return;

    const allowed = ["image/png", "image/jpeg", "image/webp"];
    if (!allowed.includes(file.type)) {
        displayMessage(
            "assistant",
            "⚠️ Please select a PNG, JPG/JPEG, or WEBP image."
        );
        this.value = "";
        return;
    }

    if (file.size > 12 * 1024 * 1024) {
        displayMessage(
            "assistant",
            "⚠️ Image is too large. Maximum size is 12 MB."
        );
        this.value = "";
        return;
    }

    visionFile = file;
    document.getElementById("visionName").textContent =
        "🖼️ " + file.name + " ready";

    displayMessage(
        "assistant",
        "🖼️ " + file.name +
        " is ready. Ask me to describe it, read text from it, or explain what is shown."
    );
});

function clearVision() {
    visionFile = null;
    document.getElementById("visionInput").value = "";
    document.getElementById("visionName").textContent = "No image selected";
}

async function sendVisionMessage(question) {
    const formData = new FormData();
    formData.append("file", visionFile);
    formData.append("question", question);
    formData.append("token", authToken);

    const response = await fetch("/vision", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    if (!response.ok || data.error) {
        throw new Error(data.error || "Vision request failed.");
    }

    return data.reply;
}

function makeChatTitle(messages) {
    const firstUser = messages.find(m => m.role === "user");
    if (!firstUser) return "New Chat";
    let title = String(firstUser.content || "").trim();
    return title.length > 36 ? title.substring(0, 36) + "..." : title;
}

function saveChats() {
    localStorage.setItem("raizen_saved_chats_" + loggedInUsername.toLowerCase(), JSON.stringify(savedChats));
}

function saveCurrentChat() {
    if (!chatHistory.length) return;
    if (!currentChatId) {
        currentChatId = Date.now().toString() + Math.random().toString(36).slice(2, 8);
    }
    const chatData = {
        id: currentChatId,
        title: makeChatTitle(chatHistory),
        messages: chatHistory.slice(-100),
        updatedAt: Date.now()
    };
    const index = savedChats.findIndex(c => c.id === currentChatId);
    if (index >= 0) savedChats[index] = chatData;
    else savedChats.unshift(chatData);
    savedChats.sort((a,b) => b.updatedAt - a.updatedAt);
    saveChats();
    localStorage.setItem("raizen_chat_history_" + loggedInUsername.toLowerCase(), JSON.stringify(chatHistory.slice(-30)));
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById("historyList");
    if (!list) return;
    list.innerHTML = "";
    if (!savedChats.length) {
        list.innerHTML = '<div class="history-empty">No saved chats yet.</div>';
        return;
    }
    savedChats.sort((a,b) => b.updatedAt - a.updatedAt);
    savedChats.forEach(chat => {
        const item = document.createElement("div");
        item.className = "history-item";
        const title = document.createElement("div");
        title.className = "history-title";
        title.textContent = chat.title || "New Chat";
        const date = document.createElement("div");
        date.className = "history-date";
        date.textContent = new Date(chat.updatedAt).toLocaleString();
        const del = document.createElement("button");
        del.className = "history-delete";
        del.textContent = "🗑";
        del.title = "Delete chat";
        del.onclick = function(e) { e.stopPropagation(); deleteChat(chat.id); };
        item.appendChild(title); item.appendChild(date); item.appendChild(del);
        item.onclick = function() { openChat(chat.id); };
        list.appendChild(item);
    });
}

function openChat(id) {
    const selected = savedChats.find(c => c.id === id);
    if (!selected) return;
    currentChatId = selected.id;
    chatHistory = Array.isArray(selected.messages) ? [...selected.messages] : [];
    const chat = document.getElementById("chat");
    chat.innerHTML = "";
    if (!chatHistory.length) {
        chat.innerHTML = '<div id="welcome" class="assistant message">⚡ Welcome to RAIZEN. Ask me anything.</div>';
    } else {
        chatHistory.forEach(item => displayMessage(item.role, item.content));
    }
    toggleHistory();
}

function deleteChat(id) {
    savedChats = savedChats.filter(c => c.id !== id);
    if (currentChatId === id) {
        currentChatId = null;
        chatHistory = [];
        document.getElementById("chat").innerHTML = '<div id="welcome" class="assistant message">⚡ Welcome to RAIZEN. Ask me anything.</div>';
    }
    saveChats();
    renderHistory();
}

function toggleHistory() {
    const drawer = document.getElementById("historyDrawer");
    if (!drawer) return;
    drawer.classList.toggle("open");
    if (drawer.classList.contains("open")) renderHistory();
}

function clearAllHistory() {
    if (!savedChats.length) return;
    if (!confirm("Delete all saved chat history?")) return;
    savedChats = [];
    currentChatId = null;
    chatHistory = [];
    localStorage.removeItem("raizen_saved_chats_" + loggedInUsername.toLowerCase());
    localStorage.removeItem("raizen_chat_history_" + loggedInUsername.toLowerCase());
    document.getElementById("chat").innerHTML = '<div id="welcome" class="assistant message">⚡ Welcome to RAIZEN. Ask me anything.</div>';
    renderHistory();
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
    if (!chat) {
        console.error("RAIZEN: chat container not found");
        return null;
    }
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

function usePrompt(text) {
    const input = document.getElementById("messageInput");
    input.value = text;
    input.focus();
}

async function generateImageFromInput() {
    const input = document.getElementById("messageInput");
    const button = document.getElementById("sendButton");
    const prompt = input.value.trim();

    if (!prompt || button.disabled) return;

    input.value = "";
    button.disabled = true;

    displayMessage("user", prompt);

    const thinking = displayMessage(
        "assistant",
        "🎨 RAIZEN is creating your image...",
        "thinking"
    );

    try {
        const response = await fetch("/generate-image", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({token: authToken, prompt: prompt})
        });

        const data = await response.json();

        thinking.remove();

        if (data.error) {
            displayMessage("assistant", data.error);
            chatHistory.push({role: "user", content: prompt});
            chatHistory.push({role: "assistant", content: data.error});
        } else {
            displayGeneratedImage(prompt, data.image);
            chatHistory.push({role: "user", content: prompt});
            chatHistory.push({
                role: "assistant",
                content: "🎨 Generated an image for: " + prompt
            });
        }

        saveCurrentChat();
    } catch (error) {
        thinking.remove();
        const msg = "⚠️ Image generation failed. Please try again.";
        displayMessage("assistant", msg);
        chatHistory.push({role: "user", content: prompt});
        chatHistory.push({role: "assistant", content: msg});
        saveCurrentChat();
    }

    button.disabled = false;
    input.focus();
}

function displayGeneratedImage(prompt, dataUrl) {
    const chat = document.getElementById("chat");

    const div = document.createElement("div");
    div.className = "message assistant";

    const title = document.createElement("div");
    title.innerHTML = "<strong>🎨 Generated Image</strong><br>" + escapeHtml(prompt);

    const img = document.createElement("img");
    img.src = dataUrl;
    img.alt = prompt;
    img.style.maxWidth = "100%";
    img.style.width = "768px";
    img.style.borderRadius = "18px";
    img.style.marginTop = "12px";
    img.style.display = "block";

    const download = document.createElement("a");
    download.href = dataUrl;
    download.download = "raizen-generated-image.png";
    download.textContent = "⬇️ Save Image";
    download.style.display = "inline-block";
    download.style.marginTop = "10px";

    div.appendChild(title);
    div.appendChild(img);
    div.appendChild(download);

    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;

    return div;
}

window.sendMessage = async function sendMessage() {
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
        visionFile
            ? "👁️ RAIZEN is analyzing the image..."
            : "⚡ RAIZEN is thinking...",
        "thinking"
    );

    try {
        if (visionFile) {
            const reply = await sendVisionMessage(message);

            if (thinking) thinking.remove();
            displayMessage("assistant", reply);

            chatHistory.push({
                role: "assistant",
                content: reply
            });

            saveCurrentChat();
            button.disabled = false;
            input.focus();
            return;
        }

        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                token: authToken,
                message: message,
                history: chatHistory.slice(-30),
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

        if (thinking) thinking.remove();

        const reply =
            data.reply ||
            "Sorry, I could not generate a response.";

        displayMessage("assistant", reply);

        chatHistory.push({
            role: "assistant",
            content: reply
        });

        saveCurrentChat();

    } catch (error) {
        if (thinking) thinking.remove();

        const reply =
            "⚠️ Something went wrong. Please try again.";

        displayMessage("assistant", reply);

        chatHistory.push({
            role: "assistant",
            content: reply
        });

        saveCurrentChat();
    }

    button.disabled = false;
    input.focus();
}

function clearChat() {
    chatHistory = [];
    currentChatId = null;
    localStorage.removeItem("raizen_chat_history_" + loggedInUsername.toLowerCase());
    document.getElementById("chat").innerHTML = '<div id="welcome" class="assistant message">⚡ Welcome to RAIZEN. Ask me anything.</div>';
}

function newChat() {
    if (chatHistory.length) saveCurrentChat();
    chatHistory = [];
    currentChatId = null;
    document.getElementById("chat").innerHTML = '<div id="welcome" class="assistant message">⚡ Welcome to RAIZEN. Ask me anything.</div>';
    const drawer = document.getElementById("historyDrawer");
    if (drawer && drawer.classList.contains("open")) drawer.classList.remove("open");
}

document.getElementById("messageInput").addEventListener("keydown", function(event) {
    if (event.key === "Enter") { event.preventDefault(); sendMessage(); }
});

function loadChatHistory() {
    try {
        const saved = localStorage.getItem("raizen_saved_chats_" + loggedInUsername.toLowerCase());
        savedChats = saved ? JSON.parse(saved) : [];
        if (!Array.isArray(savedChats)) savedChats = [];

        if (!savedChats.length) {
            const legacy = localStorage.getItem("raizen_chat_history_" + loggedInUsername.toLowerCase());
            if (legacy) {
                const old = JSON.parse(legacy);
                if (Array.isArray(old) && old.length) {
                    savedChats = [{ id: Date.now().toString(), title: makeChatTitle(old), messages: old, updatedAt: Date.now() }];
                    saveChats();
                }
            }
        }

        if (savedChats.length) {
            savedChats.sort((a,b) => b.updatedAt - a.updatedAt);
            const latest = savedChats[0];
            currentChatId = latest.id;
            chatHistory = Array.isArray(latest.messages) ? [...latest.messages] : [];
            const chat = document.getElementById("chat");
            chat.innerHTML = "";
            chatHistory.forEach(item => displayMessage(item.role, item.content));
        }
    } catch (error) {
        savedChats = [];
        chatHistory = [];
        currentChatId = null;
    }
    renderHistory();
}
window.addEventListener("DOMContentLoaded", checkLogin);
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(content=HTML)



@app.post("/register")
async def register(request: AuthRequest):
    username = request.username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_]{3,20}", username):
        return {"ok": False, "message": "Username must be 3–20 characters using letters, numbers, or underscore."}
    if len(request.password) < 8:
        return {"ok": False, "message": "Password must be at least 8 characters."}

    salt, password_hash = hash_password(request.password)
    created_at = datetime.now(timezone.utc)

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO raizen_users
                       (username_key, username, salt, password_hash, created_at)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (username.lower(), username, salt, password_hash, created_at)
                )
            conn.commit()
        token = create_auth_token(username)
        return {"ok": True, "message": "Account created successfully.", "username": username, "token": token}
    except psycopg.errors.UniqueViolation:
        return {"ok": False, "message": "That username already exists."}
    except Exception as error:
        print("REGISTER ERROR:", error)
        return {"ok": False, "message": "Account service is temporarily unavailable. Please try again."}


@app.post("/login")
async def login(request: AuthRequest):
    username = request.username.strip()
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT username, salt, password_hash FROM raizen_users WHERE username_key = %s",
                    (username.lower(),)
                )
                user = cur.fetchone()

        if not user or not verify_password(request.password, user["salt"], user["password_hash"]):
            return {"ok": False, "message": "Invalid username or password."}

        token = create_auth_token(user["username"])
        return {"ok": True, "message": "Login successful.", "username": user["username"], "token": token}
    except Exception as error:
        print("LOGIN ERROR:", error)
        return {"ok": False, "message": "Account service is temporarily unavailable. Please try again."}


@app.post("/logout")
async def logout(token: str = ""):
    if token:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM raizen_sessions WHERE token_hash = %s", (hash_token(token),))
                conn.commit()
        except Exception as error:
            print("LOGOUT ERROR:", error)
    return {"ok": True}


@app.get("/creator-dashboard")
async def creator_dashboard(token: str = ""):
    username = get_authenticated_username(token)
    if not username:
        return {"ok": False, "message": "Authentication required."}

    creator_names = {"raihan", "raihankausar", "raihankausarkausar"}
    if username.lower() not in creator_names:
        return {"ok": False, "message": "Creator access denied."}

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT username, created_at FROM raizen_users ORDER BY created_at DESC"
                )
                users = cur.fetchall()
        return {
            "ok": True,
            "creator": "Raihan Kausar",
            "total_users": len(users),
            "users": [
                {"username": user["username"], "created_at": user["created_at"].isoformat()}
                for user in users
            ]
        }
    except Exception as error:
        print("CREATOR DASHBOARD ERROR:", error)
        return {"ok": False, "message": "Could not load creator dashboard."}


@app.get("/auth-check")
async def auth_check(token: str = ""):
    username = get_authenticated_username(token)
    return {"ok": bool(username), "username": username or ""}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), token: str = Form("")):
    if not get_authenticated_username(token):
        return {"error": "Please log in to upload documents."}
    filename = file.filename or "document"
    if not filename.lower().endswith((".pdf", ".docx", ".txt")):
        return {"error": "Unsupported file type. Please upload PDF, DOCX, or TXT."}

    try:
        raw_bytes = await file.read()
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            return {"error": "File is too large. Maximum size is 50 MB."}

        text = extract_document_text(filename, raw_bytes)
        if not text:
            return {"error": "I could not extract readable text from this file."}

        return {
            "filename": filename,
            "characters": len(text),
            "text": text
        }
    except Exception as error:
        print("FILE ERROR:", error)
        return {"error": "Could not read this document."}


@app.post("/vision")
async def vision(file: UploadFile = File(...), question: str = Form(""), token: str = Form("")):
    if not get_authenticated_username(token):
        return {"error": "Please log in to use Vision."}
    filename = file.filename or "image"
    content_type = file.content_type or ""

    allowed_types = {
        "image/png": "png",
        "image/jpeg": "jpeg",
        "image/webp": "webp"
    }

    if content_type not in allowed_types:
        return {
            "error": "Unsupported image type. Please use PNG, JPG/JPEG, or WEBP."
        }

    try:
        raw_bytes = await file.read()

        if len(raw_bytes) > MAX_IMAGE_BYTES:
            return {"error": "Image is too large. Maximum size is 12 MB."}

        encoded = base64.b64encode(raw_bytes).decode("utf-8")
        data_url = f"data:{content_type};base64,{encoded}"

        user_question = question.strip()
        if not user_question:
            user_question = (
                "Describe this image clearly. Mention important visible details "
                "and read any clearly visible text when possible."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are RAIZEN's vision assistant. Analyze the supplied image "
                    "carefully. Be accurate, concise, and do not invent details. "
                    "If text is blurry or unreadable, say so."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    },
                    {
                        "type": "text",
                        "text": user_question
                    }
                ]
            }
        ]

        response = vision_client.chat.completions.create(
            model=VISION_MODEL,
            messages=messages,
            max_tokens=1000
        )

        reply = response.choices[0].message.content

        if not reply:
            return {"error": "The vision model returned an empty response."}

        return {
            "reply": reply,
            "model": VISION_MODEL
        }

    except Exception as error:
        print("VISION ERROR:", error)
        return {
            "error": (
                "Vision is temporarily unavailable. "
                "Please try the image again."
            )
        }


@app.post("/chat")
async def chat(request: ChatRequest):
    username = get_authenticated_username(request.token)
    if not username:
        return {"reply": "🔐 Please log in to use RAIZEN."}
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
                    "🌤️ Please mention a city, "
                    "for example: weather in Cuttack."
                )
            }

        result = weather_search(city)

        if result:
            return {"reply": result}

        return {
            "reply": (
                "⚠️ I couldn't retrieve live weather data right now. "
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
            + build_document_context(request.document_text, message)
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

    for item in request.history[-MAX_HISTORY_MESSAGES:]:
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
        response = chat_completion_with_retry(
            messages,
            MAX_CHAT_OUTPUT_TOKENS
        )

        reply = response.choices[0].message.content

        if not reply:
            reply = "Sorry, I could not generate a response."

        return {"reply": reply}

    except Exception as error:
        print("RAIZEN ERROR:", error)

        return {
            "reply": (
                "⚠️ RAIZEN is temporarily unable to respond. "
                "Please try again."
            )
        }


class ImageRequest(BaseModel):
    token: str = ""
    prompt: str
    negative_prompt: str = ""


@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    if not get_authenticated_username(request.token):
        return {"error": "Please log in to create images."}
    prompt = request.prompt.strip()

    if not prompt:
        return {"error": "Please enter an image prompt."}

    if len(prompt) > 1000:
        return {"error": "Image prompt is too long. Keep it under 1000 characters."}

    try:
        image = image_client.text_to_image(
            prompt,
            model=IMAGE_MODEL,
            negative_prompt=request.negative_prompt.strip() or None,
            width=768,
            height=768,
            num_inference_steps=4
        )

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "image": "data:image/png;base64," + encoded,
            "model": IMAGE_MODEL
        }

    except Exception as error:
        print("IMAGE GENERATION ERROR:", error)
        return {
            "error": "Image generation is temporarily unavailable. Please try again."
        }


@app.get("/health")
async def health():
    database_ok = False
    if DATABASE_URL:
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ok")
                    database_ok = bool(cur.fetchone())
        except Exception as error:
            print("HEALTH DATABASE ERROR:", error)

    return {
        "status": "ok" if bool(HF_TOKEN) else "degraded",
        "token_loaded": bool(HF_TOKEN),
        "database_configured": bool(DATABASE_URL),
        "database_connected": database_ok,
        "auth_required": False,
        "model": MODEL,
        "vision_model": VISION_MODEL,
        "vision": True,
        "live_search": True,
        "image_generation": True,
        "image_model": IMAGE_MODEL
    }
