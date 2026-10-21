"""YouTube Data API metadata for a track's source video (the optional `youtube` extra)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse


def extract_youtube_id(url: str) -> str | None:
    """Video ID from youtu.be/<id>, youtube.com/watch?v=<id>, /shorts/<id> or /embed/<id>."""
    try:
        u = urlparse(url)
    except Exception:
        return None
    if u.netloc in ("youtu.be",):
        return u.path.strip("/")
    if "youtube.com" in u.netloc:
        if u.path == "/watch":
            return parse_qs(u.query).get("v", [None])[0]
        m = re.match(r"^/(shorts|embed)/([^/?]+)", u.path)
        if m:
            return m.group(2)
    return None


def fetch_youtube_metadata(video_id: str, api_key: str) -> dict:
    """Title, channel and publish date; an {"error": ...} dict on any failure."""
    try:
        import requests
    except ImportError:
        return {"error": "requests not installed. pip install requests"}

    r = requests.get("https://www.googleapis.com/youtube/v3/videos",
                     params={"key": api_key, "id": video_id, "part": "snippet"}, timeout=20)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}", "body": r.text[:500]}
    data = r.json()
    if not data.get("items"):
        return {"error": "No items returned", "videoId": video_id}
    snip = data["items"][0].get("snippet", {})
    return {
        "videoId": video_id,
        "title": snip.get("title"),
        "channelTitle": snip.get("channelTitle"),
        "channelId": snip.get("channelId"),
        "publishedAt": snip.get("publishedAt"),
    }


def video_metadata(url: str, api_key: str) -> dict:
    vid = extract_youtube_id(url)
    return fetch_youtube_metadata(vid, api_key) if vid else {"error": "Could not parse video ID from URL."}
