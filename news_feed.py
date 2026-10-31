"""
Crypto news plug-in.

Two kinds of real-world information, used honestly:

1. Fear & Greed Index (alternative.me) - daily market sentiment 0..100 with
   history back to 2018. It is a TRAINED model feature (fng, fng_chg_7, fng_ma_30):
   the walk-forward test can measure whether it actually helps.

2. Headlines (Cointelegraph, CoinDesk, Decrypt, The Block RSS) - no free
   history exists, so the model cannot learn from them yet. Today they are used as
   a RULES-BASED risk filter (hack / exploit / delisting / lawsuit news on a coin
   blocks new entries; the most severe news forces an exit), and every headline is
   archived daily to logs/news/headlines.jsonl. Once ~180 days are collected,
   ml_autoresearch.py tests a headline-sentiment feature automatically.

Only text and numbers are downloaded; nothing fetched is ever executed.
"""

from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
FNG_PATH = ROOT / "reference_data" / "fear_greed.csv"
NEWS_DIR = ROOT / "logs" / "news"
ARCHIVE_PATH = NEWS_DIR / "headlines.jsonl"
UA = {"User-Agent": "Mozilla/5.0 (crypto-quant research bot)"}

RSS_FEEDS = {
    "cointelegraph": "https://cointelegraph.com/rss",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "decrypt": "https://decrypt.co/feed",
    "theblock": "https://www.theblock.co/rss.xml",
}

# Words that name each coin in headlines (base symbol -> aliases). Short tickers
# are matched as whole words only, so "OP" doesn't match "OPEN".
COIN_ALIASES: Dict[str, List[str]] = {
    "XRP": ["xrp", "ripple"], "HBAR": ["hbar", "hedera"], "ONDO": ["ondo"],
    "SOL": ["solana", "sol"], "AVAX": ["avalanche", "avax"], "NEAR": ["near protocol", "near"],
    "LINK": ["chainlink", "link"], "AAVE": ["aave"], "UNI": ["uniswap", "uni"],
    "ARB": ["arbitrum", "arb"], "OP": ["optimism"], "SUI": ["sui"], "PENDLE": ["pendle"],
    "DOGE": ["dogecoin", "doge"], "SHIB": ["shiba inu", "shib"], "PEPE": ["pepe"],
    "BONK": ["bonk"], "WIF": ["dogwifhat", "wif"], "FLOKI": ["floki"],
}

# Severe: a rug/hack/delisting in progress -> sell the position.
SEVERE_PATTERNS = [
    r"\bhack(ed|er|ers)?\b", r"\bexploit(ed)?\b", r"\bdrain(ed)?\b", r"\brug ?pull", r"\bdelist",
    r"\bhalts? (withdrawals|trading)\b", r"\binsolven", r"\bbankrupt",
]
# Caution: bad news worth sitting out new entries for.
CAUTION_PATTERNS = [
    r"\bsec (sues|charges|lawsuit)\b", r"\blawsuit\b", r"\bfraud\b", r"\bscam\b", r"\bvulnerab",
    r"\boutage\b", r"\bdepeg", r"\bunlock", r"\binvestigat", r"\bsell[- ]off\b", r"\bliquidat",
]
POS_WORDS = {"surge", "surges", "rally", "rallies", "soar", "soars", "record", "approval", "approved",
             "adoption", "partnership", "gain", "gains", "bullish", "inflows", "breakout", "upgrade"}
NEG_WORDS = {"crash", "crashes", "plunge", "plunges", "slump", "drop", "drops", "fall", "falls",
             "bearish", "outflows", "fear", "hack", "exploit", "lawsuit", "ban", "fraud", "selloff"}


# --------------------------------------------------------------------------- #
# Fear & Greed (trained feature)                                              #
# --------------------------------------------------------------------------- #
def update_fear_greed(timeout: float = 20.0) -> pd.DataFrame:
    """Download the full Fear & Greed history and cache it. Falls back to the cache offline."""
    # The index updates once a day; don't re-download it every hourly cycle.
    if FNG_PATH.exists() and time.time() - FNG_PATH.stat().st_mtime < 6 * 3600:
        return load_fear_greed()
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=0&format=json", timeout=timeout, headers=UA)
        r.raise_for_status()
        rows = [
            {"date": pd.Timestamp(int(x["timestamp"]), unit="s").normalize(), "fng": float(x["value"])}
            for x in r.json().get("data", [])
        ]
        df = pd.DataFrame(rows).drop_duplicates("date").set_index("date").sort_index()
        if len(df) > 100:
            FNG_PATH.parent.mkdir(exist_ok=True)
            df.to_csv(FNG_PATH)
            return df
    except Exception as exc:
        print(f"[news] Fear & Greed download failed ({exc}); using cache")
    return load_fear_greed()


def load_fear_greed() -> pd.DataFrame:
    if not FNG_PATH.exists():
        return pd.DataFrame(columns=["fng"])
    return pd.read_csv(FNG_PATH, parse_dates=["date"]).set_index("date").sort_index()


def fear_greed_features(fng: pd.DataFrame) -> pd.DataFrame:
    """Features known at the close of each day (index value published at 00:00 UTC for that day)."""
    if fng.empty:
        return pd.DataFrame(columns=["fng", "fng_chg_7", "fng_ma_30"])
    s = fng["fng"].astype(float).asfreq("D").ffill()
    return pd.DataFrame({"fng": s, "fng_chg_7": s - s.shift(7), "fng_ma_30": s.rolling(30).mean()})


# --------------------------------------------------------------------------- #
# Headlines (rules today, archived for future training)                       #
# --------------------------------------------------------------------------- #
def _parse_rss(source: str, xml_bytes: bytes) -> List[dict]:
    items = []
    root = ET.fromstring(xml_bytes)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        pub = it.findtext("pubDate") or ""
        try:
            ts = parsedate_to_datetime(pub).astimezone(timezone.utc)
        except Exception:
            ts = datetime.now(timezone.utc)
        desc = re.sub(r"<[^>]+>", " ", it.findtext("description") or "")[:400]
        items.append({"source": source, "published": ts.isoformat(), "title": title,
                      "summary": " ".join(desc.split()), "link": (it.findtext("link") or "").strip()})
    return items


def fetch_headlines(timeout: float = 20.0) -> List[dict]:
    out: List[dict] = []
    for source, url in RSS_FEEDS.items():
        try:
            r = requests.get(url, timeout=timeout, headers=UA)
            r.raise_for_status()
            out.extend(_parse_rss(source, r.content))
        except Exception as exc:
            print(f"[news] {source} feed failed: {exc}")
    return out


def archive_headlines(items: List[dict]) -> int:
    """Append unseen headlines (dedupe by title) to the long-term archive."""
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    if ARCHIVE_PATH.exists():
        with ARCHIVE_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["title"])
                except Exception:
                    continue
    new = [x for x in items if x["title"] not in seen]
    with ARCHIVE_PATH.open("a", encoding="utf-8") as f:
        for x in new:
            x = dict(x, sentiment=headline_sentiment(x["title"]), coins=coins_mentioned(x["title"]))
            f.write(json.dumps(x) + "\n")
    return len(new)


def headline_sentiment(text: str) -> int:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return len(words & POS_WORDS) - len(words & NEG_WORDS)


def coins_mentioned(text: str) -> List[str]:
    t = text.lower()
    return [base for base, aliases in COIN_ALIASES.items()
            if any(re.search(rf"\b{re.escape(a)}\b", t) for a in aliases)]


def coin_news_risk(items: List[dict], lookback_hours: int = 48) -> Dict[str, dict]:
    """
    Per coin: {"level": "severe"|"caution", "headlines": [...]}.
    Only headlines where the coin is named in the TITLE count (summaries are too loose).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    risk: Dict[str, dict] = {}
    for x in items:
        try:
            if datetime.fromisoformat(x["published"]) < cutoff:
                continue
        except Exception:
            continue
        title = x["title"].lower()
        level = None
        if any(re.search(p, title) for p in SEVERE_PATTERNS):
            level = "severe"
        elif any(re.search(p, title) for p in CAUTION_PATTERNS):
            level = "caution"
        if level is None:
            continue
        for coin in coins_mentioned(x["title"]):
            cur = risk.setdefault(coin, {"level": level, "headlines": []})
            if level == "severe":
                cur["level"] = "severe"
            cur["headlines"].append(f"[{x['source']}] {x['title']}")
    return risk


def market_mood(items: List[dict], lookback_hours: int = 24) -> Optional[float]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    scores = []
    for x in items:
        try:
            if datetime.fromisoformat(x["published"]) >= cutoff:
                scores.append(headline_sentiment(x["title"]))
        except Exception:
            continue
    return float(sum(scores) / len(scores)) if scores else None


def archive_days() -> int:
    if not ARCHIVE_PATH.exists():
        return 0
    days = set()
    with ARCHIVE_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                days.add(json.loads(line)["published"][:10])
            except Exception:
                continue
    return len(days)


def news_snapshot() -> dict:
    """One call for the live runner: refresh everything, archive, return the risk view."""
    fng = update_fear_greed()
    items = fetch_headlines()
    new = archive_headlines(items)
    latest = fng["fng"].iloc[-1] if not fng.empty else None
    return {
        "fear_greed": float(latest) if latest is not None else None,
        "headlines": len(items),
        "new_archived": new,
        "mood_24h": market_mood(items),
        "coin_risk": coin_news_risk(items),
        "fetched_at": time.time(),
    }


if __name__ == "__main__":
    snap = news_snapshot()
    print(f"Fear & Greed: {snap['fear_greed']}  headlines: {snap['headlines']}  "
          f"newly archived: {snap['new_archived']}  24h mood: {snap['mood_24h']}")
    print(f"Archive covers {archive_days()} days")
    for coin, info in snap["coin_risk"].items():
        print(f"  {coin}: {info['level'].upper()}")
        for h in info["headlines"][:3]:
            print(f"     {h}")
