import os, time, requests
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import poisson

KEY = os.environ["KEY"]
H = {"X-Auth-Token": KEY}
BASE = "https://api.football-data.org/v4/competitions/FL1/matches"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])
cache = {"t": 0, "data": []}

def proba(lh, la):
    p1 = pn = p2 = 0
    for i in range(10):
        for j in range(10):
            p = poisson.pmf(i, lh) * poisson.pmf(j, la)
            if i > j: p1 += p
            elif i == j: pn += p
            else: p2 += p
    return p1, pn, p2

def calculer():
    rows = []
    for season in [2023, 2024, 2025, 2026]:
        r = requests.get(f"{BASE}?season={season}&status=FINISHED", headers=H)
        for m in r.json().get("matches", []):
            s = m["score"]["fullTime"]
            if s["home"] is None:
                continue
            rows.append((m["homeTeam"]["shortName"], m["awayTeam"]["shortName"], s["home"], s["away"]))
    d = pd.DataFrame(rows, columns=["home", "away", "hg", "ag"])
    k = 20
    mh, ma = d.hg.mean(), d.ag.mean()

    def force(g, col, m):
        s = d.groupby(g)[col].sum()
        n = d.groupby(g)[col].count()
        return ((s + k * m) / (n + k)) / m

    ah, dh = force("home", "hg", mh), force("home", "ag", ma)
    aa, da = force("away", "ag", ma), force("away", "hg", mh)

    r = requests.get(f"{BASE}?season=2026&status=SCHEDULED,TIMED", headers=H)
    out = []
    for m in r.json().get("matches", [])[:10]:
        h, a = m["homeTeam"]["shortName"], m["awayTeam"]["shortName"]
        try:
            lh = ah[h] * da[a] * mh
            la = aa[a] * dh[h] * ma
        except KeyError:
            continue
        p1, pn, p2 = proba(lh, la)
        out.append({"date": m["utcDate"][:10], "dom": h, "ext": a,
                    "p1": round(p1 * 100), "pn": round(pn * 100), "p2": round(p2 * 100)})
    return out

@app.get("/")
def accueil():
    return {"statut": "ok"}

@app.get("/predictions")
def predictions():
    if time.time() - cache["t"] > 6 * 3600:
        try:
            cache["data"] = calculer()
            cache["t"] = time.time()
        except Exception:
            pass
    return cache["data"]
