from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="European Football Screener", page_icon="⚽", layout="wide")
BASE = "https://site.api.espn.com/apis"
UNAVAILABLE = "UNAVAILABLE"
MIN_MATCHES = 5
EARLY_SHARE = 0.085

LEAGUES = {
    "England - Premier League":"eng.1","England - Championship":"eng.2","England - League One":"eng.3","England - League Two":"eng.4",
    "Spain - LaLiga":"esp.1","Spain - Segunda Division":"esp.2","Italy - Serie A":"ita.1","Italy - Serie B":"ita.2",
    "Germany - Bundesliga":"ger.1","Germany - 2. Bundesliga":"ger.2","France - Ligue 1":"fra.1","France - Ligue 2":"fra.2",
    "Netherlands - Eredivisie":"ned.1","Belgium - Pro League":"bel.1","Portugal - Primeira Liga":"por.1","Scotland - Premiership":"sco.1",
    "Austria - Bundesliga":"aut.1","Switzerland - Super League":"sui.1","Denmark - Superliga":"den.1","Norway - Eliteserien":"nor.1",
    "Sweden - Allsvenskan":"swe.1","Finland - Veikkausliiga":"fin.1","Greece - Super League":"gre.1","Turkey - Super Lig":"tur.1",
    "Czechia - First League":"cze.1","Poland - Ekstraklasa":"pol.1","Romania - Liga 1":"rou.1","Croatia - HNL":"cro.1",
    "Serbia - SuperLiga":"srb.1","Ukraine - Premier League":"ukr.1","Russia - Premier League":"rus.1","Ireland - Premier Division":"irl.1",
    "Iceland - Besta deild":"isl.1","Hungary - NB I":"hun.1"
}
HEADERS={"User-Agent":"Mozilla/5.0 FootballScreener/1.0","Accept":"application/json"}

def standings_url(s): return f"{BASE}/v2/sports/soccer/{s}/standings"
def scoreboard_url(s): return f"{BASE}/site/v2/sports/soccer/{s}/scoreboard"
def injuries_url(s): return f"{BASE}/site/v2/sports/soccer/{s}/injuries"

@st.cache_data(ttl=900, show_spinner=False)
def get_json(url: str, params: Optional[dict]=None) -> dict:
    r=requests.get(url,params=params,headers=HEADERS,timeout=20)
    r.raise_for_status()
    return r.json()

def walk(node: Any):
    if isinstance(node,dict):
        if isinstance(node.get("standings"),dict) and isinstance(node["standings"].get("entries"),list): yield node
        for v in node.values(): yield from walk(v)
    elif isinstance(node,list):
        for v in node: yield from walk(v)

def parse_standings(data):
    groups=list(walk(data))
    if not groups: return pd.DataFrame()
    group=max(groups,key=lambda g:len(g["standings"]["entries"]))
    rows=[]
    for e in group["standings"]["entries"]:
        team=e.get("team",{})
        stats={x.get("name"):x.get("value") for x in e.get("stats",[])}
        if stats.get("rank") is None or stats.get("gamesPlayed") is None: continue
        rows.append({"team_id":str(team.get("id","")),"team":team.get("displayName",UNAVAILABLE),"rank":int(stats["rank"]),"GP":int(stats["gamesPlayed"])})
    return pd.DataFrame(rows)

def parse_events(data):
    rows=[]
    for event in data.get("events",[]):
        comps=event.get("competitions") or []
        if not comps: continue
        comp=comps[0]; cs=comp.get("competitors") or []
        home=next((x for x in cs if x.get("homeAway")=="home"),None)
        away=next((x for x in cs if x.get("homeAway")=="away"),None)
        if not home or not away: continue
        def info(c):
            t=c.get("team") or {}; return str(c.get("id") or t.get("id") or ""),t.get("displayName",UNAVAILABLE)
        def score(c):
            v=c.get("score")
            if isinstance(v,dict): v=v.get("value",v.get("displayValue"))
            try:return float(v)
            except:return None
        hi,hn=info(home); ai,an=info(away)
        state=((comp.get("status") or {}).get("type") or {}).get("state",UNAVAILABLE)
        rows.append({"event_id":str(event.get("id","")),"date":event.get("date",UNAVAILABLE),"home_id":hi,"home":hn,"away_id":ai,"away":an,"state":state,"home_score":score(home),"away_score":score(away)})
    return rows

@st.cache_data(ttl=1800, show_spinner=False)
def completed_season_events(slug: str, today_iso: str):
    today=datetime.fromisoformat(today_iso).date(); start=today-timedelta(days=240)
    seen={}; cursor=start
    while cursor<=today:
        stop=min(cursor+timedelta(days=20),today)
        # ESPN accepts a compact start-end date string for historical chunks.
        try:
            d=get_json(scoreboard_url(slug),{"dates":f"{cursor:%Y%m%d}-{stop:%Y%m%d}","limit":1000})
            for e in parse_events(d):
                if e["event_id"]: seen[e["event_id"]]=e
        except Exception:
            # Fallback to one request per day if this league rejects ranges.
            day=cursor
            while day<=stop:
                try:
                    d=get_json(scoreboard_url(slug),{"dates":day.strftime("%Y%m%d")})
                    for e in parse_events(d):
                        if e["event_id"]: seen[e["event_id"]]=e
                except Exception: pass
                day+=timedelta(days=1)
        cursor=stop+timedelta(days=1)
    return list(seen.values())

def splits_and_means(events):
    s=defaultdict(lambda:{"home_GP":0,"home_GF":0.,"home_GA":0.,"away_GP":0,"away_GF":0.,"away_GA":0.})
    hg=ag=n=0.
    for e in events:
        if e["state"]!="post" or e["home_score"] is None or e["away_score"] is None: continue
        h,a=e["home_id"],e["away_id"]; hs,aws=e["home_score"],e["away_score"]
        s[h]["home_GP"]+=1;s[h]["home_GF"]+=hs;s[h]["home_GA"]+=aws
        s[a]["away_GP"]+=1;s[a]["away_GF"]+=aws;s[a]["away_GA"]+=hs
        hg+=hs;ag+=aws;n+=1
    return dict(s),(hg/n if n else None),(ag/n if n else None)

def model(h,a,s,mh,ma):
    hs,aws=s.get(h),s.get(a)
    if not hs or not aws or not mh or not ma or not hs["home_GP"] or not aws["away_GP"]:
        return {"lambda_home":UNAVAILABLE,"lambda_away":UNAVAILABLE,"P_BTTS":UNAVAILABLE,"P_NO_GOAL_10":UNAVAILABLE}
    lh=(hs["home_GF"]/hs["home_GP"]/mh)*(aws["away_GA"]/aws["away_GP"]/mh)*mh
    la=(aws["away_GF"]/aws["away_GP"]/ma)*(hs["home_GA"]/hs["home_GP"]/ma)*ma
    return {"lambda_home":round(lh,3),"lambda_away":round(la,3),"P_BTTS":f"{((1-math.exp(-lh))*(1-math.exp(-la))):.1%}","P_NO_GOAL_10":f"{math.exp(-(lh+la)*EARLY_SHARE):.1%}"}

def parse_injuries(data,wanted_ids,wanted_names):
    out=[]; containers=data.get("injuries") or data.get("items") or []
    if isinstance(containers,dict): containers=list(containers.values())
    for item in containers if isinstance(containers,list) else []:
        team=item.get("team") or {}; tid=str(team.get("id") or item.get("teamId") or ""); tn=team.get("displayName") or item.get("teamName") or UNAVAILABLE
        if tid not in wanted_ids and tn not in wanted_names: continue
        for x in item.get("injuries") or item.get("items") or [item]:
            athlete=x.get("athlete") or {}; status=x.get("status")
            if isinstance(status,dict): status=status.get("name") or status.get("type")
            detail=x.get("details") or x.get("description") or x.get("comment") or UNAVAILABLE
            if isinstance(detail,dict): detail=detail.get("detail") or detail.get("type") or str(detail)
            out.append({"Team":tn,"Player":athlete.get("displayName") or x.get("displayName") or UNAVAILABLE,"Status":status or UNAVAILABLE,"Details":detail})
    return out

def screen(hr,ar,n,s1,s2):
    labels=[]
    if s1 and ((hr<=4 and ar>=n-3) or (ar<=4 and hr>=n-3)): labels.append("S1")
    if s2 and ((5<=hr<=7 and ar>=n-2) or (5<=ar<=7 and hr>=n-2)): labels.append("S2")
    return labels

st.title("⚽ European Football Fixture Screener")
st.caption("ESPN public site API • ranking screeners • Poisson estimates")
with st.sidebar:
    st.header("Screen settings")
    selected=st.multiselect("Leagues",list(LEAGUES),default=list(LEAGUES)[:5])
    days=st.slider("Fixture window (days)",1,30,7)
    use_s1=st.checkbox("S1: Top-4 vs Bottom-4",True)
    use_s2=st.checkbox("S2: 5th-7th vs Bottom-3",True)
    use_models=st.checkbox("Poisson models",True)
    use_injuries=st.checkbox("Injuries",True)
    run=st.button("Run screen",type="primary",use_container_width=True)
st.info("CS% gate: UNAVAILABLE | FTS% gate: UNAVAILABLE — the requested free ESPN endpoints do not provide these gates reliably.")

if run:
    if not selected: st.warning("Select at least one league."); st.stop()
    if not (use_s1 or use_s2): st.warning("Enable S1, S2, or both."); st.stop()
    today=datetime.now(timezone.utc).date(); hits=[]; injuries=[]; warnings=[]
    progress=st.progress(0,text="Starting…")
    for idx,name in enumerate(selected,1):
        slug=LEAGUES[name]; progress.progress((idx-1)/len(selected),text=f"Loading {name}…")
        try:
            table=parse_standings(get_json(standings_url(slug)))
            if table.empty: warnings.append(f"{name}: standings UNAVAILABLE"); continue
            eligible=table[table.GP>=MIN_MATCHES].sort_values("rank")
            if len(eligible)<8: warnings.append(f"{name}: fewer than 8 eligible teams"); continue
            ranks=dict(zip(eligible.team_id,eligible["rank"])); n=len(eligible)
            upcoming=[]
            for offset in range(days+1):
                d=today+timedelta(days=offset)
                try:
                    upcoming.extend(x for x in parse_events(get_json(scoreboard_url(slug),{"dates":d.strftime("%Y%m%d")})) if x["state"]=="pre")
                except Exception as exc:
                    warnings.append(f"{name}: fixtures UNAVAILABLE for {d}: {type(exc).__name__}")
            sp,mh,ma=({},None,None)
            if use_models: sp,mh,ma=splits_and_means(completed_season_events(slug,today.isoformat()))
            league_hits=[]
            for f in upcoming:
                hr,ar=ranks.get(f["home_id"]),ranks.get(f["away_id"])
                if hr is None or ar is None: continue
                labels=screen(hr,ar,n,use_s1,use_s2)
                if not labels: continue
                row={"League":name,"Screener":", ".join(labels),"Kickoff (UTC)":f["date"],"Home":f["home"],"Home rank":hr,"Away":f["away"],"Away rank":ar,"CS% gate":UNAVAILABLE,"FTS% gate":UNAVAILABLE,"_home_id":f["home_id"],"_away_id":f["away_id"]}
                if use_models: row.update(model(f["home_id"],f["away_id"],sp,mh,ma))
                league_hits.append(row)
            hits.extend(league_hits)
            if use_injuries and league_hits:
                ids={str(r[k]) for r in league_hits for k in ("_home_id","_away_id")}; names={r[k] for r in league_hits for k in ("Home","Away")}
                try:
                    found=parse_injuries(get_json(injuries_url(slug)),ids,names)
                    if found:
                        for x in found:x["League"]=name
                        injuries.extend(found)
                    else: injuries.append({"League":name,"Team":", ".join(sorted(names)),"Player":UNAVAILABLE,"Status":UNAVAILABLE,"Details":"No usable injury records returned by ESPN."})
                except Exception as exc: injuries.append({"League":name,"Team":", ".join(sorted(names)),"Player":UNAVAILABLE,"Status":UNAVAILABLE,"Details":f"ESPN injury endpoint unavailable: {type(exc).__name__}"})
        except Exception as exc:
            warnings.append(f"{name}: {type(exc).__name__}: {exc}")
    progress.progress(1.0,text="Complete")
    st.subheader(f"Screener hits ({len(hits)})")
    if hits:
        df=pd.DataFrame(hits).drop(columns=["_home_id","_away_id"],errors="ignore")
        st.dataframe(df,use_container_width=True,hide_index=True)
        st.download_button("Download hits as CSV",df.to_csv(index=False).encode(),"football_screener_hits.csv","text/csv")
    else: st.warning("No fixtures matched the enabled screeners in the selected window.")
    if use_injuries:
        st.subheader("Injuries for involved teams")
        if injuries: st.dataframe(pd.DataFrame(injuries),use_container_width=True,hide_index=True)
        else: st.info("UNAVAILABLE — no screener-hit teams or no injury data returned.")
    if warnings:
        with st.expander(f"Warnings / unavailable data ({len(warnings)})"):
            for w in warnings: st.warning(w)
else:
    st.write("Choose leagues and options in the sidebar, then select **Run screen**.")
    st.caption("Probabilities are model estimates, not guarantees or betting advice.")
