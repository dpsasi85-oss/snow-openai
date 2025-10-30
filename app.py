# app.py — Conversational Snowflake ↔ ChatGPT (concise + history)
# Includes: no-autoscroll, collapsible Charts & SQL (default collapsed),
# session history (table, CSV, re-run), alias repair, GROUP BY fix, date filter injection,
# compact weather enrichment, no playbooks.

import os, re, json, time
from datetime import date
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit.components.v1 as components
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector import ProgrammingError
from openai import OpenAI
import altair as alt
import requests
import html


# Local guardrails (available in your repo)
from guardrails import is_safe_sql_detail, enforce_limit
from contextlib import contextmanager
import streamlit as st

# ───────────────── Boot & basic UI
load_dotenv()
st.set_page_config(page_title="Ask about your assets", layout="wide")

import streamlit as st, os
if hasattr(st, "secrets"):
    for k, v in st.secrets.items():
        if isinstance(v, (dict, list)):
            continue
        os.environ.setdefault(k, str(v))
        
st.markdown("""
<style>
  [data-testid="stSidebar"] { display: none; }
  .hero { margin-top: 4px; }
  .hero h1 { font-size: 2rem; margin: 0; }
  .hero p { color:#6b7280; margin:.25rem 0 0; }
  .sql-note { color:#6b7280; font-style: italic; }
  section[data-testid="stChatInput"] textarea { min-height: 80px; font-size: 1rem; }
  .stChatFloatingInputContainer { padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)
@contextmanager
def circle_loader(text: str = "Working"):
    ph = st.empty()
    ph.markdown(f"""
    <style>
      .cl-wrap {{ display:flex; align-items:center; gap:10px; padding:10px 12px;
                  border:1px solid #e5e7eb; border-radius:12px; background:#fafafa; }}
      .cl-ball {{ width:12px; height:12px; border-radius:50%; background:#9ca3af;
                  animation: cl-bounce .6s ease-in-out infinite; }}
      @keyframes cl-bounce {{ 0%,100% {{transform:translateY(0)}} 50% {{transform:translateY(-6px)}} }}
    </style>
    <div class="cl-wrap"><div class="cl-ball"></div>
      <div style="font-weight:600; color:#374151;">{text}</div>
    </div>
    """, unsafe_allow_html=True)
    try:
        yield
    finally:
        ph.empty()

# Keep viewport pinned to top (avoid chat input auto-focus scrolling)
def keep_viewport_top(blur_chat: bool = True):
    components.html(
        """
        <script>
        (function(){
          const t=(f)=>setTimeout(f,60),
                topNow=()=>{const m=parent.document.querySelector('section.main'); if(m) m.scrollTo(0,0); parent.scrollTo(0,0)},
                blur=()=>{const ta=parent.document.querySelector("section[data-testid='stChatInput'] textarea"); if(ta) ta.blur()};
          topNow(); t(()=>{ topNow(); %s });
          const mo=new MutationObserver(()=>{ topNow(); %s; mo.disconnect() });
          mo.observe(parent.document.body,{childList:true,subtree:true});
        })();
        </script>
        """ % ("blur();" if blur_chat else "", "blur();" if blur_chat else ""), height=0)
keep_viewport_top(True)

# Header
c1, c2 = st.columns([1,6])
with c1: st.image("logo.png", width=80)
with c2:
    st.markdown('<div class="hero"><h1>Ask about your assets</h1>'
                '<p>Conversational insights • Weather-aware • Guardrailed SQL</p></div>',
                unsafe_allow_html=True)

# Filters (simple, collapsed by default)
st.session_state.setdefault("_filters", {"lookback_days":30, "include_weather":True, "weather_days":3})
with st.expander("Filters", expanded=False):
    F = st.session_state._filters
    F["lookback_days"]  = st.slider("Lookback (days)", 0, 365, F["lookback_days"])
    F["include_weather"] = st.checkbox("Include weather forecast by site", value=F["include_weather"])
    F["weather_days"]   = st.slider("Forecast horizon (days)", 1, 7, F["weather_days"])

# ───────────────── Config & clients
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
SF_ACCOUNT       = os.getenv("SNOWFLAKE_ACCOUNT")
SF_USER          = os.getenv("SNOWFLAKE_USER")
SF_PASSWORD      = os.getenv("SNOWFLAKE_PASSWORD")
SF_WAREHOUSE     = os.getenv("SNOWFLAKE_WAREHOUSE")
SF_DB            = os.getenv("SNOWFLAKE_DATABASE")
SF_SCHEMA        = os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
SF_ROLE          = os.getenv("SNOWFLAKE_ROLE")
PRIMARY_MODEL    = os.getenv("OPENAI_MODEL_PRIMARY", "gpt-4o-mini")
FALLBACK_MODEL   = os.getenv("OPENAI_MODEL_FALLBACK", "gpt-3.5-turbo")
OPENAI_RETRIES   = int(os.getenv("OPENAI_RETRIES", "2"))

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in env."); st.stop()

@st.cache_resource(show_spinner=False)
def _openai(): return OpenAI(api_key=OPENAI_API_KEY)
client = _openai()

@st.cache_resource(show_spinner=False)
def sf_conn():
    return snowflake.connector.connect(
        account=SF_ACCOUNT, user=SF_USER, password=SF_PASSWORD,
        warehouse=SF_WAREHOUSE, database=SF_DB, schema=SF_SCHEMA, role=SF_ROLE,
        client_session_keep_alive=True,
        session_parameters={"QUERY_TAG":"SF_CHATGPT_POC","STATEMENT_TIMEOUT_IN_SECONDS":30},
    )

@st.cache_data(ttl=60, show_spinner=False)
def run_sql(sql: str) -> pd.DataFrame:
    t0 = time.perf_counter()
    with sf_conn().cursor() as cur:
        cur.execute(sql)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["_elapsed_s"] = time.perf_counter() - t0
    return df

# ───────────────── Schema helpers for LLM
ALLOWED_TABLES = ["ASSET_DIM","SITE_DIM","ASSET_DAILY_FACT","MAINTENANCE_EVENTS","WORK_ORDERS","ASSET_MGMT"]

@st.cache_data(ttl=300)
def describe_schema(allowed: List[str]) -> str:
    if not allowed: return ""
    ph = ",".join(["%s"]*len(allowed))
    q = f"""
    SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
    FROM {SF_DB}.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA=%s AND TABLE_NAME IN ({ph})
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    with sf_conn().cursor() as cur:
        cur.execute(q, (SF_SCHEMA, *allowed)); rows = cur.fetchall()
    out, cur_tbl = [], None
    for tbl, col, dt in rows:
        if tbl != cur_tbl:
            if cur_tbl is not None and out and out[-1].endswith(","): out[-1]=out[-1][:-1]; out.append(")")
            out.append(f"TABLE {tbl}("); cur_tbl = tbl
        out.append(f"  {col} {dt},")
    if out and out[-1].endswith(","): out[-1]=out[-1][:-1]
    if out: out.append(")")
    return "\n".join(out)

SCHEMA_TEXT = describe_schema(ALLOWED_TABLES)
RELATIONSHIPS_TEXT = (
    "- ASSET_DIM.ASSET_ID = ASSET_DAILY_FACT.ASSET_ID = MAINTENANCE_EVENTS.ASSET_ID = WORK_ORDERS.ASSET_ID\n"
    "- SITE_DIM.SITE_ID  = ASSET_DAILY_FACT.SITE_ID  = MAINTENANCE_EVENTS.SITE_ID  = WORK_ORDERS.SITE_ID\n"
    "- MAINTENANCE_EVENTS.EVENT_ID = WORK_ORDERS.EVENT_ID\n"
    "Guidance: Use explicit JOIN ... ON ..., aggregate correctly, and limit rows."
)

# ───────────────── Weather (minimal, cached)
@st.cache_data(ttl=600)
def get_site_coords() -> pd.DataFrame:
    try:
        return run_sql("""
            SELECT SITE_ID, SITE_NAME, LAT, LON
            FROM SITE_DIM
            WHERE LAT IS NOT NULL AND LON IS NOT NULL
            ORDER BY SITE_NAME
        """)
    except Exception:
        return pd.DataFrame(columns=["SITE_ID","SITE_NAME","LAT","LON"])

@st.cache_data(ttl=1200)
def fetch_forecast(lat: float, lon: float, days: int) -> Dict[str, Any]:
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": float(lat), "longitude": float(lon),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "windspeed_unit": "kmh", "timezone": "auto",
    }, timeout=10); r.raise_for_status()
    d=r.json().get("daily",{})
    cut=lambda x: x[:days] if isinstance(x,list) else x
    return {"date":cut(d.get("time",[])), "tmax_c":cut(d.get("temperature_2m_max",[])),
            "tmin_c":cut(d.get("temperature_2m_min",[])), "precip_mm":cut(d.get("precipitation_sum",[])),
            "wind_max_kmh":cut(d.get("windspeed_10m_max",[]))}

@st.cache_data(ttl=1200)
def build_weather_bundle(days: int=3) -> Dict[str,Any]:
    sites=get_site_coords(); out={"sites":{}}
    if sites.empty: return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(fetch_forecast,float(r.LAT),float(r.LON),days):(r.SITE_ID,r.SITE_NAME,r.LAT,r.LON) for _,r in sites.iterrows()}
        for f,(sid,sname,lat,lon) in list(futs.items()):
            try:
                fc=f.result(); lst=[]
                for i in range(len(fc["date"])):
                    flags={"heat_risk":(fc["tmax_c"][i] is not None and fc["tmax_c"][i]>=42),
                           "rain_risk":(fc["precip_mm"][i] is not None and fc["precip_mm"][i]>=5),
                           "wind_risk":(fc["wind_max_kmh"][i] is not None and fc["wind_max_kmh"][i]>=40)}
                    lst.append({k:fc[k][i] for k in ("date","tmax_c","tmin_c","precip_mm","wind_max_kmh")} | {"flags":flags})
                out["sites"][sid]={"site_name":sname,"lat":float(lat),"lon":float(lon),"forecast":lst}
            except Exception as e:
                out["sites"][sid]={"site_name":sname,"lat":float(lat),"lon":float(lon),"error":str(e),"forecast":[]}
    return out

def summarize_weather_risks(wx: Dict[str,Any]|None)->List[Dict[str,Any]]:
    out=[]
    if not wx or not wx.get("sites"): return out
    for sid,node in wx["sites"].items():
        earliest=None; agg={"heat":False,"rain":False,"wind":False}
        for d in node.get("forecast",[]):
            f=d.get("flags",{}); risky=any([f.get("heat_risk"),f.get("rain_risk"),f.get("wind_risk")])
            if risky and earliest is None: earliest=d.get("date")
            agg["heat"]|=bool(f.get("heat_risk")); agg["rain"]|=bool(f.get("rain_risk")); agg["wind"]|=bool(f.get("wind_risk"))
        out.append({"site_id":str(sid),"site_name":node.get("site_name") or sid,"first_risk_date":earliest,
                    "heat":agg["heat"],"rain":agg["rain"],"wind":agg["wind"],"lat":node.get("lat"),"lon":node.get("lon")})
    return out

def weather_enrich_df(df: pd.DataFrame, wx: Dict[str,Any]|None) -> pd.DataFrame:
    if df is None or df.empty or not wx: return df
    colsU=[c.upper() for c in df.columns]; has_id="SITE_ID" in colsU; has_name="SITE_NAME" in colsU
    if not (has_id or has_name): return df
    dig=summarize_weather_risks(wx)
    m_id={d["site_id"]:d for d in dig if d.get("site_id")}
    m_nm={d["site_name"].upper():d for d in dig if d.get("site_name")}
    df2=df.copy(); df2["WX_FIRST_RISK_DATE"], df2["WX_RISK_TYPES"] = None, ""
    sid_col=next((c for c in df.columns if c.upper()=="SITE_ID"),None)
    sname_col=next((c for c in df.columns if c.upper()=="SITE_NAME"),None)
    for i,row in df2.iterrows():
        pick=None
        if has_id and sid_col: pick=m_id.get(str(row[sid_col]))
        if not pick and has_name and sname_col: pick=m_nm.get(str(row[sname_col]).upper())
        if pick:
            kinds=[k for k in ("heat","rain","wind") if pick.get(k)]
            df2.at[i,"WX_FIRST_RISK_DATE"]=pick.get("first_risk_date")
            df2.at[i,"WX_RISK_TYPES"]=", ".join(kinds)
    return df2

# ───────────────── LLM + SQL helpers
BASE_SQL_RULES = (
    "You are a Snowflake SQL expert. Return ONE read-only SELECT using ONLY the provided tables. "
    "Rules: one statement; no DML/DDL; use explicit JOIN ... ON ...; add LIMIT<=1000 if missing; follow relationships; "
    "be careful with aggregations; use base table names only. Return JSON {\"sql\":...,\"rationale\":...}."
    "Aggregate base measures once at the grouping grain. Never wrap SUM/AVG around already-aggregated expressions. Use analytic window functions for shares and cumulative totals."
)
STRICT_SQL_RULES = (
    "Rewrite into a single SELECT with proper GROUP BY and LIMIT 1000 if missing. No CTEs, no schema prefixes, no comments."
)

def _retry_wait(i: int): time.sleep(0.6*(2**i))

def _openai_json(system: str, user: str) -> Dict[str,Any]:
    err=None
    for a in range(OPENAI_RETRIES+1):
        for m in (PRIMARY_MODEL,FALLBACK_MODEL):
            try:
                r=client.chat.completions.create(
                    model=m,temperature=0.1,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                    response_format={"type":"json_object"},
                )
                return json.loads(r.choices[0].message.content)
            except Exception as e: err=e
        if a<OPENAI_RETRIES: _retry_wait(a)
    raise RuntimeError(f"OpenAI JSON failed: {err}")

def _openai_text(system: str, user: str) -> str:
    err=None
    for a in range(OPENAI_RETRIES+1):
        for m in (PRIMARY_MODEL,FALLBACK_MODEL):
            try:
                r=client.chat.completions.create(
                    model=m,temperature=0.0,
                    messages=[{"role":"system","content":system},{"role":"user","content":user}],
                )
                return r.choices[0].message.content.strip()
            except Exception as e: err=e
        if a<OPENAI_RETRIES: _retry_wait(a)
    raise RuntimeError(f"OpenAI text failed: {err}")

def normalize_sql(sql: str) -> str:
    if not sql: return sql
    sql=re.sub(r"/\*.*?\*/"," ",sql,flags=re.DOTALL); sql=re.sub(r"--[^\n]*"," ",sql)
    def strip_prefix(s): return s.split(".")[-1].strip().strip('"')
    sql=re.sub(
        r"\b(FROM|JOIN)\s+((?:\"[^\"]+\"|[A-Za-z_][\w\$]*)(?:\.(?:\"[^\"]+\"|[A-Za-z_][\w\$]*)){1,2})",
        lambda m:f"{m.group(1)} "+strip_prefix(m.group(2)), sql, flags=re.IGNORECASE
    )
    return re.sub(r"\s+"," ",sql).strip().rstrip(";")

GROUPBY_FIX_RULES = (
    "Ensure every non-aggregated select column is in GROUP BY and aggregate sensible measures (SUM *_HOURS, *_$, OUTPUT_UNITS; AVG *_PCT). "
    "Single SELECT, no CTEs/prefixes/comments, add LIMIT 1000 if missing."
    "Arrange in ascending or descending order based on the context of the request"
)

def propose_sql(question: str, schema_text: str, rel_text: str, allowed: List[str], max_rows:int=1000):
    plan=_openai_json(BASE_SQL_RULES, f"Schema\n{schema_text}\n\n{rel_text}\n\nQuestion\n{question}\nReturn JSON.")
    sql1=(plan.get("sql") or "").strip(); rat=plan.get("rationale","") or "LLM plan"
    for candidate in (sql1, normalize_sql(sql1), _openai_text(STRICT_SQL_RULES+"\n"+GROUPBY_FIX_RULES, sql1)):
        ok,_,_=is_safe_sql_detail(candidate, allowed_tables=allowed)
        if ok: return {"sql": enforce_limit(candidate, max_rows), "rationale": rat}
    raise ValueError("Could not produce a safe SQL.")

TRAILING_CLAUSE_RE = re.compile(r"\b(QUALIFY|HAVING|GROUP\s+BY|ORDER\s+BY|LIMIT)\b", re.IGNORECASE)

def inject_global_filters(sql: str, days: int) -> str:
    s=sql.strip().rstrip(";"); preds=[]
    if days:
        cands=[]
        if re.search(r"\bFACT_DATE\b", s, flags=re.IGNORECASE): cands.append("FACT_DATE")
        cands += [f"{m.group(1)}.FACT_DATE" for m in re.finditer(r"\b([A-Za-z_][\w]*)\.FACT_DATE\b", s, flags=re.IGNORECASE)]
        if re.search(r"\bEVENT_DATE\b", s, flags=re.IGNORECASE): cands.append("EVENT_DATE")
        cands += [f"{m.group(1)}.EVENT_DATE" for m in re.finditer(r"\b([A-Za-z_][\w]*)\.EVENT_DATE\b", s, flags=re.IGNORECASE)]
        seen=set(); uniq=[x for x in cands if (x.upper() not in seen and not seen.add(x.upper()))]
        if uniq:
            lhs="COALESCE("+", ".join([f"TRY_TO_DATE({c})" for c in uniq[:2]])+")"
            rhs=f"DATEADD(day, -{int(days)}, CURRENT_DATE())"
            preds.append(f"{lhs} >= {rhs}")
    if not preds: return s
    extra=" AND ".join(preds); m=TRAILING_CLAUSE_RE.search(s); i=m.start() if m else len(s)
    pre, suf = s[:i].rstrip(), s[i:]
    pre += (" AND "+extra) if re.search(r"\bWHERE\b", pre, flags=re.IGNORECASE) else (" WHERE "+extra)
    return (pre+" "+suf).strip()

ALIAS_MAP = {"SD":"SITE_DIM","AD":"ASSET_DIM","ADF":"ASSET_DAILY_FACT","ME":"MAINTENANCE_EVENTS","WO":"WORK_ORDERS"}
ALIAS_DEF_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w\$]*)(?:\s+AS)?\s+([A-Za-z_][\w\$]*)", re.IGNORECASE)
TABLE_TOKEN_RE = re.compile(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w\$]*)(?!\s+[A-Za-z_])", re.IGNORECASE)

def auto_add_missing_alias(sql: str) -> str:
    used={a.upper() for a in re.findall(r"\b([A-Za-z][\w_]*)\.[A-Za-z_][\w_]*\b", sql)}
    defined={m.group(2).upper() for m in ALIAS_DEF_RE.finditer(sql)}
    tables=[m.group(1).upper() for m in TABLE_TOKEN_RE.finditer(sql)]
    missing=[a for a in used if a in ALIAS_MAP and a not in defined]
    out=sql
    for a in missing:
        base=ALIAS_MAP[a]
        if base.upper() in tables:
            out=re.sub(rf"\b(FROM|JOIN)\s+{base}\b(?!\s+[A-Za-z_])", rf"\1 {base} {a}", out, count=1, flags=re.IGNORECASE)
    return out

# ───────────────── Summarizer
def summarize_answer(question: str, df: pd.DataFrame, context: Dict[str,Any]) -> str:
    sample=df.head(40).copy()
    for c in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[c]): sample[c]=sample[c].astype(str)
    sample_json=json.dumps(sample.to_dict(orient="records"), default=str)[:3500]
    ctx_json=json.dumps(context, default=str)[:6000]
    r=client.chat.completions.create(
        model=PRIMARY_MODEL, temperature=0.25,
        messages=[
            {"role":"system","content":(
                "You are an asset-management advisor for mining ops. Write crisp guidance grounded in the provided data. "
                "Output 2–3 short paragraphs + 3 action bullets. Always cite sites/assets/types with figures and absolute dates."
                "Ensure that the dates proposed for maintenance are never past dates"
                "Do not use Markdown emphasis.If you mention column names with underscores, wrap them in backticks."

            )},
            {"role":"user","content":(
                f"Question:\n{question}\n\nResult sample (JSON rows):\n{sample_json}\n\nContext: {ctx_json}\n\n"
                "Structure:\n• Opening: one-sentence takeaway.\n• Analysis with concrete figures.\n"
                "• Weather: risks & scheduling moves.\n• Bullets: 'Site ops', 'Maintenance', 'Work orders & staffing'."
            )},
        ],
    )
    return r.choices[0].message.content

# ───────────────── Session history setup
st.session_state.setdefault("history", [])

# ───────────────── App flow
if "greeted" not in st.session_state:
    with st.chat_message("assistant"):
        st.markdown("Good Day! 👋 Ask about your asset data and I'll crunch it—with weather context where helpful.")
    st.session_state.greeted=True

# Allow re-run from history to preseed the chat box
seed = st.session_state.pop("preseed_prompt", None)
prompt = seed or st.chat_input("Ask a question about the assets… e.g., least downtime sites in last 30 days")

if prompt:
    with st.chat_message("user"):
        st.write(prompt)
    with circle_loader("Crunching your question…"):
        lb = int(st.session_state._filters["lookback_days"])
        weather_days = int(st.session_state._filters["weather_days"])
        include_weather = bool(st.session_state._filters["include_weather"])

    lb = int(st.session_state._filters["lookback_days"])
    weather_days = int(st.session_state._filters["weather_days"])
    include_weather = bool(st.session_state._filters["include_weather"])

    # ── Build SQL (no status UI)
    plan = propose_sql(
        prompt + f"\n\n[Constraints]\n- Restrict dates to last {lb} days when date columns exist.",
        SCHEMA_TEXT, RELATIONSHIPS_TEXT, ALLOWED_TABLES
    )
    sql = inject_global_filters(plan["sql"].strip(), lb)
    sql = auto_add_missing_alias(sql)

    # ── Execute (no status/row count shown)
    try:
        df = run_sql(sql)
    except ProgrammingError as e:
        msg = str(e)
        if re.search(r"invalid identifier '([A-Za-z0-9_]+)\.[A-Za-z0-9_]+'", msg, flags=re.IGNORECASE):
            sql2 = auto_add_missing_alias(sql); df = run_sql(sql2); sql = sql2
        elif re.search(r"neither an aggregate nor in the group by", msg, flags=re.IGNORECASE):
            fixed = _openai_text(STRICT_SQL_RULES + "\n" + GROUPBY_FIX_RULES, sql)
            fixed = normalize_sql(fixed); fixed = auto_add_missing_alias(fixed)
            df = run_sql(fixed); sql = fixed
        else:
            raise

    # ── Optional weather and advisor note (silent)
    wx = build_weather_bundle(weather_days) if include_weather else {}
    digest = summarize_weather_risks(wx) if include_weather else []
    advisor = summarize_answer(
        prompt, df,
        {"as_of": str(date.today()), "window_days": lb, "weather_digest": digest}
    )

    with st.chat_message("assistant"):
        safe = html.escape(advisor).replace("\n\n", "<br><br>").replace("\n", "<br>")
        st.markdown(f"<div class='advisor'>{safe}</div>", unsafe_allow_html=True)

        if df.empty:
            st.info("No rows returned.")
        else:
            df = weather_enrich_df(df, wx)
            st.dataframe(df, use_container_width=True)
            st.caption("Weather columns: WX_FIRST_RISK_DATE (earliest risky day), WX_RISK_TYPES (heat, rain, wind)")

            # Collapsible charts (default collapsed)
            with st.expander("📈 Charts & Visuals", expanded=False):
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if num_cols:
                    x = df.columns[0]; y = num_cols[0]
                    gp = df.copy()
                    if not pd.api.types.is_datetime64_any_dtype(gp[x]):
                        try: gp[x] = pd.to_datetime(gp[x])
                        except: pass
                    ch = alt.Chart(gp).mark_bar().encode(
                        x=alt.X(x), y=alt.Y(y), tooltip=list(gp.columns)
                    ).properties(height=420, width="container")
                    st.altair_chart(ch.interactive(), use_container_width=True)
                else:
                    st.caption("No numeric columns to chart.")

        # Collapsible SQL (default collapsed)
        with st.expander("🧾 SQL query", expanded=False):
            st.markdown("<div class='sql-note'>This is the SQL my circuits ran on ☕🤖:</div>", unsafe_allow_html=True)
            st.code(sql, language="sql")

        st.download_button(
            "Download result as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Log to session history (internal only; not shown as a status)
    st.session_state.history.append({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "q": prompt,
        "rows": int(len(df)),
        "ms": int(1000*df.attrs.get("_elapsed_s", 0)),
        "sql": sql,
    })

# Session history panel
if st.session_state.history:
    with st.expander("🧭 Session history", expanded=False):
        hist = st.session_state.history
        st.caption(f"Runs: {len(hist)}")
        try:
            hist_df = pd.DataFrame(hist)
            st.dataframe(hist_df, use_container_width=True)
            st.download_button(
                "Download history (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="session_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        except Exception:
            pass
        # Re-run a past query
        options = [(i, h.get("q","")) for i, h in enumerate(hist)]
        if options:
            sel = st.selectbox(
                "Re-run a past query",
                options=options,
                index=len(options)-1,
                format_func=lambda t: f"{t[0]+1}. {t[1][:80]}" if isinstance(t, tuple) else str(t)
            )
            cA, cB = st.columns([1,1])
            with cA:
                if st.button("Re-run selected"):
                    st.session_state.preseed_prompt = sel[1]
                    st.rerun()
            with cB:
                if st.button("Clear history"):
                    st.session_state.history = []
                    st.rerun()
else:
    st.caption("No session history yet — run a query to populate it.")
