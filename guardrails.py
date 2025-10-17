import re

# Allowed read-only patterns; forbid data-changing / session / metadata ops
FORBIDDEN_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|COPY|PUT|REMOVE|LIST|UNLOAD|CALL|"
    r"CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE|UNSET|SET|USE|"
    r"BEGIN|COMMIT|ROLLBACK|SHOW|DESCRIBE|EXPLAIN|ANALYZE"
    r")\b",
    re.IGNORECASE
)

# Disallow querying Snowflake metadata or cross-account/system views via FROM/JOIN
DISALLOWED_SOURCES = re.compile(
    r"\b(?:INFORMATION_SCHEMA|ACCOUNT_USAGE|SNOWFLAKE|SNOWFLAKE_SAMPLE_DATA)\b",
    re.IGNORECASE
)

# Optional: limit fan-out to reduce accidental Cartesian products
MAX_TABLE_REFERENCES = 6


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return re.sub(r"\s+", " ", sql).strip()


def _single_statement(sql: str) -> bool:
    """
    Allow 0 or 1 trailing semicolon but disallow multiple statements.
    """
    s = sql.strip()
    if s.count(";") == 0:
        return True
    if s.endswith(";") and s[:-1].count(";") == 0:
        return True
    return False


def _from_join_targets(sql: str):
    """
    Return list of raw tokens referenced after FROM / JOIN.
    We keep raw to also detect disallowed sources by schema/catalog prefix.
    """
    return re.findall(
        r"\b(?:FROM|JOIN)\s+((?:\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_\$]*)(?:\.(?:\"[^\"]+\"|[A-Za-z_][A-Za-z0-9_\$]*)){0,2})",
        sql,
        flags=re.IGNORECASE
    )


def _used_tables(sql: str):
    """
    Return set of base table names referenced in FROM / JOIN.
    Strips db/schema prefixes and quotes: DB.SCHEMA.TABLE -> TABLE
    """
    out = set()
    for c in _from_join_targets(sql):
        parts = [p.strip().strip('"') for p in c.split('.')]
        out.add(parts[-1])
    return out


def is_safe_sql_detail(sql: str, allowed_tables=None):
    """
    Returns (ok: bool, reason: str, debug: dict)
    - Ensures single statement
    - Ensures read-only (no DDL/DML/etc)
    - Ensures only allowed tables referenced (if provided)
    - Blocks metadata/system sources
    - Optionally caps number of table refs
    """
    if not sql or not isinstance(sql, str):
        return False, "empty_sql", {"normalized_sql": sql}

    raw = sql
    s = _strip_comments(sql)

    # must start with SELECT or WITH
    if not re.match(r"^\s*(SELECT|WITH)\b", s, flags=re.IGNORECASE):
        return False, "not_readonly_select", {"normalized_sql": s}

    # single statement
    if not _single_statement(raw):
        return False, "multiple_statements", {"normalized_sql": s}

    # forbidden keywords
    if FORBIDDEN_KEYWORDS.search(s):
        return False, "forbidden_keyword", {"normalized_sql": s}

    # Block metadata/system sources
    for target in _from_join_targets(s):
        if DISALLOWED_SOURCES.search(target):
            return False, "disallowed_source", {"target": target, "normalized_sql": s}

    # allowed tables enforcement
    if allowed_tables is not None:
        used = _used_tables(s)
        if used and not set(t.upper() for t in used).issubset(set(x.upper() for x in (allowed_tables or []))):
            return False, "disallowed_table", {"normalized_sql": s, "used_tables": sorted(used), "allowed": allowed_tables}

    # Cap the number of table references (best-effort safety)
    if len(_from_join_targets(s)) > MAX_TABLE_REFERENCES:
        return False, "too_many_tables", {"count": len(_from_join_targets(s)), "limit": MAX_TABLE_REFERENCES}

    return True, "ok", {"normalized_sql": s}


def enforce_limit(sql: str, max_rows: int = 1000) -> str:
    """
    Ensure a LIMIT exists and is <= max_rows.
    If a LIMIT exists and > max_rows, clamp it.
    Insert LIMIT before trailing semicolon (if any).
    """
    if not sql: return sql
    s = sql.strip()
    semi = ";" if s.endswith(";") else ""
    core = s[:-1] if semi else s

    # Find existing LIMIT
    m = re.search(r"\bLIMIT\s+(\d+)\b", core, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n > max_rows:
            core = core[:m.start(1)] + str(max_rows) + core[m.end(1):]
        return core + semi

    # Insert LIMIT at the end (before semicolon)
    return f"{core} LIMIT {int(max_rows)}{semi}"
