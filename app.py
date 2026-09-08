"""
Leads Hunter — Gradio UI
Run with: .venv/bin/python app.py
"""

import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

_state = {
    "status":            "idle",   # idle | running | done | error
    "log":               [],       # raw stdout lines (debug accordion)
    "events":            [],       # (type, ts, text) — Runtime tab
    "error":             None,
    "run_start":         None,
    "qualifier_output":  None,
    "copywriter_output": None,
    "current_agent":     None,
    "current_task":      None,
}

_MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Per-run timing state
# ---------------------------------------------------------------------------

_timing: dict = {
    "phases":        [],    # {"phase", "started_at", "duration_s"} — appended as each agent finishes
    "run_start_ts":  None,  # float (time.time())
    "run_end_ts":    None,
    "_phase_start":  None,  # temp: wall-clock start of the agent currently running
    "_phase_name":   None,  # temp: display name of the agent currently running
    "_prefix":       "",    # "" for main run, "Retry N" for retries
}

# ---------------------------------------------------------------------------
# Event system — drives the Runtime tab
# ---------------------------------------------------------------------------

_EVENT_COLORS = {
    "system":  "#94a3b8",   # slate
    "agent":   "#60a5fa",   # blue
    "task":    "#c4b5fd",   # violet
    "tool":    "#fb923c",   # orange
    "score":   "#a78bfa",   # purple
    "retry":   "#fbbf24",   # amber
    "success": "#4ade80",   # green
    "error":   "#f87171",   # red
    "timing":  "#38bdf8",   # sky
}

_EVENT_ICONS = {
    "system":  "⚙ ",
    "agent":   "🤖",
    "task":    "📋",
    "tool":    "🔧",
    "score":   "📊",
    "retry":   "⚠ ",
    "success": "✅",
    "error":   "❌",
    "timing":  "⏱ ",
}


def _add_event(event_type: str, text: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    _state["events"].append((event_type, ts, text))


def _format_runtime_html(events: list) -> str:
    """Render structured pipeline events as a dark terminal-style HTML block."""
    if not events:
        return (
            '<div id="runtime-log" style="'
            "height:580px;overflow-y:auto;font-family:'JetBrains Mono','Fira Mono',monospace;"
            "font-size:13px;line-height:1.8;background:#0f172a;color:#e2e8f0;"
            'padding:18px 22px;border-radius:10px">'
            '<span style="color:#475569;font-style:italic">'
            "Pipeline not started yet. Click  ▶ Run Process  on the Overview tab."
            "</span></div>"
        )

    rows = []
    for event_type, ts, text in events:
        color = _EVENT_COLORS.get(event_type, "#e2e8f0")
        icon  = _EVENT_ICONS.get(event_type, "  ")
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        indent = "padding-left:28px;" if text.startswith("  ↻") else ""
        rows.append(
            f'<div style="display:flex;gap:12px;padding:3px 0;'
            f'border-bottom:1px solid #1e293b;{indent}">'
            f'<span style="color:#334155;flex-shrink:0;font-size:11px;padding-top:1px">[{ts}]</span>'
            f'<span style="flex-shrink:0">{icon}</span>'
            f'<span style="color:{color}">{escaped}</span>'
            f"</div>"
        )

    body = "\n".join(rows)
    return (
        '<div id="runtime-log" style="'
        "height:580px;overflow-y:auto;font-family:'JetBrains Mono','Fira Mono',monospace;"
        "font-size:13px;line-height:1.7;background:#0f172a;color:#e2e8f0;"
        'padding:18px 22px;border-radius:10px">'
        + body
        + "</div>"
    )


def _format_full_log_html(lines: list) -> str:
    """Render raw stdout log as a compact dark HTML block (debug accordion)."""
    _LOG_COLORS = {
        "⚙": "#94a3b8", "❌": "#f87171",
        "🔧": "#fb923c", "📋": "#4ade80", "🤖": "#60a5fa",
    }
    parts = []
    for line in lines:
        first = line.lstrip()[:1]
        color = _LOG_COLORS.get(first, "#64748b")
        escaped = (
            line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        parts.append(f'<span style="color:{color}">{escaped}</span>')

    body = "<br>".join(parts) if parts else '<span style="color:#475569">No output yet.</span>'
    return (
        '<div id="full-log" style="'
        "height:380px;overflow-y:auto;font-family:'JetBrains Mono','Fira Mono',monospace;"
        "font-size:11px;line-height:1.5;background:#0f172a;color:#e2e8f0;"
        'padding:14px 16px;border-radius:8px;white-space:pre-wrap">'
        + body
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Smart stdout parser — populates _state["events"] from CrewAI output
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


_AGENT_ALIASES = {
    "linkedin lead hunter":       "LinkedIn Lead Hunter",
    "lead qualifier":             "Lead Qualifier",
    "gtm copywriter":             "GTM Copywriter",
    "outreach quality evaluator": "Outreach Quality Evaluator",
}

_AGENT_TASK_NAMES = {
    "LinkedIn Lead Hunter":       "Lead Discovery",
    "Lead Qualifier":             "Lead Qualification",
    "GTM Copywriter":             "Message Writing",
    "Outreach Quality Evaluator": "Message Evaluation",
}

# Panel state carried across lines — CrewAI emits Rich-style multi-line panels
_panel_state: dict = {
    "expect_tool":       False,   # inside a "Tool Execution Started" panel
    "tool_event_idx":    -1,      # events[] index of the tool event awaiting args
    "tool_name":         "",
    "args_buffer":       "",
    "pending_tool_args": {},      # tool_name → FIFO list of pre-parsed summaries
                                  # emitted via [TOOL_ARGS] marker from the tool
                                  # itself. Markers fire before Rich renders the
                                  # panel, so they must be queued and consumed
                                  # when the `Tool:` line is finally parsed.
}

# Pre-compiled patterns for parsing the Args: line of a tool panel
_SQL_OP_RE        = re.compile(r"\b(INSERT|UPDATE|SELECT|DELETE)\b", re.IGNORECASE)
_SQL_TABLE_RE     = re.compile(r"\b(?:INTO|UPDATE|FROM)\s+(\w+)", re.IGNORECASE)
_SQL_NAME_RE      = re.compile(r"name\s*=\s*'([^']+)'", re.IGNORECASE)
_SQL_VALUES_RE    = re.compile(r"VALUES\s*\(\s*'([^']+)'", re.IGNORECASE)
_SQL_MSG_TYPE_RE  = re.compile(r"'(linkedin_invite|followup_email)'")
_DATASTAX_CLIENT_RE = re.compile(r"DataStax client:\s*([^'\"]+?)['\"]")
_ARGS_KV_RE       = re.compile(r"['\"]?(\w+)['\"]?\s*:\s*['\"]([^'\"]+)['\"]")
_PROFILE_ID_RE    = re.compile(r"^p\d+$")


def _load_profile_id_to_name() -> dict[str, str]:
    """Map mock profile id (pNNN) → 'First Last', used to resolve lead_ids the
    Copywriter/Evaluator embed directly in write_query args."""
    path = os.path.join(os.path.dirname(__file__), "data", "mock_profiles.json")
    try:
        with open(path, encoding="utf-8") as f:
            profiles = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        p["id"]: f"{p.get('first_name','').strip()} {p.get('last_name','').strip()}".strip()
        for p in profiles if p.get("id")
    }


_PROFILE_ID_TO_NAME = _load_profile_id_to_name()


def _kv(key: str, blob: str) -> str:
    """Extract a single string-valued key from a Python-dict-shaped blob."""
    m = re.search(rf"['\"]?{key}['\"]?\s*:\s*['\"]([^'\"]+)['\"]", blob)
    return m.group(1) if m else ""


def _extract_tool_args_summary(tool_name: str, args_blob: str) -> str:
    """Return a concise human-readable summary of a tool call's args."""
    if not args_blob:
        return ""

    # --- SQL tools ----------------------------------------------------------
    # Include message_type when the target is the `messages` table so the
    # Copywriter's INSERT pair and the Evaluator's UPDATE pair are
    # distinguishable at a glance (linkedin_invite vs followup_email).
    if tool_name in ("write_query", "read_query"):
        op    = _SQL_OP_RE.search(args_blob)
        tbl   = _SQL_TABLE_RE.search(args_blob)
        nm    = _SQL_NAME_RE.search(args_blob)
        val   = _SQL_VALUES_RE.search(args_blob)
        mtype = _SQL_MSG_TYPE_RE.search(args_blob)
        entity = (nm.group(1) if nm else None) or (val.group(1) if val else None)
        # Copywriter/Evaluator embed the mock profile id (pNNN) as lead_id in
        # write_query VALUES — resolve it to the human-readable name for the UI.
        if entity and _PROFILE_ID_RE.match(entity):
            entity = _PROFILE_ID_TO_NAME.get(entity, entity)
        parts = []
        if op and tbl:
            parts.append(f"{op.group(1).upper()} {tbl.group(1)}")
        elif op:
            parts.append(op.group(1).upper())
        if entity:
            parts.append(f"'{entity}'")
        if mtype:
            parts.append(mtype.group(1))
        return " ".join(parts)

    # --- Copywriter RAG lookup ---------------------------------------------
    # Args always carry message_type + industry + seniority — show all three
    # so the two calls per lead (linkedin_invite, email) are distinguishable.
    if tool_name == "get_successful_templates":
        mt  = _kv("message_type", args_blob)
        ind = _kv("industry",     args_blob)
        sen = _kv("seniority",    args_blob)
        parts = [p for p in (mt, ind, sen) if p]
        return ", ".join(parts)

    # --- Qualifier scoring --------------------------------------------------
    # Args are a dict starting with `tier1_signals`: [list…], which CrewAI
    # truncates before reaching the identifying fields (seniority, industry).
    # Preference:
    #   1. "DataStax client: X" pattern inside tier1_signals — gives the company
    #   2. seniority + industry if visible before truncation
    #   3. Last fully-quoted signal — most specific string that survived
    #      truncation (typically a Certification or Experience line)
    if tool_name == "calculate_lead_score":
        dsc = _DATASTAX_CLIENT_RE.search(args_blob)
        if dsc:
            return dsc.group(1).strip()
        sen = _kv("seniority", args_blob)
        ind = _kv("industry",  args_blob)
        if ind and sen: return f"{ind}, {sen}"
        if sen:         return f"seniority='{sen}'"
        if ind:         return f"industry='{ind}'"
        # Last-ditch: the last '...' string with a prefixed label
        # (e.g., "Technology: X", "Certification: Y"). Truncated fragments
        # lack a closing quote so findall naturally skips them.
        signals = [s for s in re.findall(r"'([^']+)'", args_blob) if ":" in s]
        if signals:
            last = signals[-1]
            return last[:70] + ("…" if len(last) > 70 else "")
        return ""

    # --- Generic fallback ---------------------------------------------------
    for key in ("name", "lead_name", "company", "company_name", "query", "industry", "seniority"):
        v = _kv(key, args_blob)
        if v:
            return f"{key}='{v[:60]}…'" if len(v) > 60 else f"{key}='{v}'"

    m = _ARGS_KV_RE.search(args_blob)
    if m:
        v = m.group(2)
        return f"{m.group(1)}='{v[:60]}…'" if len(v) > 60 else f"{m.group(1)}='{v}'"

    return ""


def _finalize_tool_args() -> None:
    """Parse the accumulated args buffer and patch the last tool event."""
    idx = _panel_state["tool_event_idx"]
    buf = _panel_state["args_buffer"]
    _panel_state["tool_event_idx"] = -1
    _panel_state["args_buffer"]    = ""
    if idx < 0 or not buf or idx >= len(_state["events"]):
        return
    summary = _extract_tool_args_summary(_panel_state["tool_name"], buf)
    if summary:
        ev = _state["events"][idx]
        _state["events"][idx] = (ev[0], ev[1], f"{ev[2]}  ({summary})")


def _parse_line(raw_line: str) -> None:
    """
    Parse one ANSI-stripped line from CrewAI stdout and emit a structured event.

    CrewAI emits Rich-style multi-line panels like:
      ╭─── 🤖 Agent Started ───╮
      │  Agent: LinkedIn Lead Hunter │
      │  Task: <desc>                │
      ╰──────────────────────────────╯

    We detect panel headers (emoji markers in horizontal box lines) to set
    state, then interpret "Agent:" / "Tool:" content lines accordingly.
    The `Agent:` line reappears inside "Agent Final Answer" panels too — we
    dedupe by current_agent so the "started" event only fires on transition.
    """
    t  = _strip_ansi(raw_line).strip()
    tl = t.lower()

    if not t:
        return

    # --- [TOOL_ARGS] marker ---------------------------------------------------
    # Tools can print `[TOOL_ARGS] <tool_name> | k=v | k=v ...` to bypass
    # CrewAI's stdout truncation of the Args: panel line. Rich renders tool
    # panels asynchronously — markers arrive BEFORE the `Tool:` event is
    # emitted, and multiple markers can be concatenated onto one stdout write
    # (no newlines between them). We split them apart, queue each summary per
    # tool name, and consume one FIFO when the matching `Tool:` line is parsed.
    if "[TOOL_ARGS]" in t:
        # Markers sit at the start; panel box-drawing (if any) starts at `╭`.
        box_idx       = t.find("╭")
        markers_part  = t if box_idx == -1 else t[:box_idx]
        leftover      = "" if box_idx == -1 else t[box_idx:]

        for seg in markers_part.split("[TOOL_ARGS]"):
            seg = seg.strip()
            if not seg:
                continue
            parts = [p.strip() for p in seg.split("|") if p.strip()]
            if not parts:
                continue
            tool_name = parts[0]
            parsed = {}
            for p in parts[1:]:
                if "=" in p:
                    k, v = p.split("=", 1)
                    parsed[k.strip()] = v.strip()
            summary_bits = []
            if "industry"  in parsed: summary_bits.append(parsed["industry"])
            if "seniority" in parsed: summary_bits.append(parsed["seniority"])
            if "size"      in parsed: summary_bits.append(parsed["size"])
            if "tier1"     in parsed: summary_bits.append(f"T1={parsed['tier1']}")
            if "tier2"     in parsed: summary_bits.append(f"T2={parsed['tier2']}")
            summary = ", ".join(summary_bits)
            if summary:
                _panel_state["pending_tool_args"].setdefault(tool_name, []).append(summary)

        if not leftover:
            return
        # Fall through to normal parsing for the panel box-drawing tail.
        t  = leftover
        tl = t.lower()

    # --- Panel footer ---------------------------------------------------------
    # Any line beginning with `╰` closes the current Rich panel. If we were
    # buffering tool args, this is where we parse them into a summary.
    if t.startswith("╰"):
        _finalize_tool_args()
        return

    # --- Panel headers --------------------------------------------------------
    if "Tool Execution Started" in t:
        _panel_state["expect_tool"] = True
        return
    if "Tool Execution Completed" in t:
        _finalize_tool_args()
        _panel_state["expect_tool"] = False
        return

    # Task complete — either the Final Answer panel or the Task Completion panel
    if ("Agent Final Answer" in t) or ("Task Completion" in t):
        _finalize_tool_args()
        if _state.get("current_task"):
            task = _state["current_task"]
            _add_event("success", f"Task: {task} — complete")
            _state["current_task"] = None  # dedupe the sibling panel
            if _timing["_phase_start"] and _timing["_phase_name"]:
                duration = time.time() - _timing["_phase_start"]
                _timing["phases"].append({
                    "phase":      _timing["_phase_name"],
                    "started_at": datetime.fromtimestamp(_timing["_phase_start"], tz=timezone.utc).strftime("%H:%M:%S"),
                    "duration_s": round(duration, 1),
                })
                _add_event("timing", f"{_timing['_phase_name']} — {duration:.1f}s")
                _timing["_phase_start"] = None
                _timing["_phase_name"]  = None
        return

    # --- Panel content --------------------------------------------------------
    # Strip Rich box-drawing chars + spaces from both edges
    clean = t.strip("│╭╰─╮╯ ").strip()

    # "Agent: X" — new agent transition (deduped by current_agent)
    if clean.startswith("Agent:"):
        raw_name = clean[len("Agent:"):].split("│")[0].strip()
        canonical = _AGENT_ALIASES.get(raw_name.lower(), raw_name)
        if canonical and canonical != _state.get("current_agent"):
            _state["current_agent"] = canonical
            task_label = _AGENT_TASK_NAMES.get(canonical, "work")
            _state["current_task"] = task_label
            _add_event("agent", f"{canonical} — started")
            _add_event("task",  f"Task: {task_label} — started")
            phase_label = f"{_timing['_prefix']} — {canonical}" if _timing["_prefix"] else canonical
            _timing["_phase_start"] = time.time()
            _timing["_phase_name"]  = phase_label
        return

    # "Tool: X" — only when inside a Tool Execution Started panel
    if clean.startswith("Tool:") and _panel_state.get("expect_tool"):
        tool_name = clean[len("Tool:"):].split("│")[0].strip()
        if tool_name and len(tool_name) < 120:
            # If this tool already pushed a [TOOL_ARGS] marker, consume it (FIFO)
            # and emit the event fully-annotated — no further Args: parsing needed.
            queue   = _panel_state["pending_tool_args"].get(tool_name, [])
            summary = queue.pop(0) if queue else ""
            if summary:
                _add_event("tool", f"→ {tool_name}  ({summary})")
                _panel_state["tool_event_idx"] = -1   # skip Args: parsing for this panel
            else:
                _add_event("tool", f"→ {tool_name}")
                _panel_state["tool_event_idx"] = len(_state["events"]) - 1
            _panel_state["tool_name"]      = tool_name
            _panel_state["args_buffer"]    = ""
        _panel_state["expect_tool"] = False  # one emit per panel
        return

    # "Args: {...}" — start buffering. Long arg blobs wrap onto subsequent
    # lines inside the same panel, so keep appending until the footer `╰`
    # fires and `_finalize_tool_args()` patches the tool event with a summary.
    if clean.startswith("Args:") and _panel_state["tool_event_idx"] >= 0:
        _panel_state["args_buffer"] = clean[len("Args:"):].strip()
        return
    if _panel_state["tool_event_idx"] >= 0 and _panel_state["args_buffer"] and clean:
        _panel_state["args_buffer"] += " " + clean
        return

    # Errors (skip noise like "error handling", "no error")
    if any(kw in tl for kw in ("error", "exception", "traceback")):
        if not any(skip in tl for skip in ("no error", "without error", "error handling", "error_handling")):
            if len(t) < 300:
                _add_event("error", t[:200])


def _parse_for_events(text: str) -> None:
    """Split a (possibly multi-line) write() chunk and parse each line."""
    for raw_line in text.splitlines():
        _parse_line(raw_line)


def _classify_log_line(text: str) -> str:
    """Return an emoji prefix for the raw full-log."""
    t = text.lower()
    if "[ui]" in t:
        return "⚙  "
    if any(kw in t for kw in ("error", "exception", "traceback", "failed")):
        return "❌  "
    if any(kw in t for kw in ("action input:", "action:", "using tool")):
        return "🔧  "
    if "observation:" in t:
        return "📋  "
    if any(kw in t for kw in ("> entering", "> finished", "agent ")):
        return "🤖  "
    return "    "


class _LogCapture:
    """Redirect stdout into _state["log"] and parse events in real time."""

    def __init__(self, original):
        self._original = original
        debug_path = os.path.join(os.path.dirname(__file__), "output", "debug_run.log")
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        self._debug_file = open(debug_path, "w", encoding="utf-8")

    def write(self, text):
        if text.strip():
            prefix = _classify_log_line(text)
            _state["log"].append(prefix + text)
            _parse_for_events(text)
            try:
                self._debug_file.write(text)
                self._debug_file.flush()
            except Exception:
                pass
        self._original.write(text)

    def flush(self):
        self._original.flush()
        try:
            self._debug_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self._debug_file.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scoring helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _write_dry_run_entry(log_path: str, lead_name: str, message_type: str, content: str) -> None:
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 72
    entry = (
        f"\n{sep}\n"
        f"[DRY RUN]  {ts}\n"
        f"Lead:      {lead_name}\n"
        f"Type:      {message_type}\n"
        f"{sep}\n"
        f"{content}\n"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)


def _write_timing_report() -> None:
    """Write per-phase wall-clock timings to output/timing_report.json after each run."""
    start_ts = _timing.get("run_start_ts")
    end_ts   = _timing.get("run_end_ts")
    report = {
        "run_start":        datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if start_ts else None,
        "run_end":          datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if end_ts else None,
        "total_duration_s": round(end_ts - start_ts, 1) if (start_ts and end_ts) else None,
        "phases":           _timing["phases"],
    }
    out_path = os.path.join(os.path.dirname(__file__), "output", "timing_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _revalidate(msg):
    """Re-run EvaluatedMessage.compute_scores after mutating content fields.

    Pydantic model_validators only run on parse — direct attribute writes bypass
    them. Rebuilding via model_validate forces a fresh pass of compute_scores so
    dimension totals stay consistent with the features/content.
    """
    from core.models import EvaluatedMessage
    return EvaluatedMessage.model_validate(msg.model_dump())


def _enforce_threshold(output) -> None:
    """Totals + statuses already come out of compute_scores; here we just
    tally the output-level approved/rejected counters."""
    if not output:
        return
    output.approved = sum(
        1 for m in output.messages
        if m.linkedin_status == "APPROVED" and m.email_status == "APPROVED"
    )
    output.rejected = len(output.messages) - output.approved


def _truncate_linkedin_invites(output) -> None:
    """Trim invites over the 300-char hard limit at a word boundary, then
    re-run the validator so linkedin_char_count / linkedin_format / total /
    status reflect the trimmed text instead of the original."""
    if not output:
        return
    for idx, msg in enumerate(output.messages):
        if len(msg.linkedin_invite) > 300:
            trimmed = msg.linkedin_invite[:300].rsplit(" ", 1)[0].rstrip(".,;:")
            _state["log"].append(
                f"⚙   [UI] truncated LinkedIn invite for {msg.lead_name}: "
                f"{len(msg.linkedin_invite)} → {len(trimmed)} chars\n"
            )
            msg.linkedin_invite = trimmed
            output.messages[idx] = _revalidate(msg)


# ---------------------------------------------------------------------------
# Python DB safety net — fills any rows the agents missed writing via MCP
# ---------------------------------------------------------------------------

def _python_db_sync_leads(qualifier_output) -> None:
    """Upsert qualified leads (status, score, qualification_notes) to the DB.

    Split out from the post-crew sync so it can run right after the Qualifier
    task finishes — the Leads tab then reflects scores + notes while later
    agents are still executing.
    """
    from database import get_connection

    if not qualifier_output:
        return

    with get_connection() as conn:
        for lead in qualifier_output.leads:
            signals_json = ",".join(lead.tier1_signals + lead.tier2_signals)
            conn.execute(
                """
                INSERT INTO leads
                  (name, title, company, linkedin_url, industry, seniority,
                   signals, discovery_source, status, score, qualification_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'linkedin_mock', ?, ?, ?)
                ON CONFLICT(name, company) DO UPDATE SET
                  status=excluded.status,
                  score=excluded.score,
                  qualification_notes=excluded.qualification_notes
                """,
                (lead.name, lead.title, lead.company, lead.linkedin_url,
                 lead.industry, lead.seniority, signals_json,
                 lead.status.lower(), lead.score, lead.qualification_notes),
            )


def _python_db_sync_messages(qualifier_output, copywriter_output, eval_output) -> None:
    """Upsert message content + eval scores to the DB. Runs after Evaluator."""
    from database import get_connection

    if not qualifier_output or not eval_output:
        return

    approach_map = {}
    if copywriter_output:
        for m in copywriter_output.messages:
            approach_map[m.lead_name] = m.approach

    with get_connection() as conn:
        lead_id_map = {
            lead.name: conn.execute(
                "SELECT id FROM leads WHERE name=? AND company=?",
                (lead.name, lead.company)
            ).fetchone()
            for lead in qualifier_output.leads
        }

        for msg in eval_output.messages:
            row = lead_id_map.get(msg.lead_name)
            if not row:
                continue
            lead_id  = row["id"]
            approach = approach_map.get(msg.lead_name, "")
            email_content = f"Subject: {msg.email_subject}\n\n{msg.email_body}"

            conn.execute(
                """
                INSERT INTO messages
                  (lead_id, type, content, approach, eval_score, eval_notes,
                   eval_status, dry_run)
                VALUES (?, 'linkedin_invite', ?, ?, ?, ?, ?, 1)
                ON CONFLICT(lead_id, type) DO UPDATE SET
                  content=excluded.content,
                  approach=excluded.approach,
                  eval_score=excluded.eval_score,
                  eval_notes=excluded.eval_notes,
                  eval_status=excluded.eval_status
                """,
                (lead_id, msg.linkedin_invite, approach,
                 msg.linkedin_score, msg.linkedin_notes, msg.linkedin_status),
            )
            conn.execute(
                """
                INSERT INTO messages
                  (lead_id, type, content, approach, eval_score, eval_notes,
                   eval_status, dry_run)
                VALUES (?, 'followup_email', ?, ?, ?, ?, ?, 1)
                ON CONFLICT(lead_id, type) DO UPDATE SET
                  content=excluded.content,
                  approach=excluded.approach,
                  eval_score=excluded.eval_score,
                  eval_notes=excluded.eval_notes,
                  eval_status=excluded.eval_status
                """,
                (lead_id, email_content, approach,
                 msg.email_score, msg.email_notes, msg.email_status),
            )


def _python_db_sync_messages_content(qualifier_output, copywriter_output) -> None:
    """Upsert message content + approach (no eval fields). Runs right after Copywriter.

    Split from the full messages sync so the Messages tab can show actual drafts
    while Evaluator is still running. Eval columns stay NULL until _python_db_sync_messages
    fills them in after the Evaluator finishes.
    """
    from database import get_connection

    if not qualifier_output or not copywriter_output:
        return

    with get_connection() as conn:
        lead_id_map = {
            lead.name: conn.execute(
                "SELECT id FROM leads WHERE name=? AND company=?",
                (lead.name, lead.company)
            ).fetchone()
            for lead in qualifier_output.leads
        }

        for msg in copywriter_output.messages:
            row = lead_id_map.get(msg.lead_name)
            if not row:
                continue
            lead_id = row["id"]
            email_content = f"Subject: {msg.email_subject}\n\n{msg.email_body}"

            conn.execute(
                """
                INSERT INTO messages (lead_id, type, content, approach, dry_run)
                VALUES (?, 'linkedin_invite', ?, ?, 1)
                ON CONFLICT(lead_id, type) DO UPDATE SET
                  content=excluded.content,
                  approach=excluded.approach
                """,
                (lead_id, msg.linkedin_invite, msg.approach),
            )
            conn.execute(
                """
                INSERT INTO messages (lead_id, type, content, approach, dry_run)
                VALUES (?, 'followup_email', ?, ?, 1)
                ON CONFLICT(lead_id, type) DO UPDATE SET
                  content=excluded.content,
                  approach=excluded.approach
                """,
                (lead_id, email_content, msg.approach),
            )


def _python_db_sync(qualifier_output, copywriter_output, eval_output) -> None:
    """Convenience wrapper — runs both leads and messages syncs.

    Kept for the final end-of-run call; the leads slice is also called
    separately from a task callback right after the Qualifier finishes.
    """
    _python_db_sync_leads(qualifier_output)
    _python_db_sync_messages(qualifier_output, copywriter_output, eval_output)


def _on_qualifier_task_done(task_output) -> None:
    """CrewAI task callback fired when the Qualifier task completes.

    Writes qualification data to the DB immediately so the Leads tab shows
    scores + notes mid-run rather than waiting for the full pipeline.
    """
    try:
        qo = task_output.pydantic
    except AttributeError:
        qo = None
    if not qo:
        return
    _state["qualifier_output"] = qo
    _python_db_sync_leads(qo)
    _add_event(
        "success",
        f"Lead Qualifier — complete "
        f"({qo.qualified} qualified, {qo.blocked} blocked, {qo.skipped} skipped)",
    )
    _add_event("system", "Leads DB sync complete — scores visible in Leads tab")


def _on_copywriter_task_done(task_output) -> None:
    """CrewAI task callback fired when the Copywriter task completes.

    Writes message content + approach to the DB immediately so the Messages tab
    shows the actual drafts while the Evaluator is still running.
    """
    try:
        co = task_output.pydantic
    except AttributeError:
        co = None
    if not co or not hasattr(co, "messages"):
        return
    _state["copywriter_output"] = co
    _python_db_sync_messages_content(_state.get("qualifier_output"), co)
    _add_event(
        "success",
        f"GTM Copywriter — complete "
        f"({len(co.messages)} message pairs written)",
    )
    _add_event("system", "Messages DB sync complete — drafts visible in Messages tab")


# ---------------------------------------------------------------------------
# Pipeline runner (background thread, no human pause)
# ---------------------------------------------------------------------------

def _run_pipeline():
    global _state
    _state.update(
        status="running", log=[], events=[], error=None,
        run_start=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        qualifier_output=None,
        copywriter_output=None,
        current_agent=None,
        current_task=None,
    )
    _timing.update(
        phases=[], run_start_ts=time.time(), run_end_ts=None,
        _phase_start=None, _phase_name=None, _prefix="",
    )
    _panel_state["expect_tool"]       = False
    _panel_state["tool_event_idx"]    = -1
    _panel_state["tool_name"]         = ""
    _panel_state["args_buffer"]       = ""
    _panel_state["pending_tool_args"] = {}

    original_stdout = sys.stdout
    sys.stdout = _LogCapture(original_stdout)

    _add_event("system", "Pipeline initialising — 4 agents, 2 MCP servers (LinkedIn mock + SQLite)")

    try:
        from database import init_db
        from core.rag import init_rag
        from core.crew import build_crew

        _log_path = os.path.join(os.path.dirname(__file__), "output", "dry_run.log")
        if os.path.exists(_log_path):
            os.remove(_log_path)

        init_db()
        from database import clear_run_data
        clear_run_data()
        init_rag()

        linkedin_adapter, sqlite_adapter, crew = build_crew()

        from crewai import Crew, Process, Task as CrewTask
        from core.models import CopywriterOutput, EvaluatorOutput as EvalModel

        # Mid-run DB sync callbacks: Leads tab populates after Qualifier finishes,
        # Messages tab populates after Copywriter finishes — rather than waiting for Evaluator.
        crew.tasks[1].callback = _on_qualifier_task_done
        crew.tasks[2].callback = _on_copywriter_task_done

        with linkedin_adapter, sqlite_adapter:
            result = crew.kickoff()

            # Per-agent completion summaries (Hunter, Qualifier, Copywriter)
            try:
                hunter_output = result.tasks_output[0].pydantic
                if hunter_output and hasattr(hunter_output, "profiles"):
                    _add_event(
                        "success",
                        f"LinkedIn Lead Hunter — complete "
                        f"({len(hunter_output.profiles)} profiles discovered)",
                    )
            except Exception:
                pass

            # Fallback: if the task callback didn't fire (shouldn't happen), set
            # state + emit the success event here so the rest of the pipeline still works.
            try:
                if not _state.get("qualifier_output"):
                    _state["qualifier_output"] = result.tasks_output[1].pydantic
                    qo = _state["qualifier_output"]
                    if qo:
                        _add_event(
                            "success",
                            f"Lead Qualifier — complete "
                            f"({qo.qualified} qualified, {qo.blocked} blocked, {qo.skipped} skipped)",
                        )
            except Exception:
                pass

            # Fallback: if the copywriter task callback didn't fire, set state here.
            try:
                if not _state.get("copywriter_output"):
                    _state["copywriter_output"] = result.tasks_output[2].pydantic
                    co = _state["copywriter_output"]
                    if co and hasattr(co, "messages"):
                        _add_event(
                            "success",
                            f"GTM Copywriter — complete "
                            f"({len(co.messages)} message pairs written)",
                        )
            except Exception:
                pass

            eval_output = result.pydantic
            if eval_output is None:
                try:
                    eval_output = result.tasks_output[3].pydantic
                except Exception:
                    pass

            _enforce_threshold(eval_output)

            # Emit per-lead evaluator scores
            if eval_output:
                _add_event("agent", "Outreach Quality Evaluator — scores")
                for msg in eval_output.messages:
                    li_icon  = "✅" if msg.linkedin_status == "APPROVED" else "❌"
                    em_icon  = "✅" if msg.email_status   == "APPROVED" else "❌"
                    outcome  = (
                        "APPROVED"
                        if msg.linkedin_status == "APPROVED" and msg.email_status == "APPROVED"
                        else "NEEDS RETRY"
                    )
                    _add_event(
                        "score",
                        f"{msg.lead_name:<22}  LinkedIn: {msg.linkedin_score:>3}/100 {li_icon}   "
                        f"Email: {msg.email_score:>3}/100 {em_icon}   [{outcome}]",
                    )
                approved_now = sum(
                    1 for m in eval_output.messages
                    if m.linkedin_status == "APPROVED" and m.email_status == "APPROVED"
                )
                _add_event(
                    "success",
                    f"Outreach Quality Evaluator — complete "
                    f"({approved_now}/{len(eval_output.messages)} approved)",
                )

            # ------------------------------------------------------------------
            # Automatic retry loop — no human pause
            # ------------------------------------------------------------------
            sender_name  = os.getenv("SENDER_NAME", "Your Name")
            sender_title = os.getenv("SENDER_TITLE", "Solutions Architect")

            for retry_num in range(_MAX_RETRIES):
                if not eval_output:
                    break

                missing_leads = []
                qo = _state.get("qualifier_output")
                if qo and hasattr(qo, "leads"):
                    messaged_names = {m.lead_name for m in eval_output.messages}
                    for lead in qo.leads:
                        if lead.status == "QUALIFIED" and lead.name not in messaged_names:
                            missing_leads.append(lead)

                rejected = [
                    m for m in eval_output.messages
                    if m.linkedin_status == "REJECTED" or m.email_status == "REJECTED"
                ]

                if not rejected and not missing_leads:
                    break

                _add_event(
                    "retry",
                    f"Retry {retry_num + 1}/{_MAX_RETRIES} — "
                    f"{len(rejected)} rejected, {len(missing_leads)} missing — rewriting",
                )
                _state["log"].append(
                    f"\n⚙   [UI] retry {retry_num + 1}/{_MAX_RETRIES} — "
                    f"{len(rejected)} rejected, {len(missing_leads)} missing\n"
                )

                lead_by_name  = (
                    {l.name: l for l in qo.leads}
                    if qo and hasattr(qo, "leads") else {}
                )
                feedback_lines = []
                for m in rejected:
                    lead   = lead_by_name.get(m.lead_name)
                    t1     = "; ".join(lead.tier1_signals) if lead and lead.tier1_signals else "n/a"
                    t2     = "; ".join(lead.tier2_signals) if lead and lead.tier2_signals else "n/a"
                    co     = f" @ {lead.company} ({lead.seniority}, {lead.industry})" if lead else ""
                    li_len = len(m.linkedin_invite)
                    entry  = (
                        f"Lead: {m.lead_name}{co}\n"
                        f"  Tier-1 signals: {t1}\n"
                        f"  Tier-2 signals: {t2}\n"
                        f"  Previous LinkedIn invite ({li_len} chars — "
                        f"{'OVER LIMIT' if li_len > 300 else 'within limit'}): "
                        f"{m.linkedin_invite}\n"
                        f"  Previous email subject: {m.email_subject}"
                    )
                    if m.linkedin_status == "REJECTED":
                        entry += f"\n  LINKEDIN REJECTED (score {m.linkedin_score}/100): {m.linkedin_notes}"
                    if m.email_status == "REJECTED":
                        entry += f"\n  EMAIL REJECTED (score {m.email_score}/100): {m.email_notes}"
                    feedback_lines.append(entry)

                for lead in missing_leads:
                    t1 = "; ".join(lead.tier1_signals) if lead.tier1_signals else "unknown"
                    t2 = "; ".join(lead.tier2_signals) if lead.tier2_signals else "unknown"
                    feedback_lines.append(
                        f"Lead: {lead.name} @ {lead.company} "
                        f"({lead.seniority}, {lead.industry})\n"
                        f"  Tier-1 signals: {t1}\n"
                        f"  Tier-2 signals: {t2}\n"
                        f"  Issue: No message was generated — write fresh messages."
                    )

                feedback_text = "\n\n".join(feedback_lines)
                n_needed      = len(rejected) + len(missing_leads)

                retry_copy_task = CrewTask(
                    description=(
                        f"Write or rewrite message sets for these {n_needed} lead(s).\n"
                        f"Evaluator feedback to address:\n\n{feedback_text}\n\n"
                        "LINKEDIN INVITE RULES (must all pass to score ≥70):\n"
                        "  1. Hard limit: ≤300 characters including spaces.\n"
                        "  2. Name their specific role or company.\n"
                        "  3. Reference a concrete signal (Cassandra/DataStax tech, cert, or industry).\n"
                        "  4. Hint at a relevant topic (IBM acquisition, exploring alternatives) — "
                        "     do NOT name ScyllaDB.\n"
                        "  5. Warm, human tone — no buzzwords.\n"
                        "  6. End with a soft connection ask.\n"
                        "  Target 150-280 chars — a good example: "
                        "'Hi [Name], noticed your Cassandra work at [Co] — "
                        "with the IBM acquisition shaking things up I've been chatting with "
                        "engineers exploring options. Would love to connect.'\n\n"
                        "EMAIL RULES: 100-200 words, subject line, name ScyllaDB explicitly, "
                        "connect DataStax pain to ScyllaDB value. "
                        f"Sign: {sender_name}, {sender_title}, ScyllaDB."
                    ),
                    expected_output=(
                        f"A CopywriterOutput with exactly {n_needed} LeadMessages entries "
                        "— one per lead listed above."
                    ),
                    output_pydantic=CopywriterOutput,
                    agent=crew.agents[2],
                )
                retry_eval_task = CrewTask(
                    description=(
                        "Re-evaluate the rewritten messages by EXTRACTING FEATURES into "
                        "the EvaluatedMessage schema. Follow the feature definitions in "
                        "your backstory exactly.\n\n"
                        "FOR EACH REWRITTEN MESSAGE SET:\n"
                        "  1. Copy lead_id, lead_name, linkedin_invite, email_subject, "
                        "     email_body VERBATIM from the copywriter output.\n"
                        "  2. Read the invite character by character and fill every li_* "
                        "     feature field by literal observation.\n"
                        "  3. Read the email body and fill every em_* feature field.\n"
                        "  4. Cross-reference the lead's tier-1 / tier-2 signals against "
                        "     each message; list only signals that appear verbatim in the "
                        "     text in li_tier1_refs / em_tier1_refs / li_tier2_refs / "
                        "     em_tier2_refs.\n"
                        "  5. Fill linkedin_notes / email_notes with a one-liner each "
                        "     (char/word count + strongest signal).\n\n"
                        "DO NOT output dimension scores, totals, statuses, or char/word "
                        "counts — Python computes all of those from your features. Leave "
                        "those fields as defaults.\n\n"
                        "Do NOT call any SQL tools — Python handles DB writes."
                    ),
                    expected_output=(
                        "An EvaluatorOutput with one EvaluatedMessage per rewritten lead. "
                        "Each EvaluatedMessage contains lead_id, lead_name, the three "
                        "content fields verbatim, and every li_* / em_* feature field "
                        "filled by literal observation. Dimension, total, status, and "
                        "count fields remain at their defaults."
                    ),
                    output_pydantic=EvalModel,
                    agent=crew.agents[3],
                    context=[retry_copy_task],
                )
                retry_crew = Crew(
                    agents=crew.agents,
                    tasks=[retry_copy_task, retry_eval_task],
                    process=Process.sequential,
                    verbose=True,
                )
                _timing["_prefix"] = f"Retry {retry_num + 1}"
                retry_result = retry_crew.kickoff()

                new_output = retry_result.pydantic
                if new_output and new_output.messages:
                    replaced_names = (
                        {m.lead_name for m in rejected}
                        | {l.name for l in missing_leads}
                    )
                    eval_output.messages = (
                        [m for m in eval_output.messages if m.lead_name not in replaced_names]
                        + new_output.messages
                    )
                    _enforce_threshold(eval_output)

                    # Emit updated scores after retry
                    for msg in new_output.messages:
                        li_icon = "✅" if msg.linkedin_status == "APPROVED" else "❌"
                        em_icon = "✅" if msg.email_status   == "APPROVED" else "❌"
                        outcome = (
                            "APPROVED"
                            if msg.linkedin_status == "APPROVED" and msg.email_status == "APPROVED"
                            else "STILL NEEDS WORK"
                        )
                        _add_event(
                            "score",
                            f"  ↻ {msg.lead_name:<20}  LinkedIn: {msg.linkedin_score:>3}/100 {li_icon}   "
                            f"Email: {msg.email_score:>3}/100 {em_icon}   [{outcome}]",
                        )

            _truncate_linkedin_invites(eval_output)
            _enforce_threshold(eval_output)

            # Python safety net: ensure everything is in the DB even if an agent
            # skipped a write_query call. Uses upsert so agent writes are never overwritten.
            _python_db_sync(_state.get("qualifier_output"), _state.get("copywriter_output"), eval_output)
            _add_event("system", "DB sync complete")

            # Write dry_run.log from final (post-retry) eval output (Python handles file writes).
            if eval_output:
                for _msg in eval_output.messages:
                    if _msg.linkedin_status == "APPROVED":
                        _write_dry_run_entry(
                            _log_path, _msg.lead_name,
                            "linkedin_invite", _msg.linkedin_invite,
                        )
                    if _msg.email_status == "APPROVED":
                        _write_dry_run_entry(
                            _log_path, _msg.lead_name,
                            "followup_email",
                            f"Subject: {_msg.email_subject}\n\n{_msg.email_body}",
                        )

        # Final summary event
        if eval_output:
            approved = sum(
                1 for m in eval_output.messages
                if m.linkedin_status == "APPROVED" and m.email_status == "APPROVED"
            )
            total  = len(eval_output.messages)
            qo     = _state.get("qualifier_output")
            b_cnt  = qo.blocked  if qo else "?"
            sk_cnt = qo.skipped  if qo else "?"
            _add_event(
                "success",
                f"Pipeline complete — {approved}/{total} messages approved  |  "
                f"{b_cnt} blocked  |  {sk_cnt} skipped",
            )
        else:
            _add_event("success", "Pipeline complete")

        _state["status"] = "done"

    except Exception as e:
        import traceback
        msg = str(e)
        # Classify rate-limit errors so the UI can show a recoverable message
        # instead of a stack-trace dump. LiteLLM has already retried 3× with
        # backoff before this exception bubbles up, so a 429 here means the
        # TPM budget is genuinely saturated — user action required.
        is_rate_limit = ("429" in msg) or ("rate limit" in msg.lower()) or ("rate_limit" in msg.lower())
        if is_rate_limit:
            _state["error"]  = (
                "Rate limit hit after 3 retries — your OpenAI/Anthropic tokens-per-minute "
                "budget is saturated. Wait ~10 seconds and click Run Process again."
            )
            _add_event("error", "Rate limit — retries exhausted. Wait 10s and re-run.")
        else:
            _state["error"]  = msg
            _add_event("error", f"Pipeline error: {msg[:200]}")
        _state["status"] = "error"
        _state["log"].append(f"\n❌   {traceback.format_exc()}\n")
    finally:
        _timing["run_end_ts"] = time.time()
        try:
            _write_timing_report()
        except Exception:
            pass
        try:
            sys.stdout.close()
        except Exception:
            pass
        sys.stdout = original_stdout


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _status_text() -> str:
    return {
        "idle":    "⚪ Ready",
        "running": "🔄 Running… — see Runtime tab for live logs",
        "done":    "✅ Complete",
        "error":   f"❌ {_state['error']}",
    }.get(_state["status"], _state["status"])


def start_pipeline():
    if _state["status"] == "running":
        return "🔄 Already running"
    t = threading.Thread(target=_run_pipeline, daemon=True)
    t.start()
    return "🔄 Running… — see Runtime tab for live logs"


def refresh_runtime():
    html  = _format_runtime_html(_state["events"])
    full  = _format_full_log_html(_state["log"])
    return html, full, _status_text()


# ---------------------------------------------------------------------------
# Leads tab (auto-refresh)
# ---------------------------------------------------------------------------

def load_leads():
    # While running: serve cached QualifierOutput so table populates before DB sync completes
    if _state["status"] == "running" and _state.get("qualifier_output"):
        qo = _state["qualifier_output"]
        if qo and qo.leads:
            by_status = {}
            for lead in qo.leads:
                by_status[lead.status] = by_status.get(lead.status, 0) + 1
            stats = (
                f"**Discovered:** {len(qo.leads)}  |  "
                f"**Qualified:** {by_status.get('QUALIFIED', 0)}  |  "
                f"**Blocked:** {by_status.get('BLOCKED', 0)}  |  "
                f"**Skipped:** {by_status.get('SKIPPED', 0)}"
                "  *(live — pipeline running)*"
            )
            table = [
                [l.name, l.company, l.industry or "—", l.seniority or "—",
                 l.status, l.score, l.qualification_notes or "—"]
                for l in qo.leads
            ]
            return stats, table

    # No run yet this session — show nothing
    run_start = _state.get("run_start")
    if not run_start:
        return "Run the pipeline to see leads.", []

    # Read from DB filtered to current run
    try:
        from database import get_connection, init_db
        init_db()
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT name, company, industry, seniority, status, score, "
                "qualification_notes FROM leads ORDER BY score DESC NULLS LAST"
            ).fetchall()
            counts = conn.execute(
                "SELECT status, COUNT(*) n FROM leads GROUP BY status"
            ).fetchall()

        if not rows:
            return "No leads yet — run the pipeline first.", []

        total     = sum(r["n"] for r in counts)
        by_status = {r["status"]: r["n"] for r in counts}
        stats = (
            f"**Discovered:** {total}  |  "
            f"**Qualified:** {by_status.get('qualified', 0)}  |  "
            f"**Blocked:** {by_status.get('blocked', 0)}  |  "
            f"**Skipped:** {by_status.get('skipped', 0)}"
        )
        table = [
            [r["name"], r["company"], r["industry"] or "—",
             r["seniority"] or "—", r["status"], r["score"] or "—",
             r["qualification_notes"] or "—"]
            for r in rows
        ]
        return stats, table

    except Exception as e:
        return f"Error: {e}", []


# ---------------------------------------------------------------------------
# Messages tab (auto-refresh)
# ---------------------------------------------------------------------------

def load_all_messages():
    run_start = _state.get("run_start")
    if not run_start:
        return [], []

    try:
        from database import get_connection, init_db
        init_db()
        with get_connection() as conn:
            rows = conn.execute("""
                SELECT m.id, l.name, l.company, m.type,
                       m.eval_status, m.eval_score, m.approach,
                       substr(m.content, 1, 200) AS preview,
                       m.created_at
                FROM messages m
                JOIN leads l ON l.id = m.lead_id
                ORDER BY m.created_at DESC
            """).fetchall()

        if not rows:
            return [], []

        table = [
            [r["name"], r["company"], r["type"],
             r["eval_status"] or "—", r["eval_score"] or "—",
             r["approach"] or "—",
             r["preview"] or "—",
             r["created_at"]]
            for r in rows
        ]
        return table, [r["id"] for r in rows]
    except Exception as e:
        return [[f"Error: {e}", "", "", "", "", "", "", ""]], []


def show_full_message(evt: gr.SelectData, message_ids):
    """Look up the full, untruncated content for the clicked Messages row."""
    if not message_ids or evt.index[0] >= len(message_ids):
        return "Select a row above to view the full message."

    message_id = message_ids[evt.index[0]]
    try:
        from database import get_connection, init_db
        init_db()
        with get_connection() as conn:
            row = conn.execute(
                "SELECT content FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
        return row["content"] if row else "Message not found."
    except Exception as e:
        return f"Error: {e}"



# ---------------------------------------------------------------------------
# Self-Improvement tab
# ---------------------------------------------------------------------------

def load_rag_stats():
    try:
        from core.rag import init_rag, _get_collection
        init_rag()
        col   = _get_collection()
        total = col.count()
        if total == 0:
            return "ChromaDB is empty — seed data not yet loaded.", []
        results     = col.get(include=["metadatas"])
        by_approach = {}
        by_response = {}
        for meta in results["metadatas"]:
            a = meta.get("approach", "unknown")
            r = meta.get("response_type", "unknown")
            by_approach[a] = by_approach.get(a, 0) + 1
            by_response[r] = by_response.get(r, 0) + 1
        stats  = f"**Total templates in ChromaDB: {total}**\n\n"
        stats += "**By approach:** " + "  |  ".join(f"{k}: {v}" for k, v in by_approach.items()) + "\n\n"
        stats += "**By response:** " + "  |  ".join(f"{k}: {v}" for k, v in by_response.items())
        rows = [
            [m.get("approach", "—"), m.get("message_type", "—"),
             m.get("industry", "—"), m.get("seniority", "—"),
             m.get("response_type", "—"), m.get("eval_score", "—")]
            for m in results["metadatas"]
        ]
        return stats, rows
    except Exception as e:
        return f"Error: {e}", []




# ---------------------------------------------------------------------------
# Landing page content
# ---------------------------------------------------------------------------

_LANDING_MD = """
## What This Does

An end-to-end **GTM automation pipeline** that finds DataStax users on LinkedIn, qualifies them as
ScyllaDB prospects, writes personalised outreach messages, evaluates them against quality thresholds,
and logs a complete dry run — **no real messages are ever sent**.

**Expected run time: ~4–5 minutes** (~270 s end-to-end for 12 mock profiles through all four agents,
plus the Python safety-net upserts and dry-run logging). Progress streams into the **Runtime** tab
and the Leads / Messages tabs fill progressively as each agent finishes.

**Fresh run every time.** Because the POC works off a fixed mock profile set
(`data/mock_profiles.json`), each run clears the `leads` and `messages` tables before
kickoff — so what you see in the Leads / Messages tabs is always the current run, never
an accumulation. `message_feedback` and the ChromaDB RAG store are preserved across
runs so the self-improvement loop keeps learning.

---

## Framework: CrewAI

Sequential 4-agent pipeline with typed inter-agent handoffs via Pydantic (`output_pydantic`).

**Why CrewAI over alternatives:**
- **First-class agent personas** — essential for the Copywriter's tone and consistency
- `output_pydantic` enforces typed data contracts between agents out of the box
- **Multi-provider LLM** support (OpenAI + Anthropic) via a single config flag
- Sequential process maps cleanly to this pipeline's linear flow

---

## Pipeline Architecture

```
Hunter  →  Qualifier  →  Copywriter  →  Evaluator
```

| Agent | Role | Tools | MCP Server | LLM |
|-------|------|-------|------------|-----|
| GTM Lead Hunter | Discovers DataStax users from mock LinkedIn | `search_linkedin_profiles` — query mock LinkedIn via MCP<br>`write_query` — sqlite INSERT via MCP | LinkedIn mock (stdio, custom) + SQLite (uvx) | GPT-4.1 |
| GTM Qualifier | Scores leads 0–100, classifies QUALIFIED / SKIPPED / BLOCKED | `validate_lead` — DataStax/IBM employer guardrail<br>`calculate_lead_score` — deterministic 0–100 base score<br>`get_company_profile` — known-client Tier-1 lookup<br>`write_query` — sqlite UPDATE via MCP | SQLite (uvx) | GPT-4o |
| GTM Copywriter | Writes LinkedIn invite + email per lead | `get_successful_templates` — ChromaDB RAG few-shot retrieval<br>`search_recent_news` — mock news search for recency hook<br>`write_query` / `read_query` — sqlite I/O via MCP | SQLite (uvx) | Claude Haiku |
| GTM Evaluator | Extracts message features (booleans/lists/counts); Python `@model_validator` computes 4 dimensions × 2 channels | `write_query` / `read_query` — sqlite I/O via MCP | SQLite (uvx) | GPT-4o-mini |

**A note on DB writes:** Each agent's prompt instructs it to call `write_query` as the last step of its task, but LLMs occasionally skip it (tool-call budget, latency). `_python_db_sync` in `app.py` runs after the crew finishes and upserts any missing rows/fields via parameterized queries. This is deliberate — the MCP path demonstrates distributed per-agent writes; the Python safety net guarantees data integrity for the POC. Apostrophe-prone fields (`qualification_notes`, `eval_notes`, message content) are intentionally excluded from MCP writes and handled only by the Python path, since LLM-generated raw SQL breaks on single quotes.

**Databases:**
- `data/leads.db` — SQLite · tables: `leads`, `messages`, `message_feedback`
- `data/chroma_db/` — ChromaDB · embeddings of high-performing past messages (sentence-transformers)

---

## Lead Discovery Query

The Hunter calls `search_linkedin_profiles(query)` on the LinkedIn MCP server. When no query is
passed, the server falls back to the default DataStax keyword set:

```
cassandra, apache cassandra,
datastax, datastax enterprise, dse,
astra db, astradb,
cql,
migrat          # tier-2 signal — admits weak leads whose only hit is a
                # "migration" mention in recent_posts; the Qualifier then
                # scores them < 60 and marks them SKIPPED
```

**Two-layer filtering:**
- **Layer 1 (MCP server):** case-insensitive keyword match across each profile's
  `technologies`, `employment_history`, `recent_posts`, and `certifications`.
  Also strips out DataStax / IBM / IBM-DataStax employees at source.
- **Layer 2 (Hunter agent):** fine-grained signal identification — decides which
  signals are Tier-1 (strong: tech stack, employer is a known DataStax client,
  DataStax certifications) vs Tier-2 (weak: post mentions, migration chatter).

Signals are **pre-computed deterministically** inside the MCP server (not LLM work)
and passed through to the Hunter ready-to-use — this keeps scoring factual rather
than inferential.

---

## Project Layout

```
leads-hunter/
├── app.py                  # Gradio UI + pipeline orchestrator (primary entry)
├── main.py                 # Headless CLI runner
├── database.py             # SQLite schema + migrations
├── ingest_feedback.py      # CRM-webhook stand-in — SQLite → ChromaDB
├── requirements.txt, .env*
│
├── core/                   # Shared pipeline modules (imported, not run directly)
│   ├── config.py           #   LLM routing (DEV_MODE vs submission)
│   ├── crew.py             #   Crew assembly + MCP adapter lifecycle
│   ├── tasks.py            #   CrewAI Task definitions + expected outputs
│   ├── models.py           #   Pydantic contracts between agents
│   └── rag.py              #   ChromaDB management (add / query / stats)
│
├── agents/                 # Agent definitions (hunter, qualifier, copywriter, evaluator)
├── tools/                  # @tool functions + SanitizedSQLTool wrapper
├── servers/                # Custom MCP server for mock LinkedIn
├── data/                   # SQLite DB, ChromaDB store, mock profiles, seed templates
└── output/                 # Dry-run log (gitignored, cleared each run)
```

**Design rule:** the four runnable scripts stay in the root so the documented commands
(`.venv/bin/python app.py`, `main.py`, `ingest_feedback.py`, `database.py`) remain unchanged.
Everything else that is only *imported* lives under `core/` or the existing topic folders.

---

## Self-Improvement Loop

Before writing messages, the **Copywriter retrieves the 3 most similar approved past messages**
from ChromaDB (matched by industry + seniority) as few-shot examples.
The store grows as responses come in — simulated via the **Self-Improvement** tab.

Only `replied` and `accepted` responses are ingested into ChromaDB.
Negative feedback is recorded in SQLite only (for audit, not learning).

**Bootstrapping the store (POC):** On the first run, `init_rag()` seeds ChromaDB from
`data/seed_templates.json` (8 mock historical messages spanning different approaches,
industries, and seniorities) so the Copywriter has non-empty few-shot context from message #1.
After the initial seed the collection persists; subsequent runs re-use it and grow it.

**In production:** seed data would be replaced by a **CRM webhook → `ingest_feedback.py`** flow.
When a prospect replies or accepts a connection in Salesforce / HubSpot / LinkedIn Sales Navigator,
the CRM fires a webhook at an internal endpoint; the handler writes a row into `message_feedback`
and calls `rag.add_message()` to embed the winning message into ChromaDB.
The same gating rule applies — only `replied` / `accepted` are ingested; negative outcomes stay
in SQL for audit only.

---

## Mock Data

LinkedIn profiles are **Apollo.io-shaped JSON** (`data/mock_profiles.json`).
`_load_profiles()` in the MCP server has a comment showing the exact 5-line swap to go live
with a real Apollo API key. The `recent_posts` field is mock-only — in production it would
come from a secondary Proxycurl `/person/posts` call.

Simulated lead responses (replied / accepted / ignored / rejected) are entered via the
Self-Improvement tab. In production: CRM webhook → `ingest_feedback.py`.

---

## Challenges Overcome

| Bug | Root cause | Fix |
|-----|-----------|-----|
| Evaluator scoring | LLM wrote a single dimension score (e.g. 24/25) into the total field | Split `EvaluatedMessage` into 4 explicit dimension fields; `@model_validator` computes totals in Python — never trust LLM arithmetic |
| Retry context gap | Copywriter rewrites had no lead profile data, producing generic messages | Retry task now passes full tier-1/tier-2 signals, rejected message text, char count, and per-channel scores |
| LinkedIn 300-char limit | LLM regularly exceeded the platform hard limit | Mechanical word-boundary truncation after retries; `linkedin_format` dimension restored to 20 so scoring is fair |
| Status case mismatch | Agents inserted UPPERCASE status values; UI queries were lowercase | All queries now use `LOWER(status)` / `UPPER(eval_status)` |
| SQL in agent prompts | Agents need to know the DB schema to write correct queries | **Intentional design choice:** SQL templates are hardcoded in each task's prompt. Schema discovery via `list_tables`/`describe_table` was too slow. Trade-off: prompts must be updated manually if the schema changes. |
"""

# ---------------------------------------------------------------------------
# Extensions tab content
# ---------------------------------------------------------------------------

_EXTENSIONS_MD = """
## Beyond the Assignment

These are features I would build for a real pilot deployment.
I built the foundation for all of these in this POC but scoped them out
to keep the submission focused.

---

### 1. Human-in-the-Loop Review *(Pilot Phase)*

Before messages go to the dry-run log, a GTM team member reviews and approves each one.
The UI would allow:
- **Message approval / rejection** per lead with optional comment
- **Lead exclusions** — uncheck a lead to remove them from the outbox
- **Query string editing** — adjust the LinkedIn search terms without touching code

*Why it matters:* The first 50 outreach attempts in a real pilot should always have a human
in the loop. Trust is built incrementally — automation earns its autonomy.

---

### 2. Self-Improving Lead Qualification

Currently the self-improvement loop applies only to **messages** (via ChromaDB).
The same mechanism should apply to **lead qualification**:
track which signals in `tier1_signals` and `tier2_signals` actually correlate with
replied/accepted responses, and feed that back into `calculate_lead_score.py`.

*Foundation already built:* `message_feedback` table and `ingest_feedback.py` already capture
response types per message. Extending this to leads is additive.

---

### 3. Two-Phase Outreach *(LinkedIn → Email)*

Currently the pipeline generates both a LinkedIn invite and a follow-up email for each lead in a single pass.
In a real outreach workflow, the email would only be sent **after** the LinkedIn connection is accepted — the invite comes first, and the follow-up email is triggered by acceptance.

For this dry-run POC, both messages are generated upfront so the full message pair can be evaluated in one pipeline run. In production, this would require a CRM webhook to detect acceptance before the email step is triggered.

---

### 4. Parallel Lead Qualification *(Batch `validate_lead`)*

Currently `validate_lead` is called once per lead, sequentially — the Qualifier processes each lead one at a time.
In production with 100+ leads per run, this creates unnecessary latency: N leads means N blocking tool calls before any scoring begins.

A batch variant of `validate_lead` could accept a list of `(name, company)` pairs, run all DataStax/IBM employee checks in a single pass, and return statuses for the whole batch.
Combined with parallelising `get_company_profile` and `calculate_lead_score` per lead, this could reduce Qualifier wall time by 50–70%.

*Foundation already built:* `validate_lead` is a synchronous single-call `@tool` in `tools/validate_lead.py`.
A batch version is an additive change — the existing single-call path can stay as a fallback.

---

### 5. Keyword Management UI

Today, LinkedIn search keywords are hardcoded in `servers/linkedin_mock_server.py`.
In production an internal GTM team page would allow:
- Add / remove / toggle search keywords
- Tag keywords as Tier 1 or Tier 2 signals
- Preview how keyword changes affect the current mock lead pool before going live

*Why it matters:* The GTM team understands the market better than the engineer who shipped
the pipeline. Keyword control should live with them, not in a pull request.
"""

# ---------------------------------------------------------------------------
# JS — auto-scroll all log divs on content change
# ---------------------------------------------------------------------------

_JS = """
() => {
    function setupScroll() {
        ['runtime-log', 'full-log'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                new MutationObserver(() => { el.scrollTop = el.scrollHeight; })
                    .observe(el, { childList: true, subtree: true, characterData: true });
            }
        });
        const missing = !document.getElementById('runtime-log');
        if (missing) { setTimeout(setupScroll, 600); }
    }
    setupScroll();
    return [];
}
"""

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
.gradio-container { font-family: 'Inter', sans-serif !important; }
.prose h2 { border-bottom: 2px solid #e5e7eb; padding-bottom: 0.35em; margin-top: 1.6em; }
.prose h3 { margin-top: 1.2em; color: #374151; }
.prose table thead th { background: #f1f5f9 !important; font-weight: 600 !important; }
.prose code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px; }
table thead th { background: #f1f5f9 !important; font-weight: 600 !important; }
"""

# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(title="Leads Hunter — ScyllaDB GTM Pipeline", css=_CSS, js=_JS) as demo:

        with gr.Tabs():

            # ----------------------------------------------------------------
            # Tab 1: Overview (Landing Page)
            # ----------------------------------------------------------------
            with gr.Tab("🏠 Overview"):
                gr.Markdown("# 🎯 Leads Hunter — ScyllaDB GTM Pipeline")
                gr.Markdown("*Dry run — no messages are ever sent*")

                with gr.Row():
                    run_btn    = gr.Button("▶  Run Process", variant="primary", scale=1)
                    status_box = gr.Textbox(
                        label="Status", value="⚪ Ready",
                        interactive=False, scale=3,
                    )

                gr.Markdown(
                    "*After clicking Run, switch to the **Runtime** tab to watch the pipeline live.*",
                    container=False,
                )

                gr.Markdown(_LANDING_MD)

            # ----------------------------------------------------------------
            # Tab 2: Runtime
            # ----------------------------------------------------------------
            with gr.Tab("⚡ Runtime"):
                gr.Markdown("### Live Pipeline Events")
                gr.Markdown(
                    "Each line represents one meaningful event — agent transitions, "
                    "tool calls with key arguments, evaluator scores, and retry loops. "
                    "Streams live as the pipeline runs.",
                    container=False,
                )
                runtime_log = gr.HTML(value=_format_runtime_html([]))

                with gr.Accordion("Full debug log (raw CrewAI output)", open=False):
                    full_log_box = gr.HTML(value=_format_full_log_html([]))

            # ----------------------------------------------------------------
            # Tab 3: Leads
            # ----------------------------------------------------------------
            with gr.Tab("👥 Leads"):
                gr.Markdown(
                    "Auto-updates while the pipeline runs. "
                    "Shows live in-memory data during qualification, then switches to the DB.",
                    container=False,
                )
                leads_stats = gr.Markdown("Run the pipeline to see leads.")
                leads_table = gr.Dataframe(
                    headers=["Name", "Company", "Industry", "Seniority",
                             "Status", "Score", "Notes"],
                    datatype=["str", "str", "str", "str", "str", "number", "str"],
                    column_widths=["120px", "120px", "120px", "100px",
                                    "100px", "80px", "auto"],
                    interactive=False,
                    wrap=True,
                )

            # ----------------------------------------------------------------
            # Tab 4: Messages
            # ----------------------------------------------------------------
            with gr.Tab("✉ Messages"):
                gr.Markdown(
                    "LinkedIn invite + follow-up email per lead. Drafts appear as soon as the "
                    "Copywriter finishes; evaluator scores fill in after the Evaluator runs. "
                    "Auto-updates every 3 s.",
                    container=False,
                )
                messages_table = gr.Dataframe(
                    headers=["Lead", "Company", "Type", "Eval Status", "Eval Score",
                             "Approach", "Preview", "Created"],
                    datatype=["str", "str", "str", "str", "number",
                              "str", "str", "str"],
                    column_widths=["120px", "120px", "130px", "110px", "90px",
                                    "150px", "auto", "120px"],
                    interactive=False,
                    wrap=True,
                )
                messages_ids_state = gr.State([])
                full_message_box = gr.Textbox(
                    label="Full message (click a row above)",
                    lines=10,
                    interactive=False,
                )

            # ----------------------------------------------------------------
            # Tab 6: Self-Improvement
            # ----------------------------------------------------------------
            with gr.Tab("🔄 Self-Improvement"):
                gr.Markdown(
                    "### ChromaDB — Successful Message Templates\n"
                    "Before writing, the Copywriter retrieves the top-3 most similar "
                    "approved past messages as few-shot examples. "
                    "Only `replied` and `accepted` responses are ingested — negative "
                    "feedback is recorded in SQLite only. Auto-updates every 5 s."
                )
                rag_stats = gr.Markdown("Store stats will appear automatically after the first run.")
                rag_table = gr.Dataframe(
                    headers=["Approach", "Type", "Industry", "Seniority", "Response", "Score"],
                    interactive=False,
                )

            # ----------------------------------------------------------------
            # Tab 7: Extensions
            # ----------------------------------------------------------------
            with gr.Tab("🚀 Extensions"):
                gr.Markdown(_EXTENSIONS_MD)

        # ----------------------------------------------------------------
        # Timers — all tabs auto-update, no manual refresh buttons
        # ----------------------------------------------------------------
        log_timer  = gr.Timer(0.3)
        data_timer = gr.Timer(5)

        # Run button
        run_btn.click(start_pipeline, outputs=[status_box])

        # Runtime + status — live (every 0.3 s)
        log_timer.tick(
            refresh_runtime,
            outputs=[runtime_log, full_log_box, status_box],
        )

        # Leads, Messages, RAG stats — every 5 s
        data_timer.tick(load_leads,        outputs=[leads_stats, leads_table])
        data_timer.tick(load_all_messages, outputs=[messages_table, messages_ids_state])
        data_timer.tick(load_rag_stats,    outputs=[rag_stats, rag_table])

        messages_table.select(
            show_full_message,
            inputs=[messages_ids_state],
            outputs=[full_message_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, theme=gr.themes.Soft())
