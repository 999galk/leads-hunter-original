import os
from crewai import Crew, Process
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv
from core.config import llm_hunter, llm_qualifier, llm_copywriter, llm_evaluator
from core.rag import init_rag
from tools.sanitize_sql import wrap_sqlite_tools

load_dotenv()

_ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON          = os.path.join(_ROOT, ".venv", "bin", "python")
_LINKEDIN_SERVER = os.path.join(_ROOT, "servers", "linkedin_mock_server.py")
_DB_PATH         = os.path.join(_ROOT, "data", "leads.db")


def build_crew() -> tuple[MCPServerAdapter, MCPServerAdapter, Crew]:
    """
    Instantiate all agents, build tasks, and return both MCP adapters + the Crew.

    Caller must manage both adapter lifecycles:

        linkedin_adapter, sqlite_adapter, crew = build_crew()
        with linkedin_adapter, sqlite_adapter:
            result = crew.kickoff()

    Two MCP servers:
      - LinkedIn mock (stdio, custom) — Hunter only for profile search
      - mcp-server-sqlite (uvx, community) — all agents for DB persistence
        Hunter inserts discovered leads, Qualifier updates qualification data,
        Copywriter inserts message drafts, Evaluator updates eval scores.
    """
    init_rag()

    from agents.hunter import create_hunter_agent
    from agents.qualifier import create_qualifier_agent
    from agents.copywriter import create_copywriter_agent
    from agents.evaluator import create_evaluator_agent

    # -- MCP server: LinkedIn mock (Hunter only for profile search) ------------
    linkedin_adapter = MCPServerAdapter(StdioServerParameters(
        command=_PYTHON,
        args=[_LINKEDIN_SERVER],
        env={**os.environ},
    ))

    # -- MCP server: sqlite (external community server via uvx) ----------------
    # Tools are wrapped in SanitizedSQLTool which retries once with a cleaned
    # prefix when the LLM hallucinates trailing JSON/f-string garbage after
    # valid SQL (only under strict certainty — see tools/sanitize_sql.py).
    # NOTE: mcp-server-sqlite (2025.4.25) still uses the removed @server.list_resources()
    # decorator, which crashes under the mcp 2.x SDK that uvx resolves by default.
    # Pin mcp to a 1.x that still exposes it so uvx builds a compatible env.
    sqlite_adapter = MCPServerAdapter(StdioServerParameters(
        command="uvx",
        args=["--from", "mcp-server-sqlite", "--with", "mcp==1.9.4",
              "mcp-server-sqlite", "--db-path", _DB_PATH],
        env={**os.environ},
    ))

    sqlite_tools = wrap_sqlite_tools(sqlite_adapter.tools)

    # -- Agents ----------------------------------------------------------------
    agents = {
        "hunter":     create_hunter_agent(llm=llm_hunter, linkedin_tools=linkedin_adapter.tools, sqlite_tools=sqlite_tools),
        "qualifier":  create_qualifier_agent(llm=llm_qualifier, sqlite_tools=sqlite_tools),
        "copywriter": create_copywriter_agent(llm=llm_copywriter, sqlite_tools=sqlite_tools),
        "evaluator":  create_evaluator_agent(llm=llm_evaluator, sqlite_tools=sqlite_tools),
    }

    from core.tasks import create_tasks
    tasks = create_tasks(agents)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    return linkedin_adapter, sqlite_adapter, crew
