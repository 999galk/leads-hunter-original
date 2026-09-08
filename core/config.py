import os
import litellm
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# DEV_MODE=true  → Claude Haiku for Copywriter (cheaper, faster iteration)
# DEV_MODE=false → Claude Sonnet for Copywriter (submission-quality writing)
# Hunter / Qualifier / Evaluator use the same models in both modes.
# ---------------------------------------------------------------------------
DEV_MODE = os.getenv("DEV_MODE", "true").lower() == "true"

# Retries on transient LLM errors (429 rate limits, 500s, connection drops).
# LiteLLM applies these to every completion call made through it (which is how
# CrewAI routes all its model traffic). Exponential backoff between attempts,
# and the Retry-After header is honored when present.
# Set module-level (not per-LLM via additional_params, which gets forwarded
# into the provider SDK call and rejected as an unknown kwarg).
litellm.num_retries = 3

# NOTE: all agents now run on Anthropic (OpenAI key was exhausted). The SQL
# tools remain wrapped by SanitizedSQLTool as a safety net against any trailing
# garbage after valid SQL, regardless of which model composes the query.
#
# `temperature` is NOT passed to any Anthropic model here: the installed
# anthropic SDK (1.4.0) has dropped temperature/top_p/top_k from
# Messages.create() entirely — confirmed via inspect.signature(), no
# `temperature` param at all, for any model — so passing it raises
# "unexpected keyword argument 'temperature'" regardless of tier. There's also
# no replacement lever wired up: crewai 1.15's native Anthropic path
# (crewai.llms.providers.anthropic.completion.AnthropicCompletion) doesn't
# merge additional_params into the request, and its `thinking` field only
# supports the deprecated budget_tokens shape (which itself 400s on Sonnet 5),
# so `thinking` is left unset too — models run their default adaptive
# thinking. Revisit if crewai adds output_config.effort support.
_ANTHROPIC = os.environ["ANTHROPIC_API_KEY"]
if DEV_MODE:
    # Development: Sonnet 5 for Hunter/Qualifier (SQL composition + output
    # headroom). Evaluator uses Haiku 4.5 since it emits structured Pydantic
    # output and isn't exposed to SQL string composition.
    llm_hunter     = LLM(model="claude-sonnet-5",            api_key=_ANTHROPIC)
    llm_qualifier  = LLM(model="claude-sonnet-5",            api_key=_ANTHROPIC)
    llm_copywriter = LLM(model="claude-haiku-4-5-20251001",  api_key=_ANTHROPIC)
    llm_evaluator  = LLM(model="claude-haiku-4-5-20251001",  api_key=_ANTHROPIC)
else:
    # Production: quality models for submission
    # Hunter     → Sonnet 5 (room for 12 profiles + write_query INSERTs + HunterOutput in one turn)
    # Qualifier  → Sonnet 5 (accurate scoring, per-profile so no output-token pressure)
    # Copywriter → Sonnet 5 (best writing quality)
    # Evaluator  → Haiku 4.5 (structured scoring, doesn't need heavy model)
    llm_hunter     = LLM(model="claude-sonnet-5",            api_key=_ANTHROPIC)
    llm_qualifier  = LLM(model="claude-sonnet-5",            api_key=_ANTHROPIC)
    llm_copywriter = LLM(model="claude-sonnet-5",            api_key=_ANTHROPIC)
    llm_evaluator  = LLM(model="claude-haiku-4-5-20251001",  api_key=_ANTHROPIC)
