# agent_eval.py
# Evaluates a LangGraph-based agent on Accuracy, Efficiency, Reliability, and User outcomes.
# Works in two modes:
#  - use_mocks=True: stubs UI/DB/LLM calls so evaluations run quickly and deterministically.
#  - use_mocks=False: runs your real graph; ensure your environment has required services enabled.
#
# Metrics definitions follow graph/agent evaluation best practices:
# - Accuracy: end-to-end task success and optional emotion label checks. [web:10][web:9]
# - Efficiency: latency and step count per run. [web:4]
# - Reliability: schema/JSON validity and action execution success. [web:4]
# - User outcomes: acceptance/click-through proxies from interaction stubs or production signals. [web:4]
#
# To use:
#   1) Set AGENT_MODULE to your module path that exposes create_workflow() or run_agent_system(...).
#   2) Run: python agent_eval.py
#
# Notes:
# - If your graph returns a pydantic state, this script will extract attributes safely.
# - If it returns a dict, this script will use keys directly.

import time
import statistics
import traceback
import importlib
import random
from types import SimpleNamespace
from contextlib import contextmanager

# ----------------------------
# Configuration
# ----------------------------
AGENT_MODULE = "your_agent_module_name"  # e.g., "emofi_agent"
USE_MOCKS = True                         # Set False to exercise the real system
SEED = 42                                # Repro for mock paths and choices
RUNS_PER_CASE = 3                        # Repetitions per test case to stabilize results

# Acceptance and opening probabilities in mock mode
MOCK_ACCEPT_PROB = 0.9      # Probability that user accepts recommendations (interrupt)
MOCK_CLICK_PROB = 0.95      # Probability that user clicks a recommendation
MOCK_OPEN_PROB = 0.98       # Probability that an app opens successfully

# ----------------------------
# Test dataset
# ----------------------------
# Each case defines:
# - emotions: list[str] observed by your pipeline
# - expect_negative: whether the agent should treat this as a "negative" emotion path
# - expected_avg: optional expected majority label for the average_emotion node
TEST_CASES = [
    {
        "name": "negative_angry",
        "emotions": ["Angry", "Angry", "Sad"],
        "expect_negative": True,
        "expected_avg": "Angry",
    },
    {
        "name": "positive_happy",
        "emotions": ["Happy", "Happy", "Surprise"],
        "expect_negative": False,
        "expected_avg": "Happy",
    },
    {
        "name": "neutral_mix",
        "emotions": ["Neutral", "Neutral", "Happy"],
        "expect_negative": False,
        "expected_avg": "Neutral",
    },
]

# ----------------------------
# Utilities
# ----------------------------
def _get_attr(state, key, default=None):
    if state is None:
        return default
    # dict-like
    if isinstance(state, dict):
        return state.get(key, default)
    # pydantic or object
    return getattr(state, key, default)

def _normalize_rec_options(rec_options):
    """
    Normalizes recommendation_options, which may be:
      - list[list[AppRecommendation]] (pydantic models)
      - list[list[dict]]
      - None
    Returns: list of list of dicts with keys app_name, app_url, search_query, is_local
    """
    out = []
    if not rec_options:
        return out
    for pair in rec_options:
        row = []
        for opt in pair:
            if isinstance(opt, dict):
                row.append({
                    "app_name": opt.get("app_name"),
                    "app_url": opt.get("app_url"),
                    "search_query": opt.get("search_query", ""),
                    "is_local": bool(opt.get("is_local", False)),
                })
            else:
                # pydantic model style
                row.append({
                    "app_name": getattr(opt, "app_name", None),
                    "app_url": getattr(opt, "app_url", None),
                    "search_query": getattr(opt, "search_query", ""),
                    "is_local": bool(getattr(opt, "is_local", False)),
                })
        out.append(row)
    return out

def _check_four_words(s):
    if not isinstance(s, str):
        return False
    words = [w for w in s.strip().split() if w]
    return len(words) == 4

def _check_web_url(url):
    return isinstance(url, str) and url.startswith("https://")

def _validate_recommendations(recs, rec_opts):
    """
    Reliability validation for the recommendation node outputs:
    - Must have exactly 3 recommendations.
    - Each recommendation must be exactly 4 words.
    - Each recommendation must have exactly 2 options.
    - No duplicate app_name across all options.
    - Local apps: search_query=="".
    - Web apps: url starts with https:// and contains <search_query>.
    Returns: (is_valid: bool, violations: list[str])
    """
    violations = []
    if not isinstance(recs, list) or len(recs) != 3:
        violations.append("recommendation_count_mismatch")
    if not isinstance(rec_opts, list) or len(rec_opts) != 3:
        violations.append("recommendation_options_count_mismatch")

    # Early exit if structure off
    if violations:
        return False, violations

    all_apps = set()
    for i in range(3):
        if not _check_four_words(recs[i]):
            violations.append(f"recommendation_{i}_not_four_words")
        if not isinstance(rec_opts[i], list) or len(rec_opts[i]) != 2:
            violations.append(f"recommendation_{i}_options_count_mismatch")
            continue
        for j, opt in enumerate(rec_opts[i]):
            app_name = opt.get("app_name")
            app_url = opt.get("app_url")
            search_query = opt.get("search_query", "")
            is_local = opt.get("is_local", False)

            if not app_name:
                violations.append(f"rec_{i}_opt_{j}_missing_app_name")
            if not app_url:
                violations.append(f"rec_{i}_opt_{j}_missing_app_url")

            # Duplicate app check (within and across)
            if app_name:
                key = app_name.lower()
                if key in all_apps:
                    violations.append(f"duplicate_app_{app_name}")
                all_apps.add(key)

            if is_local:
                if search_query not in ("", None):
                    violations.append(f"rec_{i}_opt_{j}_local_search_query_not_empty")
            else:
                if not _check_web_url(app_url):
                    violations.append(f"rec_{i}_opt_{j}_web_url_invalid")
                if "<search_query>" not in app_url:
                    violations.append(f"rec_{i}_opt_{j}_missing_search_placeholder")

    return len(violations) == 0, violations

# ----------------------------
# Monkeypatch helpers
# ----------------------------
@contextmanager
def patched(module, name, new):
    old = getattr(module, name)
    setattr(module, name, new)
    try:
        yield
    finally:
        setattr(module, name, old)

@contextmanager
def maybe_patch(module, name, new, enable=True):
    if not enable:
        yield
    else:
        with patched(module, name, new):
            yield

# ----------------------------
# Mocks
# ----------------------------
def _mock_send_notification(title, recommendations, recommendation_options):
    # Simulate user click-through
    if random.random() > MOCK_CLICK_PROB:
        return None  # no selection
    # Choose first option deterministically
    # Return the exact structure expected by open_recommendations(chosen_recommendation)
    # Use the first recommendation’s first option
    try:
        opt = recommendation_options[0][0]
    except Exception:
        return None
    return opt

class _DummyWebHandle:
    @property
    def current_url(self):
        return "https://example.com"

def _mock_open_recommendations(chosen_recommendation):
    # Simulate success/failure opening local/web
    if random.random() > MOCK_OPEN_PROB:
        return False
    is_local = bool(chosen_recommendation.get("is_local", False))
    if is_local:
        return True, 12345, "local"
    return True, _DummyWebHandle(), "web"

def _mock_get_connection():
    return SimpleNamespace(name="FAKE_DB")

def _mock_get_apps(_conn):
    # Provide a few local apps to satisfy “prefer local” logic
    return [
        "media | VLC | vlc | C:\\Program Files\\VideoLAN\\VLC\\vlc.exe",
        "editor | Notepad++ | npp | C:\\Program Files\\Notepad++\\notepad++.exe",
        "browser | Chrome | chrome | C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    ]

def _mock_add_agent_recommendations(*args, **kwargs):
    # No-op: record not persisted in mock mode
    return True

def _mock_wait_for_close_agent(state):
    # Immediately stop waiting to avoid timeouts in evaluation
    return {
        "continue_waiting": False,
        "open_app_handle": None,
        "app_type": None
    }

def _mock_recommendation_agent(state):
    # Deterministic, schema-valid triad of recommendations
    recs = [
        "Take a quick break",
        "Listen to relaxing music",
        "Play a short game",
    ]
    opts = [
        [
            {"app_name": "VLC", "app_url": r"C:\Program Files\VideoLAN\VLC\vlc.exe", "search_query": "", "is_local": True},
            {"app_name": "YouTube", "app_url": "https://www.youtube.com/results?search_query=<search_query>", "search_query": "lofi beats", "is_local": False},
        ],
        [
            {"app_name": "Spotify", "app_url": "https://open.spotify.com/search/<search_query>", "search_query": "calm piano", "is_local": False},
            {"app_name": "Notepad++", "app_url": r"C:\Program Files\Notepad++\notepad++.exe", "search_query": "", "is_local": True},
        ],
        [
            {"app_name": "Poki", "app_url": "https://poki.com/en/search/<search_query>", "search_query": "arcade", "is_local": False},
            {"app_name": "Chrome", "app_url": r"C:\Program Files\Google\Chrome\Application\chrome.exe", "search_query": "", "is_local": True},
        ],
    ]
    state.recommendation = recs
    state.recommendation_options = opts
    return {"recommendation": recs, "recommendation_options": opts}

# ----------------------------
# Step counting wrapper
# ----------------------------
def _wrap_node_for_steps(module, name, step_counter):
    original = getattr(module, name)

    def wrapper(state):
        step_counter["steps"] += 1
        return original(state)
    return wrapper

# ----------------------------
# Evaluator
# ----------------------------
def run_one(agent_mod, emotions, use_mocks):
    """
    Runs a single end-to-end invocation, returning:
      dict(success, accepted, clicked, opened, latency, steps, reliability_pass, reliability_violations, avg_emotion_ok)
    """
    random.seed(SEED + hash(tuple(emotions)) % 10_000_000)
    step_counter = {"steps": 0}

    # Decide which entrypoint is available
    run_fn = getattr(agent_mod, "run_agent_system", None)
    compiled_graph = None
    if run_fn is None:
        # Try to compile the graph
        create_graph = getattr(agent_mod, "create_workflow", None)
        if not create_graph:
            raise RuntimeError("Neither run_agent_system nor create_workflow found in agent module.")
        compiled_graph = create_graph()

    # Patch points
    # - Count steps on key nodes
    nodes_to_wrap = [
        "average_emotion_agent",
        "interrupt_check_agent",
        "recommendation_agent",
        "task_execution_agent",
        "wait_for_close_agent",
        "task_exit_agent",
    ]
    wrappers = {}
    for n in nodes_to_wrap:
        if hasattr(agent_mod, n):
            wrappers[n] = _wrap_node_for_steps(agent_mod, n, step_counter)

    # Track user outcomes via mocks
    accepted = None
    clicked = None
    opened = None

    def _mock_interrupt_check_agent(state):
        # Mirror original behavior: if negative emotion, simulate user response
        negative = ["Angry", "Sad", "Fear", "Disgust", "Stress", "Boring"]
        cont = False
        if getattr(state, "average_emotion", None) in negative:
            # Acceptance draws from MOCK_ACCEPT_PROB
            accept = random.random() < MOCK_ACCEPT_PROB
            cont = bool(accept)
        nonlocal accepted
        accepted = cont
        return {"continue_workflow": cont}

    def _mock_task_execution_agent(state):
        # Simulate selection and opening outcome
        nonlocal clicked, opened
        recs = getattr(state, "recommendation", [])
        opts = getattr(state, "recommendation_options", [])
        if not recs or "No action needed" in recs:
            return {"executed": False}

        # Simulate click
        if random.random() >= MOCK_CLICK_PROB:
            clicked = False
            return {"executed": False}
        clicked = True

        # Choose first option and simulate open
        chosen = None
        try:
            chosen = _normalize_rec_options(opts)[0][0]
        except Exception:
            pass
        if not chosen:
            return {"executed": False}

        if random.random() >= MOCK_OPEN_PROB:
            opened = False
            return {"executed": False}
        opened = True

        # Return open handle info to flow into wait_for_close
        if chosen.get("is_local", False):
            return {
                "executed": True,
                "open_app_handle": 9999,
                "app_type": "local",
                "continue_waiting": True,
                "wait_start_time": time.time()
            }
        else:
            return {
                "executed": True,
                "open_app_handle": _DummyWebHandle(),
                "app_type": "web",
                "continue_waiting": True,
                "wait_start_time": time.time()
            }

    patches = []
    # Step wrappers
    for n, fn in wrappers.items():
        patches.append(patched(agent_mod, n, fn))

    # Optional mocks
    if use_mocks:
        if hasattr(agent_mod, "send_notification"):
            patches.append(patched(agent_mod, "send_notification", _mock_send_notification))
        if hasattr(agent_mod, "open_recommendations"):
            patches.append(patched(agent_mod, "open_recommendations", _mock_open_recommendations))
        if hasattr(agent_mod, "get_connection"):
            patches.append(patched(agent_mod, "get_connection", _mock_get_connection))
        if hasattr(agent_mod, "get_apps"):
            patches.append(patched(agent_mod, "get_apps", _mock_get_apps))
        if hasattr(agent_mod, "add_agent_recommendations"):
            patches.append(patched(agent_mod, "add_agent_recommendations", _mock_add_agent_recommendations))
        # Avoid long waiting loops
        if hasattr(agent_mod, "wait_for_close_agent"):
            patches.append(patched(agent_mod, "wait_for_close_agent", _mock_wait_for_close_agent))
        # Deterministic recommendation generator
        if hasattr(agent_mod, "recommendation_agent"):
            patches.append(patched(agent_mod, "recommendation_agent", _mock_recommendation_agent))
        # Deterministic interrupt acceptance
        if hasattr(agent_mod, "interrupt_check_agent"):
            patches.append(patched(agent_mod, "interrupt_check_agent", _mock_interrupt_check_agent))
        # Deterministic task execution
        if hasattr(agent_mod, "task_execution_agent"):
            patches.append(patched(agent_mod, "task_execution_agent", _mock_task_execution_agent))

    # Run the flow
    t0 = time.time()
    try:
        with _nested(*patches):
            if run_fn:
                final_state = run_fn(emotions)
            else:
                # Build initial state manually if needed
                AgentState = getattr(agent_mod, "AgentState", None)
                if AgentState:
                    initial_state = AgentState(
                        emotions=emotions,
                        average_emotion=None,
                        continue_workflow=None,
                        recommendation=None,
                        recommendation_options=[],
                        executed=False,
                        action_executed=None,
                        action_time_start=0.0
                    )
                else:
                    initial_state = {"emotions": emotions}
                final_state = compiled_graph.invoke(initial_state, config={"recursion_limit": 100})
    except Exception:
        latency = time.time() - t0
        return {
            "success": False,
            "accepted": bool(accepted),
            "clicked": bool(clicked),
            "opened": bool(opened),
            "latency": latency,
            "steps": step_counter["steps"],
            "reliability_pass": False,
            "reliability_violations": ["exception_during_run"],
            "avg_emotion_ok": False,
            "exception": traceback.format_exc(limit=5),
        }

    latency = time.time() - t0

    # Extract fields
    executed = bool(_get_attr(final_state, "executed", False))
    recommendation = _get_attr(final_state, "recommendation", [])
    recommendation_options = _get_attr(final_state, "recommendation_options", [])
    avg_emotion = _get_attr(final_state, "average_emotion", None)

    # Reliability checks on recommendations
    norm_opts = _normalize_rec_options(recommendation_options)
    reliability_pass, violations = _validate_recommendations(recommendation, norm_opts)

    # End-to-end success (opened app is a strong proxy for task success)
    # If mocks disabled, fall back to executed=True as success proxy
    if USE_MOCKS:
        success = bool(opened)
    else:
        success = bool(executed)

    return {
        "success": success,
        "accepted": bool(accepted),
        "clicked": bool(clicked),
        "opened": bool(opened),
        "latency": latency,
        "steps": step_counter["steps"],
        "reliability_pass": reliability_pass,
        "reliability_violations": violations,
        "avg_emotion": avg_emotion,
        "recommendation": recommendation,
        "recommendation_options": norm_opts,
    }

@contextmanager
def _nested(*cms):
    if not cms:
        yield
        return
    with cms[0]:
        with _nested(*cms[1:]):
            yield

def aggregate(results, expected_avg=None, expect_negative=None):
    """
    Aggregates metrics over multiple runs for one test case.
    """
    # Accuracy-like signals
    successes = [r["success"] for r in results]
    success_rate = sum(1 for s in successes if s) / max(1, len(successes))

    avg_ok = None
    if expected_avg is not None:
        avg_ok_list = [r.get("avg_emotion") == expected_avg for r in results]
        avg_ok = sum(1 for a in avg_ok_list if a) / max(1, len(avg_ok_list))

    # Efficiency
    latencies = [r["latency"] for r in results]
    steps = [r["steps"] for r in results]
    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else max(latencies) if latencies else 0.0
    avg_steps = statistics.mean(steps) if steps else 0.0

    # Reliability
    reliables = [r["reliability_pass"] for r in results]
    reliability_rate = sum(1 for x in reliables if x) / max(1, len(reliables))
    violation_counts = {}
    for r in results:
        for v in r.get("reliability_violations", []):
            violation_counts[v] = violation_counts.get(v, 0) + 1

    # User outcomes (mock-mode proxies)
    accepts = [r.get("accepted") for r in results if r.get("accepted") is not None]
    clicks = [r.get("clicked") for r in results if r.get("clicked") is not None]
    opens = [r.get("opened") for r in results if r.get("opened") is not None]
    accept_rate = sum(1 for a in accepts if a) / max(1, len(accepts)) if accepts else None
    click_rate = sum(1 for c in clicks if c) / max(1, len(clicks)) if clicks else None
    open_rate = sum(1 for o in opens if o) / max(1, len(opens)) if opens else None

    return {
        "success_rate": success_rate,
        "avg_emotion_accuracy": avg_ok,
        "avg_latency_sec": avg_latency,
        "p95_latency_sec": p95_latency,
        "avg_steps": avg_steps,
        "reliability_rate": reliability_rate,
        "violation_counts": violation_counts,
        "accept_rate": accept_rate,
        "click_rate": click_rate,
        "open_rate": open_rate,
        "num_runs": len(results),
    }

def main():
    random.seed(SEED)
    agent_mod = importlib.import_module(AGENT_MODULE)

    all_case_reports = []
    for case in TEST_CASES:
        per_run = []
        for _ in range(RUNS_PER_CASE):
            per_run.append(run_one(agent_mod, case["emotions"], USE_MOCKS))
        report = aggregate(
            per_run,
            expected_avg=case.get("expected_avg"),
            expect_negative=case.get("expect_negative"),
        )
        all_case_reports.append((case["name"], report))

    # Global aggregation across cases
    # Flatten
    flat_runs = []
    for case in TEST_CASES:
        for _ in range(RUNS_PER_CASE):
            flat_runs.append(run_one(agent_mod, case["emotions"], USE_MOCKS))
    global_report = aggregate(flat_runs)

    # Pretty print
    print("\n=== Per-case metrics ===")
    for name, rep in all_case_reports:
        print(f"- {name}: {rep}")

    print("\n=== Global metrics ===")
    print(global_report)

if __name__ == "__main__":
    main()
