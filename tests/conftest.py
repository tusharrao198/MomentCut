import os
import sys

# Make the repo root importable so tests can import the app modules directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests always run the app in self-host (BYOK) mode. app.py freezes
# BILLING_ENABLED at import time and load_dotenv never overrides an existing
# variable, so this must be set here — before any test module imports app — or
# the suite's behavior would depend on the developer's personal .env.
os.environ["BILLING_ENABLED"] = "0"
# Test modules opt into source context themselves. Keep a developer's .env
# from turning it on for otherwise unrelated API and pipeline tests.
os.environ["SOURCE_CONTEXT_MODE"] = "disabled"
# Hook-grounding fixtures exercise the upstream Gemini-enabled default. Keep a
# developer's local-only setting from turning that model call into a no-op.
os.environ["GEMINI_CLIP_ANALYSIS_ENABLED"] = "true"
# CED is an explicit macOS opt-in. Keep a developer's native .env from
# affecting unrelated checkpoint, API, or portable-route tests.
os.environ["AUDIO_EVENTS_MODE"] = "disabled"
