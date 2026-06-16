# ── Model ────────────────────────────────────────────────────
MODEL       = "gpt-3.5-turbo"
TEMPERATURE = 0.3          # slight creativity for more natural review tone

# ── Resume handling ───────────────────────────────────────────
# Trim very long resumes to avoid hitting token limits.
# 6000 chars ≈ ~1200 words — more than enough for any resume.
MAX_RESUME_CHARS = 6000
