# ── Model ────────────────────────────────────────────────────
MODEL       = "gpt-3.5-turbo"
TEMPERATURE = 0            # 0 = consistent/deterministic output

# ── Text splitting ────────────────────────────────────────────
# chunk_size:    max characters per chunk (~800 words at 4000 chars)
# chunk_overlap: characters shared between adjacent chunks so we
#                don't lose context at the boundary between chunks
CHUNK_SIZE    = 4000
CHUNK_OVERLAP = 200
