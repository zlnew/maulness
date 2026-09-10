-- Tasks table (Direct runs & Multi-route tickets)
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    discord_thread_id INTEGER UNIQUE,
    title TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    workspace_path TEXT NOT NULL,
    mode TEXT CHECK(mode IN ('direct', 'multi')) NOT NULL,
    status TEXT CHECK(status IN ('planning', 'building', 'review', 'done', 'failed')) NOT NULL DEFAULT 'planning',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Active & Historical Agent Sessions
CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
    profile TEXT NOT NULL,              -- 'builder', 'planner', 'reviewer', 'triage'
    engine TEXT NOT NULL,               -- 'acp' or 'gemini_api'
    acp_session_id TEXT,
    pid INTEGER,
    status TEXT DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Approvals (HITL coordinate table)
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
    rpc_request_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args TEXT NOT NULL,            -- JSON string
    status TEXT DEFAULT 'pending',      -- 'pending', 'approved', 'rejected', 'timeout'
    discord_message_id INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Event Journal
CREATE TABLE IF NOT EXISTS task_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,           -- 'thought', 'message', 'tool_call', 'status_change'
    payload TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Session Memories (Compacted summaries)
CREATE TABLE IF NOT EXISTS session_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

