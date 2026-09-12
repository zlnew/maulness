-- Tasks table (Platform-agnostic, Repo-optional)
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    repo_name TEXT,
    workspace_path TEXT,
    origin_platform TEXT NOT NULL DEFAULT 'cli',
    origin_channel_id TEXT,
    origin_thread_id TEXT,
    mode TEXT CHECK(mode IN ('direct', 'multi')) NOT NULL DEFAULT 'direct',
    status TEXT CHECK(status IN ('planning', 'building', 'review', 'done', 'failed', 'suspended_afk')) NOT NULL DEFAULT 'planning',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tasks_origin ON tasks(origin_platform, origin_channel_id);
CREATE INDEX IF NOT EXISTS idx_tasks_repo ON tasks(repo_name);

-- Active & Historical Agent Sessions
CREATE TABLE IF NOT EXISTS agent_sessions (
    id TEXT PRIMARY KEY,
    task_id TEXT REFERENCES tasks(id) ON DELETE CASCADE,
    profile TEXT NOT NULL,              -- 'builder', 'planner', 'reviewer', 'triage'
    engine TEXT NOT NULL,               -- 'acp', 'sdk', or 'gemini_api'
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
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected', 'timeout')),
    platform TEXT NOT NULL DEFAULT 'discord',
    platform_message_id TEXT,           -- Platform-specific message ID / timestamp
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Active Channel Conversations (multi-turn session persistence across platforms)
CREATE TABLE IF NOT EXISTS channel_conversations (
    platform TEXT NOT NULL DEFAULT 'discord',
    channel_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    profile_name TEXT NOT NULL DEFAULT 'default',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (platform, channel_id)
);

-- Conversation Message History (multi-turn context persistence for all providers)
CREATE TABLE IF NOT EXISTS conversation_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conv_messages_conv_id ON conversation_messages(conversation_id);

-- Durable Event-Sourced Step Journal (Kernel v2)
CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    event_type TEXT NOT NULL,       -- 'prompt', 'thought', 'tool_call', 'tool_result', 'gate_eval', 'checkpoint'
    event_payload TEXT NOT NULL,    -- JSON string
    idempotency_key TEXT UNIQUE,    -- SHA256(task_id:stage:step_index:event_type:payload_signature)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_agent_events_task_stage ON agent_events(task_id, stage, step_index);
CREATE INDEX IF NOT EXISTS idx_agent_events_idempotency ON agent_events(idempotency_key);

-- Session Memories (Compacted summaries)
CREATE TABLE IF NOT EXISTS session_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


