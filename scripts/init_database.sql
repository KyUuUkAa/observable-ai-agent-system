BEGIN;

CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY,
    title VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    role VARCHAR NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_message_role CHECK (role IN ('user', 'assistant')),
    CONSTRAINT fk_messages_conversation
        FOREIGN KEY (conversation_id)
        REFERENCES conversations (id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
    ON messages (conversation_id);

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id VARCHAR PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agent_messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    message_data TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT agent_messages_session_id_fkey
        FOREIGN KEY (session_id)
        REFERENCES agent_sessions (session_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_messages_session_time
    ON agent_messages (session_id, created_at);

CREATE TABLE IF NOT EXISTS oracle_recognition_records (
    id UUID PRIMARY KEY,
    conversation_id UUID NULL,
    image_reference_id VARCHAR(32) NULL,
    original_filename TEXT NOT NULL,
    content_type VARCHAR(255) NOT NULL,
    image_sha256 CHAR(64) NOT NULL,
    image_data BYTEA NOT NULL,
    image_width INTEGER NOT NULL,
    image_height INTEGER NOT NULL,
    top1_class_id INTEGER NOT NULL,
    top1_class_code VARCHAR(64) NOT NULL,
    top1_confidence DOUBLE PRECISION NOT NULL,
    top5 JSONB NOT NULL,
    model_info JSONB NOT NULL,
    model_version VARCHAR(128) NOT NULL,
    review_threshold DOUBLE PRECISION NOT NULL,
    review_status VARCHAR(32) NOT NULL,
    review_notes TEXT NULL,
    reviewed_at TIMESTAMPTZ NULL,
    execution_trace JSONB NOT NULL DEFAULT '{}'::JSONB,
    source VARCHAR(32) NOT NULL DEFAULT 'single',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_oracle_record_conversation
        FOREIGN KEY (conversation_id)
        REFERENCES conversations (id)
        ON DELETE SET NULL,
    CONSTRAINT check_oracle_review_status
        CHECK (
            review_status IN (
                'pending',
                'auto_accepted',
                'accepted',
                'rejected'
            )
        ),
    CONSTRAINT check_oracle_confidence
        CHECK (top1_confidence >= 0 AND top1_confidence <= 1),
    CONSTRAINT check_oracle_review_threshold
        CHECK (review_threshold >= 0 AND review_threshold <= 1)
);

CREATE INDEX IF NOT EXISTS idx_oracle_records_created_at
    ON oracle_recognition_records (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_oracle_records_review_status
    ON oracle_recognition_records (review_status, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_oracle_records_class_code
    ON oracle_recognition_records (top1_class_code);

CREATE INDEX IF NOT EXISTS idx_oracle_records_image_sha256
    ON oracle_recognition_records (image_sha256);

COMMIT;
