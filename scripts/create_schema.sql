-- Stage 6: Database Schema for rl-emails
-- Creates tables per docs/v2/SCHEMA.md

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Drop existing tables if rebuilding
DROP TABLE IF EXISTS attachments CASCADE;
DROP TABLE IF EXISTS threads CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS emails CASCADE;
DROP TABLE IF EXISTS raw_emails CASCADE;

-- ============================================
-- raw_emails (Source of Truth)
-- Immutable raw data from MBOX parse
-- ============================================
CREATE TABLE raw_emails (
    id SERIAL PRIMARY KEY,
    message_id TEXT UNIQUE NOT NULL,

    -- Raw headers (exactly as parsed from MBOX)
    thread_id TEXT,
    in_reply_to TEXT,
    references_raw TEXT,

    date_raw TEXT,
    from_raw TEXT,
    to_raw TEXT,
    cc_raw TEXT,
    bcc_raw TEXT,
    subject_raw TEXT,

    -- Raw content
    body_text TEXT,
    body_html TEXT,

    -- Gmail metadata
    labels_raw TEXT,

    -- MBOX metadata
    mbox_offset BIGINT,
    raw_size_bytes INTEGER,

    imported_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_raw_emails_message_id ON raw_emails(message_id);
CREATE INDEX idx_raw_emails_thread_id ON raw_emails(thread_id);

-- ============================================
-- emails (Enriched/Derived)
-- Derived from raw_emails with parsed fields
-- ============================================
CREATE TABLE emails (
    id SERIAL PRIMARY KEY,
    raw_email_id INTEGER REFERENCES raw_emails(id),
    message_id TEXT UNIQUE NOT NULL,
    thread_id TEXT,
    in_reply_to TEXT,

    -- Parsed headers
    date_parsed TIMESTAMPTZ,
    from_email TEXT,
    from_name TEXT,
    to_emails TEXT[],
    cc_emails TEXT[],
    subject TEXT,

    -- Content
    body_text TEXT,
    body_preview TEXT,
    word_count INTEGER,

    -- Parsed Gmail metadata
    labels TEXT[],

    -- Attachment summary
    has_attachments BOOLEAN DEFAULT FALSE,
    attachment_count INTEGER DEFAULT 0,
    attachment_types TEXT[],
    total_attachment_bytes BIGINT DEFAULT 0,

    -- Ownership
    is_sent BOOLEAN DEFAULT FALSE,

    -- Computed enrichment (Stage 5)
    action TEXT,
    timing TEXT,
    response_time_seconds INTEGER,
    priority_score FLOAT,

    -- Processing metadata
    enriched_at TIMESTAMPTZ,
    enrichment_version INTEGER DEFAULT 1,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_emails_thread_id ON emails(thread_id);
CREATE INDEX idx_emails_from_email ON emails(from_email);
CREATE INDEX idx_emails_date ON emails(date_parsed);
CREATE INDEX idx_emails_in_reply_to ON emails(in_reply_to);
CREATE INDEX idx_emails_is_sent ON emails(is_sent);
CREATE INDEX idx_emails_labels ON emails USING GIN(labels);
CREATE INDEX idx_emails_subject_trgm ON emails USING GIN(subject gin_trgm_ops);

-- ============================================
-- attachments
-- Individual attachment metadata
-- ============================================
CREATE TABLE attachments (
    id SERIAL PRIMARY KEY,
    raw_email_id INTEGER REFERENCES raw_emails(id),
    email_id INTEGER REFERENCES emails(id),

    -- Attachment metadata
    filename TEXT,
    content_type TEXT,
    size_bytes INTEGER,
    content_disposition TEXT,

    -- Hashing for dedup
    content_hash TEXT,

    -- Storage
    stored_path TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_attachments_raw_email_id ON attachments(raw_email_id);
CREATE INDEX idx_attachments_email_id ON attachments(email_id);
CREATE INDEX idx_attachments_content_type ON attachments(content_type);

-- ============================================
-- threads (Aggregated)
-- Thread-level aggregations
-- ============================================
CREATE TABLE threads (
    id SERIAL PRIMARY KEY,
    thread_id TEXT UNIQUE NOT NULL,
    subject TEXT,

    -- Participants
    participants TEXT[],
    your_role TEXT,

    -- Email stats
    email_count INTEGER DEFAULT 0,
    your_email_count INTEGER DEFAULT 0,
    your_reply_count INTEGER DEFAULT 0,

    -- Attachment stats
    has_attachments BOOLEAN DEFAULT FALSE,
    total_attachment_count INTEGER DEFAULT 0,
    attachment_types TEXT[],

    -- Timing
    started_at TIMESTAMPTZ,
    last_activity TIMESTAMPTZ,
    your_first_reply_at TIMESTAMPTZ,

    -- Computed
    avg_response_time_seconds INTEGER,
    thread_duration_seconds INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_threads_thread_id ON threads(thread_id);
CREATE INDEX idx_threads_has_attachments ON threads(has_attachments);

-- ============================================
-- users (Aggregated)
-- Email addresses with communication stats
-- ============================================
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,

    -- Relationship
    is_you BOOLEAN DEFAULT FALSE,

    -- Communication stats
    emails_from INTEGER DEFAULT 0,
    emails_to INTEGER DEFAULT 0,
    threads_with INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,

    -- Response patterns
    avg_response_time_seconds INTEGER,
    reply_rate FLOAT,

    -- Labels
    is_important_sender BOOLEAN DEFAULT FALSE,
    first_contact TIMESTAMPTZ,
    last_contact TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_is_you ON users(is_you);

-- ============================================
-- email_features (Pre-computed ML Features)
-- Per-email feature vectors for RL pipeline
-- ============================================
CREATE TABLE email_features (
    id SERIAL PRIMARY KEY,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL,

    -- Relationship features (from CommunicationGraph)
    sender_response_deviation FLOAT,
    sender_frequency_rank FLOAT,
    inferred_hierarchy FLOAT,
    relationship_strength FLOAT,
    emails_from_sender_7d INTEGER DEFAULT 0,
    emails_from_sender_30d INTEGER DEFAULT 0,
    emails_from_sender_90d INTEGER DEFAULT 0,
    response_rate_to_sender FLOAT,
    avg_thread_depth FLOAT,
    days_since_last_email FLOAT,
    cc_affinity_score FLOAT,

    -- Service classification
    is_service_email BOOLEAN DEFAULT FALSE,
    service_type TEXT,
    service_email_confidence FLOAT,
    has_list_unsubscribe_header BOOLEAN DEFAULT FALSE,
    has_unsubscribe_url BOOLEAN DEFAULT FALSE,
    unsubscribe_phrase_count INTEGER DEFAULT 0,

    -- Task features
    task_count INTEGER DEFAULT 0,
    has_deadline BOOLEAN DEFAULT FALSE,
    deadline_urgency FLOAT,
    is_assigned_to_user BOOLEAN DEFAULT FALSE,
    estimated_effort TEXT,
    has_deliverable BOOLEAN DEFAULT FALSE,

    -- Urgency scoring
    urgency_score FLOAT,
    urgency_bucket TEXT,

    -- Computed priority scores
    project_score FLOAT,
    topic_score FLOAT,
    task_score FLOAT,
    people_score FLOAT,
    temporal_score FLOAT,
    service_score FLOAT,
    relationship_score FLOAT,
    overall_priority FLOAT,

    -- Embeddings (stored as arrays)
    feature_vector FLOAT[],
    content_embedding FLOAT[],
    embedding_model TEXT,
    embedding_dim INTEGER,

    -- Processing metadata
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    feature_version INTEGER DEFAULT 1,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_email_features_email_id ON email_features(email_id);
CREATE INDEX idx_email_features_message_id ON email_features(message_id);
CREATE INDEX idx_email_features_is_service ON email_features(is_service_email);
CREATE INDEX idx_email_features_urgency ON email_features(urgency_score);
CREATE INDEX idx_email_features_priority ON email_features(overall_priority);
CREATE INDEX idx_email_features_service_type ON email_features(service_type);

-- ============================================
-- tasks
-- Extracted tasks from emails
-- ============================================
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    task_id TEXT UNIQUE NOT NULL,
    email_id INTEGER REFERENCES emails(id) ON DELETE CASCADE,

    -- Task details
    description TEXT NOT NULL,
    deadline TIMESTAMPTZ,
    deadline_text TEXT,
    assignee_hint TEXT,
    complexity TEXT CHECK (complexity IN ('trivial', 'quick', 'medium', 'substantial', 'unknown')),
    task_type TEXT CHECK (task_type IN ('review', 'send', 'schedule', 'decision', 'research', 'create', 'follow_up', 'other')),
    urgency_score FLOAT CHECK (urgency_score >= 0 AND urgency_score <= 1),
    source_text TEXT,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_tasks_email_id ON tasks(email_id);
CREATE INDEX idx_tasks_task_type ON tasks(task_type);
CREATE INDEX idx_tasks_complexity ON tasks(complexity);
CREATE INDEX idx_tasks_deadline ON tasks(deadline);
CREATE INDEX idx_tasks_urgency_score ON tasks(urgency_score);

-- ============================================
-- Summary
-- ============================================
-- Tables created:
--   raw_emails: Immutable source data
--   emails: Enriched/derived data
--   attachments: Attachment metadata
--   threads: Thread-level aggregations
--   users: Communication stats per user
--   email_features: Pre-computed ML features per email
--   tasks: Extracted tasks from emails
