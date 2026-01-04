-- SurrealDB Schema for Email RL System
-- Shared schema for both Enron and Gmail databases
--
-- Usage:
--   surreal import -c "surreal://root:root@localhost:8000" -ns rl_emails -db enron db/schema.surql
--   surreal import -c "surreal://root:root@localhost:8000" -ns rl_emails -db gmail db/schema.surql

-- ============================================================================
-- TABLE DEFINITIONS
-- ============================================================================

-- Users table: email addresses with metadata
DEFINE TABLE users SCHEMAFULL;
DEFINE FIELD email ON users TYPE string;
DEFINE FIELD name ON users TYPE option<string>;
DEFINE FIELD org_level ON users TYPE option<string>;  -- e.g., "Vice President, Trading"
DEFINE FIELD department ON users TYPE option<string>;
DEFINE FIELD is_internal ON users TYPE bool DEFAULT true;
-- Baseline response metrics (computed from data)
DEFINE FIELD avg_response_time_hours ON users TYPE option<float>;
DEFINE FIELD reply_rate ON users TYPE option<float>;
DEFINE FIELD emails_sent ON users TYPE int DEFAULT 0;
DEFINE FIELD emails_received ON users TYPE int DEFAULT 0;
DEFINE FIELD created_at ON users TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON users TYPE datetime DEFAULT time::now();

DEFINE INDEX users_email ON users COLUMNS email UNIQUE;

-- Threads table: conversation groupings
DEFINE TABLE threads SCHEMAFULL;
DEFINE FIELD thread_id ON threads TYPE string;  -- Message-ID of first email or X-GM-THRID
DEFINE FIELD subject ON threads TYPE string;
DEFINE FIELD email_count ON threads TYPE int DEFAULT 0;
DEFINE FIELD participants ON threads TYPE array<string> DEFAULT [];  -- email addresses
DEFINE FIELD started_at ON threads TYPE option<datetime>;
DEFINE FIELD last_activity ON threads TYPE option<datetime>;
-- User-specific counts (relative to user_email)
DEFINE FIELD your_email_count ON threads TYPE int DEFAULT 0;
DEFINE FIELD your_reply_count ON threads TYPE int DEFAULT 0;
-- Computed timing fields
DEFINE FIELD thread_duration_seconds ON threads TYPE option<float>;
DEFINE FIELD avg_response_time_seconds ON threads TYPE option<float>;
-- Attachment fields
DEFINE FIELD has_attachments ON threads TYPE bool DEFAULT false;
DEFINE FIELD total_attachment_count ON threads TYPE int DEFAULT 0;
DEFINE FIELD created_at ON threads TYPE datetime DEFAULT time::now();

DEFINE INDEX threads_thread_id ON threads COLUMNS thread_id UNIQUE;

-- Emails table: full email records with embeddings
DEFINE TABLE emails SCHEMAFULL;
DEFINE FIELD message_id ON emails TYPE string;
DEFINE FIELD date ON emails TYPE option<datetime>;
DEFINE FIELD date_str ON emails TYPE string DEFAULT "";  -- Original date string for parsing failures
DEFINE FIELD subject ON emails TYPE string DEFAULT "";
DEFINE FIELD body ON emails TYPE string DEFAULT "";
DEFINE FIELD from_email ON emails TYPE string;
DEFINE FIELD to_emails ON emails TYPE array<string> DEFAULT [];
DEFINE FIELD cc_emails ON emails TYPE array<string> DEFAULT [];
DEFINE FIELD bcc_emails ON emails TYPE array<string> DEFAULT [];
DEFINE FIELD in_reply_to ON emails TYPE option<string>;
DEFINE FIELD references ON emails TYPE array<string> DEFAULT [];

-- Enron-specific fields
DEFINE FIELD x_from ON emails TYPE option<string>;  -- Sender with org info
DEFINE FIELD x_to ON emails TYPE option<string>;
DEFINE FIELD x_folder ON emails TYPE option<string>;
DEFINE FIELD x_origin ON emails TYPE option<string>;
DEFINE FIELD folder ON emails TYPE option<string>;
DEFINE FIELD file_path ON emails TYPE option<string>;
DEFINE FIELD attachments ON emails TYPE array<string> DEFAULT [];
DEFINE FIELD enron_user ON emails TYPE option<string>;  -- Mailbox owner

-- Gmail-specific fields
DEFINE FIELD gmail_thread_id ON emails TYPE option<string>;  -- X-GM-THRID
DEFINE FIELD labels ON emails TYPE array<string> DEFAULT [];

-- ML fields
DEFINE FIELD action ON emails TYPE option<string>;  -- REPLIED, FORWARDED, etc.
DEFINE FIELD timing ON emails TYPE option<string>;  -- IMMEDIATE, HOUR, DAY, etc.
DEFINE FIELD split ON emails TYPE option<string>;   -- train, val, test

-- Embedding vectors (384-dim from all-MiniLM-L6-v2)
DEFINE FIELD embedding ON emails TYPE option<array<float>>;
DEFINE FIELD embedding_model ON emails TYPE option<string>;

-- Computed features (cached for fast retrieval)
DEFINE FIELD features ON emails TYPE option<object>;

DEFINE FIELD created_at ON emails TYPE datetime DEFAULT time::now();

DEFINE INDEX emails_message_id ON emails COLUMNS message_id UNIQUE;
DEFINE INDEX emails_from ON emails COLUMNS from_email;
DEFINE INDEX emails_date ON emails COLUMNS date;
DEFINE INDEX emails_action ON emails COLUMNS action;
DEFINE INDEX emails_split ON emails COLUMNS split;
DEFINE INDEX emails_gmail_thread ON emails COLUMNS gmail_thread_id;

-- Sender features table: per-sender aggregated metrics
DEFINE TABLE sender_features SCHEMAFULL;
DEFINE FIELD sender_email ON sender_features TYPE string;
DEFINE FIELD total_emails ON sender_features TYPE int DEFAULT 0;
DEFINE FIELD avg_response_rate ON sender_features TYPE option<float>;
DEFINE FIELD is_service ON sender_features TYPE bool DEFAULT false;
DEFINE FIELD service_type ON sender_features TYPE option<string>;  -- e.g., "newsletter", "notification", "billing"
DEFINE FIELD last_interaction ON sender_features TYPE option<datetime>;
DEFINE FIELD relationship_score ON sender_features TYPE option<float>;
DEFINE FIELD created_at ON sender_features TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON sender_features TYPE datetime DEFAULT time::now();

DEFINE INDEX sender_features_email ON sender_features COLUMNS sender_email UNIQUE;

-- ============================================================================
-- EDGE DEFINITIONS (Graph Relationships)
-- ============================================================================

-- sent_by: email -> user (who sent this email)
DEFINE TABLE sent_by SCHEMAFULL TYPE RELATION IN emails OUT users;
DEFINE FIELD created_at ON sent_by TYPE datetime DEFAULT time::now();

-- received_by: email -> user (who received this email, includes to/cc/bcc)
DEFINE TABLE received_by SCHEMAFULL TYPE RELATION IN emails OUT users;
DEFINE FIELD field ON received_by TYPE string;  -- "to", "cc", or "bcc"
DEFINE FIELD created_at ON received_by TYPE datetime DEFAULT time::now();

-- belongs_to: email -> thread (thread membership)
DEFINE TABLE belongs_to SCHEMAFULL TYPE RELATION IN emails OUT threads;
DEFINE FIELD position ON belongs_to TYPE int DEFAULT 0;  -- Order in thread
DEFINE FIELD created_at ON belongs_to TYPE datetime DEFAULT time::now();

-- replies_to: email -> email (direct reply relationship)
DEFINE TABLE replies_to SCHEMAFULL TYPE RELATION IN emails OUT emails;
DEFINE FIELD created_at ON replies_to TYPE datetime DEFAULT time::now();

-- communicates: user -> user (aggregated communication pattern)
DEFINE TABLE communicates SCHEMAFULL TYPE RELATION IN users OUT users;
DEFINE FIELD email_count ON communicates TYPE int DEFAULT 0;
DEFINE FIELD reply_count ON communicates TYPE int DEFAULT 0;
DEFINE FIELD avg_response_time_hours ON communicates TYPE option<float>;
DEFINE FIELD first_email ON communicates TYPE option<datetime>;
DEFINE FIELD last_email ON communicates TYPE option<datetime>;
DEFINE FIELD updated_at ON communicates TYPE datetime DEFAULT time::now();

-- ============================================================================
-- VIEWS (Computed Tables)
-- ============================================================================

-- View: training data with embeddings
DEFINE TABLE training_data AS
    SELECT
        id,
        message_id,
        subject,
        body,
        from_email,
        to_emails,
        action,
        timing,
        embedding,
        features,
        date
    FROM emails
    WHERE split = "train" AND action IS NOT NONE
;

-- View: validation data
DEFINE TABLE validation_data AS
    SELECT
        id,
        message_id,
        subject,
        body,
        from_email,
        to_emails,
        action,
        timing,
        embedding,
        features,
        date
    FROM emails
    WHERE split = "val" AND action IS NOT NONE
;

-- View: test data
DEFINE TABLE test_data AS
    SELECT
        id,
        message_id,
        subject,
        body,
        from_email,
        to_emails,
        action,
        timing,
        embedding,
        features,
        date
    FROM emails
    WHERE split = "test" AND action IS NOT NONE
;

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- Function to get emails between two users
DEFINE FUNCTION fn::emails_between($user1: string, $user2: string) {
    RETURN SELECT * FROM emails
        WHERE (from_email = $user1 AND $user2 IN to_emails)
           OR (from_email = $user2 AND $user1 IN to_emails)
        ORDER BY date ASC;
};

-- Function to get user's response time baseline
DEFINE FUNCTION fn::response_baseline($user_email: string) {
    LET $user = (SELECT * FROM users WHERE email = $user_email LIMIT 1);
    RETURN {
        avg_response_time: $user[0].avg_response_time_hours,
        reply_rate: $user[0].reply_rate,
        emails_sent: $user[0].emails_sent,
        emails_received: $user[0].emails_received
    };
};

-- Function to compute communication frequency between users
-- Simplified to avoid IF/THEN/ELSE syntax issues in SurrealDB 2.4.0
DEFINE FUNCTION fn::comm_frequency($from_user: string, $to_user: string) {
    LET $edge = (SELECT * FROM communicates
                 WHERE in.email = $from_user AND out.email = $to_user
                 LIMIT 1);
    RETURN $edge[0];
};

-- Function to count emails from sender to recipient in a time window
-- $reference_time: ISO datetime string to compute window from
-- $window_days: number of days to look back
DEFINE FUNCTION fn::sender_frequency_window(
    $sender: string,
    $recipient: string,
    $reference_time: datetime,
    $window_days: int
) {
    LET $window_start = $reference_time - ($window_days * 1d);
    LET $count = (SELECT count() FROM emails
                  WHERE from_email = $sender
                    AND $recipient IN to_emails
                    AND date >= $window_start
                    AND date < $reference_time
                  GROUP ALL);
    RETURN $count[0].count ?? 0;
};

-- Function to get time-windowed email frequency for all standard windows (7d, 30d, 90d)
-- Returns object with emails_7d, emails_30d, emails_90d
DEFINE FUNCTION fn::sender_frequency_all_windows(
    $sender: string,
    $recipient: string,
    $reference_time: datetime
) {
    RETURN {
        emails_7d: fn::sender_frequency_window($sender, $recipient, $reference_time, 7),
        emails_30d: fn::sender_frequency_window($sender, $recipient, $reference_time, 30),
        emails_90d: fn::sender_frequency_window($sender, $recipient, $reference_time, 90)
    };
};

-- ============================================================================
-- EVENTS (Auto-update triggers)
-- ============================================================================

-- Auto-update user email counts when emails are created
DEFINE EVENT email_created ON emails WHEN $event = "CREATE" THEN {
    -- Update sender stats
    UPDATE users SET emails_sent += 1 WHERE email = $value.from_email;
    -- Update recipient stats
    FOR $to IN $value.to_emails {
        UPDATE users SET emails_received += 1 WHERE email = $to;
    };
};
