-- Populate threads table from emails data
-- Derives thread-level aggregations from the emails table

INSERT INTO threads (
    thread_id,
    subject,
    participants,
    your_role,
    email_count,
    your_email_count,
    your_reply_count,
    has_attachments,
    total_attachment_count,
    attachment_types,
    started_at,
    last_activity,
    your_first_reply_at,
    avg_response_time_seconds,
    thread_duration_seconds
)
SELECT
    thread_id,
    -- Subject: from first email in thread (earliest date)
    (SELECT subject FROM emails e2
     WHERE e2.thread_id = e.thread_id
     ORDER BY e2.date_parsed ASC NULLS LAST
     LIMIT 1) AS subject,
    -- Participants: all unique email addresses (from + to + cc)
    (SELECT ARRAY_AGG(DISTINCT addr)
     FROM (
         SELECT from_email AS addr FROM emails e3 WHERE e3.thread_id = e.thread_id AND from_email IS NOT NULL
         UNION
         SELECT UNNEST(to_emails) AS addr FROM emails e4 WHERE e4.thread_id = e.thread_id AND to_emails IS NOT NULL
         UNION
         SELECT UNNEST(cc_emails) AS addr FROM emails e5 WHERE e5.thread_id = e.thread_id AND cc_emails IS NOT NULL
     ) all_addrs WHERE addr IS NOT NULL
    ) AS participants,
    -- Your role: sender if you sent any, recipient if received to you, cc if only cc'd
    CASE
        WHEN BOOL_OR(is_sent) THEN 'sender'
        WHEN BOOL_OR('me@nik-patel.com' = ANY(to_emails)) THEN 'recipient'
        WHEN BOOL_OR('me@nik-patel.com' = ANY(cc_emails)) THEN 'cc'
        ELSE 'none'
    END AS your_role,
    -- Email count
    COUNT(*) AS email_count,
    -- Your email count
    COUNT(*) FILTER (WHERE is_sent = TRUE) AS your_email_count,
    -- Your reply count (sent and is a reply)
    COUNT(*) FILTER (WHERE is_sent = TRUE AND in_reply_to IS NOT NULL) AS your_reply_count,
    -- Has attachments
    BOOL_OR(has_attachments) AS has_attachments,
    -- Total attachment count
    COALESCE(SUM(attachment_count), 0)::INTEGER AS total_attachment_count,
    -- Attachment types: unique types across all emails
    (SELECT ARRAY_AGG(DISTINCT atype)
     FROM (
         SELECT UNNEST(attachment_types) AS atype
         FROM emails e6
         WHERE e6.thread_id = e.thread_id AND attachment_types IS NOT NULL
     ) atypes WHERE atype IS NOT NULL
    ) AS attachment_types,
    -- Started at (first email)
    MIN(date_parsed) AS started_at,
    -- Last activity (most recent email)
    MAX(date_parsed) AS last_activity,
    -- Your first reply
    MIN(date_parsed) FILTER (WHERE is_sent = TRUE AND in_reply_to IS NOT NULL) AS your_first_reply_at,
    -- Average response time (where available)
    AVG(response_time_seconds)::INTEGER AS avg_response_time_seconds,
    -- Thread duration in seconds
    EXTRACT(EPOCH FROM (MAX(date_parsed) - MIN(date_parsed)))::INTEGER AS thread_duration_seconds
FROM emails e
WHERE thread_id IS NOT NULL
GROUP BY thread_id;
