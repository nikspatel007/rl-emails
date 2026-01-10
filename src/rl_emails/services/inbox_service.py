"""Inbox service for priority inbox functionality."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.schemas.inbox import (
    EmailSummary,
    InboxStats,
    PriorityContext,
    PriorityEmail,
    PriorityInboxResponse,
)


class InboxService:
    """Service for priority inbox operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize service with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self._session = session

    async def get_priority_inbox(
        self,
        user_id: UUID | None = None,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> PriorityInboxResponse:
        """Get priority-sorted inbox with context.

        Args:
            user_id: Optional user UUID to scope query (for multi-tenant).
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Priority inbox response with emails and stats.
        """
        # Build the main query joining emails with priority data
        query = text(
            """
            SELECT
                e.id,
                e.message_id,
                e.thread_id,
                e.subject,
                e.from_email,
                e.from_name,
                e.date_parsed,
                e.body_preview,
                e.is_sent,
                e.action,
                e.has_attachments,
                e.labels,
                ep.priority_rank,
                ep.priority_score,
                pc.sender_email,
                pc.sender_importance,
                pc.sender_reply_rate,
                pc.thread_length,
                pc.is_business_hours,
                pc.age_hours,
                pc.people_score,
                pc.temporal_score,
                pc.relationship_score,
                pc.overall_priority,
                (SELECT COUNT(*) FROM tasks t WHERE t.email_id = e.id AND t.status = 'pending') as task_count,
                (SELECT p.name FROM projects p
                 JOIN email_project_links epl ON epl.project_id = p.id
                 WHERE epl.email_id = e.id LIMIT 1) as project_name
            FROM emails e
            LEFT JOIN email_priority ep ON ep.email_id = e.id
            LEFT JOIN priority_contexts pc ON pc.email_id = e.id
            WHERE e.is_sent = false
            ORDER BY ep.priority_rank ASC NULLS LAST, e.date_parsed DESC
            LIMIT :limit OFFSET :offset
            """
        )

        result = await self._session.execute(query, {"limit": limit, "offset": offset})
        rows = result.fetchall()

        # Get total count
        count_query = text(
            """
            SELECT COUNT(*) FROM emails e WHERE e.is_sent = false
            """
        )
        count_result = await self._session.execute(count_query)
        total = count_result.scalar() or 0

        # Build response
        emails = []
        for row in rows:
            email_summary = EmailSummary(
                id=row[0],
                message_id=row[1],
                thread_id=row[2],
                subject=row[3],
                from_email=row[4],
                from_name=row[5],
                date_parsed=row[6],
                body_preview=row[7],
                is_sent=row[8] or False,
                action=row[9],
                has_attachments=row[10] or False,
                labels=row[11],
            )

            priority_context = None
            if row[14] is not None:  # sender_email
                priority_context = PriorityContext(
                    email_id=row[0],
                    sender_email=row[14],
                    sender_importance=row[15],
                    sender_reply_rate=row[16],
                    thread_length=row[17],
                    is_business_hours=row[18],
                    age_hours=row[19],
                    people_score=row[20],
                    temporal_score=row[21],
                    relationship_score=row[22],
                    overall_priority=row[23],
                )

            priority_email = PriorityEmail(
                email=email_summary,
                priority_rank=row[12] or 0,
                priority_score=row[13] or 0.0,
                context=priority_context,
                task_count=row[24] or 0,
                project_name=row[25],
            )
            emails.append(priority_email)

        # Get stats
        stats = await self._get_inbox_stats(user_id)

        return PriorityInboxResponse(
            emails=emails,
            total=total,
            limit=limit,
            offset=offset,
            has_more=offset + len(emails) < total,
            pending_tasks=stats.pending_tasks,
            urgent_count=stats.urgent_emails,
            from_real_people_count=stats.from_real_people,
        )

    async def get_inbox_stats(self, user_id: UUID | None = None) -> InboxStats:
        """Get inbox statistics.

        Args:
            user_id: Optional user UUID to scope query.

        Returns:
            Inbox statistics.
        """
        return await self._get_inbox_stats(user_id)

    async def _get_inbox_stats(self, user_id: UUID | None = None) -> InboxStats:
        """Get inbox statistics (internal).

        Args:
            user_id: Optional user UUID to scope query.

        Returns:
            Inbox statistics.
        """
        # Total emails (received, not sent)
        total_query = text("SELECT COUNT(*) FROM emails WHERE is_sent = false")
        total_result = await self._session.execute(total_query)
        total_emails = total_result.scalar() or 0

        # Pending tasks
        pending_query = text("SELECT COUNT(*) FROM tasks WHERE status = 'pending'")
        pending_result = await self._session.execute(pending_query)
        pending_tasks = pending_result.scalar() or 0

        # Urgent emails (priority_score > 0.7)
        urgent_query = text(
            """
            SELECT COUNT(*)
            FROM email_priority ep
            WHERE ep.priority_score > 0.7
            """
        )
        urgent_result = await self._session.execute(urgent_query)
        urgent_emails = urgent_result.scalar() or 0

        # From real people (using priority context sender_importance)
        real_people_query = text(
            """
            SELECT COUNT(*)
            FROM priority_contexts pc
            WHERE pc.sender_importance > 0
            """
        )
        real_people_result = await self._session.execute(real_people_query)
        from_real_people = real_people_result.scalar() or 0

        # Average priority score
        avg_query = text("SELECT AVG(priority_score) FROM email_priority")
        avg_result = await self._session.execute(avg_query)
        avg_priority = avg_result.scalar()

        # Oldest unanswered email age
        oldest_query = text(
            """
            SELECT EXTRACT(EPOCH FROM (NOW() - MIN(e.date_parsed))) / 3600
            FROM emails e
            LEFT JOIN emails reply ON reply.in_reply_to = e.message_id
            WHERE e.is_sent = false
            AND e.action = 'received'
            AND reply.id IS NULL
            AND e.date_parsed > NOW() - INTERVAL '30 days'
            """
        )
        oldest_result = await self._session.execute(oldest_query)
        oldest_unanswered = oldest_result.scalar()

        return InboxStats(
            total_emails=total_emails,
            unread_count=0,  # Would need label tracking for accurate count
            pending_tasks=pending_tasks,
            urgent_emails=urgent_emails,
            from_real_people=from_real_people,
            avg_priority_score=float(avg_priority) if avg_priority else None,
            oldest_unanswered_hours=float(oldest_unanswered) if oldest_unanswered else None,
        )
