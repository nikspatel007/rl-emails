# Future Phases

Items deferred from Phase 3 for future implementation.

---

## Background Pipeline Worker

### Story
As a system, I need background workers to process emails through the ML pipeline so that classification and priority scores are always up to date.

### Deliverables
1. Pipeline worker service
2. Job queue with PostgreSQL
3. Worker health monitoring
4. Batch processing optimization
5. Error handling and retries

### Implementation Design

```python
# src/rl_emails/workers/pipeline_worker.py
class PipelineWorker:
    """Background worker for ML pipeline processing."""

    async def run(self) -> None:
        """Run the worker loop."""
        while self._running:
            job = await self._queue.dequeue(timeout=5.0)
            if job:
                await self._process_job(job)

    async def _process_job(self, job: PipelineJob) -> None:
        """Process a pipeline job."""
        # Run pipeline stages
        result = await self._processor.process_batch(
            emails=job.emails,
            run_embeddings=True,
            run_llm=True,
        )

        # Extract entities after pipeline
        await self._entity_extractor.extract_all(job.user_id)

        await self._queue.complete(job.id, result)
```

### Acceptance Criteria

- [ ] Worker processes jobs from queue
- [ ] PostgreSQL queue with SKIP LOCKED
- [ ] Job retry with exponential backoff
- [ ] Entity extraction runs after pipeline
- [ ] Graceful shutdown
- [ ] 100% test coverage on new code

---

## Scheduled Sync & Watch Renewal

### Story
As a system administrator, I need scheduled tasks to maintain sync freshness and watch renewals.

### Deliverables
1. Scheduled task framework
2. Gmail watch renewal job
3. Periodic full sync job
4. Stale connection detection
5. Admin monitoring endpoints

### Implementation Design

```python
# src/rl_emails/workers/scheduler.py
class Scheduler:
    """Simple async scheduler for periodic tasks."""

    def add_job(self, name: str, func: Callable, interval: timedelta) -> None:
        """Add a scheduled job."""
        self._jobs.append(ScheduledJob(name=name, func=func, interval=interval))

    async def start(self) -> None:
        """Start the scheduler."""
        while self._running:
            for job in self._jobs:
                if self._should_run(job):
                    await job.func()
            await asyncio.sleep(60)
```

### Acceptance Criteria

- [ ] Scheduler runs periodic tasks
- [ ] Gmail watch renewal before expiration
- [ ] Periodic full sync for consistency
- [ ] Admin endpoint for job status
- [ ] 100% test coverage on new code

---

## Integration Testing & Documentation

### Story
As a team, we need comprehensive integration tests and documentation so that the system is production-ready.

### Deliverables
1. End-to-end integration tests
2. API documentation (OpenAPI + guides)
3. Deployment documentation
4. Performance benchmarks
5. Security audit checklist

### Test Scenarios

```python
# tests/integration/test_full_flow.py
class TestFullUserFlow:
    """End-to-end integration tests."""

    async def test_user_onboarding_flow(self, client, mock_clerk, mock_gmail):
        """Test complete user onboarding."""
        # 1. Authenticate with Clerk
        # 2. Connect Gmail
        # 3. Trigger sync
        # 4. Wait for pipeline completion
        # 5. Verify projects detected
        # 6. Verify tasks extracted
        # 7. Check priority inbox

    async def test_real_time_sync(self, client, mock_clerk, mock_gmail):
        """Test real-time webhook sync."""
        # 1. Setup Gmail connection
        # 2. Simulate webhook notification
        # 3. Verify new email appears
        # 4. Verify tasks updated
```

### Acceptance Criteria

- [ ] Integration tests cover all user flows
- [ ] API documentation complete
- [ ] Deployment guide with Docker examples
- [ ] Performance benchmarks documented
- [ ] Security checklist completed
- [ ] 100% coverage on business logic

---

## Additional Future Features

### IMAP Provider
Support for non-Gmail email via IMAP protocol.

### Outlook Provider
Microsoft Graph API integration for Office 365/Outlook.com accounts.

### Mobile-Optimized Endpoints
Response compression, field selection, and pagination optimizations for mobile clients.

### Push Notifications
Mobile push notifications for urgent emails via Firebase/APNs.

### ML Model Training
Custom model training from user feedback to improve classification accuracy.

### Multi-Organization Support
Shared workspaces and team collaboration features.

### Email Compose & Send
Draft composition, send scheduling, and template support.

### Analytics Dashboard
Usage metrics, email patterns, and productivity insights.
