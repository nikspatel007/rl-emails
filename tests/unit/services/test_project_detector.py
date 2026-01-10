"""Tests for ProjectDetectorService."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest import mock

import pytest

from rl_emails.models.cluster_metadata import ClusterMetadata
from rl_emails.services.project_detector import (
    ProjectDetectionConfig,
    ProjectDetectionResult,
    ProjectDetectionSummary,
    ProjectDetectorService,
)


class TestProjectDetectionConfig:
    """Tests for ProjectDetectionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ProjectDetectionConfig()
        assert config.min_size == 5
        assert config.min_engagement == 0.2
        assert config.stale_days == 14
        assert config.completed_days == 30
        assert config.min_coherence == 0.5
        assert config.min_participants == 2

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ProjectDetectionConfig(
            min_size=10,
            min_engagement=0.5,
            stale_days=7,
            completed_days=21,
            min_coherence=0.7,
            min_participants=3,
        )
        assert config.min_size == 10
        assert config.min_engagement == 0.5
        assert config.stale_days == 7
        assert config.completed_days == 21
        assert config.min_coherence == 0.7
        assert config.min_participants == 3


class TestProjectDetectionResult:
    """Tests for ProjectDetectionResult dataclass."""

    def test_project_result(self) -> None:
        """Test result for detected project."""
        result = ProjectDetectionResult(
            cluster_id=1,
            is_project=True,
            status="active",
            confidence=0.85,
            reason="size=50, engagement=75.0%",
        )
        assert result.cluster_id == 1
        assert result.is_project is True
        assert result.status == "active"
        assert result.confidence == 0.85

    def test_non_project_result(self) -> None:
        """Test result for non-project cluster."""
        result = ProjectDetectionResult(
            cluster_id=2,
            is_project=False,
            status=None,
            confidence=0.2,
            reason="size=2",
        )
        assert result.is_project is False
        assert result.status is None


class TestProjectDetectionSummary:
    """Tests for ProjectDetectionSummary dataclass."""

    def test_summary(self) -> None:
        """Test detection summary."""
        results = [
            ProjectDetectionResult(1, True, "active", 0.8, "test"),
            ProjectDetectionResult(2, True, "stale", 0.6, "test"),
            ProjectDetectionResult(3, False, None, 0.2, "test"),
        ]
        summary = ProjectDetectionSummary(
            clusters_analyzed=3,
            projects_detected=2,
            active_projects=1,
            stale_projects=1,
            completed_projects=0,
            detection_results=results,
        )
        assert summary.clusters_analyzed == 3
        assert summary.projects_detected == 2
        assert summary.active_projects == 1
        assert len(summary.detection_results) == 3


class TestProjectDetectorService:
    """Tests for ProjectDetectorService."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.fixture
    def custom_config_service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance with custom config."""
        config = ProjectDetectionConfig(
            min_size=3,
            min_engagement=0.1,
            stale_days=7,
        )
        return ProjectDetectorService(session=mock_session, config=config)

    def test_init_default_config(self, mock_session: mock.MagicMock) -> None:
        """Test service initialization with default config."""
        service = ProjectDetectorService(session=mock_session)
        assert service.config.min_size == 5
        assert service.config.min_engagement == 0.2

    def test_init_custom_config(self, mock_session: mock.MagicMock) -> None:
        """Test service initialization with custom config."""
        config = ProjectDetectionConfig(min_size=10)
        service = ProjectDetectorService(session=mock_session, config=config)
        assert service.config.min_size == 10


class TestProjectDetectorServiceDetermineStatus:
    """Tests for _determine_project_status method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    def test_status_active(self, service: ProjectDetectorService) -> None:
        """Test active status for recent activity."""
        recent = datetime.now(UTC) - timedelta(days=3)
        status = service._determine_project_status(recent)
        assert status == "active"

    def test_status_stale(self, service: ProjectDetectorService) -> None:
        """Test stale status for older activity."""
        older = datetime.now(UTC) - timedelta(days=20)
        status = service._determine_project_status(older)
        assert status == "stale"

    def test_status_completed(self, service: ProjectDetectorService) -> None:
        """Test completed status for very old activity."""
        very_old = datetime.now(UTC) - timedelta(days=45)
        status = service._determine_project_status(very_old)
        assert status == "completed"

    def test_status_none_activity(self, service: ProjectDetectorService) -> None:
        """Test stale status for None activity."""
        status = service._determine_project_status(None)
        assert status == "stale"

    def test_status_naive_datetime(self, service: ProjectDetectorService) -> None:
        """Test status with naive datetime (no timezone)."""
        naive = datetime.now() - timedelta(days=5)
        status = service._determine_project_status(naive)
        assert status == "active"


class TestProjectDetectorServiceCalculateScore:
    """Tests for _calculate_project_score method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    def test_high_score_cluster(self, service: ProjectDetectorService) -> None:
        """Test scoring a high-engagement cluster."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=50,
            pct_replied=75.0,
            coherence_score=0.85,
        )
        activity = {
            "last_activity": datetime.now(UTC) - timedelta(days=2),
            "email_count": 50,
            "participant_count": 8,
        }
        score, reason = service._calculate_project_score(metadata, activity)
        assert score > 0.5
        assert "size=50" in reason
        assert "engagement=75.0%" in reason

    def test_low_score_cluster(self, service: ProjectDetectorService) -> None:
        """Test scoring a low-engagement cluster."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=2,
            pct_replied=5.0,
        )
        activity = {
            "last_activity": datetime.now(UTC) - timedelta(days=100),
            "email_count": 2,
            "participant_count": 1,
        }
        score, reason = service._calculate_project_score(metadata, activity)
        assert score < 0.3

    def test_score_with_none_values(self, service: ProjectDetectorService) -> None:
        """Test scoring with None values."""
        metadata = ClusterMetadata(
            dimension="content",
            cluster_id=1,
            size=0,
        )
        activity = {
            "last_activity": None,
            "email_count": None,
            "participant_count": None,
        }
        score, reason = service._calculate_project_score(metadata, activity)
        assert score == 0.0


class TestProjectDetectorServiceDetectProjects:
    """Tests for detect_projects method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_detect_no_clusters(self, service: ProjectDetectorService) -> None:
        """Test detection with no clusters."""
        with mock.patch.object(service.repository, "list_by_dimension", return_value=([], 0)):
            summary = await service.detect_projects()
            assert summary.clusters_analyzed == 0
            assert summary.projects_detected == 0

    @pytest.mark.asyncio
    async def test_detect_with_projects(self, service: ProjectDetectorService) -> None:
        """Test detection finding projects."""
        clusters = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                pct_replied=75.0,
            ),
            ClusterMetadata(
                dimension="content",
                cluster_id=2,
                size=3,
                pct_replied=10.0,
            ),
        ]

        activity_data = {
            "last_activity": datetime.now(UTC) - timedelta(days=2),
            "email_count": 50,
            "participant_count": 8,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 2)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
            mock.patch.object(service.repository, "mark_as_project"),
        ):
            summary = await service.detect_projects()
            assert summary.clusters_analyzed == 2
            # At least one should be detected as project
            assert len(summary.detection_results) == 2


class TestProjectDetectorServiceUpdateStatuses:
    """Tests for update_project_statuses method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_update_no_projects(self, service: ProjectDetectorService) -> None:
        """Test update with no projects."""
        with mock.patch.object(service.repository, "list_projects", return_value=([], 0)):
            changes = await service.update_project_statuses()
            assert changes["unchanged"] == 0
            assert changes["to_active"] == 0

    @pytest.mark.asyncio
    async def test_update_status_changes(self, service: ProjectDetectorService) -> None:
        """Test updating project statuses."""
        projects = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                is_project=True,
                project_status="active",
            ),
        ]

        # Simulate activity that would make it stale
        activity_data = {
            "last_activity": datetime.now(UTC) - timedelta(days=20),
            "email_count": 50,
            "participant_count": 5,
        }

        with (
            mock.patch.object(service.repository, "list_projects", return_value=(projects, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
            mock.patch.object(service.repository, "mark_as_project"),
        ):
            changes = await service.update_project_statuses()
            # Status should change from active to stale
            assert changes["to_stale"] == 1


class TestProjectDetectorServiceGetSummary:
    """Tests for get_project_summary method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_get_summary(self, service: ProjectDetectorService) -> None:
        """Test getting project summary."""
        mock_stats = {
            "total_clusters": 50,
            "labeled_clusters": 30,
            "project_clusters": 10,
            "active_projects": 5,
        }

        active_projects = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                auto_label="Q4 Review",
                pct_replied=75.0,
            ),
        ]

        with (
            mock.patch.object(service.repository, "get_stats", return_value=mock_stats),
            mock.patch.object(
                service.repository, "list_projects", return_value=(active_projects, 1)
            ),
        ):
            summary = await service.get_project_summary()

            assert summary["total_projects"] == 10
            assert summary["active_projects"] == 5
            assert summary["total_content_clusters"] == 50
            assert len(summary["top_active_projects"]) == 1  # type: ignore[arg-type]


class TestProjectDetectorServiceEnrichMetadata:
    """Tests for enrich_cluster_metadata method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_enrich_metadata(self, service: ProjectDetectorService) -> None:
        """Test enriching cluster metadata."""
        clusters = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
            ),
        ]

        activity_data = {
            "last_activity": datetime.now(UTC),
            "email_count": 50,
            "participant_count": 8,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
        ):
            updated = await service.enrich_cluster_metadata()
            assert updated == 1
            assert clusters[0].participant_count == 8


class TestGetClusterActivity:
    """Tests for _get_cluster_activity method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_get_cluster_activity_found(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting activity when cluster found."""
        mock_row = (
            datetime.now(UTC),  # last_activity
            50,  # email_count
            8,  # participant_count
        )
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        activity = await service._get_cluster_activity(cluster_id=1)

        assert activity["last_activity"] == mock_row[0]
        assert activity["email_count"] == 50
        assert activity["participant_count"] == 8

    @pytest.mark.asyncio
    async def test_get_cluster_activity_not_found(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting activity when cluster not found."""
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        activity = await service._get_cluster_activity(cluster_id=999)

        assert activity["last_activity"] is None
        assert activity["email_count"] == 0
        assert activity["participant_count"] == 0

    @pytest.mark.asyncio
    async def test_get_cluster_activity_null_values(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting activity with null values."""
        mock_row = (None, None, None)
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        activity = await service._get_cluster_activity(cluster_id=1)

        assert activity["last_activity"] is None
        assert activity["email_count"] == 0
        assert activity["participant_count"] == 0

    @pytest.mark.asyncio
    async def test_get_cluster_activity_with_user_id(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting activity with user_id filter."""
        import uuid

        user_id = uuid.uuid4()
        mock_row = (datetime.now(UTC), 25, 5)
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        activity = await service._get_cluster_activity(cluster_id=1, user_id=user_id)

        assert activity["email_count"] == 25
        assert activity["participant_count"] == 5


class TestDetectProjectsAdvanced:
    """Advanced tests for detect_projects method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_detect_projects_with_non_datetime_activity(
        self, service: ProjectDetectorService
    ) -> None:
        """Test detection with non-datetime last_activity."""
        clusters = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                pct_replied=75.0,
            ),
        ]

        # Activity with None for last_activity
        activity_data = {
            "last_activity": None,
            "email_count": 50,
            "participant_count": 8,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
            mock.patch.object(service, "_calculate_project_score", return_value=(0.8, "test")),
            mock.patch.object(service.repository, "mark_as_project"),
        ):
            summary = await service.detect_projects()
            assert summary.projects_detected == 1
            # Status should be stale since last_activity is not datetime
            assert summary.stale_projects == 1

    @pytest.mark.asyncio
    async def test_detect_projects_updates_last_activity(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test detection updates last_activity_at."""
        last_activity = datetime.now(UTC) - timedelta(days=2)
        clusters = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                pct_replied=75.0,
            ),
        ]

        activity_data = {
            "last_activity": last_activity,
            "email_count": 50,
            "participant_count": 8,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
            mock.patch.object(service, "_calculate_project_score", return_value=(0.8, "test")),
            mock.patch.object(service.repository, "mark_as_project"),
        ):
            await service.detect_projects()
            # Check that last_activity_at was updated
            assert clusters[0].last_activity_at == last_activity

    @pytest.mark.asyncio
    async def test_detect_projects_completed_status(self, service: ProjectDetectorService) -> None:
        """Test detection with completed project."""
        clusters = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                pct_replied=75.0,
            ),
        ]

        # Very old activity
        activity_data = {
            "last_activity": datetime.now(UTC) - timedelta(days=60),
            "email_count": 50,
            "participant_count": 8,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
            mock.patch.object(service, "_calculate_project_score", return_value=(0.8, "test")),
            mock.patch.object(service.repository, "mark_as_project"),
        ):
            summary = await service.detect_projects()
            assert summary.completed_projects == 1


class TestUpdateProjectStatusesAdvanced:
    """Advanced tests for update_project_statuses method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_update_unchanged_status(self, service: ProjectDetectorService) -> None:
        """Test update when status remains unchanged."""
        projects = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                is_project=True,
                project_status="active",
            ),
        ]

        # Activity that keeps it active
        activity_data = {
            "last_activity": datetime.now(UTC) - timedelta(days=2),
            "email_count": 50,
            "participant_count": 5,
        }

        with (
            mock.patch.object(service.repository, "list_projects", return_value=(projects, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
        ):
            changes = await service.update_project_statuses()
            assert changes["unchanged"] == 1
            assert changes["to_active"] == 0

    @pytest.mark.asyncio
    async def test_update_non_datetime_activity(self, service: ProjectDetectorService) -> None:
        """Test update when last_activity is not datetime."""
        projects = [
            ClusterMetadata(
                dimension="content",
                cluster_id=1,
                size=50,
                is_project=True,
                project_status="active",
            ),
        ]

        # Activity with None for last_activity
        activity_data = {
            "last_activity": None,
            "email_count": 50,
            "participant_count": 5,
        }

        with (
            mock.patch.object(service.repository, "list_projects", return_value=(projects, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
            mock.patch.object(service.repository, "mark_as_project"),
        ):
            changes = await service.update_project_statuses()
            # Status should change to stale
            assert changes["to_stale"] == 1


class TestGetTemporalActivity:
    """Tests for _get_temporal_activity method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_get_temporal_activity_found(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting temporal activity when found."""
        now = datetime.now(UTC)
        mock_row = (
            now - timedelta(days=30),  # first_activity
            now - timedelta(days=1),  # last_activity
            100,  # total_count
            15,  # emails_7d
            50,  # emails_30d
            10,  # emails_prev_7d
        )
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        temporal = await service._get_temporal_activity(cluster_id=1)

        assert temporal["first_activity"] == mock_row[0]
        assert temporal["last_activity"] == mock_row[1]
        assert temporal["total_count"] == 100
        assert temporal["emails_7d"] == 15
        assert temporal["emails_30d"] == 50
        assert temporal["emails_prev_7d"] == 10

    @pytest.mark.asyncio
    async def test_get_temporal_activity_not_found(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting temporal activity when not found."""
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        temporal = await service._get_temporal_activity(cluster_id=999)

        assert temporal["first_activity"] is None
        assert temporal["last_activity"] is None
        assert temporal["total_count"] == 0
        assert temporal["emails_7d"] == 0

    @pytest.mark.asyncio
    async def test_get_temporal_activity_null_values(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting temporal activity with null values."""
        mock_row = (None, None, None, None, None, None)
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        temporal = await service._get_temporal_activity(cluster_id=1)

        assert temporal["first_activity"] is None
        assert temporal["total_count"] == 0

    @pytest.mark.asyncio
    async def test_get_temporal_activity_with_user_id(
        self, service: ProjectDetectorService, mock_session: mock.MagicMock
    ) -> None:
        """Test getting temporal activity with user_id."""
        import uuid

        user_id = uuid.uuid4()
        mock_row = (datetime.now(UTC), datetime.now(UTC), 50, 10, 25, 8)
        mock_result = mock.MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_session.execute.return_value = mock_result

        temporal = await service._get_temporal_activity(cluster_id=1, user_id=user_id)

        assert temporal["total_count"] == 50
        mock_session.execute.assert_called_once()


class TestCalculateTemporalStrength:
    """Tests for _calculate_temporal_strength method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        return mock.MagicMock()

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    def test_high_activity_temporal_strength(self, service: ProjectDetectorService) -> None:
        """Test temporal strength with high activity."""
        temporal = {
            "first_activity": datetime.now(UTC) - timedelta(days=30),
            "last_activity": datetime.now(UTC) - timedelta(hours=1),
            "total_count": 100,
            "emails_7d": 20,
            "emails_30d": 80,
            "emails_prev_7d": 15,
        }

        strength, velocity, trend = service._calculate_temporal_strength(temporal)

        assert strength > 0.5  # High strength
        assert velocity > 0  # Has velocity
        assert trend > 0  # Positive trend (more activity this week)

    def test_no_activity_temporal_strength(self, service: ProjectDetectorService) -> None:
        """Test temporal strength with no activity."""
        temporal = {
            "first_activity": None,
            "last_activity": None,
            "total_count": 0,
            "emails_7d": 0,
            "emails_30d": 0,
            "emails_prev_7d": 0,
        }

        strength, velocity, trend = service._calculate_temporal_strength(temporal)

        assert strength < 0.2  # Low strength
        assert velocity == 0
        assert trend == 0

    def test_declining_activity_temporal_strength(self, service: ProjectDetectorService) -> None:
        """Test temporal strength with declining activity."""
        temporal = {
            "first_activity": datetime.now(UTC) - timedelta(days=60),
            "last_activity": datetime.now(UTC) - timedelta(days=10),
            "total_count": 100,
            "emails_7d": 2,
            "emails_30d": 30,
            "emails_prev_7d": 10,  # More activity before
        }

        strength, velocity, trend = service._calculate_temporal_strength(temporal)

        assert trend < 0  # Negative trend

    def test_new_activity_temporal_strength(self, service: ProjectDetectorService) -> None:
        """Test temporal strength with new activity where there was none."""
        temporal = {
            "first_activity": datetime.now(UTC) - timedelta(days=3),
            "last_activity": datetime.now(UTC) - timedelta(hours=2),
            "total_count": 10,
            "emails_7d": 10,
            "emails_30d": 10,
            "emails_prev_7d": 0,  # No prior activity
        }

        strength, velocity, trend = service._calculate_temporal_strength(temporal)

        assert trend == 1.0  # Maximum positive trend

    def test_naive_datetime_temporal_strength(self, service: ProjectDetectorService) -> None:
        """Test temporal strength with naive datetime."""
        temporal = {
            "first_activity": datetime.now() - timedelta(days=30),
            "last_activity": datetime.now() - timedelta(hours=1),
            "total_count": 50,
            "emails_7d": 10,
            "emails_30d": 40,
            "emails_prev_7d": 8,
        }

        strength, velocity, trend = service._calculate_temporal_strength(temporal)

        # Should handle naive datetime correctly
        assert strength > 0


class TestComputeTemporalStrength:
    """Tests for compute_temporal_strength method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_compute_temporal_strength(self, service: ProjectDetectorService) -> None:
        """Test computing temporal strength for clusters."""
        now = datetime.now(UTC)
        clusters = [
            ClusterMetadata(dimension="content", cluster_id=1, size=50),
            ClusterMetadata(dimension="content", cluster_id=2, size=30),
        ]

        temporal_data = {
            "first_activity": now - timedelta(days=30),
            "last_activity": now - timedelta(days=1),
            "total_count": 100,
            "emails_7d": 15,
            "emails_30d": 50,
            "emails_prev_7d": 10,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 2)),
            mock.patch.object(service, "_get_temporal_activity", return_value=temporal_data),
        ):
            updated = await service.compute_temporal_strength()

            assert updated == 2
            assert clusters[0].temporal_strength is not None
            assert clusters[0].activity_velocity is not None
            assert clusters[0].activity_trend is not None
            assert clusters[0].first_activity_at == temporal_data["first_activity"]
            assert clusters[0].last_activity_at == temporal_data["last_activity"]
            assert clusters[0].emails_last_7d == 15
            assert clusters[0].emails_last_30d == 50

    @pytest.mark.asyncio
    async def test_compute_temporal_strength_with_user_id(
        self, service: ProjectDetectorService
    ) -> None:
        """Test computing temporal strength with user_id."""
        import uuid

        user_id = uuid.uuid4()
        clusters = [ClusterMetadata(dimension="content", cluster_id=1, size=50)]

        temporal_data = {
            "first_activity": datetime.now(UTC) - timedelta(days=10),
            "last_activity": datetime.now(UTC) - timedelta(hours=1),
            "total_count": 25,
            "emails_7d": 10,
            "emails_30d": 25,
            "emails_prev_7d": 5,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_temporal_activity", return_value=temporal_data),
        ):
            updated = await service.compute_temporal_strength(user_id=user_id)

            assert updated == 1

    @pytest.mark.asyncio
    async def test_compute_temporal_strength_no_clusters(
        self, service: ProjectDetectorService
    ) -> None:
        """Test computing temporal strength with no clusters."""
        with mock.patch.object(service.repository, "list_by_dimension", return_value=([], 0)):
            updated = await service.compute_temporal_strength()
            assert updated == 0

    @pytest.mark.asyncio
    async def test_compute_temporal_strength_non_integer_values(
        self, service: ProjectDetectorService
    ) -> None:
        """Test computing temporal strength when emails_7d/prev are not integers."""
        clusters = [ClusterMetadata(dimension="content", cluster_id=1, size=50)]

        # Return non-integer values for emails_7d and emails_prev_7d
        temporal_data = {
            "first_activity": datetime.now(UTC) - timedelta(days=10),
            "last_activity": datetime.now(UTC) - timedelta(hours=1),
            "total_count": 25,
            "emails_7d": None,  # Non-integer
            "emails_30d": None,  # Non-integer
            "emails_prev_7d": "not a number",  # Non-integer
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_temporal_activity", return_value=temporal_data),
        ):
            updated = await service.compute_temporal_strength()

            assert updated == 1
            # temporal_strength should still be computed (defaults to 0.0 for trend)
            assert clusters[0].temporal_strength is not None
            # emails_last_7d should NOT be set since it's not an integer
            assert clusters[0].emails_last_7d is None
            assert clusters[0].emails_last_30d is None

    @pytest.mark.asyncio
    async def test_compute_temporal_strength_non_datetime_values(
        self, service: ProjectDetectorService
    ) -> None:
        """Test computing temporal strength when activity dates are not datetime."""
        clusters = [ClusterMetadata(dimension="content", cluster_id=1, size=50)]

        # Return non-datetime values for first_activity and last_activity
        temporal_data = {
            "first_activity": "2024-01-01",  # String, not datetime
            "last_activity": None,  # None, not datetime
            "total_count": 25,
            "emails_7d": 10,
            "emails_30d": 25,
            "emails_prev_7d": 5,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_temporal_activity", return_value=temporal_data),
        ):
            updated = await service.compute_temporal_strength()

            assert updated == 1
            # first_activity_at and last_activity_at should NOT be set
            assert clusters[0].first_activity_at is None
            assert clusters[0].last_activity_at is None
            # But numeric fields should still be set
            assert clusters[0].emails_last_7d == 10
            assert clusters[0].emails_last_30d == 25


class TestEnrichMetadataAdvanced:
    """Advanced tests for enrich_cluster_metadata method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def service(self, mock_session: mock.MagicMock) -> ProjectDetectorService:
        """Create a service instance."""
        return ProjectDetectorService(session=mock_session)

    @pytest.mark.asyncio
    async def test_enrich_metadata_with_null_activity(
        self, service: ProjectDetectorService
    ) -> None:
        """Test enriching with null activity values."""
        clusters = [
            ClusterMetadata(dimension="content", cluster_id=1, size=10),
        ]

        activity_data = {
            "last_activity": None,
            "email_count": 0,
            "participant_count": None,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
        ):
            updated = await service.enrich_cluster_metadata()
            assert updated == 1
            # participant_count should not be set since it's None
            assert clusters[0].participant_count is None

    @pytest.mark.asyncio
    async def test_enrich_metadata_with_user_id(self, service: ProjectDetectorService) -> None:
        """Test enriching with user_id filter."""
        import uuid

        user_id = uuid.uuid4()
        clusters = [ClusterMetadata(dimension="content", cluster_id=1, size=20)]

        activity_data = {
            "last_activity": datetime.now(UTC),
            "email_count": 20,
            "participant_count": 5,
        }

        with (
            mock.patch.object(service.repository, "list_by_dimension", return_value=(clusters, 1)),
            mock.patch.object(service, "_get_cluster_activity", return_value=activity_data),
        ):
            updated = await service.enrich_cluster_metadata(user_id=user_id)
            assert updated == 1
            assert clusters[0].participant_count == 5
