"""Tests for ClusterMetadataRepository."""

from __future__ import annotations

from unittest import mock
from uuid import uuid4

import pytest

from rl_emails.models.cluster_metadata import ClusterMetadata
from rl_emails.repositories.cluster_metadata import ClusterMetadataRepository
from rl_emails.schemas.cluster_metadata import ClusterMetadataCreate


class TestClusterMetadataRepository:
    """Tests for ClusterMetadataRepository."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        session.add = mock.MagicMock()
        session.flush = mock.AsyncMock()
        session.refresh = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    def test_init(self, mock_session: mock.MagicMock) -> None:
        """Test repository initialization."""
        repo = ClusterMetadataRepository(mock_session)
        assert repo.session is mock_session


class TestClusterMetadataRepositoryCreate:
    """Tests for create method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        session.add = mock.MagicMock()
        session.flush = mock.AsyncMock()
        session.refresh = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_create_minimal(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test creating with minimal fields."""
        data = ClusterMetadataCreate(
            dimension="content",
            cluster_id=0,
        )
        await repository.create(data)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_awaited_once()
        mock_session.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_with_user(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test creating with user_id."""
        user_id = uuid4()
        data = ClusterMetadataCreate(
            dimension="content",
            cluster_id=0,
            user_id=user_id,
            size=50,
        )
        await repository.create(data)
        mock_session.add.assert_called_once()


class TestClusterMetadataRepositoryUpsert:
    """Tests for upsert method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        session.add = mock.MagicMock()
        session.flush = mock.AsyncMock()
        session.refresh = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_upsert_new(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test upserting a new record."""
        # Mock get_by_dimension_and_cluster to return None (not found)
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        data = ClusterMetadataCreate(
            dimension="content",
            cluster_id=5,
            size=100,
        )
        await repository.upsert(data)
        # Should create new
        mock_session.add.assert_called()

    @pytest.mark.asyncio
    async def test_upsert_existing(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test upserting an existing record."""
        existing = ClusterMetadata(
            dimension="content",
            cluster_id=5,
            size=50,
            is_project=False,
        )
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute.return_value = mock_result

        data = ClusterMetadataCreate(
            dimension="content",
            cluster_id=5,
            size=100,
        )
        await repository.upsert(data)
        # Should update existing - size was updated
        assert existing.size == 100


class TestClusterMetadataRepositoryGet:
    """Tests for get_by_dimension_and_cluster method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_get_found(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test getting an existing record."""
        existing = ClusterMetadata(
            dimension="content",
            cluster_id=5,
            size=50,
            is_project=False,
        )
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_dimension_and_cluster(
            dimension="content",
            cluster_id=5,
        )
        assert result is existing

    @pytest.mark.asyncio
    async def test_get_not_found(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test getting a non-existent record."""
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_dimension_and_cluster(
            dimension="content",
            cluster_id=999,
        )
        assert result is None


class TestClusterMetadataRepositoryList:
    """Tests for list methods."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_list_by_dimension(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test listing by dimension."""
        clusters = [
            ClusterMetadata(dimension="content", cluster_id=0, size=10, is_project=False),
            ClusterMetadata(dimension="content", cluster_id=1, size=20, is_project=False),
        ]

        # Mock for count query - returns scalar
        mock_count_result = mock.MagicMock()
        mock_count_result.scalar.return_value = 2

        # Mock for data query
        mock_data_result = mock.MagicMock()
        mock_scalars = mock.MagicMock()
        mock_scalars.all.return_value = clusters
        mock_data_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_data_result]

        result, total = await repository.list_by_dimension(
            dimension="content",
            limit=50,
        )
        assert len(result) == 2
        assert total == 2

    @pytest.mark.asyncio
    async def test_list_unlabeled(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test listing unlabeled clusters."""
        clusters = [
            ClusterMetadata(dimension="content", cluster_id=0, size=10, is_project=False),
        ]

        mock_result = mock.MagicMock()
        mock_scalars = mock.MagicMock()
        mock_scalars.all.return_value = clusters
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.list_unlabeled(limit=50)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_projects(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test listing projects."""
        projects = [
            ClusterMetadata(
                dimension="content",
                cluster_id=0,
                size=50,
                is_project=True,
                project_status="active",
            ),
        ]

        # Mock for count query
        mock_count_result = mock.MagicMock()
        mock_count_result.scalar.return_value = 1

        # Mock for data query
        mock_data_result = mock.MagicMock()
        mock_scalars = mock.MagicMock()
        mock_scalars.all.return_value = projects
        mock_data_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_data_result]

        result, total = await repository.list_projects(limit=50)
        assert len(result) == 1
        assert result[0].is_project is True


class TestClusterMetadataRepositoryUpdate:
    """Tests for update methods."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_update_label(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test updating cluster label."""
        mock_result = mock.MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repository.update_label(
            dimension="content",
            cluster_id=5,
            label="Q4 Budget Review",
        )
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_mark_as_project(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test marking cluster as project."""
        mock_result = mock.MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repository.mark_as_project(
            dimension="content",
            cluster_id=5,
            status="active",
        )
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert result is True


class TestClusterMetadataRepositoryStats:
    """Tests for get_stats method."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_get_stats(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test getting dimension stats."""
        # Create mock row with named attributes
        mock_row = mock.MagicMock()
        mock_row.total_clusters = 50
        mock_row.total_emails = 1000
        mock_row.avg_cluster_size = 20.0
        mock_row.largest_cluster = 100
        mock_row.smallest_cluster = 5
        mock_row.labeled_clusters = 30
        mock_row.project_clusters = 10
        mock_row.active_projects = 5

        mock_result = mock.MagicMock()
        mock_result.one.return_value = mock_row
        mock_session.execute.return_value = mock_result

        stats = await repository.get_stats("content")
        assert stats["total_clusters"] == 50
        assert stats["total_emails"] == 1000
        assert stats["avg_cluster_size"] == 20.0
        assert stats["labeled_clusters"] == 30
        assert stats["project_clusters"] == 10


class TestClusterMetadataRepositoryDelete:
    """Tests for delete methods."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_delete_by_user(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test deleting by user."""
        user_id = uuid4()
        mock_result = mock.MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result

        result = await repository.delete_by_user(user_id)
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert result == 5

    @pytest.mark.asyncio
    async def test_truncate_by_dimension(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test truncating by dimension."""
        mock_result = mock.MagicMock()
        mock_result.rowcount = 10
        mock_session.execute.return_value = mock_result

        result = await repository.truncate_by_dimension("content")
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert result == 10


class TestClusterMetadataRepositoryAdditional:
    """Additional tests for missing coverage."""

    @pytest.fixture
    def mock_session(self) -> mock.MagicMock:
        """Create a mock async session."""
        session = mock.MagicMock()
        session.execute = mock.AsyncMock()
        session.commit = mock.AsyncMock()
        session.add = mock.MagicMock()
        session.refresh = mock.AsyncMock()
        return session

    @pytest.fixture
    def repository(self, mock_session: mock.MagicMock) -> ClusterMetadataRepository:
        """Create a repository instance."""
        return ClusterMetadataRepository(mock_session)

    @pytest.mark.asyncio
    async def test_get_by_id_found(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test getting metadata by ID when found."""
        existing = ClusterMetadata(
            dimension="content",
            cluster_id=5,
            size=50,
            is_project=False,
        )
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(1)
        assert result is existing

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test getting metadata by ID when not found."""
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_id(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_dimension_with_user_id(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test getting with user_id filter."""
        user_id = uuid4()
        existing = ClusterMetadata(
            dimension="content",
            cluster_id=5,
            user_id=user_id,
            size=50,
            is_project=False,
        )
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute.return_value = mock_result

        result = await repository.get_by_dimension_and_cluster(
            dimension="content",
            cluster_id=5,
            user_id=user_id,
        )
        assert result is existing

    @pytest.mark.asyncio
    async def test_list_by_dimension_with_user_and_project_filter(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test listing with user_id and is_project filters."""
        user_id = uuid4()
        clusters = [
            ClusterMetadata(
                dimension="content",
                cluster_id=0,
                user_id=user_id,
                size=10,
                is_project=True,
            ),
        ]

        mock_count_result = mock.MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_data_result = mock.MagicMock()
        mock_scalars = mock.MagicMock()
        mock_scalars.all.return_value = clusters
        mock_data_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_data_result]

        result, total = await repository.list_by_dimension(
            dimension="content",
            user_id=user_id,
            is_project=True,
            limit=50,
        )
        assert len(result) == 1
        assert total == 1

    @pytest.mark.asyncio
    async def test_list_projects_with_user_and_status(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test listing projects with user_id and status filters."""
        user_id = uuid4()
        projects = [
            ClusterMetadata(
                dimension="content",
                cluster_id=0,
                user_id=user_id,
                size=50,
                is_project=True,
                project_status="active",
            ),
        ]

        mock_count_result = mock.MagicMock()
        mock_count_result.scalar.return_value = 1

        mock_data_result = mock.MagicMock()
        mock_scalars = mock.MagicMock()
        mock_scalars.all.return_value = projects
        mock_data_result.scalars.return_value = mock_scalars

        mock_session.execute.side_effect = [mock_count_result, mock_data_result]

        result, total = await repository.list_projects(
            user_id=user_id,
            status="active",
            limit=50,
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_unlabeled_with_dimension_and_user(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test listing unlabeled with dimension and user filters."""
        user_id = uuid4()
        clusters = [
            ClusterMetadata(
                dimension="content",
                cluster_id=0,
                user_id=user_id,
                size=10,
                is_project=False,
            ),
        ]

        mock_result = mock.MagicMock()
        mock_scalars = mock.MagicMock()
        mock_scalars.all.return_value = clusters
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repository.list_unlabeled(
            dimension="content",
            user_id=user_id,
            limit=50,
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_update_found(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test updating metadata when found."""
        from rl_emails.schemas.cluster_metadata import ClusterMetadataUpdate

        existing = ClusterMetadata(
            dimension="content",
            cluster_id=5,
            size=50,
            is_project=False,
        )
        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        mock_session.execute.return_value = mock_result

        update_data = ClusterMetadataUpdate(is_project=True, project_status="active")
        result = await repository.update(1, update_data)

        assert result is not None
        assert result.is_project is True
        assert result.project_status == "active"
        mock_session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_update_not_found(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test updating metadata when not found."""
        from rl_emails.schemas.cluster_metadata import ClusterMetadataUpdate

        mock_result = mock.MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        update_data = ClusterMetadataUpdate(is_project=True)
        result = await repository.update(999, update_data)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_label_with_user_id(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test updating label with user_id."""
        user_id = uuid4()
        mock_result = mock.MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repository.update_label(
            dimension="content",
            cluster_id=5,
            label="Q4 Budget Review",
            user_id=user_id,
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_update_project_status(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test bulk updating project status."""
        mock_result = mock.MagicMock()
        mock_result.rowcount = 3
        mock_session.execute.return_value = mock_result

        result = await repository.update_project_status(
            cluster_ids=[1, 2, 3],
            status="completed",
        )
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert result == 3

    @pytest.mark.asyncio
    async def test_update_project_status_with_user_id(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test bulk updating project status with user_id."""
        user_id = uuid4()
        mock_result = mock.MagicMock()
        mock_result.rowcount = 2
        mock_session.execute.return_value = mock_result

        result = await repository.update_project_status(
            cluster_ids=[1, 2],
            status="stale",
            user_id=user_id,
        )
        assert result == 2

    @pytest.mark.asyncio
    async def test_get_stats_with_user_id(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test getting stats with user_id."""
        user_id = uuid4()
        mock_row = mock.MagicMock()
        mock_row.total_clusters = 25
        mock_row.total_emails = 500
        mock_row.avg_cluster_size = 20.0
        mock_row.largest_cluster = 50
        mock_row.smallest_cluster = 5
        mock_row.labeled_clusters = 15
        mock_row.project_clusters = 5
        mock_row.active_projects = 3

        mock_result = mock.MagicMock()
        mock_result.one.return_value = mock_row
        mock_session.execute.return_value = mock_result

        stats = await repository.get_stats("content", user_id=user_id)
        assert stats["total_clusters"] == 25

    @pytest.mark.asyncio
    async def test_truncate_by_dimension_with_user_id(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test truncating by dimension with user_id."""
        user_id = uuid4()
        mock_result = mock.MagicMock()
        mock_result.rowcount = 5
        mock_session.execute.return_value = mock_result

        result = await repository.truncate_by_dimension("content", user_id=user_id)
        mock_session.execute.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert result == 5

    @pytest.mark.asyncio
    async def test_mark_as_project_with_user_id(
        self,
        repository: ClusterMetadataRepository,
        mock_session: mock.MagicMock,
    ) -> None:
        """Test marking as project with user_id."""
        user_id = uuid4()
        mock_result = mock.MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repository.mark_as_project(
            dimension="content",
            cluster_id=5,
            status="active",
            user_id=user_id,
        )
        assert result is True
