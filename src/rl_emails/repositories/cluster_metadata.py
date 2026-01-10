"""ClusterMetadata repository for database operations."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rl_emails.models.cluster_metadata import ClusterMetadata
from rl_emails.schemas.cluster_metadata import (
    ClusterMetadataCreate,
    ClusterMetadataUpdate,
)

ClusterDimension = Literal["people", "content", "behavior", "service", "temporal"]


class ClusterMetadataRepository:
    """Repository for cluster metadata database operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize repository with database session.

        Args:
            session: Async SQLAlchemy session.
        """
        self.session = session

    async def create(self, data: ClusterMetadataCreate) -> ClusterMetadata:
        """Create cluster metadata record.

        Args:
            data: Cluster metadata creation data.

        Returns:
            Created cluster metadata.
        """
        metadata = ClusterMetadata(**data.model_dump())
        self.session.add(metadata)
        await self.session.commit()
        await self.session.refresh(metadata)
        return metadata

    async def upsert(self, data: ClusterMetadataCreate) -> ClusterMetadata:
        """Create or update cluster metadata.

        Args:
            data: Cluster metadata data.

        Returns:
            Created or updated cluster metadata.
        """
        # Try to find existing
        existing = await self.get_by_dimension_and_cluster(
            dimension=data.dimension,
            cluster_id=data.cluster_id,
            user_id=data.user_id,
        )

        if existing:
            # Update existing
            update_data = data.model_dump(exclude={"dimension", "cluster_id", "user_id", "org_id"})
            for key, value in update_data.items():
                setattr(existing, key, value)
            await self.session.commit()
            await self.session.refresh(existing)
            return existing

        # Create new
        return await self.create(data)

    async def get_by_id(self, metadata_id: int) -> ClusterMetadata | None:
        """Get cluster metadata by ID.

        Args:
            metadata_id: Metadata ID.

        Returns:
            ClusterMetadata if found, None otherwise.
        """
        result = await self.session.execute(
            select(ClusterMetadata).where(ClusterMetadata.id == metadata_id)
        )
        return result.scalar_one_or_none()

    async def get_by_dimension_and_cluster(
        self,
        dimension: str,
        cluster_id: int,
        user_id: UUID | None = None,
    ) -> ClusterMetadata | None:
        """Get cluster metadata by dimension and cluster ID.

        Args:
            dimension: Clustering dimension.
            cluster_id: Cluster ID.
            user_id: Optional user ID for multi-tenant.

        Returns:
            ClusterMetadata if found, None otherwise.
        """
        conditions = [
            ClusterMetadata.dimension == dimension,
            ClusterMetadata.cluster_id == cluster_id,
        ]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)
        else:
            conditions.append(ClusterMetadata.user_id.is_(None))

        result = await self.session.execute(select(ClusterMetadata).where(and_(*conditions)))
        return result.scalar_one_or_none()

    async def list_by_dimension(
        self,
        dimension: str,
        user_id: UUID | None = None,
        *,
        is_project: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ClusterMetadata], int]:
        """List cluster metadata by dimension.

        Args:
            dimension: Clustering dimension.
            user_id: Optional user ID for multi-tenant.
            is_project: Filter by project status.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (metadata list, total count).
        """
        conditions = [ClusterMetadata.dimension == dimension]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)
        if is_project is not None:
            conditions.append(ClusterMetadata.is_project == is_project)

        # Get total count
        count_query = select(func.count()).select_from(ClusterMetadata).where(and_(*conditions))
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get metadata
        query = (
            select(ClusterMetadata)
            .where(and_(*conditions))
            .order_by(ClusterMetadata.size.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        metadata_list = list(result.scalars().all())

        return metadata_list, total

    async def list_projects(
        self,
        user_id: UUID | None = None,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[ClusterMetadata], int]:
        """List clusters flagged as projects.

        Args:
            user_id: Optional user ID for multi-tenant.
            status: Filter by project status (active, stale, completed).
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (project clusters, total count).
        """
        conditions = [
            ClusterMetadata.is_project.is_(True),
            ClusterMetadata.dimension == "content",
        ]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)
        if status is not None:
            conditions.append(ClusterMetadata.project_status == status)

        # Get total count
        count_query = select(func.count()).select_from(ClusterMetadata).where(and_(*conditions))
        count_result = await self.session.execute(count_query)
        total = count_result.scalar() or 0

        # Get projects, ordered by last activity
        query = (
            select(ClusterMetadata)
            .where(and_(*conditions))
            .order_by(ClusterMetadata.last_activity_at.desc().nulls_last())
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        projects = list(result.scalars().all())

        return projects, total

    async def list_unlabeled(
        self,
        dimension: str | None = None,
        user_id: UUID | None = None,
        *,
        limit: int = 100,
    ) -> list[ClusterMetadata]:
        """List clusters without auto-generated labels.

        Args:
            dimension: Optional dimension filter.
            user_id: Optional user ID for multi-tenant.
            limit: Maximum number of results.

        Returns:
            List of unlabeled cluster metadata.
        """
        query = select(ClusterMetadata).where(ClusterMetadata.auto_label.is_(None))

        if dimension is not None:
            query = query.where(ClusterMetadata.dimension == dimension)
        if user_id is not None:
            query = query.where(ClusterMetadata.user_id == user_id)

        query = query.order_by(ClusterMetadata.size.desc()).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update(
        self,
        metadata_id: int,
        data: ClusterMetadataUpdate,
    ) -> ClusterMetadata | None:
        """Update cluster metadata.

        Args:
            metadata_id: Metadata ID.
            data: Update data.

        Returns:
            Updated metadata if found, None otherwise.
        """
        metadata = await self.get_by_id(metadata_id)
        if metadata is None:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(metadata, key, value)

        await self.session.commit()
        await self.session.refresh(metadata)
        return metadata

    async def update_label(
        self,
        dimension: str,
        cluster_id: int,
        label: str,
        user_id: UUID | None = None,
    ) -> bool:
        """Update auto-label for a cluster.

        Args:
            dimension: Clustering dimension.
            cluster_id: Cluster ID.
            label: New label.
            user_id: Optional user ID for multi-tenant.

        Returns:
            True if updated, False if not found.
        """
        conditions = [
            ClusterMetadata.dimension == dimension,
            ClusterMetadata.cluster_id == cluster_id,
        ]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)

        stmt = update(ClusterMetadata).where(and_(*conditions)).values(auto_label=label)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return bool(getattr(result, "rowcount", 0) > 0)

    async def mark_as_project(
        self,
        dimension: str,
        cluster_id: int,
        status: str = "active",
        user_id: UUID | None = None,
    ) -> bool:
        """Mark a cluster as a project.

        Args:
            dimension: Clustering dimension.
            cluster_id: Cluster ID.
            status: Project status (active, stale, completed).
            user_id: Optional user ID for multi-tenant.

        Returns:
            True if updated, False if not found.
        """
        conditions = [
            ClusterMetadata.dimension == dimension,
            ClusterMetadata.cluster_id == cluster_id,
        ]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)

        stmt = (
            update(ClusterMetadata)
            .where(and_(*conditions))
            .values(is_project=True, project_status=status)
        )
        result = await self.session.execute(stmt)
        await self.session.commit()
        return bool(getattr(result, "rowcount", 0) > 0)

    async def update_project_status(
        self,
        cluster_ids: list[int],
        status: str,
        user_id: UUID | None = None,
    ) -> int:
        """Bulk update project status.

        Args:
            cluster_ids: List of cluster IDs to update.
            status: New project status.
            user_id: Optional user ID for multi-tenant.

        Returns:
            Number of records updated.
        """
        conditions = [
            ClusterMetadata.cluster_id.in_(cluster_ids),
            ClusterMetadata.dimension == "content",
            ClusterMetadata.is_project.is_(True),
        ]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)

        stmt = update(ClusterMetadata).where(and_(*conditions)).values(project_status=status)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return int(getattr(result, "rowcount", 0))

    async def get_stats(
        self,
        dimension: str,
        user_id: UUID | None = None,
    ) -> dict[str, int | float]:
        """Get statistics for a dimension.

        Args:
            dimension: Clustering dimension.
            user_id: Optional user ID for multi-tenant.

        Returns:
            Statistics dictionary.
        """
        conditions = [ClusterMetadata.dimension == dimension]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)

        # Get aggregate stats
        query = select(
            func.count().label("total_clusters"),
            func.sum(ClusterMetadata.size).label("total_emails"),
            func.avg(ClusterMetadata.size).label("avg_cluster_size"),
            func.max(ClusterMetadata.size).label("largest_cluster"),
            func.min(ClusterMetadata.size).label("smallest_cluster"),
            func.count().filter(ClusterMetadata.auto_label.isnot(None)).label("labeled_clusters"),
            func.count().filter(ClusterMetadata.is_project.is_(True)).label("project_clusters"),
            func.count()
            .filter(
                and_(
                    ClusterMetadata.is_project.is_(True),
                    ClusterMetadata.project_status == "active",
                )
            )
            .label("active_projects"),
        ).where(and_(*conditions))

        result = await self.session.execute(query)
        row = result.one()

        return {
            "total_clusters": row.total_clusters or 0,
            "total_emails": row.total_emails or 0,
            "avg_cluster_size": float(row.avg_cluster_size or 0),
            "largest_cluster_size": row.largest_cluster or 0,
            "smallest_cluster_size": row.smallest_cluster or 0,
            "labeled_clusters": row.labeled_clusters or 0,
            "project_clusters": row.project_clusters or 0,
            "active_projects": row.active_projects or 0,
        }

    async def delete_by_user(self, user_id: UUID) -> int:
        """Delete all cluster metadata for a user.

        Args:
            user_id: User ID.

        Returns:
            Number of records deleted.
        """
        from sqlalchemy import delete

        stmt = delete(ClusterMetadata).where(ClusterMetadata.user_id == user_id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return int(getattr(result, "rowcount", 0))

    async def truncate_by_dimension(
        self,
        dimension: str,
        user_id: UUID | None = None,
    ) -> int:
        """Delete all cluster metadata for a dimension.

        Args:
            dimension: Clustering dimension.
            user_id: Optional user ID for multi-tenant.

        Returns:
            Number of records deleted.
        """
        from sqlalchemy import delete

        conditions = [ClusterMetadata.dimension == dimension]
        if user_id is not None:
            conditions.append(ClusterMetadata.user_id == user_id)

        stmt = delete(ClusterMetadata).where(and_(*conditions))
        result = await self.session.execute(stmt)
        await self.session.commit()
        return int(getattr(result, "rowcount", 0))
