"""Tests for Stage 9: Cluster emails."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_09_cluster_emails
from rl_emails.pipeline.stages.base import StageResult


class TestCreateTables:
    """Tests for create_tables function."""

    def test_creates_tables(self) -> None:
        """Test table creation."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        stage_09_cluster_emails.create_tables(conn)

        assert mock_cursor.execute.call_count == 2
        conn.commit.assert_called_once()


class TestLoadPeopleFeatures:
    """Tests for load_people_features function."""

    def test_loads_features(self) -> None:
        """Test loading people features."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 0.5, 0.3, 24.0, 10, 0.2, 0),
            (2, 0.8, 0.6, 12.0, 20, 0.4, 1),
        ]

        email_ids, features = stage_09_cluster_emails.load_people_features(conn)

        assert len(email_ids) == 2
        assert features.shape == (2, 6)

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        email_ids, features = stage_09_cluster_emails.load_people_features(conn)

        assert email_ids == []
        assert features.shape == (0,)


class TestLoadEmbeddings:
    """Tests for load_embeddings function."""

    def test_loads_embeddings(self) -> None:
        """Test loading embeddings."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, "[0.1, 0.2, 0.3]"),
            (2, "[0.4, 0.5, 0.6]"),
        ]

        email_ids, embeddings = stage_09_cluster_emails.load_embeddings(conn)

        assert len(email_ids) == 2
        assert embeddings.shape == (2, 3)
        assert embeddings[0][0] == 0.1

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        email_ids, embeddings = stage_09_cluster_emails.load_embeddings(conn)

        assert email_ids == []


class TestLoadBehaviorFeatures:
    """Tests for load_behavior_features function."""

    def test_loads_features(self) -> None:
        """Test loading behavior features."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 1, 2.5, 1),
            (2, 2, 24.0, 3),
        ]

        email_ids, features = stage_09_cluster_emails.load_behavior_features(conn)

        assert len(email_ids) == 2
        assert features.shape == (2, 3)

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        email_ids, features = stage_09_cluster_emails.load_behavior_features(conn)

        assert email_ids == []


class TestLoadServiceFeatures:
    """Tests for load_service_features function."""

    def test_loads_features(self) -> None:
        """Test loading service features."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 1, 0.5, 1, 0, 2),
            (2, 0, 0.8, 0, 1, 0),
        ]

        email_ids, features = stage_09_cluster_emails.load_service_features(conn)

        assert len(email_ids) == 2
        assert features.shape == (2, 5)

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        email_ids, features = stage_09_cluster_emails.load_service_features(conn)

        assert email_ids == []


class TestLoadTemporalFeatures:
    """Tests for load_temporal_features function."""

    def test_loads_features(self) -> None:
        """Test loading temporal features."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            (1, 9, 1, 1, 0),
            (2, 14, 5, 0, 1),
        ]

        email_ids, features = stage_09_cluster_emails.load_temporal_features(conn)

        assert len(email_ids) == 2
        assert features.shape == (2, 4)

    def test_empty_results(self) -> None:
        """Test with no results."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        email_ids, features = stage_09_cluster_emails.load_temporal_features(conn)

        assert email_ids == []


class TestClusterPeople:
    """Tests for cluster_people function."""

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_people_features")
    def test_clusters_people(self, mock_load: MagicMock) -> None:
        """Test people clustering."""
        mock_load.return_value = (
            [1, 2, 3, 4, 5],
            np.array([[0.5, 0.3, 24.0, 10, 0.2, 0]] * 5),
        )
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_people(conn)

        assert len(result["email_ids"]) == 5
        assert len(result["labels"]) == 5
        assert result["n_clusters"] > 0

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_people_features")
    def test_empty_data(self, mock_load: MagicMock) -> None:
        """Test with empty data."""
        mock_load.return_value = ([], np.array([]))
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_people(conn)

        assert result["n_clusters"] == 0

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_people_features")
    def test_uniform_distances(self, mock_load: MagicMock) -> None:
        """Test when all points are identical (max_dist=0)."""
        # All identical features will result in max_dist=0
        mock_load.return_value = (
            [1, 2],
            np.array([[0.5, 0.3, 24.0, 10, 0.2, 0], [0.5, 0.3, 24.0, 10, 0.2, 0]]),
        )
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_people(conn)

        # Should not crash, probs should be all 1.0
        assert len(result["probs"]) == 2
        assert all(p == 1.0 for p in result["probs"])


class TestClusterContent:
    """Tests for cluster_content function."""

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_embeddings")
    def test_clusters_content_kmeans(self, mock_load: MagicMock) -> None:
        """Test content clustering with K-Means fallback."""
        mock_load.return_value = (
            [1, 2, 3, 4, 5],
            np.random.rand(5, 100),  # 5 embeddings of 100 dims
        )
        conn = MagicMock()

        with patch.object(stage_09_cluster_emails, "HAS_HDBSCAN", False):
            with patch.object(stage_09_cluster_emails, "HAS_UMAP", False):
                result = stage_09_cluster_emails.cluster_content(conn)

        assert len(result["email_ids"]) == 5
        assert result["n_clusters"] > 0

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_embeddings")
    def test_empty_data(self, mock_load: MagicMock) -> None:
        """Test with empty data."""
        mock_load.return_value = ([], np.array([]))
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_content(conn)

        assert result["n_clusters"] == 0

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_embeddings")
    def test_uniform_distances_kmeans(self, mock_load: MagicMock) -> None:
        """Test when all embeddings are identical (max_dist=0)."""
        mock_load.return_value = (
            [1, 2],
            np.array([[0.5] * 100, [0.5] * 100]),
        )
        conn = MagicMock()

        with patch.object(stage_09_cluster_emails, "HAS_HDBSCAN", False):
            with patch.object(stage_09_cluster_emails, "HAS_UMAP", False):
                result = stage_09_cluster_emails.cluster_content(conn)

        assert len(result["probs"]) == 2

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_embeddings")
    def test_with_hdbscan(self, mock_load: MagicMock) -> None:
        """Test content clustering with HDBSCAN."""
        mock_load.return_value = (
            list(range(100)),
            np.random.rand(100, 50),
        )
        conn = MagicMock()

        # Mock hdbscan module
        mock_hdbscan = MagicMock()
        mock_clusterer = MagicMock()
        mock_clusterer.fit_predict.return_value = np.array([0] * 50 + [1] * 50)
        mock_clusterer.probabilities_ = np.array([0.9] * 100)
        mock_hdbscan.HDBSCAN.return_value = mock_clusterer

        with patch.object(stage_09_cluster_emails, "HAS_HDBSCAN", True):
            with patch.object(stage_09_cluster_emails, "hdbscan", mock_hdbscan):
                with patch.object(stage_09_cluster_emails, "HAS_UMAP", False):
                    result = stage_09_cluster_emails.cluster_content(conn)

        assert len(result["email_ids"]) == 100
        mock_hdbscan.HDBSCAN.assert_called_once()

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_embeddings")
    def test_with_umap(self, mock_load: MagicMock) -> None:
        """Test content clustering with UMAP dimensionality reduction."""
        mock_load.return_value = (
            list(range(100)),
            np.random.rand(100, 1536),
        )
        conn = MagicMock()

        # Mock umap module
        mock_umap_module = MagicMock()
        mock_reducer = MagicMock()
        mock_reducer.fit_transform.return_value = np.random.rand(100, 50)
        mock_umap_module.UMAP.return_value = mock_reducer

        with patch.object(stage_09_cluster_emails, "HAS_UMAP", True):
            with patch.object(stage_09_cluster_emails, "umap", mock_umap_module):
                with patch.object(stage_09_cluster_emails, "HAS_HDBSCAN", False):
                    result = stage_09_cluster_emails.cluster_content(conn)

        assert len(result["email_ids"]) == 100
        mock_umap_module.UMAP.assert_called_once()


class TestClusterBehavior:
    """Tests for cluster_behavior function."""

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_behavior_features")
    def test_clusters_behavior(self, mock_load: MagicMock) -> None:
        """Test behavior clustering."""
        mock_load.return_value = (
            [1, 2, 3, 4, 5],
            np.array([[1, 2.5, 1]] * 5),
        )
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_behavior(conn)

        assert len(result["email_ids"]) == 5
        assert result["n_clusters"] > 0

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_behavior_features")
    def test_empty_data(self, mock_load: MagicMock) -> None:
        """Test with empty data."""
        mock_load.return_value = ([], np.array([]))
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_behavior(conn)

        assert result["n_clusters"] == 0


class TestClusterService:
    """Tests for cluster_service function."""

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_service_features")
    def test_clusters_service(self, mock_load: MagicMock) -> None:
        """Test service clustering."""
        mock_load.return_value = (
            [1, 2, 3, 4, 5],
            np.array([[1, 0.5, 1, 0, 2]] * 5),
        )
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_service(conn)

        assert len(result["email_ids"]) == 5
        assert result["n_clusters"] > 0

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_service_features")
    def test_empty_data(self, mock_load: MagicMock) -> None:
        """Test with empty data."""
        mock_load.return_value = ([], np.array([]))
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_service(conn)

        assert result["n_clusters"] == 0


class TestClusterTemporal:
    """Tests for cluster_temporal function."""

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_temporal_features")
    def test_clusters_temporal(self, mock_load: MagicMock) -> None:
        """Test temporal clustering."""
        mock_load.return_value = (
            [1, 2, 3, 4, 5],
            np.array([[9, 1, 1, 0]] * 5),
        )
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_temporal(conn)

        assert len(result["email_ids"]) == 5
        assert result["n_clusters"] > 0

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.load_temporal_features")
    def test_empty_data(self, mock_load: MagicMock) -> None:
        """Test with empty data."""
        mock_load.return_value = ([], np.array([]))
        conn = MagicMock()

        result = stage_09_cluster_emails.cluster_temporal(conn)

        assert result["n_clusters"] == 0


class TestSaveClusters:
    """Tests for save_clusters function."""

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.execute_values")
    def test_saves_clusters(self, mock_execute_values: MagicMock) -> None:
        """Test saving cluster assignments."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        results: dict[str, Any] = {
            "people": {
                "email_ids": [1, 2],
                "labels": np.array([0, 1]),
                "probs": np.array([0.9, 0.8]),
            },
            "content": {
                "email_ids": [1, 2],
                "labels": np.array([0, 0]),
                "probs": np.array([0.7, 0.7]),
            },
            "behavior": {"email_ids": [1, 2], "labels": np.array([0, 1])},
            "service": {"email_ids": [1, 2], "labels": np.array([0, 0])},
            "temporal": {"email_ids": [1, 2], "labels": np.array([1, 1])},
        }

        saved = stage_09_cluster_emails.save_clusters(conn, results)

        assert saved == 2
        mock_execute_values.assert_called_once()
        conn.commit.assert_called_once()


class TestComputeClusterMetadata:
    """Tests for compute_cluster_metadata function."""

    def test_computes_metadata(self) -> None:
        """Test metadata computation."""
        conn = MagicMock()
        mock_cursor = MagicMock()
        conn.cursor.return_value.__enter__.return_value = mock_cursor

        # Mock stats query results for each dimension
        mock_cursor.fetchall.side_effect = [
            [(0, 100, 25.0, 12.5, 0.5)],  # people stats
            [(0, 80, 20.0, 24.0, 0.4)],  # content stats
            [(0, 60, 15.0, 48.0, 0.3)],  # behavior stats
            [(0, 40, 10.0, 72.0, 0.2)],  # service stats
            [(0, 20, 5.0, 96.0, 0.1)],  # temporal stats
        ]
        # Mock representative query
        mock_cursor.fetchone.side_effect = [(1,)] * 5

        stage_09_cluster_emails.compute_cluster_metadata(conn)

        # TRUNCATE + 5 stats queries + 5 representative queries + 5 inserts = multiple calls
        assert mock_cursor.execute.call_count > 5
        conn.commit.assert_called_once()


class TestRun:
    """Tests for run function."""

    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.psycopg2.connect")
    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.cluster_people")
    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.cluster_content")
    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.cluster_behavior")
    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.cluster_service")
    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.cluster_temporal")
    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.save_clusters")
    @patch("rl_emails.pipeline.stages.stage_09_cluster_emails.compute_cluster_metadata")
    def test_run_success(
        self,
        mock_metadata: MagicMock,
        mock_save: MagicMock,
        mock_temporal: MagicMock,
        mock_service: MagicMock,
        mock_behavior: MagicMock,
        mock_content: MagicMock,
        mock_people: MagicMock,
        mock_connect: MagicMock,
    ) -> None:
        """Test successful run."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_connect.return_value = mock_conn

        # Mock cluster results
        mock_people.return_value = {
            "email_ids": [1],
            "labels": np.array([0]),
            "probs": np.array([1.0]),
            "n_clusters": 1,
        }
        mock_content.return_value = {
            "email_ids": [1],
            "labels": np.array([0]),
            "probs": np.array([1.0]),
            "n_clusters": 1,
        }
        mock_behavior.return_value = {
            "email_ids": [1],
            "labels": np.array([0]),
            "probs": np.array([1.0]),
            "n_clusters": 1,
        }
        mock_service.return_value = {
            "email_ids": [1],
            "labels": np.array([0]),
            "probs": np.array([1.0]),
            "n_clusters": 1,
        }
        mock_temporal.return_value = {
            "email_ids": [1],
            "labels": np.array([0]),
            "probs": np.array([1.0]),
            "n_clusters": 1,
        }
        mock_save.return_value = 100

        config = Config(database_url="postgresql://test")
        result = stage_09_cluster_emails.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 100
        mock_conn.close.assert_called_once()
