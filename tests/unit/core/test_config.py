"""Tests for rl_emails.core.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from rl_emails.core.config import Config


class TestConfig:
    """Tests for Config class."""

    def test_from_env_with_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading config from environment with DATABASE_URL."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/testdb")
        monkeypatch.delenv("MBOX_PATH", raising=False)
        monkeypatch.delenv("YOUR_EMAIL", raising=False)

        config = Config.from_env()

        assert config.database_url == "postgresql://localhost/testdb"
        assert config.mbox_path is None
        assert config.your_email is None

    def test_from_env_missing_database_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that missing DATABASE_URL raises ValueError."""
        monkeypatch.delenv("DATABASE_URL", raising=False)

        with pytest.raises(ValueError, match="DATABASE_URL is required"):
            Config.from_env()

    def test_from_env_with_all_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading config with all values set."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/testdb")
        monkeypatch.setenv("MBOX_PATH", "/path/to/mbox")
        monkeypatch.setenv("YOUR_EMAIL", "me@example.com")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")

        config = Config.from_env()

        assert config.database_url == "postgresql://localhost/testdb"
        assert config.mbox_path == Path("/path/to/mbox")
        assert config.your_email == "me@example.com"
        assert config.openai_api_key == "sk-test"
        assert config.anthropic_api_key == "ant-test"

    def test_validate_with_valid_config(self) -> None:
        """Test validation with valid config."""
        config = Config(database_url="postgresql://localhost/testdb")
        assert config.validate() == []

    def test_validate_with_missing_database_url(self) -> None:
        """Test validation with missing database URL."""
        config = Config(database_url="")
        assert "DATABASE_URL" in config.validate()

    def test_has_openai(self) -> None:
        """Test has_openai method."""
        config = Config(database_url="test", openai_api_key="sk-test")
        assert config.has_openai() is True

        config_no_key = Config(database_url="test")
        assert config_no_key.has_openai() is False

    def test_has_anthropic(self) -> None:
        """Test has_anthropic method."""
        config = Config(database_url="test", anthropic_api_key="ant-test")
        assert config.has_anthropic() is True

        config_no_key = Config(database_url="test")
        assert config_no_key.has_anthropic() is False

    def test_has_llm(self) -> None:
        """Test has_llm method."""
        config_openai = Config(database_url="test", openai_api_key="sk-test")
        assert config_openai.has_llm() is True

        config_anthropic = Config(database_url="test", anthropic_api_key="ant-test")
        assert config_anthropic.has_llm() is True

        config_both = Config(
            database_url="test", openai_api_key="sk-test", anthropic_api_key="ant-test"
        )
        assert config_both.has_llm() is True

        config_none = Config(database_url="test")
        assert config_none.has_llm() is False

    def test_from_env_with_env_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test loading config from .env file."""
        # Create a temp .env file
        env_file = tmp_path / ".env"
        env_file.write_text("DATABASE_URL=postgresql://from_file/db\nYOUR_EMAIL=file@example.com\n")

        # Clear environment variables
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("YOUR_EMAIL", raising=False)

        config = Config.from_env(env_file=env_file)

        assert config.database_url == "postgresql://from_file/db"
        assert config.your_email == "file@example.com"

    def test_env_overrides_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Test that environment variables override .env file."""
        # Create a temp .env file
        env_file = tmp_path / ".env"
        env_file.write_text("DATABASE_URL=postgresql://from_file/db\n")

        # Set environment variable (should override file)
        monkeypatch.setenv("DATABASE_URL", "postgresql://from_env/db")

        config = Config.from_env(env_file=env_file)

        assert config.database_url == "postgresql://from_env/db"


class TestConfigMultiTenant:
    """Tests for Config multi-tenant features."""

    def test_is_multi_tenant_false_by_default(self) -> None:
        """Test is_multi_tenant is False when no user_id."""
        config = Config(database_url="test")
        assert config.is_multi_tenant is False

    def test_is_multi_tenant_true_with_user_id(self) -> None:
        """Test is_multi_tenant is True when user_id set."""
        import uuid

        config = Config(database_url="test", user_id=uuid.uuid4())
        assert config.is_multi_tenant is True

    def test_with_user_creates_new_config(self) -> None:
        """Test with_user creates a new config with user context."""
        import uuid

        config = Config(database_url="test", openai_api_key="sk-test")
        user_id = uuid.uuid4()
        org_id = uuid.uuid4()

        new_config = config.with_user(user_id, org_id)

        # Verify it's a new instance
        assert new_config is not config
        # Verify user context is set
        assert new_config.user_id == user_id
        assert new_config.org_id == org_id
        # Verify other values preserved
        assert new_config.database_url == "test"
        assert new_config.openai_api_key == "sk-test"
        # Verify original unchanged
        assert config.user_id is None
        assert config.org_id is None

    def test_with_user_org_id_optional(self) -> None:
        """Test with_user works without org_id."""
        import uuid

        config = Config(database_url="test")
        user_id = uuid.uuid4()

        new_config = config.with_user(user_id)

        assert new_config.user_id == user_id
        assert new_config.org_id is None


class TestConfigGoogleOAuth:
    """Tests for Config Google OAuth features."""

    def test_has_google_oauth_true(self) -> None:
        """Test has_google_oauth returns True when both fields set."""
        config = Config(
            database_url="test",
            google_client_id="client-id-123",
            google_client_secret="client-secret-456",
        )
        assert config.has_google_oauth() is True

    def test_has_google_oauth_false_missing_id(self) -> None:
        """Test has_google_oauth returns False when missing client_id."""
        config = Config(
            database_url="test",
            google_client_secret="client-secret-456",
        )
        assert config.has_google_oauth() is False

    def test_has_google_oauth_false_missing_secret(self) -> None:
        """Test has_google_oauth returns False when missing client_secret."""
        config = Config(
            database_url="test",
            google_client_id="client-id-123",
        )
        assert config.has_google_oauth() is False

    def test_has_google_oauth_false_empty(self) -> None:
        """Test has_google_oauth returns False with no OAuth config."""
        config = Config(database_url="test")
        assert config.has_google_oauth() is False

    def test_default_redirect_uri(self) -> None:
        """Test default redirect URI."""
        config = Config(database_url="test")
        assert config.google_redirect_uri == "http://localhost:8000/auth/google/callback"

    def test_custom_redirect_uri(self) -> None:
        """Test custom redirect URI."""
        config = Config(
            database_url="test",
            google_redirect_uri="https://myapp.com/oauth/callback",
        )
        assert config.google_redirect_uri == "https://myapp.com/oauth/callback"

    def test_from_env_with_google_oauth(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading Google OAuth config from environment."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/testdb")
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "env-client-id")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "env-client-secret")
        monkeypatch.setenv("GOOGLE_REDIRECT_URI", "https://env.example.com/callback")

        config = Config.from_env()

        assert config.google_client_id == "env-client-id"
        assert config.google_client_secret == "env-client-secret"
        assert config.google_redirect_uri == "https://env.example.com/callback"
        assert config.has_google_oauth() is True

    def test_from_env_google_oauth_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test Google OAuth defaults when not in environment."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/testdb")
        monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
        monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("GOOGLE_REDIRECT_URI", raising=False)

        config = Config.from_env()

        assert config.google_client_id is None
        assert config.google_client_secret is None
        assert config.google_redirect_uri == "http://localhost:8000/auth/google/callback"
        assert config.has_google_oauth() is False
