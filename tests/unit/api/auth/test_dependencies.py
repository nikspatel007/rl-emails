"""Tests for authentication dependencies."""

from __future__ import annotations

from unittest import mock

import pytest
from starlette.datastructures import State
from starlette.requests import Request

from rl_emails.api.auth.clerk import ClerkJWTValidator, ClerkUser
from rl_emails.api.auth.config import ClerkConfig
from rl_emails.api.auth.dependencies import (
    _extract_bearer_token,
    get_api_key_user,
    get_current_user,
    get_current_user_optional,
    get_current_user_or_api_key,
    get_jwt_validator,
)
from rl_emails.api.auth.exceptions import AuthenticationError, InvalidTokenError


class TestExtractBearerToken:
    """Tests for _extract_bearer_token helper."""

    def test_valid_bearer_token(self) -> None:
        """Test extracting valid bearer token."""
        token = _extract_bearer_token("Bearer my-jwt-token")

        assert token == "my-jwt-token"

    def test_bearer_case_insensitive(self) -> None:
        """Test bearer prefix is case insensitive."""
        token = _extract_bearer_token("bearer my-token")
        assert token == "my-token"

        token = _extract_bearer_token("BEARER my-token")
        assert token == "my-token"

    def test_no_header(self) -> None:
        """Test returns None when no header."""
        token = _extract_bearer_token(None)

        assert token is None

    def test_empty_header(self) -> None:
        """Test returns None for empty header."""
        token = _extract_bearer_token("")

        assert token is None

    def test_non_bearer_scheme(self) -> None:
        """Test returns None for non-bearer schemes."""
        token = _extract_bearer_token("Basic dXNlcjpwYXNz")

        assert token is None

    def test_missing_token(self) -> None:
        """Test returns None when token part missing."""
        token = _extract_bearer_token("Bearer")

        assert token is None

    def test_extra_parts(self) -> None:
        """Test returns None when extra parts present."""
        token = _extract_bearer_token("Bearer token extra")

        assert token is None


class TestGetJWTValidator:
    """Tests for get_jwt_validator dependency."""

    def test_raises_when_not_configured(self) -> None:
        """Test raises AuthenticationError when Clerk not configured."""
        config = ClerkConfig(secret_key="", issuer="")  # Explicitly empty

        with pytest.raises(AuthenticationError, match="Authentication not configured"):
            get_jwt_validator(config)

    def test_creates_validator_when_configured(self) -> None:
        """Test creates validator when properly configured."""
        # Clear global validator
        import rl_emails.api.auth.dependencies as deps

        deps._validator = None

        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )

        validator = get_jwt_validator(config)

        assert isinstance(validator, ClerkJWTValidator)

    def test_reuses_validator(self) -> None:
        """Test reuses same validator instance."""
        import rl_emails.api.auth.dependencies as deps

        deps._validator = None

        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )

        validator1 = get_jwt_validator(config)
        validator2 = get_jwt_validator(config)

        assert validator1 is validator2


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.fixture
    def mock_request(self) -> Request:
        """Create mock request."""
        request = mock.MagicMock(spec=Request)
        request.state = State()
        return request

    @pytest.fixture
    def mock_validator(self) -> mock.MagicMock:
        """Create mock validator."""
        return mock.MagicMock(spec=ClerkJWTValidator)

    @pytest.mark.anyio
    async def test_requires_authorization_header(
        self, mock_request: Request, mock_validator: mock.MagicMock
    ) -> None:
        """Test raises error when no authorization header."""
        with pytest.raises(AuthenticationError, match="Authorization header required"):
            await get_current_user(mock_request, None, mock_validator)

    @pytest.mark.anyio
    async def test_requires_bearer_token(
        self, mock_request: Request, mock_validator: mock.MagicMock
    ) -> None:
        """Test raises error when not bearer token."""
        with pytest.raises(AuthenticationError, match="Authorization header required"):
            await get_current_user(mock_request, "Basic xxx", mock_validator)

    @pytest.mark.anyio
    async def test_validates_token(
        self, mock_request: Request, mock_validator: mock.MagicMock
    ) -> None:
        """Test validates token and returns user."""
        expected_user = ClerkUser(id="user_123", email="user@example.com")
        mock_validator.validate_token.return_value = expected_user

        user = await get_current_user(mock_request, "Bearer valid-token", mock_validator)

        assert user == expected_user
        mock_validator.validate_token.assert_called_once_with("valid-token")

    @pytest.mark.anyio
    async def test_sets_user_on_request_state(
        self, mock_request: Request, mock_validator: mock.MagicMock
    ) -> None:
        """Test sets user on request.state."""
        expected_user = ClerkUser(id="user_123")
        mock_validator.validate_token.return_value = expected_user

        await get_current_user(mock_request, "Bearer token", mock_validator)

        assert mock_request.state.user == expected_user

    @pytest.mark.anyio
    async def test_propagates_validation_errors(
        self, mock_request: Request, mock_validator: mock.MagicMock
    ) -> None:
        """Test propagates token validation errors."""
        mock_validator.validate_token.side_effect = InvalidTokenError("Bad token")

        with pytest.raises(InvalidTokenError):
            await get_current_user(mock_request, "Bearer bad-token", mock_validator)


class TestGetCurrentUserOptional:
    """Tests for get_current_user_optional dependency."""

    @pytest.fixture
    def mock_request(self) -> Request:
        """Create mock request."""
        request = mock.MagicMock(spec=Request)
        request.state = State()
        return request

    @pytest.mark.anyio
    async def test_returns_none_when_no_token(self, mock_request: Request) -> None:
        """Test returns None when no token provided."""
        result = await get_current_user_optional(mock_request, None, None)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_not_configured(self, mock_request: Request) -> None:
        """Test returns None when not configured."""
        config = ClerkConfig()  # Not configured

        result = await get_current_user_optional(mock_request, "Bearer token", config)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_config_is_none_with_token(self, mock_request: Request) -> None:
        """Test returns None when config is None but token exists."""
        # Token provided but config is None - should return None
        result = await get_current_user_optional(mock_request, "Bearer token", None)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_user_when_valid(self, mock_request: Request) -> None:
        """Test returns user when token is valid."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )
        expected_user = ClerkUser(id="user_123")

        with mock.patch("rl_emails.api.auth.dependencies.ClerkJWTValidator") as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_token.return_value = expected_user

            result = await get_current_user_optional(mock_request, "Bearer valid-token", config)

            assert result == expected_user
            assert mock_request.state.user == expected_user

    @pytest.mark.anyio
    async def test_returns_none_on_validation_error(self, mock_request: Request) -> None:
        """Test returns None when validation fails."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
        )

        with mock.patch("rl_emails.api.auth.dependencies.ClerkJWTValidator") as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_token.side_effect = InvalidTokenError()

            result = await get_current_user_optional(mock_request, "Bearer bad-token", config)

            assert result is None


class TestGetApiKeyUser:
    """Tests for get_api_key_user dependency."""

    @pytest.fixture
    def mock_request(self) -> Request:
        """Create mock request."""
        request = mock.MagicMock(spec=Request)
        request.state = State()
        return request

    @pytest.mark.anyio
    async def test_returns_none_when_no_api_key(self, mock_request: Request) -> None:
        """Test returns None when no API key provided."""
        result = await get_api_key_user(mock_request, None, None)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_no_config(self, mock_request: Request) -> None:
        """Test returns None when no config."""
        result = await get_api_key_user(mock_request, "some-key", None)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_invalid_key(self, mock_request: Request) -> None:
        """Test returns None when API key not in allowed list."""
        config = ClerkConfig(api_keys_raw="valid-key-1,valid-key-2")

        result = await get_api_key_user(mock_request, "invalid-key", config)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_service_user_when_valid(self, mock_request: Request) -> None:
        """Test returns service user when API key is valid."""
        config = ClerkConfig(api_keys_raw="valid-key-1,valid-key-2")

        result = await get_api_key_user(mock_request, "valid-key-1", config)

        assert result is not None
        assert result.id == "service"
        assert result.first_name == "Service"
        assert result.last_name == "Account"
        assert result.metadata["auth_method"] == "api_key"

    @pytest.mark.anyio
    async def test_sets_user_on_request_state(self, mock_request: Request) -> None:
        """Test sets user on request.state."""
        config = ClerkConfig(api_keys_raw="valid-key")

        await get_api_key_user(mock_request, "valid-key", config)

        assert mock_request.state.user is not None
        assert mock_request.state.user.id == "service"


class TestGetCurrentUserOrApiKey:
    """Tests for get_current_user_or_api_key dependency."""

    @pytest.fixture
    def mock_request(self) -> Request:
        """Create mock request."""
        request = mock.MagicMock(spec=Request)
        request.state = State()
        return request

    @pytest.mark.anyio
    async def test_prefers_jwt_over_api_key(self, mock_request: Request) -> None:
        """Test prefers JWT authentication over API key."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
            api_keys_raw="valid-key",
        )
        jwt_user = ClerkUser(id="jwt_user")

        with mock.patch("rl_emails.api.auth.dependencies.ClerkJWTValidator") as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_token.return_value = jwt_user

            result = await get_current_user_or_api_key(
                mock_request, "Bearer token", "valid-key", config
            )

            assert result.id == "jwt_user"

    @pytest.mark.anyio
    async def test_falls_back_to_api_key(self, mock_request: Request) -> None:
        """Test falls back to API key when JWT fails."""
        config = ClerkConfig(
            secret_key="sk_test_xxx",
            issuer="https://clerk.example.com",
            api_keys_raw="valid-key",
        )

        with mock.patch("rl_emails.api.auth.dependencies.ClerkJWTValidator") as MockValidator:
            mock_validator = MockValidator.return_value
            mock_validator.validate_token.side_effect = InvalidTokenError()

            result = await get_current_user_or_api_key(
                mock_request, "Bearer bad-token", "valid-key", config
            )

            assert result.id == "service"

    @pytest.mark.anyio
    async def test_api_key_only(self, mock_request: Request) -> None:
        """Test works with only API key."""
        config = ClerkConfig(api_keys_raw="valid-key")

        result = await get_current_user_or_api_key(mock_request, None, "valid-key", config)

        assert result.id == "service"

    @pytest.mark.anyio
    async def test_raises_when_neither_valid(self, mock_request: Request) -> None:
        """Test raises error when neither JWT nor API key valid."""
        config = ClerkConfig(api_keys_raw="valid-key")

        with pytest.raises(AuthenticationError, match="Valid JWT or API key required"):
            await get_current_user_or_api_key(mock_request, None, "invalid-key", config)

    @pytest.mark.anyio
    async def test_raises_when_no_credentials(self, mock_request: Request) -> None:
        """Test raises error when no credentials provided."""
        config = ClerkConfig()

        with pytest.raises(AuthenticationError, match="Valid JWT or API key required"):
            await get_current_user_or_api_key(mock_request, None, None, config)
