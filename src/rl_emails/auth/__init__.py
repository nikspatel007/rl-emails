"""Authentication module for OAuth2 flows."""

from rl_emails.auth.oauth import GoogleTokens, OAuthError

__all__ = [
    "GoogleTokens",
    "OAuthError",
]
