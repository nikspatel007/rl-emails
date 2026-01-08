#!/usr/bin/env python3
"""Quick script to set up a user for Gmail sync.

Usage:
    python scripts/setup_user.py your@email.com
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from rl_emails.core.config import Config


async def setup_user(email: str) -> None:
    """Create organization and user for Gmail sync."""
    config = Config.from_env()

    # Convert to async URL
    db_url = config.database_url
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url)

    async with AsyncSession(engine) as session:
        # Check if user already exists
        result = await session.execute(
            text("SELECT id FROM org_users WHERE email = :email"),
            {"email": email}
        )
        existing = result.fetchone()

        if existing:
            print(f"User already exists!")
            print(f"User ID: {existing[0]}")
            print(f"\nUse this command to connect Gmail:")
            print(f"  uv run rl-emails auth connect --user {existing[0]}")
            return

        # Create organization
        org_id = uuid4()
        await session.execute(
            text("""
                INSERT INTO organizations (id, name, slug, created_at, updated_at)
                VALUES (:id, :name, :slug, NOW(), NOW())
            """),
            {
                "id": org_id,
                "name": f"{email.split('@')[0]}'s Organization",
                "slug": email.split('@')[0].lower().replace('.', '-'),
            }
        )

        # Create user
        user_id = uuid4()
        await session.execute(
            text("""
                INSERT INTO org_users (id, org_id, email, role, created_at, updated_at)
                VALUES (:id, :org_id, :email, :role, NOW(), NOW())
            """),
            {
                "id": user_id,
                "org_id": org_id,
                "email": email,
                "role": "owner",
            }
        )

        await session.commit()

        print(f"User created successfully!")
        print(f"")
        print(f"  Email: {email}")
        print(f"  User ID: {user_id}")
        print(f"  Org ID: {org_id}")
        print(f"")
        print(f"Next steps:")
        print(f"  1. Connect Gmail:")
        print(f"     uv run rl-emails auth connect --user {user_id}")
        print(f"")
        print(f"  2. After authorizing, sync your emails:")
        print(f"     uv run rl-emails sync --user {user_id}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/setup_user.py your@email.com")
        sys.exit(1)

    email = sys.argv[1]
    if "@" not in email:
        print(f"Invalid email: {email}")
        sys.exit(1)

    asyncio.run(setup_user(email))


if __name__ == "__main__":
    main()
