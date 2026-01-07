#!/usr/bin/env python3
"""Interactive query tool for SurrealDB email databases.

Usage:
    # Interactive mode
    ./scripts/query_db.py enron
    ./scripts/query_db.py gmail

    # Single query mode
    ./scripts/query_db.py enron 'SELECT count() FROM emails GROUP ALL'
    ./scripts/query_db.py gmail 'SELECT * FROM users LIMIT 5'

Example queries:
    # Count emails
    SELECT count() FROM emails GROUP ALL

    # Find emails by sender
    SELECT * FROM emails WHERE from_email = 'john.smith@enron.com' LIMIT 10

    # Get user communication patterns
    SELECT *, ->communicates->users AS contacts FROM users WHERE email = 'jeff.skilling@enron.com'

    # Find threads with many replies
    SELECT *, <-belongs_to<-emails AS messages FROM threads ORDER BY array::len(messages) DESC LIMIT 5

    # Graph traversal - who did this person email?
    SELECT ->sent_by->users.email AS sender, ->received_by->users.email AS recipients
    FROM emails WHERE from_email CONTAINS 'skilling' LIMIT 5
"""
from __future__ import annotations

import argparse
import json
import os
import readline  # noqa: F401 - enables line editing in input()
import sys
from typing import Any

from dotenv import load_dotenv
from surrealdb import Surreal

# Load environment variables from .env file
load_dotenv()

# Database configurations (can be overridden via environment)
DB_CONFIG = {
    'enron': {'port': 8000, 'db': 'enron'},
    'gmail': {'port': 8001, 'db': 'gmail'},
}

NAMESPACE = os.environ.get('SURREALDB_NAMESPACE', 'rl_emails')
SURREALDB_USER = os.environ.get('SURREALDB_USER', 'root')
SURREALDB_PASS = os.environ.get('SURREALDB_PASS', 'root')


def format_result(result: Any, indent: int = 2) -> str:
    """Format query result for display."""
    if result is None:
        return 'null'
    if isinstance(result, (list, dict)):
        return json.dumps(result, indent=indent, default=str)
    return str(result)


def run_query(db: Any, query: str) -> None:
    """Execute a query and print results."""
    try:
        result = db.query(query)
        if result:
            print(format_result(result))
        else:
            print('(no results)')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)


def interactive_mode(db: Any, db_name: str) -> None:
    """Run interactive query session."""
    print(f'Connected to {db_name} database')
    print('Type your SurrealQL queries, or:')
    print('  .help     - Show example queries')
    print('  .tables   - List all tables')
    print('  .count    - Count records in all tables')
    print('  .exit     - Exit')
    print('')

    while True:
        try:
            query = input(f'{db_name}> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if not query:
            continue

        if query.lower() in ('.exit', '.quit', 'exit', 'quit'):
            print('Goodbye!')
            break

        if query.lower() == '.help':
            print(__doc__)
            continue

        if query.lower() == '.tables':
            run_query(db, 'INFO FOR DB')
            continue

        if query.lower() == '.count':
            for table in ['emails', 'users', 'threads']:
                result = db.query(f'SELECT count() FROM {table} GROUP ALL')
                count = result[0].get('count', 0) if result else 0
                print(f'  {table}: {count}')
            continue

        # Execute the query
        run_query(db, query)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Interactive query tool for SurrealDB email databases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'database',
        choices=['enron', 'gmail'],
        help='Database to query',
    )
    parser.add_argument(
        'query',
        nargs='?',
        help='SurrealQL query to execute (omit for interactive mode)',
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help='SurrealDB host (default: localhost)',
    )

    args = parser.parse_args()

    config = DB_CONFIG[args.database]
    url = f'ws://{args.host}:{config["port"]}/rpc'

    try:
        with Surreal(url) as db:
            db.signin({'username': SURREALDB_USER, 'password': SURREALDB_PASS})
            db.use(NAMESPACE, config['db'])

            if args.query:
                # Single query mode
                run_query(db, args.query)
            else:
                # Interactive mode
                interactive_mode(db, args.database)

    except ConnectionRefusedError:
        print(
            f'Error: Could not connect to {args.database} database on port {config["port"]}',
            file=sys.stderr,
        )
        print(f'Start it first: ./scripts/start_db.sh {args.database}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
