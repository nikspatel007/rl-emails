"""Email parsers for different formats."""

from .gmail_mbox import parse_gmail_mbox, process_gmail_mbox

__all__ = ['parse_gmail_mbox', 'process_gmail_mbox']
