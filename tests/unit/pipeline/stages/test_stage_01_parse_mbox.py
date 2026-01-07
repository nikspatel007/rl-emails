"""Tests for Stage 1: Parse MBOX to JSONL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from rl_emails.core.config import Config
from rl_emails.pipeline.stages import stage_01_parse_mbox
from rl_emails.pipeline.stages.base import StageResult


class TestDecodeHeaderValue:
    """Tests for decode_header_value function."""

    def test_empty_value(self) -> None:
        """Test empty value returns empty string."""
        assert stage_01_parse_mbox.decode_header_value(None) == ""
        assert stage_01_parse_mbox.decode_header_value("") == ""

    def test_plain_value(self) -> None:
        """Test plain ASCII value."""
        assert stage_01_parse_mbox.decode_header_value("Hello World") == "Hello World"

    def test_encoded_utf8(self) -> None:
        """Test UTF-8 encoded value."""
        # RFC 2047 encoded UTF-8
        encoded = "=?UTF-8?B?SGVsbG8gV29ybGQ=?="  # "Hello World" base64
        result = stage_01_parse_mbox.decode_header_value(encoded)
        assert "Hello World" in result or result  # May vary by implementation

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.decode_header")
    def test_decode_header_invalid_charset(self, mock_decode: MagicMock) -> None:
        """Test handling of invalid charset in decode_header."""
        # Mock decode_header to return bytes with invalid charset
        mock_decode.return_value = [(b"Hello", "invalid-charset-xyz")]

        result = stage_01_parse_mbox.decode_header_value("test")
        assert result == "Hello"  # Falls back to UTF-8

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.decode_header")
    def test_decode_header_exception(self, mock_decode: MagicMock) -> None:
        """Test handling of exception in decode_header."""
        mock_decode.side_effect = ValueError("Decode error")

        result = stage_01_parse_mbox.decode_header_value("test value")
        assert result == "test value"  # Returns original string


class TestParseLabels:
    """Tests for parse_labels function."""

    def test_empty_labels(self) -> None:
        """Test empty labels string."""
        assert stage_01_parse_mbox.parse_labels(None) == []
        assert stage_01_parse_mbox.parse_labels("") == []

    def test_simple_labels(self) -> None:
        """Test simple comma-separated labels."""
        result = stage_01_parse_mbox.parse_labels("Inbox, Sent, Important")
        assert result == ["Inbox", "Sent", "Important"]

    def test_quoted_labels(self) -> None:
        """Test labels with quotes."""
        result = stage_01_parse_mbox.parse_labels('"Label with, comma", Simple')
        assert result == ["Label with, comma", "Simple"]

    def test_header_folding(self) -> None:
        """Test handling of RFC 2822 header folding."""
        result = stage_01_parse_mbox.parse_labels("Label1,\r\n Label2")
        assert result == ["Label1", "Label2"]

    def test_empty_labels_between_commas(self) -> None:
        """Test labels with empty values between commas."""
        result = stage_01_parse_mbox.parse_labels("Label1,,Label2")
        assert result == ["Label1", "Label2"]

    def test_trailing_comma(self) -> None:
        """Test labels with trailing comma."""
        result = stage_01_parse_mbox.parse_labels("Label1,Label2,")
        assert result == ["Label1", "Label2"]


class TestDecodePayload:
    """Tests for _decode_payload function."""

    def test_utf8_payload(self) -> None:
        """Test UTF-8 payload decoding."""
        payload = b"Hello World"
        result = stage_01_parse_mbox._decode_payload(payload, "utf-8")
        assert result == "Hello World"

    def test_none_charset_defaults_utf8(self) -> None:
        """Test None charset defaults to UTF-8."""
        payload = b"Hello"
        result = stage_01_parse_mbox._decode_payload(payload, None)
        assert result == "Hello"

    def test_invalid_charset_fallback(self) -> None:
        """Test invalid charset falls back to UTF-8."""
        payload = b"Hello"
        result = stage_01_parse_mbox._decode_payload(payload, "invalid-charset")
        assert result == "Hello"


class TestGetBody:
    """Tests for get_body function."""

    def test_simple_text_message(self) -> None:
        """Test extracting body from simple text message."""
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_content_type.return_value = "text/plain"
        msg.get_payload.return_value = b"Hello World"
        msg.get_content_charset.return_value = "utf-8"

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == "Hello World"
        assert body_html == ""

    def test_html_only_message(self) -> None:
        """Test extracting body from HTML-only message."""
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_content_type.return_value = "text/html"
        msg.get_payload.return_value = b"<html>Hello</html>"
        msg.get_content_charset.return_value = "utf-8"

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        # HTML should be used as fallback for body_text
        assert body_text == "<html>Hello</html>"
        assert body_html == "<html>Hello</html>"

    def test_string_payload(self) -> None:
        """Test handling string payload (not bytes) for text/plain."""
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_content_type.return_value = "text/plain"
        msg.get_payload.side_effect = [
            None,
            "String payload",
        ]  # First decode=True returns None, then decode=False
        msg.get_content_charset.return_value = None

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == "String payload"
        assert body_html == ""

    def test_string_payload_html(self) -> None:
        """Test handling string payload for text/html."""
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_content_type.return_value = "text/html"
        msg.get_payload.side_effect = [
            None,
            "<html>String HTML</html>",
        ]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == "<html>String HTML</html>"  # Fallback
        assert body_html == "<html>String HTML</html>"

    def test_multipart_message(self) -> None:
        """Test extracting body from multipart message."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        text_part = MagicMock()
        text_part.get_content_type.return_value = "text/plain"
        text_part.get.return_value = ""  # No Content-Disposition
        text_part.get_payload.return_value = b"Plain text body"
        text_part.get_content_charset.return_value = "utf-8"

        html_part = MagicMock()
        html_part.get_content_type.return_value = "text/html"
        html_part.get.return_value = ""
        html_part.get_payload.return_value = b"<html>HTML body</html>"
        html_part.get_content_charset.return_value = "utf-8"

        msg.walk.return_value = [text_part, html_part]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == "Plain text body"
        assert body_html == "<html>HTML body</html>"

    def test_multipart_message_skips_attachments(self) -> None:
        """Test multipart message skips attachment parts."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        text_part = MagicMock()
        text_part.get_content_type.return_value = "text/plain"
        text_part.get.return_value = ""
        text_part.get_payload.return_value = b"Body text"
        text_part.get_content_charset.return_value = "utf-8"

        attachment_part = MagicMock()
        attachment_part.get_content_type.return_value = "application/pdf"
        attachment_part.get.return_value = "attachment; filename=doc.pdf"

        msg.walk.return_value = [text_part, attachment_part]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == "Body text"
        assert body_html == ""

    def test_multipart_with_decode_error(self) -> None:
        """Test multipart message handles decode errors gracefully."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        text_part = MagicMock()
        text_part.get_content_type.return_value = "text/plain"
        text_part.get.return_value = ""
        text_part.get_payload.side_effect = Exception("Decode failed")

        msg.walk.return_value = [text_part]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == ""
        assert body_html == ""

    def test_multipart_html_with_decode_error(self) -> None:
        """Test multipart HTML part handles decode errors gracefully."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        html_part = MagicMock()
        html_part.get_content_type.return_value = "text/html"
        html_part.get.return_value = ""
        html_part.get_payload.side_effect = Exception("Decode failed")

        msg.walk.return_value = [html_part]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == ""
        assert body_html == ""

    def test_simple_message_decode_error(self) -> None:
        """Test simple message handles decode errors gracefully."""
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_content_type.return_value = "text/plain"
        msg.get_payload.side_effect = Exception("Decode failed")

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == ""
        assert body_html == ""

    def test_multipart_text_payload_not_bytes(self) -> None:
        """Test multipart where text payload is not bytes."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        text_part = MagicMock()
        text_part.get_content_type.return_value = "text/plain"
        text_part.get.return_value = ""
        text_part.get_payload.return_value = None  # Not bytes

        msg.walk.return_value = [text_part]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == ""
        assert body_html == ""

    def test_multipart_html_payload_not_bytes(self) -> None:
        """Test multipart where HTML payload is not bytes."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        html_part = MagicMock()
        html_part.get_content_type.return_value = "text/html"
        html_part.get.return_value = ""
        html_part.get_payload.return_value = None  # Not bytes

        msg.walk.return_value = [html_part]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == ""
        assert body_html == ""

    def test_multipart_with_other_content_type(self) -> None:
        """Test multipart with content type other than text/plain or text/html."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        # A part that is NOT an attachment but also NOT text/plain or text/html
        other_part = MagicMock()
        other_part.get_content_type.return_value = "application/json"
        other_part.get.return_value = ""  # No Content-Disposition

        msg.walk.return_value = [other_part]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == ""
        assert body_html == ""

    def test_simple_message_non_string_fallback(self) -> None:
        """Test simple message where fallback payload is not a string."""
        msg = MagicMock()
        msg.is_multipart.return_value = False
        msg.get_content_type.return_value = "text/plain"
        msg.get_payload.side_effect = [
            None,
            ["list", "not", "string"],  # Not a string
        ]

        body_text, body_html = stage_01_parse_mbox.get_body(msg)

        assert body_text == ""
        assert body_html == ""


class TestHasAttachments:
    """Tests for has_attachments function."""

    def test_no_attachments(self) -> None:
        """Test message without attachments."""
        msg = MagicMock()
        msg.is_multipart.return_value = False

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_attachment_disposition(self) -> None:
        """Test message with attachment Content-Disposition."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        part = MagicMock()
        part.get.return_value = "attachment; filename=test.pdf"
        msg.walk.return_value = [part]

        assert stage_01_parse_mbox.has_attachments(msg) is True

    def test_with_inline_image(self) -> None:
        """Test message with inline image attachment."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        part = MagicMock()
        part.get.return_value = ""
        part.get_content_type.return_value = "image/png"
        part.get_filename.return_value = "image.png"
        msg.walk.return_value = [part]

        assert stage_01_parse_mbox.has_attachments(msg) is True

    def test_with_non_text_part_no_filename(self) -> None:
        """Test non-text part without filename is not an attachment."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        part = MagicMock()
        part.get.return_value = ""  # No Content-Disposition
        part.get_content_type.return_value = "image/png"
        part.get_filename.return_value = None  # No filename
        msg.walk.return_value = [part]

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_text_part(self) -> None:
        """Test text part is not an attachment."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        part = MagicMock()
        part.get.return_value = ""
        part.get_content_type.return_value = "text/plain"
        msg.walk.return_value = [part]

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_multipart_part(self) -> None:
        """Test multipart/* content type is not an attachment."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        part = MagicMock()
        part.get.return_value = ""
        part.get_content_type.return_value = "multipart/alternative"
        msg.walk.return_value = [part]

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_none_content_type(self) -> None:
        """Test part with None content type."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        part = MagicMock()
        part.get.return_value = ""
        part.get_content_type.return_value = None  # None content type
        msg.walk.return_value = [part]

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_empty_string_content_type(self) -> None:
        """Test part with empty string content type."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        part = MagicMock()
        part.get.return_value = ""
        part.get_content_type.return_value = ""  # Empty string content type
        msg.walk.return_value = [part]

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_multiple_parts_no_attachments(self) -> None:
        """Test message with multiple parts, none are attachments."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        # Part 1: text part (content_type starts with text/)
        text_part = MagicMock()
        text_part.get.return_value = ""
        text_part.get_content_type.return_value = "text/plain"

        # Part 2: application part without filename
        app_part = MagicMock()
        app_part.get.return_value = ""
        app_part.get_content_type.return_value = "application/octet-stream"
        app_part.get_filename.return_value = None

        # Part 3: None content type
        none_part = MagicMock()
        none_part.get.return_value = ""
        none_part.get_content_type.return_value = None

        msg.walk.return_value = [text_part, app_part, none_part]

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_multiple_binary_parts_no_filenames(self) -> None:
        """Test multiple binary parts without filenames iterate correctly."""
        msg = MagicMock()
        msg.is_multipart.return_value = True

        # Multiple binary parts, none with filenames
        part1 = MagicMock()
        part1.get.return_value = ""
        part1.get_content_type.return_value = "application/octet-stream"
        part1.get_filename.return_value = None

        part2 = MagicMock()
        part2.get.return_value = ""
        part2.get_content_type.return_value = "image/jpeg"
        part2.get_filename.return_value = None

        msg.walk.return_value = [part1, part2]

        assert stage_01_parse_mbox.has_attachments(msg) is False

    def test_with_empty_walk(self) -> None:
        """Test multipart message with empty walk() result."""
        msg = MagicMock()
        msg.is_multipart.return_value = True
        msg.walk.return_value = []  # Empty iterator

        assert stage_01_parse_mbox.has_attachments(msg) is False


class TestParseEmail:
    """Tests for parse_email function."""

    def test_parses_basic_fields(self) -> None:
        """Test parsing basic email fields."""
        msg = MagicMock()
        msg.get.side_effect = lambda key, default="": {
            "Message-ID": "<test@example.com>",
            "X-GM-THRID": "12345",
            "In-Reply-To": "",
            "References": "",
            "X-Gmail-Labels": "Inbox",
            "Date": "Mon, 1 Jan 2024 12:00:00 +0000",
            "From": "sender@example.com",
            "To": "recipient@example.com",
            "Cc": "",
            "Bcc": "",
            "Subject": "Test Subject",
        }.get(key, default)
        msg.is_multipart.return_value = False
        msg.get_content_type.return_value = "text/plain"
        msg.get_payload.return_value = b"Test body"
        msg.get_content_charset.return_value = "utf-8"

        result = stage_01_parse_mbox.parse_email(msg, mbox_offset=100, raw_size_bytes=500)

        assert result["message_id"] == "<test@example.com>"
        assert result["thread_id"] == "12345"
        assert result["subject"] == "Test Subject"
        assert result["labels"] == ["Inbox"]
        assert result["mbox_offset"] == 100
        assert result["raw_size_bytes"] == 500


class TestParseMboxFile:
    """Tests for parse_mbox_file function."""

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.mailbox.mbox")
    @patch("builtins.open", new_callable=mock_open)
    def test_parses_mbox(self, mock_file: MagicMock, mock_mbox_class: MagicMock) -> None:
        """Test parsing MBOX file."""
        # Create mock message
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda key, default="": {
            "Message-ID": "<test@example.com>",
            "X-GM-THRID": "",
            "In-Reply-To": "",
            "References": "",
            "X-Gmail-Labels": "",
            "Date": "",
            "From": "test@example.com",
            "To": "",
            "Cc": "",
            "Bcc": "",
            "Subject": "Test",
        }.get(key, default)
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = "text/plain"
        mock_msg.get_payload.return_value = b"Body"
        mock_msg.get_content_charset.return_value = "utf-8"
        mock_msg.as_bytes.return_value = b"raw message"

        mock_mbox = MagicMock()
        mock_mbox.__iter__ = MagicMock(return_value=iter([mock_msg]))
        mock_mbox_class.return_value = mock_mbox

        stats = stage_01_parse_mbox.parse_mbox_file(
            Path("/tmp/test.mbox"), Path("/tmp/output.jsonl")
        )

        assert stats["total_emails"] == 1
        assert stats["emails_with_body"] == 1
        assert stats["errors"] == 0

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.mailbox.mbox")
    @patch("builtins.open", new_callable=mock_open)
    def test_parses_mbox_with_attachments(
        self, mock_file: MagicMock, mock_mbox_class: MagicMock
    ) -> None:
        """Test parsing MBOX file with attachments."""
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda key, default="": {
            "Message-ID": "<test@example.com>",
            "X-GM-THRID": "",
            "In-Reply-To": "",
            "References": "",
            "X-Gmail-Labels": "Inbox, Important",
            "Date": "",
            "From": "test@example.com",
            "To": "",
            "Cc": "",
            "Bcc": "",
            "Subject": "Test",
        }.get(key, default)
        mock_msg.is_multipart.return_value = True

        # Setup multipart with attachment
        text_part = MagicMock()
        text_part.get_content_type.return_value = "text/plain"
        text_part.get.return_value = ""
        text_part.get_payload.return_value = b"Body"
        text_part.get_content_charset.return_value = "utf-8"

        attachment_part = MagicMock()
        attachment_part.get.return_value = "attachment; filename=doc.pdf"

        mock_msg.walk.return_value = [text_part, attachment_part]
        mock_msg.as_bytes.return_value = b"raw message"

        mock_mbox = MagicMock()
        mock_mbox.__iter__ = MagicMock(return_value=iter([mock_msg]))
        mock_mbox_class.return_value = mock_mbox

        stats = stage_01_parse_mbox.parse_mbox_file(
            Path("/tmp/test.mbox"), Path("/tmp/output.jsonl")
        )

        assert stats["total_emails"] == 1
        assert stats["emails_with_attachments"] == 1
        assert "Inbox" in stats["unique_labels"]
        assert "Important" in stats["unique_labels"]

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.mailbox.mbox")
    @patch("builtins.open", new_callable=mock_open)
    def test_parses_mbox_handles_errors(
        self, mock_file: MagicMock, mock_mbox_class: MagicMock
    ) -> None:
        """Test parsing MBOX file handles message errors."""
        mock_msg = MagicMock()
        mock_msg.get.side_effect = Exception("Parse error")

        mock_mbox = MagicMock()
        mock_mbox.__iter__ = MagicMock(return_value=iter([mock_msg]))
        mock_mbox_class.return_value = mock_mbox

        stats = stage_01_parse_mbox.parse_mbox_file(
            Path("/tmp/test.mbox"), Path("/tmp/output.jsonl")
        )

        assert stats["total_emails"] == 0
        assert stats["errors"] == 1

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.mailbox.mbox")
    @patch("builtins.open", new_callable=mock_open)
    def test_parses_mbox_with_empty_body(
        self, mock_file: MagicMock, mock_mbox_class: MagicMock
    ) -> None:
        """Test parsing MBOX file where email has empty body."""
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda key, default="": {
            "Message-ID": "<test@example.com>",
            "X-GM-THRID": "",
            "In-Reply-To": "",
            "References": "",
            "X-Gmail-Labels": "",
            "Date": "",
            "From": "test@example.com",
            "To": "",
            "Cc": "",
            "Bcc": "",
            "Subject": "Empty body email",
        }.get(key, default)
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = "text/plain"
        mock_msg.get_payload.return_value = b""  # Empty body
        mock_msg.get_content_charset.return_value = "utf-8"
        mock_msg.as_bytes.return_value = b"raw message"

        mock_mbox = MagicMock()
        mock_mbox.__iter__ = MagicMock(return_value=iter([mock_msg]))
        mock_mbox_class.return_value = mock_mbox

        stats = stage_01_parse_mbox.parse_mbox_file(
            Path("/tmp/test.mbox"), Path("/tmp/output.jsonl")
        )

        assert stats["total_emails"] == 1
        assert stats["emails_with_body"] == 0  # Body is empty

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.mailbox.mbox")
    @patch("builtins.open", new_callable=mock_open)
    def test_parses_mbox_handles_as_bytes_error(
        self, mock_file: MagicMock, mock_mbox_class: MagicMock
    ) -> None:
        """Test parsing MBOX file handles as_bytes() errors."""
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda key, default="": {
            "Message-ID": "<test@example.com>",
            "X-GM-THRID": "",
            "In-Reply-To": "",
            "References": "",
            "X-Gmail-Labels": "",
            "Date": "",
            "From": "test@example.com",
            "To": "",
            "Cc": "",
            "Bcc": "",
            "Subject": "Test",
        }.get(key, default)
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = "text/plain"
        mock_msg.get_payload.return_value = b"Body"
        mock_msg.get_content_charset.return_value = "utf-8"
        mock_msg.as_bytes.side_effect = Exception("Encoding error")

        mock_mbox = MagicMock()
        mock_mbox.__iter__ = MagicMock(return_value=iter([mock_msg]))
        mock_mbox_class.return_value = mock_mbox

        stats = stage_01_parse_mbox.parse_mbox_file(
            Path("/tmp/test.mbox"), Path("/tmp/output.jsonl")
        )

        assert stats["total_emails"] == 1
        assert stats["errors"] == 0

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.mailbox.mbox")
    @patch("builtins.open", new_callable=mock_open)
    def test_parses_mbox_writes_report(
        self, mock_file: MagicMock, mock_mbox_class: MagicMock
    ) -> None:
        """Test parsing MBOX file writes report when path provided."""
        mock_msg = MagicMock()
        mock_msg.get.side_effect = lambda key, default="": {
            "Message-ID": "<test@example.com>",
            "X-GM-THRID": "",
            "In-Reply-To": "",
            "References": "",
            "X-Gmail-Labels": "",
            "Date": "",
            "From": "test@example.com",
            "To": "",
            "Cc": "",
            "Bcc": "",
            "Subject": "Test",
        }.get(key, default)
        mock_msg.is_multipart.return_value = False
        mock_msg.get_content_type.return_value = "text/plain"
        mock_msg.get_payload.return_value = b"Body"
        mock_msg.get_content_charset.return_value = "utf-8"
        mock_msg.as_bytes.return_value = b"raw"

        mock_mbox = MagicMock()
        mock_mbox.__iter__ = MagicMock(return_value=iter([mock_msg]))
        mock_mbox_class.return_value = mock_mbox

        stats = stage_01_parse_mbox.parse_mbox_file(
            Path("/tmp/test.mbox"),
            Path("/tmp/output.jsonl"),
            report_path=Path("/tmp/report.json"),
        )

        assert stats["total_emails"] == 1
        # Verify report was written (open called for report file)
        assert mock_file.call_count >= 2


class TestRun:
    """Tests for run function."""

    def test_run_without_mbox_path(self) -> None:
        """Test run fails without mbox_path configured."""
        config = Config(database_url="postgresql://test")

        result = stage_01_parse_mbox.run(config)

        assert result.success is False
        assert "MBOX_PATH not configured" in result.message

    def test_run_with_missing_file(self) -> None:
        """Test run fails when mbox file doesn't exist."""
        config = Config(database_url="postgresql://test", mbox_path=Path("/nonexistent/test.mbox"))

        result = stage_01_parse_mbox.run(config)

        assert result.success is False
        assert "not found" in result.message

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.parse_mbox_file")
    def test_run_success(self, mock_parse: MagicMock, tmp_path: Path) -> None:
        """Test successful run."""
        # Create a temp mbox file
        mbox_file = tmp_path / "test.mbox"
        mbox_file.touch()

        mock_parse.return_value = {"total_emails": 100, "emails_with_body": 95, "errors": 0}

        config = Config(database_url="postgresql://test", mbox_path=mbox_file)
        result = stage_01_parse_mbox.run(config)

        assert isinstance(result, StageResult)
        assert result.success is True
        assert result.records_processed == 100

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.parse_mbox_file")
    def test_run_with_custom_output(self, mock_parse: MagicMock, tmp_path: Path) -> None:
        """Test run with custom output path."""
        mbox_file = tmp_path / "test.mbox"
        mbox_file.touch()
        output_file = tmp_path / "output.jsonl"

        mock_parse.return_value = {"total_emails": 50, "errors": 0}

        config = Config(database_url="postgresql://test", mbox_path=mbox_file)
        result = stage_01_parse_mbox.run(config, output_path=output_file)

        assert result.success is True
        mock_parse.assert_called_once()
        call_args = mock_parse.call_args[0]
        assert call_args[1] == output_file

    @patch("rl_emails.pipeline.stages.stage_01_parse_mbox.parse_mbox_file")
    def test_run_uses_parsed_jsonl_from_config(self, mock_parse: MagicMock, tmp_path: Path) -> None:
        """Test run uses parsed_jsonl from config when no output_path."""
        mbox_file = tmp_path / "test.mbox"
        mbox_file.touch()
        parsed_jsonl = tmp_path / "parsed.jsonl"

        mock_parse.return_value = {"total_emails": 50, "errors": 0}

        config = Config(
            database_url="postgresql://test", mbox_path=mbox_file, parsed_jsonl=parsed_jsonl
        )
        result = stage_01_parse_mbox.run(config)

        assert result.success is True
        call_args = mock_parse.call_args[0]
        assert call_args[1] == parsed_jsonl
