"""
Custom exceptions and error handling for Readtube.
Provides user-friendly error messages and smart retry logic.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ReadtubeError(Exception):
    """Base exception for all Readtube errors."""

    def __init__(self, message: str, hint: Optional[str] = None):
        self.message = message
        self.hint = hint
        super().__init__(self.friendly_message)

    @property
    def friendly_message(self) -> str:
        """Return a user-friendly error message."""
        msg = f"Error: {self.message}"
        if self.hint:
            msg += f"\n\nHint: {self.hint}"
        return msg


class VideoNotFoundError(ReadtubeError):
    """Video could not be found or accessed."""

    def __init__(self, video_id: str, reason: Optional[str] = None):
        message = f"Video '{video_id}' not found"
        if reason:
            message += f": {reason}"
        hint = "Check that the video URL is correct and the video is publicly available."
        super().__init__(message, hint)


class TranscriptNotAvailableError(ReadtubeError):
    """Transcript is not available for this video."""

    def __init__(self, video_id: str, available_languages: Optional[list] = None):
        message = f"No transcript available for video '{video_id}'"
        hint = "This video may not have captions enabled."
        if available_languages:
            hint += f"\n\nAvailable languages: {', '.join(available_languages)}"
            hint += "\nTry: readtube URL --lang <code>"
        else:
            hint += "\nTry: readtube URL --list-languages to see available options."
        super().__init__(message, hint)


class RateLimitError(ReadtubeError):
    """YouTube is rate limiting requests."""

    def __init__(self, retry_after: Optional[int] = None):
        message = "YouTube is temporarily blocking requests from your IP"
        hint = """This is usually caused by:
- Too many requests in a short time
- Running from a cloud provider IP (AWS, GCP, Azure)

Solutions:
1. Wait a few minutes and try again
2. Use a VPN or different network
3. Run from a residential IP"""
        if retry_after:
            hint += f"\n\nRetry after: {retry_after} seconds"
        super().__init__(message, hint)


class NetworkError(ReadtubeError):
    """Network-related error."""

    def __init__(self, message: str):
        hint = """Check your internet connection and try again.
If the problem persists, YouTube may be experiencing issues."""
        super().__init__(message, hint)


class LLMError(ReadtubeError):
    """Error with LLM backend."""

    def __init__(self, backend: str, message: str):
        hint = f"""The {backend} backend encountered an error.

Available backends:
  python llm.py --list-backends

To use a different backend:
  python write_article.py video.json --backend <name>"""
        super().__init__(f"LLM error ({backend}): {message}", hint)


class OutputFormatError(ReadtubeError):
    """Error generating output format."""

    def __init__(self, format: str, message: str):
        hints = {
            "pdf": "PDF requires weasyprint. Install with: pip install weasyprint\nSystem deps: brew install pango cairo glib (macOS)",
            "mobi": "MOBI requires Calibre. Install from: https://calibre-ebook.com",
            "azw3": "AZW3 requires Calibre. Install from: https://calibre-ebook.com",
        }
        hint = hints.get(format, f"Check that {format} generation is properly configured.")
        super().__init__(f"Failed to generate {format}: {message}", hint)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: tuple = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions


# Default retry configs for different scenarios
YOUTUBE_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    max_delay=120.0,
    exponential_base=2.0,
    jitter=True,
)

LLM_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,
)

NETWORK_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=10.0,
    exponential_base=2.0,
    jitter=True,
)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt with exponential backoff."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter (±25%)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit error."""
    error_str = str(error).lower()
    rate_limit_indicators = [
        "rate limit",
        "too many requests",
        "429",
        "blocking",
        "ip ban",
        "blocked",
        "quota",
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is worth retrying."""
    # Rate limits are retryable
    if is_rate_limit_error(error):
        return True

    error_str = str(error).lower()

    # Network errors are retryable
    network_errors = [
        "timeout",
        "connection",
        "network",
        "temporary",
        "unavailable",
        "502",
        "503",
        "504",
    ]
    if any(err in error_str for err in network_errors):
        return True

    return False


def retry_with_backoff(
    func: Callable[[], T],
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> T:
    """
    Execute a function with retry and exponential backoff.

    Args:
        func: Function to execute
        config: Retry configuration
        on_retry: Callback called before each retry (exception, attempt, delay)

    Returns:
        Result of the function

    Raises:
        The last exception if all retries fail
    """
    if config is None:
        config = RetryConfig()

    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            return func()
        except config.retryable_exceptions as e:
            last_exception = e

            # Check if we should retry
            if attempt >= config.max_attempts - 1:
                break

            if not is_retryable_error(e):
                # Don't retry non-retryable errors
                raise

            # Calculate delay
            delay = calculate_delay(attempt, config)

            # Longer delay for rate limits
            if is_rate_limit_error(e):
                delay = max(delay, 30.0)  # At least 30 seconds for rate limits

            if on_retry:
                on_retry(e, attempt + 1, delay)
            else:
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

            time.sleep(delay)

    # All retries exhausted
    if last_exception:
        raise last_exception
    raise RuntimeError("Retry loop completed without result or exception")


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to a function."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_with_backoff(
                lambda: func(*args, **kwargs),
                config=config,
            )

        return wrapper

    return decorator


def format_error_for_user(error: Exception) -> str:
    """Format any exception into a user-friendly message."""
    if isinstance(error, ReadtubeError):
        return error.friendly_message

    error_str = str(error).lower()

    # Detect common error types and provide helpful messages
    if is_rate_limit_error(error):
        return RateLimitError().friendly_message

    if "transcript" in error_str and ("not available" in error_str or "disabled" in error_str):
        return TranscriptNotAvailableError("unknown").friendly_message

    if "video" in error_str and ("unavailable" in error_str or "not found" in error_str or "private" in error_str):
        return VideoNotFoundError("unknown", str(error)).friendly_message

    if "connection" in error_str or "timeout" in error_str or "network" in error_str:
        return NetworkError(str(error)).friendly_message

    # Generic error
    return f"Error: {error}\n\nIf this persists, please report at: https://github.com/unbalancedparentheses/readtube/issues"
