"""Custom exceptions for dataset operations."""


class DatasetError(Exception):
    """Base exception for dataset operational failures."""


class MissingVideoDataError(DatasetError):
    """Raised when real video media files for a requested dataset UID are missing.

    Synthetic or placeholder data fallback is strictly prohibited.
    """
