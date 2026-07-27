"""
data_provider/exceptions.py — shared exception hierarchy for data providers.

Every provider (BinanceDataProvider now, others later) catches its SDK's
native errors at the boundary and re-raises one of these, so callers never
need to know which SDK/exchange is behind a given DataProvider.
"""


class DataProviderError(Exception):
    """Base class for all data provider failures."""


class DataProviderNetworkError(DataProviderError):
    """Connection failed, timed out, or DNS could not resolve."""


class DataProviderHTTPError(DataProviderError):
    """Provider API responded with a non-success HTTP status."""


class DataProviderParseError(DataProviderError):
    """Response body could not be parsed into the expected shape."""
