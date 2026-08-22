"""Exception hierarchy. Every failure mode gets a named type so callers can
decide what is retryable, what halts the cycle, and what is merely skipped."""


class TraderError(Exception):
    """Base for every error raised by this package."""


class ConfigError(TraderError):
    """Configuration is missing or internally inconsistent."""


class BrokerError(TraderError):
    """The brokerage rejected a request or returned something unusable."""


class OrderRejected(BrokerError):
    """A specific order was refused. Non-fatal: skip the symbol, keep going."""


class MarketDataError(TraderError):
    """Market data could not be retrieved."""


class StaleDataError(MarketDataError):
    """Data arrived but is too old to trade on."""


class LLMError(TraderError):
    """The model call failed at the transport level."""


class LLMResponseError(LLMError):
    """The model replied, but not with something matching the schema."""


class RiskHalt(TraderError):
    """A risk limit forbids opening new exposure this cycle."""
