"""Pydantic models for ticker data: MarketData, MetricsData, TickerInfo."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from constants import MARKET_MAX_AGE_HOURS, METRICS_MAX_AGE_DAYS, STATIC_MAX_AGE_DAYS


class MarketData(BaseModel):
    """Market data region - refreshed every 24 hours."""

    price: float | None = None
    previous_close: float | None = None
    volume: int | None = None
    day_high: float | None = None
    day_low: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    updated_at: datetime | None = None

    def is_stale(self, max_age_hours: int = MARKET_MAX_AGE_HOURS) -> bool:
        if not self.updated_at:
            return True
        age = datetime.now(timezone.utc) - self.updated_at
        return age.total_seconds() > max_age_hours * 3600


class MetricsData(BaseModel):
    """Calculated metrics region - refreshed every 7 days."""

    returns: float | None = None
    volatility: float | None = None
    updated_at: datetime | None = None

    def is_stale(self, max_age_days: int = METRICS_MAX_AGE_DAYS) -> bool:
        if not self.updated_at:
            return True
        age = datetime.now(timezone.utc) - self.updated_at
        return age.days > max_age_days

    def has_data(self) -> bool:
        return self.returns is not None and self.volatility is not None


class TickerInfo(BaseModel):
    """Stock ticker with composed data regions.

    Static fields are refreshed every 30 days.
    Market and metrics have independent refresh cycles.
    """

    # Static fields
    symbol: str
    name: str = ""
    long_name: str = ""
    exchange: str = ""
    type: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    first_trade_date: str = ""
    updated_at: datetime | None = None

    # Composed data regions
    market: MarketData = Field(default_factory=MarketData)
    metrics: MetricsData = Field(default_factory=MetricsData)

    def is_static_stale(self, max_age_days: int = STATIC_MAX_AGE_DAYS) -> bool:
        if not self.updated_at:
            return True
        age = datetime.now(timezone.utc) - self.updated_at
        return age.days > max_age_days

    def to_summary(self) -> dict:
        """Minimal dict for listing (name, sector, industry)."""
        return {"name": self.name, "sector": self.sector, "industry": self.industry}

    def to_flat_dict(self) -> dict:
        """Flatten for compatibility with scripts reading cache."""
        result = {
            "symbol": self.symbol,
            "name": self.name,
            "long_name": self.long_name,
            "exchange": self.exchange,
            "type": self.type,
            "sector": self.sector,
            "industry": self.industry,
            "currency": self.currency,
            "first_trade_date": self.first_trade_date,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if self.market:
            result.update({
                "price": self.market.price,
                "previous_close": self.market.previous_close,
                "volume": self.market.volume,
                "day_high": self.market.day_high,
                "day_low": self.market.day_low,
                "week_52_high": self.market.week_52_high,
                "week_52_low": self.market.week_52_low,
                "market_updated_at": self.market.updated_at.isoformat() if self.market.updated_at else None,
            })
        if self.metrics:
            result.update({
                "returns": self.metrics.returns,
                "volatility": self.metrics.volatility,
                "metrics_updated_at": self.metrics.updated_at.isoformat() if self.metrics.updated_at else None,
            })
        return result

    @classmethod
    def from_flat_dict(cls, data: dict) -> "TickerInfo":
        """Create from flat dict (cache format)."""
        # Parse timestamps
        def parse_dt(val: str | None) -> datetime | None:
            if not val:
                return None
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                return None

        market = MarketData(
            price=data.get("price"),
            previous_close=data.get("previous_close"),
            volume=data.get("volume"),
            day_high=data.get("day_high"),
            day_low=data.get("day_low"),
            week_52_high=data.get("week_52_high"),
            week_52_low=data.get("week_52_low"),
            updated_at=parse_dt(data.get("market_updated_at")),
        )

        metrics = MetricsData(
            returns=data.get("returns"),
            volatility=data.get("volatility"),
            updated_at=parse_dt(data.get("metrics_updated_at")),
        )

        return cls(
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            long_name=data.get("long_name", ""),
            exchange=data.get("exchange", ""),
            type=data.get("type", ""),
            sector=data.get("sector", ""),
            industry=data.get("industry", ""),
            currency=data.get("currency", ""),
            first_trade_date=data.get("first_trade_date", ""),
            updated_at=parse_dt(data.get("updated_at")),
            market=market,
            metrics=metrics,
        )

    @classmethod
    def empty(cls, symbol: str) -> "TickerInfo":
        """Create placeholder for unknown ticker."""
        return cls(symbol=symbol, name=symbol)
