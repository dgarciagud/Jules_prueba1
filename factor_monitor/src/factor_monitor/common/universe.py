"""Carga y validación de `config/universe.yaml`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .sessions import SESSIONS

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "universe.yaml"
KINDS = {"target", "factor", "component"}


class UniverseError(ValueError):
    pass


@dataclass(frozen=True)
class Instrument:
    id: str
    kind: str
    dukascopy_id: str | None = None
    dukascopy_code: str | None = None
    mt5_symbol: str | None = None
    yahoo_ticker: str | None = None
    stooq_ticker: str | None = None
    session: str | None = None
    exchange: str | None = None
    point_factor: float | None = None


@dataclass(frozen=True)
class Factor:
    id: str
    instrument: str | None = None
    sign: int = 1
    long: str | None = None
    short: str | None = None
    contains: dict[str, float] = field(default_factory=dict)

    @property
    def instruments(self) -> tuple[str, ...]:
        if self.instrument is not None:
            return (self.instrument,)
        return (self.long, self.short)  # type: ignore[return-value]


@dataclass(frozen=True)
class Market:
    type: str  # "none" | "mean"
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class Target:
    id: str
    session: str
    exchange: str | None
    candidates: tuple[str, ...]
    market: Market
    basket: tuple[str, ...] = ()
    reference_etf: str | None = None


@dataclass(frozen=True)
class Exclusion:
    target: str
    factor: str
    reason: str


@dataclass(frozen=True)
class Universe:
    instruments: dict[str, Instrument]
    factors: dict[str, Factor]
    targets: dict[str, Target]
    overlap_threshold: float = 0.20

    def effective_candidates(self, target_id: str) -> tuple[tuple[str, ...], list[Exclusion]]:
        """Candidatos del objetivo tras aplicar la regla de solapamiento (§5.3)."""
        target = self.targets[target_id]
        kept, excluded = [], []
        for f in target.candidates:
            weight = abs(self.factors[f].contains.get(target_id, 0.0))
            if weight > self.overlap_threshold:
                excluded.append(
                    Exclusion(target_id, f, f"contiene al objetivo con peso {weight:.0%} > {self.overlap_threshold:.0%}")
                )
            else:
                kept.append(f)
        return tuple(kept), excluded

    def required_instruments(self, target_id: str) -> set[str]:
        """Instrumentos necesarios para modelizar el objetivo."""
        target = self.targets[target_id]
        needed = set(target.basket) if target.basket else {target_id}
        needed |= set(target.market.members)
        for f in self.effective_candidates(target_id)[0]:
            needed |= set(self.factors[f].instruments)
        return needed

    def downloadable(self) -> dict[str, Instrument]:
        """Instrumentos con ID de Dukascopy."""
        return {k: v for k, v in self.instruments.items() if v.dukascopy_id}


def load_universe(path: str | Path = DEFAULT_PATH) -> Universe:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return parse_universe(raw)


def parse_universe(raw: dict) -> Universe:
    instruments = {k: Instrument(id=k, **v) for k, v in (raw.get("instruments") or {}).items()}
    factors = {k: Factor(id=k, **v) for k, v in (raw.get("factors") or {}).items()}
    targets = {}
    for k, v in (raw.get("targets") or {}).items():
        v = dict(v)
        m = v.pop("market", {"type": "none"})
        targets[k] = Target(
            id=k,
            session=v.pop("session"),
            exchange=v.pop("exchange", None),
            candidates=tuple(v.pop("candidates", [])),
            market=Market(type=m["type"], members=tuple(m.get("members", []))),
            basket=tuple(v.pop("basket", [])),
            reference_etf=v.pop("reference_etf", None),
            **v,
        )
    universe = Universe(instruments, factors, targets, float(raw.get("overlap_threshold", 0.20)))
    _validate(universe)
    return universe


def _validate(u: Universe) -> None:
    errors = []
    for inst in u.instruments.values():
        if inst.kind not in KINDS:
            errors.append(f"{inst.id}: tipo desconocido {inst.kind!r}")
        if inst.session is not None and inst.session not in SESSIONS:
            errors.append(f"{inst.id}: sesión desconocida {inst.session!r}")
    for f in u.factors.values():
        simple = f.instrument is not None
        spread = f.long is not None and f.short is not None
        if simple == spread:
            errors.append(f"factor {f.id}: debe tener `instrument` o bien `long` y `short`")
        if f.sign not in (1, -1):
            errors.append(f"factor {f.id}: signo {f.sign} no válido")
        for i in f.instruments:
            if i is not None and i not in u.instruments:
                errors.append(f"factor {f.id}: instrumento desconocido {i}")
    for t in u.targets.values():
        if t.session not in SESSIONS:
            errors.append(f"objetivo {t.id}: sesión desconocida {t.session!r}")
        if not t.basket and t.id not in u.instruments:
            errors.append(f"objetivo {t.id}: no es un instrumento ni una cesta")
        for c in t.candidates:
            if c not in u.factors:
                errors.append(f"objetivo {t.id}: candidato desconocido {c}")
        if t.market.type not in ("none", "mean"):
            errors.append(f"objetivo {t.id}: tipo de mercado {t.market.type!r} no válido")
        if t.market.type == "mean" and not t.market.members:
            errors.append(f"objetivo {t.id}: mercado 'mean' sin miembros")
        if t.id in t.market.members:
            errors.append(f"objetivo {t.id}: el mercado no puede contener al propio objetivo (leave-one-out)")
        for m in t.market.members:
            if m not in u.instruments:
                errors.append(f"objetivo {t.id}: miembro de mercado desconocido {m}")
        for c in t.basket:
            if c not in u.instruments:
                errors.append(f"objetivo {t.id}: componente desconocido {c}")
    if errors:
        raise UniverseError("universe.yaml no válido:\n  " + "\n  ".join(errors))
