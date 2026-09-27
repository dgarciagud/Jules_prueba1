"""Universo del informe diario: índices, sectores y factores (config/daily_universe.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "config" / "daily_universe.yaml"


@dataclass(frozen=True)
class Series:
    """Una serie descargable. `id` es la clave en el almacén de precios."""

    id: str
    name: str
    role: str                 # index | sector | component | factor
    kind: str = "price"       # price | yield
    yahoo: str | None = None
    stooq: str | None = None
    index: str | None = None  # índice al que pertenece (sector/componente)
    sector: str | None = None  # sector fijado en la configuración


@dataclass
class Index:
    id: str
    name: str
    region: str
    weights_from_fund: str | None
    composition_since: str | None = None  # última revisión de la composición (limita la ventana de pesos)
    shrink: float = 0.0                   # encogimiento de los pesos estimados hacia los del fondo
    sectors: dict[str, Series] = field(default_factory=dict)       # sector -> ETF
    components: list[Series] = field(default_factory=list)          # acciones

    @property
    def sector_mode(self) -> str:
        return "etf" if self.sectors else "components"


@dataclass
class DailyUniverse:
    sector_names: dict[str, str]
    factors: dict[str, Series]
    indices: dict[str, Index]
    index_series: dict[str, Series]

    def all_series(self) -> list[Series]:
        out = list(self.index_series.values()) + list(self.factors.values())
        for ix in self.indices.values():
            out += list(ix.sectors.values()) + ix.components
        return out


def load_daily_universe(path: Path | str | None = None) -> DailyUniverse:
    cfg = yaml.safe_load(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))
    sector_names = cfg["sectors"]
    factors = {
        k: Series(id=k, name=v["name"], role="factor", kind=v.get("kind", "price"), yahoo=v.get("yahoo"), stooq=v.get("stooq"))
        for k, v in cfg["factors"].items()
    }
    indices, index_series = {}, {}
    for iid, v in cfg["indices"].items():
        index_series[iid] = Series(id=iid, name=v["name"], role="index", yahoo=v.get("yahoo"), stooq=v.get("stooq"))
        ix = Index(id=iid, name=v["name"], region=v["region"], weights_from_fund=v.get("weights_from_fund"),
                   composition_since=str(v["composition_since"]) if v.get("composition_since") else None,
                   shrink=float(v.get("shrink", 0.0)))
        for sec, s in (v.get("sectors") or {}).items():
            if sec not in sector_names:
                raise ValueError(f"{iid}: sector desconocido {sec}")
            ix.sectors[sec] = Series(id=f"{iid}:{sec}", name=f"{v['name']} · {sector_names[sec]}", role="sector",
                                     yahoo=s.get("yahoo"), stooq=s.get("stooq"), index=iid, sector=sec)  # fmt: skip
        for c in v.get("components") or []:
            if c.get("sector") and c["sector"] not in sector_names:
                raise ValueError(f"{iid}: sector desconocido {c['sector']} en {c['yahoo']}")
            ix.components.append(Series(id=c["yahoo"], name=c["name"], role="component", yahoo=c["yahoo"],
                                        stooq=c.get("stooq"), index=iid, sector=c.get("sector")))  # fmt: skip
        if not ix.sectors and not ix.components:
            raise ValueError(f"{iid}: sin sectores ni componentes")
        indices[iid] = ix
    ids = [s.id for s in [*index_series.values(), *factors.values()] + [s for ix in indices.values() for s in [*ix.sectors.values(), *ix.components]]]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"series duplicadas: {sorted(dup)}")
    return DailyUniverse(sector_names=sector_names, factors=factors, indices=indices, index_series=index_series)
