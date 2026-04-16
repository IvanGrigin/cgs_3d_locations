# -*- coding: utf-8 -*-
"""
This module registers supported supplier adapters for the main pipeline.
It is the single place that maps URLs to concrete adapter instances.
Runner and batch tools depend on this registry for site dispatch.
The registry should stay explicit rather than dynamically discovered.
Keep adapter ordering predictable and easy to inspect.
"""
from __future__ import annotations

from src.suppliers.adapters.base import SupplierAdapter
from src.suppliers.adapters.homeconcept import HomeConceptAdapter
from src.suppliers.adapters.imodern import IModernAdapter
from src.suppliers.adapters.loftdesigne import LoftDesigneAdapter
from src.suppliers.adapters.three_ddd import ThreeDDDAdapter


ADAPTER_CLASSES = (
    HomeConceptAdapter,
    IModernAdapter,
    LoftDesigneAdapter,
    ThreeDDDAdapter,
)


def build_adapters() -> list[SupplierAdapter]:
    return [adapter_cls() for adapter_cls in ADAPTER_CLASSES]


def find_adapter(url: str) -> SupplierAdapter:
    for adapter in build_adapters():
        if adapter.can_handle(url):
            return adapter
    raise ValueError(f"Нет адаптера для URL: {url}")


def get_adapter_for_url(url: str) -> SupplierAdapter:
    return find_adapter(url)
