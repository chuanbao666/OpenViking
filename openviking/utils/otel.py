# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry Metrics utilities for OpenViking."""

from typing import Optional

from opentelemetry import metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from prometheus_client import start_http_server

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

_meter: Optional[metrics.Meter] = None


def init_otel(service_name: str = "openviking", port: int = 8000) -> metrics.Meter:
    """Initialize OTEL MeterProvider and start Prometheus exporter.

    Args:
        service_name: Name of the service for OTEL resource.
        port: Port to expose Prometheus metrics on.

    Returns:
        The initialized OTEL meter.
    """
    global _meter
    if _meter is not None:
        return _meter

    resource = Resource.create({"service.name": service_name})
    reader = PrometheusMetricReader()
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)
    _meter = metrics.get_meter(service_name)

    try:
        start_http_server(port=port)
        logger.info(f"Prometheus metrics exporter started on port {port}")
    except Exception as e:
        logger.warning(f"Failed to start Prometheus exporter on port {port}: {e}")

    return _meter


def get_meter() -> metrics.Meter:
    """Get the global meter instance.

    Returns:
        The global OTEL meter.
    """
    global _meter
    if _meter is None:
        # Fallback to default if not initialized
        _meter = metrics.get_meter("openviking")
    return _meter
