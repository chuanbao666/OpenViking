# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Context Processor for OpenViking.

Handles coordinated writes and self-iteration processes
as described in the OpenViking design document.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from openviking.parse.tree_builder import TreeBuilder
from openviking.server.identity import RequestContext
from openviking.storage import VikingDBManager
from openviking.storage.viking_fs import get_viking_fs
from openviking.utils.embedding_utils import index_resource
from openviking.utils.otel import get_meter
from openviking.utils.summarizer import Summarizer
from openviking_cli.utils import get_logger
from openviking_cli.utils.storage import StoragePath

if TYPE_CHECKING:
    from openviking.parse.vlm import VLMProcessor

logger = get_logger(__name__)


class ResourceProcessor:
    """
    Handles coordinated write operations.

    When new data is added, automatically:
    1. Download if URL (prefer PDF format)
    2. Parse and structure the content (Parser writes to temp directory)
    3. Extract images/tables for mixed content
    4. Use VLM to understand non-text content
    5. TreeBuilder finalizes from temp (move to AGFS)
    6. SemanticQueue generates L0/L1 and vectorizes asynchronously
    """

    def __init__(
        self,
        vikingdb: VikingDBManager,
        media_storage: Optional["StoragePath"] = None,
        max_context_size: int = 2000,
        max_split_depth: int = 3,
    ):
        """Initialize coordinated writer."""
        self.vikingdb = vikingdb
        self.embedder = vikingdb.get_embedder()
        self.media_storage = media_storage
        self.tree_builder = TreeBuilder()
        self._vlm_processor = None
        self._media_processor = None
        self._summarizer = None

        # Initialize metrics
        meter = get_meter()
        self._process_total = meter.create_counter(
            "resource_process_total",
            description="Total number of resource processing requests",
        )
        self._process_duration = meter.create_histogram(
            "resource_process_duration_seconds",
            unit="s",
            description="Duration of resource processing",
        )
        self._phase_duration = meter.create_histogram(
            "resource_process_phase_seconds",
            unit="s",
            description="Duration of each phase in resource processing",
        )

    def _get_summarizer(self) -> "Summarizer":
        """Lazy initialization of Summarizer."""
        if self._summarizer is None:
            self._summarizer = Summarizer(self._get_vlm_processor())
        return self._summarizer

    def _get_vlm_processor(self) -> "VLMProcessor":
        """Lazy initialization of VLM processor."""
        if self._vlm_processor is None:
            from openviking.parse.vlm import VLMProcessor

            self._vlm_processor = VLMProcessor()
        return self._vlm_processor

    def _get_media_processor(self):
        """Lazy initialization of unified media processor."""
        if self._media_processor is None:
            from openviking.utils.media_processor import UnifiedResourceProcessor

            self._media_processor = UnifiedResourceProcessor(
                vlm_processor=self._get_vlm_processor(),
                storage=self.media_storage,
            )
        return self._media_processor

    async def build_index(
        self, resource_uris: List[str], ctx: RequestContext, **kwargs
    ) -> Dict[str, Any]:
        """Expose index building as a standalone method."""
        for uri in resource_uris:
            await index_resource(uri, ctx)
        return {"status": "success", "message": f"Indexed {len(resource_uris)} resources"}

    async def summarize(
        self, resource_uris: List[str], ctx: RequestContext, **kwargs
    ) -> Dict[str, Any]:
        """Expose summarization as a standalone method."""
        return await self._get_summarizer().summarize(resource_uris, ctx, **kwargs)

    async def process_resource(
        self,
        path: str,
        ctx: RequestContext,
        reason: str = "",
        instruction: str = "",
        scope: str = "resources",
        user: Optional[str] = None,
        to: Optional[str] = None,
        parent: Optional[str] = None,
        summarize: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Process and store a new resource.

        Workflow:
        1. Parse source (writes to temp directory)
        2. TreeBuilder moves to AGFS
        3. (Optional) Build vector index
        4. (Optional) Summarize
        """
        start_time = time.time()
        self._process_total.add(1, {"scope": scope})

        result = {
            "status": "success",
            "errors": [],
            "source_path": None,
        }

        # ============ Phase 1: Parse source (Parser generates L0/L1 and writes to temp) ============
        phase_start = time.time()
        try:
            media_processor = self._get_media_processor()
            viking_fs = get_viking_fs()
            # Use reason as instruction fallback so it influences L0/L1
            # generation and improves search relevance as documented.
            effective_instruction = instruction or reason
            with viking_fs.bind_request_context(ctx):
                parse_result = await media_processor.process(
                    source=path,
                    instruction=effective_instruction,
                    **kwargs,
                )
            result["source_path"] = parse_result.source_path or path
            result["meta"] = parse_result.meta

            # Only abort when no temp content was produced at all.
            # For directory imports partial success (some files failed) is
            # normal – finalization should still proceed.
            if not parse_result.temp_dir_path:
                result["status"] = "error"
                result["errors"].extend(
                    parse_result.warnings or ["Parse failed: no content generated"],
                )
                self._phase_duration.record(
                    time.time() - phase_start, {"phase": "parse", "status": "error"}
                )
                return result

            if parse_result.warnings:
                result["errors"].extend(parse_result.warnings)

            self._phase_duration.record(
                time.time() - phase_start, {"phase": "parse", "status": "success"}
            )

        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Parse error: {e}")
            logger.error(f"[ResourceProcessor] Parse error: {e}")
            import traceback

            traceback.print_exc()
            self._phase_duration.record(
                time.time() - phase_start, {"phase": "parse", "status": "error"}
            )
            return result

        # parse_result contains:
        # - root: ResourceNode tree (with L0/L1 in meta)
        # - temp_dir_path: Temporary directory path (Parser wrote all files)
        # - source_path, source_format

        # ============ Phase 2: Pass to and parent directly to TreeBuilder ============
        # ============ Phase 3: TreeBuilder finalizes from temp (scan + move to AGFS) ============
        phase_start = time.time()
        try:
            with get_viking_fs().bind_request_context(ctx):
                context_tree = await self.tree_builder.finalize_from_temp(
                    temp_dir_path=parse_result.temp_dir_path,
                    ctx=ctx,
                    scope=scope,
                    to_uri=to,
                    parent_uri=parent,
                    source_path=parse_result.source_path,
                    source_format=parse_result.source_format,
                )
                if context_tree and context_tree.root:
                    result["root_uri"] = context_tree.root.uri
            self._phase_duration.record(
                time.time() - phase_start, {"phase": "finalize", "status": "success"}
            )
        except Exception as e:
            result["status"] = "error"
            result["errors"].append(f"Finalize from temp error: {e}")
            self._phase_duration.record(
                time.time() - phase_start, {"phase": "finalize", "status": "error"}
            )

            # Cleanup temporary directory on error (via VikingFS)
            try:
                if parse_result.temp_dir_path:
                    await get_viking_fs().delete_temp(parse_result.temp_dir_path, ctx=ctx)
            except Exception:
                pass

            return result

        # ============ Phase 4: Optional Steps ============
        build_index = kwargs.get("build_index", True)
        if summarize:
            phase_start = time.time()
            # Explicit summarization request.
            # If build_index is ALSO True, we want vectorization.
            # If build_index is False, we skip vectorization.
            skip_vec = not build_index
            try:
                summarize_result = await self.summarize(
                    resource_uris=[result["root_uri"]],
                    ctx=ctx,
                    skip_vectorization=skip_vec,
                    **kwargs,
                )
                result["summarize"] = summarize_result
                self._phase_duration.record(
                    time.time() - phase_start, {"phase": "summarize", "status": "success"}
                )
            except Exception as e:
                logger.error(f"Summarize failed: {e}")
                result["warnings"] = result.get("warnings", []) + [f"Summarize failed: {e}"]
                self._phase_duration.record(
                    time.time() - phase_start, {"phase": "summarize", "status": "error"}
                )

        elif build_index:
            phase_start = time.time()
            # No explicit summary, but auto-index is requested.
            try:
                await self.build_index(
                    resource_uris=[result["root_uri"]], ctx=ctx, skip_vectorization=False, **kwargs
                )
                self._phase_duration.record(
                    time.time() - phase_start, {"phase": "index", "status": "success"}
                )
            except Exception as e:
                logger.error(f"Auto-index failed: {e}")
                result["warnings"] = result.get("warnings", []) + [f"Auto-index failed: {e}"]
                self._phase_duration.record(
                    time.time() - phase_start, {"phase": "index", "status": "error"}
                )

        self._process_duration.record(time.time() - start_time, {"status": result["status"]})
        return result
