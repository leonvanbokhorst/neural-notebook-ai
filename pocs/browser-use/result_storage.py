"""
Result Storage Module for Browser Automation
-----------------------------------------

Handles storage and analysis of browser automation results in a structured way.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class BrowserResultEncoder(json.JSONEncoder):
    """Custom JSON encoder for browser automation results."""

    def default(self, obj: Any) -> Any:
        # Handle datetime objects
        if isinstance(obj, datetime):
            return obj.isoformat()

        # Handle custom objects with to_dict method
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        # Handle custom objects with __dict__ attribute
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        try:
            # Try to convert to a basic type
            return str(obj)
        except Exception:
            return f"<non-serializable: {type(obj).__name__}>"


@dataclass
class BrowserResult:
    """Represents a single browser automation result."""

    timestamp: str
    task: str
    content: Any
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None

    @classmethod
    def create(
        cls,
        task: str,
        content: Any,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BrowserResult":
        """Factory method to create a new result."""
        return cls(
            timestamp=datetime.utcnow().isoformat(),
            task=task,
            content=content,
            metadata=metadata or {},
            success=success,
            error=error,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "task": self.task,
            "content": self._process_content(self.content),
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
        }

    def _process_content(self, content: Any) -> Any:
        """Process content to ensure it's serializable."""
        if hasattr(content, "all_results"):
            # Handle AgentHistoryList
            return [self._process_content(r) for r in content.all_results]
        elif hasattr(content, "extracted_content"):
            # Handle ActionResult
            return {
                "content": content.extracted_content,
                "success": not content.error,
                "error": str(content.error) if content.error else None,
            }
        return content


class ResultStorage:
    """Manages storage and retrieval of browser automation results."""

    def __init__(self, storage_dir: Union[str, Path]):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging for the storage module."""
        log_file = self.storage_dir / "storage.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    def store_result(self, result: BrowserResult) -> Path:
        """Store a browser automation result."""
        # Create filename based on timestamp and task
        timestamp = datetime.fromisoformat(result.timestamp)
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{self._sanitize_filename(result.task)}.json"
        file_path = self.storage_dir / filename

        # Store the result using custom encoder
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                result.to_dict(),
                f,
                indent=2,
                ensure_ascii=False,
                cls=BrowserResultEncoder,
            )

        logger.info(f"Stored result in {file_path}")
        return file_path

    def load_result(self, file_path: Union[str, Path]) -> BrowserResult:
        """Load a stored result."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BrowserResult(**data)

    def list_results(self, task_filter: Optional[str] = None) -> List[Path]:
        """List all stored results, optionally filtered by task."""
        results = list(self.storage_dir.glob("*.json"))
        if task_filter:
            results = [r for r in results if task_filter.lower() in r.stem.lower()]
        return sorted(results)

    def analyze_results(self, results: List[Path]) -> Dict[str, Any]:
        """Analyze a list of results and return statistics."""
        stats = {
            "total_results": len(results),
            "success_rate": 0,
            "common_errors": {},
            "tasks": {},
            "avg_content_length": 0,
        }

        for result_path in results:
            result = self.load_result(result_path)

            # Update success rate
            if result.success:
                stats["success_rate"] += 1

            # Track tasks
            if result.task not in stats["tasks"]:
                stats["tasks"][result.task] = 0
            stats["tasks"][result.task] += 1

            # Track errors
            if result.error:
                if result.error not in stats["common_errors"]:
                    stats["common_errors"][result.error] = 0
                stats["common_errors"][result.error] += 1

            # Calculate content length
            if isinstance(result.content, (str, list, dict)):
                stats["avg_content_length"] += len(str(result.content))

        # Finalize calculations
        if stats["total_results"] > 0:
            stats["success_rate"] = (
                stats["success_rate"] / stats["total_results"]
            ) * 100
            stats["avg_content_length"] /= stats["total_results"]

        return stats

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize the filename by removing invalid characters."""
        return "".join(c for c in filename if c.isalnum() or c in ("-", "_")).lower()[
            :50
        ]


def create_result_storage(base_dir: Optional[Union[str, Path]] = None) -> ResultStorage:
    """Create a ResultStorage instance with default or custom base directory."""
    if base_dir is None:
        base_dir = Path.cwd() / "results"
    return ResultStorage(base_dir)
