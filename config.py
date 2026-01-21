"""
Central configuration for Readtube.
Handles config file loading, defaults, and validation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('readtube')

# Default paths
CONFIG_DIR = Path.home() / '.config' / 'readtube'
CONFIG_FILE = CONFIG_DIR / 'config.json'
CACHE_DIR = Path(__file__).parent / '.transcript_cache'
OUTPUT_DIR = Path.cwd()


@dataclass
class TypographyConfig:
    """Typography settings based on Practical Typography."""
    font_family: str = "Charter, Georgia, serif"
    font_size: str = "1.1em"
    line_height: float = 1.4
    max_width: str = "65ch"
    heading_scale: List[float] = field(default_factory=lambda: [1.5, 1.2, 1.05, 1.0])


@dataclass
class OutputConfig:
    """Output format settings."""
    default_format: str = "epub"
    output_dir: str = "."
    include_cover: bool = True
    include_toc: bool = True
    page_size: str = "A5"  # For PDF


@dataclass
class FetchConfig:
    """Fetching and caching settings."""
    cache_enabled: bool = True
    cache_days: int = 7
    retry_attempts: int = 3
    retry_delay: float = 1.0
    preferred_language: Optional[str] = None
    preserve_timestamps: bool = False


@dataclass
class Config:
    """Main configuration class."""
    typography: TypographyConfig = field(default_factory=TypographyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    channels: List[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'Config':
        """Load config from file or return defaults."""
        config_path = path or CONFIG_FILE

        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return cls()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            typography=TypographyConfig(**data.get('typography', {})),
            output=OutputConfig(**data.get('output', {})),
            fetch=FetchConfig(**data.get('fetch', {})),
            channels=data.get('channels', []),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'typography': asdict(self.typography),
            'output': asdict(self.output),
            'fetch': asdict(self.fetch),
            'channels': self.channels,
        }

    def save(self, path: Optional[Path] = None) -> None:
        """Save config to file."""
        config_path = path or CONFIG_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Config saved to {config_path}")


@dataclass
class BatchJob:
    """A single job in a batch config."""
    url: str
    title: Optional[str] = None
    output_format: str = "epub"
    output_path: Optional[str] = None
    language: Optional[str] = None
    summary_mode: bool = False


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    jobs: List[BatchJob] = field(default_factory=list)
    output_dir: str = "."
    default_format: str = "epub"
    default_language: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> 'BatchConfig':
        """Load batch config from YAML or JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Batch config not found: {path}")

        with open(path, 'r') as f:
            if path.suffix in ('.yml', '.yaml'):
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config: pip install pyyaml")
            else:
                data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchConfig':
        """Create batch config from dictionary."""
        jobs = []
        for job_data in data.get('jobs', []):
            if isinstance(job_data, str):
                # Simple URL string
                jobs.append(BatchJob(url=job_data))
            else:
                jobs.append(BatchJob(**job_data))

        return cls(
            jobs=jobs,
            output_dir=data.get('output_dir', '.'),
            default_format=data.get('default_format', 'epub'),
            default_language=data.get('default_language'),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'jobs': [asdict(job) for job in self.jobs],
            'output_dir': self.output_dir,
            'default_format': self.default_format,
            'default_language': self.default_language,
        }


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
