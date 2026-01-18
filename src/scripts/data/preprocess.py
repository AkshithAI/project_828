"""
NeMo Curator Data Preprocessing Pipeline

This module provides a robust and configurable data preprocessing pipeline using NVIDIA NeMo Curator.
It supports:
- Custom dataset curation from HuggingFace or local files
- Predefined dataset pipelines (ArXiv, Common Crawl, Wikipedia)
- Multiple filtering and modification stages
- Exact, fuzzy, and semantic deduplication
- AWS S3 integration for predefined datasets

Usage:
    python preprocess.py --custom --repo_id <hf_repo_id>
    python preprocess.py --data_tag arxiv --download_dir ./data
    python preprocess.py --custom --dedup
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

# NeMo Curator imports
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader
from nemo_curator.stages.text.io.writer import JsonlWriter, ParquetWriter
from nemo_curator.stages.text.modifiers import UnicodeReformatter, UrlRemover, NewlineNormalizer
from nemo_curator.stages.text.modules import Modify, ScoreFilter, AddId
from nemo_curator.stages.text.filters import (
    WordCountFilter,
    NonAlphaNumericFilter,
    RepeatedLinesFilter,
    PunctuationFilter,
    BoilerPlateStringFilter
)
from nemo_curator import Sequential
from nemo_curator.backends.xenna import XennaExecutor

# External imports
from huggingface_hub import list_repo_files, hf_hub_download
from dotenv import load_dotenv

# Local imports
from ..helper_funcs import get_base_dir


# Logging Configuration

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger for the preprocessing pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom log format
    
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    logger = logging.getLogger("nemo_curator.preprocess")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


# Initialize default logger
logger = setup_logging()


# Custom Exceptions

class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""
    pass


class ConfigurationError(PreprocessingError):
    """Raised when configuration is invalid or missing."""
    pass


class PipelineStageError(PreprocessingError):
    """Raised when a pipeline stage fails."""
    pass


class AWSCredentialsError(PreprocessingError):
    """Raised when AWS credentials are missing or invalid."""
    pass


class DataDownloadError(PreprocessingError):
    """Raised when data download fails."""
    pass


# Enums and Data Classes

class FileFormat(Enum):
    """Supported file formats."""
    JSONL = "jsonl"
    PARQUET = "parquet"


class DataTag(Enum):
    """Supported predefined dataset tags."""
    ARXIV = "arxiv"
    COMMON_CRAWL = "common_crawl"
    WIKIPEDIA = "wiki"

@dataclass
class PipelineConfig:
    """Configuration for the preprocessing pipeline."""
    input_path: str
    output_path: str
    file_format: FileFormat = FileFormat.PARQUET
    columns: List[str] = field(default_factory=lambda: ["text"])
    text_field: str = "text"
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class ExecutorConfig:
    """Configuration for the XennaExecutor."""
    execution_mode: str = "batch"
    logging_interval: int = 10
    ignore_failures: bool = False
    slots_per_actor: int = 2
    cpu_allocation_percentage: float = 0.95


# Environment and Credentials

def load_environment() -> None:
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f"No .env file found at {env_path}. Using system environment.")


def validate_aws_credentials() -> bool:
    """
    Validate that AWS credentials are available for S3 access.
    
    Returns:
        True if credentials are valid
        
    Raises:
        AWSCredentialsError: If credentials are missing
    """
    required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
    missing = [var for var in required_vars if not os.environ.get(var)]
    
    if missing:
        raise AWSCredentialsError(
            f"Missing AWS credentials: {', '.join(missing)}. "
            f"Please set these in your .env file or environment."
        )
    
    # Optional: AWS_DEFAULT_REGION
    if not os.environ.get("AWS_DEFAULT_REGION"):
        logger.warning("AWS_DEFAULT_REGION not set. Defaulting to 'us-east-1'.")
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    
    logger.info("AWS credentials validated successfully.")
    return True


# Stage Registry - For Module Selection

class StageRegistry:
    """
    Registry for pipeline stages.
    Allows clean, declarative stage registration and retrieval.
    """
    
    def __init__(self):
        self._filters: Dict[str, Callable] = {}
        self._modifiers: Dict[str, Callable] = {}
        self._dedup: Dict[str, Callable] = {}
    
    def register_filter(self, name: str):
        """Decorator to register a filter stage builder."""
        def decorator(func: Callable):
            self._filters[name] = func
            return func
        return decorator
    
    def register_modifier(self, name: str):
        """Decorator to register a modifier stage builder."""
        def decorator(func: Callable):
            self._modifiers[name] = func
            return func
        return decorator
    
    def register_dedup(self, name: str):
        """Decorator to register a deduplication stage builder."""
        def decorator(func: Callable):
            self._dedup[name] = func
            return func
        return decorator
    
    def get_filter(self, name: str) -> Optional[Callable]:
        return self._filters.get(name)
    
    def get_modifier(self, name: str) -> Optional[Callable]:
        return self._modifiers.get(name)
    
    def get_dedup(self, name: str) -> Optional[Callable]:
        return self._dedup.get(name)
    
    @property
    def available_filters(self) -> List[str]:
        return list(self._filters.keys())
    
    @property
    def available_modifiers(self) -> List[str]:
        return list(self._modifiers.keys())
    
    @property
    def available_dedup(self) -> List[str]:
        return list(self._dedup.keys())


# Global registry
registry = StageRegistry()


# Stage Builders - Filters

@registry.register_filter("add_id")
def build_add_id_stage(params: Dict[str, Any], text_field: str = "text") -> AddId:
    """Build AddId stage from config parameters."""
    return AddId(
        id_field=params.get("id_field", "document_id"),
        id_prefix=params.get("id_prefix", "doc"),
        overwrite=params.get("overwrite", True)
    )


@registry.register_filter("WordCountFilter")
def build_word_count_filter(params: Dict[str, Any], text_field: str = "text") -> ScoreFilter:
    """Build WordCountFilter stage from config parameters."""
    return ScoreFilter(
        filter_obj=WordCountFilter(
            min_words=params.get("min_words", 50),
            max_words=params.get("max_words", 100000)
        ),
        text_field=text_field,
        score_field="word_count"
    )


@registry.register_filter("NonAlphaNumericFilter")
def build_non_alpha_numeric_filter(params: Dict[str, Any], text_field: str = "text") -> ScoreFilter:
    """Build NonAlphaNumericFilter stage from config parameters."""
    return ScoreFilter(
        filter_obj=NonAlphaNumericFilter(
            max_non_alpha_numeric_to_text_ratio=params.get("max_non_alpha_numeric_to_text_ratio", 0.25)
        ),
        text_field=text_field
    )


@registry.register_filter("RepeatedLinesFilter")
def build_repeated_lines_filter(params: Dict[str, Any], text_field: str = "text") -> ScoreFilter:
    """Build RepeatedLinesFilter stage from config parameters."""
    return ScoreFilter(
        filter_obj=RepeatedLinesFilter(
            max_repeated_line_fraction=params.get("max_repeated_line_fraction", 0.7)
        ),
        text_field=text_field
    )


@registry.register_filter("PunctuationFilter")
def build_punctuation_filter(params: Dict[str, Any], text_field: str = "text") -> ScoreFilter:
    """Build PunctuationFilter stage from config parameters."""
    return ScoreFilter(
        filter_obj=PunctuationFilter(
            max_num_sentences_without_endmark_ratio=params.get("max_num_sentences_without_endmark_ratio", 0.85)
        ),
        text_field=text_field
    )


@registry.register_filter("BoilerPlateStringFilter")
def build_boilerplate_filter(params: Dict[str, Any], text_field: str = "text") -> ScoreFilter:
    """Build BoilerPlateStringFilter stage from config parameters."""
    return ScoreFilter(
        filter_obj=BoilerPlateStringFilter(),
        text_field=text_field
    )


# Stage Builders - Modifiers

@registry.register_modifier("UnicodeReformatter")
def build_unicode_reformatter(text_field: str = "text") -> Modify:
    """Build UnicodeReformatter stage."""
    return Modify(modifier_fn=UnicodeReformatter(), input_fields=text_field)


@registry.register_modifier("NewlineNormalizer")
def build_newline_normalizer(text_field: str = "text") -> Modify:
    """Build NewlineNormalizer stage."""
    return Modify(modifier_fn=NewlineNormalizer(), input_fields=text_field)


@registry.register_modifier("UrlRemover")
def build_url_remover(text_field: str = "text") -> Modify:
    """Build UrlRemover stage."""
    return Modify(modifier_fn=UrlRemover(), input_fields=text_field)


# Stage Builders - Deduplication

@registry.register_dedup("ExactDeduplicationWorkflow")
def build_exact_dedup(input_path: str, output_path: str, params: Optional[Dict] = None) -> Sequential:
    """Build Exact Deduplication workflow."""
    from nemo_curator.stages.deduplication.exact.workflow import ExactDeduplicationWorkflow
    from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
    
    params = params or {}
    cache_path = params.get("cache_path", "./results")
    
    return Sequential([
        ExactDeduplicationWorkflow(
            input_path=input_path,
            output_path=cache_path,
            text_field=params.get("text_field", "text"),
            assign_id=params.get("assign_id", True),
            perform_removal=False,
            input_filetype=params.get("input_filetype", "parquet")
        ),
        TextDuplicatesRemovalWorkflow(
            input_path=input_path,
            ids_to_remove_path=f"{cache_path}/ExactDuplicateIds",
            output_path=output_path,
            input_filetype=params.get("input_filetype", "parquet"),
            input_id_field="_curator_dedup_id",
            ids_to_remove_duplicate_id_field="_curator_dedup_id",
            id_generator_path=f"{cache_path}/exact_id_generator.json"
        )
    ])


@registry.register_dedup("FuzzyDeduplicationWorkflow")
def build_fuzzy_dedup(input_path: str, output_path: str, params: Optional[Dict] = None) -> Sequential:
    """Build Fuzzy Deduplication workflow."""
    from nemo_curator.stages.deduplication.fuzzy.workflow import FuzzyDeduplicationWorkflow
    from nemo_curator.stages.text.deduplication.removal_workflow import TextDuplicatesRemovalWorkflow
    
    params = params or {}
    cache_path = params.get("cache_path", "./cache")
    results_path = params.get("results_path", "./results")
    
    return Sequential([
        FuzzyDeduplicationWorkflow(
            input_path=input_path,
            cache_path=cache_path,
            output_path=results_path,
            text_field=params.get("text_field", "text"),
            perform_removal=False,
            input_filetype=params.get("input_filetype", "parquet"),
            char_ngrams=params.get("char_ngrams", 24),
            num_bands=params.get("num_bands", 20),
            minhashes_per_band=params.get("minhashes_per_band", 13)
        ),
        TextDuplicatesRemovalWorkflow(
            input_path=input_path,
            ids_to_remove_path=f"{results_path}/FuzzyDuplicateIds",
            output_path=output_path,
            input_filetype=params.get("input_filetype", "parquet"),
            input_id_field="_curator_dedup_id",
            ids_to_remove_duplicate_id_field="_curator_dedup_id",
            id_generator_path=f"{results_path}/fuzzy_id_generator.json"
        )
    ])


@registry.register_dedup("TextSemanticDeduplicationWorkflow")
def build_semantic_dedup(input_path: str, output_path: str, params: Optional[Dict] = None) -> Sequential:
    """Build Semantic Deduplication workflow."""
    from nemo_curator.stages.text.deduplication.semantic import TextSemanticDeduplicationWorkflow
    
    params = params or {}
    
    return Sequential([
        TextSemanticDeduplicationWorkflow(
            input_path=input_path,
            output_path=output_path,
            cache_path=params.get("cache_path", "./sem_cache"),
            model_identifier=params.get("model_identifier", "sentence-transformers/all-MiniLM-L6-v2"),
            n_clusters=params.get("n_clusters", 100),
            eps=params.get("eps", 0.07),
            id_field=params.get("id_field", "doc_id"),
            perform_removal=params.get("perform_removal", True)
        )
    ])


# Pipeline Builders

def build_pipeline_stages(
    preprocess_params: Dict[str, Any],
    text_field: str = "text"
) -> List[Any]:
    """
    Build pipeline stages from configuration.
    
    Args:
        preprocess_params: Configuration dictionary
        text_field: Name of the text field in documents
    
    Returns:
        List of pipeline stages
    """
    stages = []
    
    # Process filters
    for filter_name in registry.available_filters:
        config = preprocess_params.get(filter_name)
        if config is None:
            continue
        
        # Handle boolean-like config (e.g., {"enabled": true})
        if isinstance(config, bool) and not config:
            continue
        
        # Handle dict config with params
        if isinstance(config, dict):
            if config.get("enabled", True) is False:
                continue
            params = config.get("params", config)
        else:
            params = {}
        
        try:
            builder = registry.get_filter(filter_name)
            if builder:
                stage = builder(params, text_field)
                stages.append(stage)
                logger.info(f"Added filter stage: {filter_name}")
        except Exception as e:
            logger.error(f"Failed to build filter '{filter_name}': {e}")
            raise PipelineStageError(f"Failed to build filter '{filter_name}': {e}")
    
    # Process modifiers
    for modifier_name in registry.available_modifiers:
        config = preprocess_params.get(modifier_name)
        if not config:
            continue
        
        # Handle boolean config
        if isinstance(config, bool) and not config:
            continue
        
        try:
            builder = registry.get_modifier(modifier_name)
            if builder:
                stage = builder(text_field)
                stages.append(stage)
                logger.info(f"Added modifier stage: {modifier_name}")
        except Exception as e:
            logger.error(f"Failed to build modifier '{modifier_name}': {e}")
            raise PipelineStageError(f"Failed to build modifier '{modifier_name}': {e}")
    
    return stages


def custom_pipeline(
    file_format: str,
    input_path: str,
    output_path: str,
    columns: List[str],
    preprocess_params: Dict[str, Any],
    executor: Any
) -> Any:
    """
    Execute custom data curation pipeline.
    
    Args:
        file_format: Input file format ('jsonl' or 'parquet')
        input_path: Path to input data
        output_path: Path to write curated data
        columns: List of columns to process
        preprocess_params: Preprocessing configuration
        executor: XennaExecutor instance
    
    Returns:
        Pipeline execution results
    
    Raises:
        PipelineStageError: If pipeline execution fails
    """
    logger.info("-" * 50)
    logger.info("Starting Custom Curation Pipeline")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"File format: {file_format}")
    logger.info(f"Columns: {columns}")
    
    try:
        # Create pipeline
        pipeline = Pipeline(
            name="text_cleaning_pipeline",
            description="Clean text data using configurable filters and modifiers"
        )
        
        # Add reader stage
        if file_format == FileFormat.JSONL.value or file_format == "jsonl":
            reader = JsonlReader(
                file_paths=input_path,
                blocksize=preprocess_params.get("block_size", "128MB"),
                fields=columns
            )
            logger.info(f"Using JSONL reader with blocksize: {preprocess_params.get('block_size', '128MB')}")
        else:
            reader = ParquetReader(
                file_paths=input_path,
                files_per_partition=preprocess_params.get("files_per_partition", 4),
                fields=columns
            )
            logger.info(f"Using Parquet reader with files_per_partition: {preprocess_params.get('files_per_partition', 4)}")
        
        pipeline.add_stage(reader)
        
        # Build and add processing stages
        text_field = columns[0] if columns else "text"
        stages = build_pipeline_stages(preprocess_params, text_field)
        
        for stage in stages:
            pipeline.add_stage(stage)
        
        logger.info(f"Added {len(stages)} processing stages to pipeline")
        
        # Add writer stage
        if file_format == FileFormat.JSONL.value or file_format == "jsonl":
            pipeline.add_stage(JsonlWriter(path=output_path))
        else:
            pipeline.add_stage(ParquetWriter(path=output_path))
        
        logger.info("Executing pipeline...")
        
        # Execute pipeline
        results = pipeline.run(executor)
        
        logger.info("Pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise PipelineStageError(f"Pipeline execution failed: {e}")


def dedup_pipeline(
    input_path: str,
    output_path: str,
    preprocess_params: Dict[str, Any]
) -> Any:
    """
    Execute deduplication pipeline.
    
    Args:
        input_path: Path to input data
        output_path: Path to write deduplicated data
        preprocess_params: Configuration containing dedup settings
    
    Returns:
        Pipeline execution results
    """
    logger.info("-" * 50)
    logger.info("Starting Deduplication Pipeline")
    
    ray_client = RayClient()
    
    try:
        ray_client.start()
        logger.info("Ray client started successfully")
        
        stages = []
        
        # Build deduplication stages based on config
        for dedup_name in registry.available_dedup:
            config = preprocess_params.get(dedup_name)
            if not config:
                continue
            
            # Handle boolean config
            if isinstance(config, bool) and not config:
                continue
            
            # Get params if available
            params = config.get("params", {}) if isinstance(config, dict) else {}
            
            try:
                builder = registry.get_dedup(dedup_name)
                if builder:
                    stage = builder(input_path, output_path, params)
                    stages.append(stage)
                    logger.info(f"Added deduplication stage: {dedup_name}")
            except Exception as e:
                logger.error(f"Failed to build dedup stage '{dedup_name}': {e}")
                raise PipelineStageError(f"Failed to build dedup stage '{dedup_name}': {e}")
        
        if not stages:
            logger.warning("No deduplication stages configured. Skipping deduplication.")
            return None
        
        # Create and run pipeline
        pipeline = Pipeline(
            name="deduplication_pipeline",
            description="Apply configured deduplication stages",
            stages=stages
        )
        
        logger.info("Executing deduplication pipeline...")
        results = pipeline.run()
        
        logger.info("Deduplication completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Deduplication pipeline failed: {e}", exc_info=True)
        raise
    finally:
        ray_client.stop()
        logger.info("Ray client stopped")


# Predefined Dataset Pipelines

def arxiv_pipeline(download_dir: str, output_path: str) -> Any:
    """
    Download and process ArXiv LaTeX sources.
    
    Args:
        download_dir: Directory to download raw data
        output_path: Path to write processed data
    
    Returns:
        Pipeline execution results
    """
    from nemo_curator.stages.text.download import ArxivDownloadExtractStage
    
    logger.info("-" * 50)
    logger.info("Starting ArXiv Pipeline")
    logger.info(f"Download directory: {download_dir}")
    logger.info(f"Output path: {output_path}")
    
    ray_client = RayClient()
    
    try:
        ray_client.start()
        logger.info("Ray client started")
        
        pipeline = Pipeline(
            name="arxiv_pipeline",
            description="Download and process ArXiv LaTeX sources"
        )
        
        arxiv_stage = ArxivDownloadExtractStage(
            download_dir=download_dir,
            url_limit=5,
            record_limit=1000,
            add_filename_column=True,
            verbose=True,
        )
        pipeline.add_stage(arxiv_stage)
        pipeline.add_stage(JsonlWriter(path=output_path))
        
        logger.info("Executing ArXiv pipeline...")
        results = pipeline.run()
        
        result_count = len(results) if results else 0
        logger.info(f"ArXiv pipeline completed with {result_count} output files")
        
        return results
        
    except Exception as e:
        logger.error(f"ArXiv pipeline failed: {e}", exc_info=True)
        raise
    finally:
        ray_client.stop()
        logger.info("Ray client stopped")


def common_crawl_pipeline(download_dir: str, output_path: str) -> Any:
    """
    Download and process Common Crawl data.
    
    Requires AWS credentials for S3 access.
    
    Args:
        download_dir: Directory to download raw data
        output_path: Path to write processed data
    
    Returns:
        Pipeline execution results
    
    Raises:
        AWSCredentialsError: If AWS credentials are missing
    """
    from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
    
    logger.info("-" * 50)
    logger.info("Starting Common Crawl Pipeline")
    
    # Validate AWS credentials for S3 access
    validate_aws_credentials()
    
    logger.info(f"Download directory: {download_dir}")
    logger.info(f"Output path: {output_path}")
    
    ray_client = RayClient()
    
    try:
        ray_client.start()
        logger.info("Ray client started")
        
        pipeline = Pipeline(
            name="common_crawl_pipeline",
            description="Download and process Common Crawl data"
        )
        
        cc_stage = CommonCrawlDownloadExtractStage(
            start_snapshot="2020-50",
            end_snapshot="2020-50",
            download_dir=download_dir,
            crawl_type="main",
            use_aws_to_download=True,
            url_limit=10,
            record_limit=1000,
        )
        pipeline.add_stage(cc_stage)
        pipeline.add_stage(JsonlWriter(path=output_path))
        
        logger.info("Executing Common Crawl pipeline...")
        results = pipeline.run()
        
        logger.info("Common Crawl pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Common Crawl pipeline failed: {e}", exc_info=True)
        raise
    finally:
        ray_client.stop()
        logger.info("Ray client stopped")


def wikipedia_pipeline(download_dir: str, output_path: str) -> Any:
    """
    Download and process Wikipedia dumps.
    
    Args:
        download_dir: Directory to download raw data
        output_path: Path to write processed data
    
    Returns:
        Pipeline execution results
    """
    from nemo_curator.stages.text.download import WikipediaDownloadExtractStage
    
    logger.info("-" * 50)
    logger.info("Starting Wikipedia Pipeline")
    logger.info(f"Download directory: {download_dir}")
    logger.info(f"Output path: {output_path}")
    
    ray_client = RayClient()
    
    try:
        ray_client.start()
        logger.info("Ray client started")
        
        pipeline = Pipeline(
            name="wikipedia_pipeline",
            description="Download and process Wikipedia dumps"
        )
        
        wikipedia_stage = WikipediaDownloadExtractStage(
            language="en",
            download_dir=download_dir,
            dump_date=None,
            url_limit=5,
            record_limit=1000,
            verbose=True
        )
        pipeline.add_stage(wikipedia_stage)
        pipeline.add_stage(JsonlWriter(path=output_path))
        
        logger.info("Executing Wikipedia pipeline...")
        results = pipeline.run()
        
        logger.info("Wikipedia pipeline completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Wikipedia pipeline failed: {e}", exc_info=True)
        raise
    finally:
        ray_client.stop()
        logger.info("Ray client stopped")


# HuggingFace Data Utilities

def get_train_filenames(repo_id: str) -> List[str]:
    """
    Retrieve parquet file paths from a HuggingFace dataset repository.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
    
    Returns:
        Sorted list of parquet file paths
    
    Raises:
        DataDownloadError: If file listing fails
    """
    logger.info(f"Fetching file list from HuggingFace repo: {repo_id}")
    
    try:
        all_files = sorted(list_repo_files(repo_id, repo_type="dataset"))
        
        # Support both parquet and jsonl formats
        supported_extensions = ('.parquet', '.jsonl')
        data_files = [f for f in all_files if f.endswith(supported_extensions)]
        
        if not data_files:
            raise DataDownloadError(
                f"No supported data files found in {repo_id}. "
                f"Supported formats: {', '.join(supported_extensions)}"
            )
        
        # Log format breakdown
        parquet_count = sum(1 for f in data_files if f.endswith('.parquet'))
        jsonl_count = sum(1 for f in data_files if f.endswith('.jsonl'))
        logger.info(f"Found {len(data_files)} data files in repository (parquet: {parquet_count}, jsonl: {jsonl_count})")
        
        return data_files
        
    except DataDownloadError:
        raise
    except Exception as e:
        logger.error(f"Failed to list files from {repo_id}: {e}")
        raise DataDownloadError(f"Failed to list files from {repo_id}: {e}")


def download_raw_files(
    repo_id: str,
    filenames: List[str],
    save_dir: str = "downloads"
) -> List[str]:
    """
    Download raw files from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        filenames: List of files to download
        save_dir: Local directory to save files
    
    Returns:
        List of local file paths
    
    Raises:
        DataDownloadError: If download fails
    """
    logger.info(f"Starting download of {len(filenames)} files to '{save_dir}'")
    
    local_paths = []
    
    for i, filename in enumerate(filenames):
        try:
            logger.debug(f"Downloading ({i+1}/{len(filenames)}): {filename}")
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=save_dir,
                local_dir_use_symlinks=False
            )
            local_paths.append(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            raise DataDownloadError(f"Failed to download {filename}: {e}")
    
    logger.info(f"Successfully downloaded {len(local_paths)} files")
    return local_paths


def get_download_paths(cache_dir: str) -> List[str]:
    """
    Get paths to downloaded data files.
    
    Args:
        cache_dir: Directory containing downloaded files
    
    Returns:
        Sorted list of file paths
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        logger.warning(f"Cache directory does not exist: {cache_dir}")
        return []
    
    data_files = sorted(
        str(p.resolve())
        for p in cache_path.rglob("*")
        if p.is_file() and p.suffix in (".parquet", ".jsonl")
    )
    
    logger.info(f"Found {len(data_files)} data files in {cache_dir}")
    return data_files


def detect_file_format(file_paths: List[str]) -> str:
    """
    Automatically detect the file format from a list of file paths.
    
    Args:
        file_paths: List of file paths to analyze
    
    Returns:
        Detected format ('parquet' or 'jsonl')
    
    Raises:
        ConfigurationError: If no files or mixed formats detected
    """
    if not file_paths:
        raise ConfigurationError("No files provided for format detection")
    
    # Count files by extension
    parquet_files = [f for f in file_paths if f.endswith('.parquet')]
    jsonl_files = [f for f in file_paths if f.endswith('.jsonl')]
    
    parquet_count = len(parquet_files)
    jsonl_count = len(jsonl_files)
    
    if parquet_count > 0 and jsonl_count > 0:
        # Mixed formats - use the majority
        if parquet_count >= jsonl_count:
            logger.warning(
                f"Mixed file formats detected (parquet: {parquet_count}, jsonl: {jsonl_count}). "
                f"Using parquet format (majority)."
            )
            return "parquet"
        else:
            logger.warning(
                f"Mixed file formats detected (parquet: {parquet_count}, jsonl: {jsonl_count}). "
                f"Using jsonl format (majority)."
            )
            return "jsonl"
    elif parquet_count > 0:
        logger.info(f"Detected file format: parquet ({parquet_count} files)")
        return "parquet"
    elif jsonl_count > 0:
        logger.info(f"Detected file format: jsonl ({jsonl_count} files)")
        return "jsonl"
    else:
        raise ConfigurationError(
            "Could not detect file format. Supported formats: .parquet, .jsonl"
        )


# Configuration Utilities

def parse_config(config_path: str) -> Dict[str, Any]:
    """
    Parse JSON configuration file.
    
    Args:
        config_path: Path to JSON config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        ConfigurationError: If config file is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in configuration file: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate preprocessing configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Check for required basic fields
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration must be a dictionary")
    
    # Log available stages
    enabled_filters = [
        name for name in registry.available_filters
        if config.get(name) and (
            config[name] is True or 
            (isinstance(config[name], dict) and config[name].get("enabled", True))
        )
    ]
    
    enabled_modifiers = [
        name for name in registry.available_modifiers
        if config.get(name)
    ]
    
    enabled_dedup = [
        name for name in registry.available_dedup
        if config.get(name)
    ]
    
    logger.info(f"Enabled filters: {enabled_filters}")
    logger.info(f"Enabled modifiers: {enabled_modifiers}")
    logger.info(f"Enabled dedup stages: {enabled_dedup}")
    
    return True


# Main Entry Point

def create_executor(config: Optional[ExecutorConfig] = None) -> XennaExecutor:
    """Create and configure XennaExecutor."""
    if config is None:
        config = ExecutorConfig()
    
    batch_config = {
        "execution_mode": config.execution_mode,
        "logging_interval": config.logging_interval,
        "ignore_failures": config.ignore_failures,
        "slots_per_actor": config.slots_per_actor,
        "cpu_allocation_percentage": config.cpu_allocation_percentage,
    }
    
    return XennaExecutor(config=batch_config)


def main():
    """Main entry point for preprocessing pipeline."""
    # Load environment variables
    load_environment()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='NeMo Curator Data Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= """
                Examples:
                  # Custom dataset curation from HuggingFace
                  python preprocess.py --custom --repo_id username/dataset-name
                  
                  # Custom curation with deduplication
                  python preprocess.py --custom --dedup --repo_id username/dataset-name
                  
                  # Predefined datasets (requires AWS credentials for Common Crawl)
                  python preprocess.py --data_tag arxiv --download_dir ./data
                  python preprocess.py --data_tag common_crawl --download_dir ./data
                  python preprocess.py --data_tag wiki --download_dir ./data
                  
                  # Custom log level
                  python preprocess.py --custom --log_level DEBUG
                """
    )
    
    parser.add_argument('--custom', action='store_true', default=True,
                        help='Use custom dataset from HuggingFace')
    parser.add_argument('--repo_id', type=str, default="karpathy/fineweb-edu-100b-shuffle",
                        help='HuggingFace repository ID')
    parser.add_argument('--dedup', action='store_true', default=False,
                        help='Enable deduplication after curation')
    parser.add_argument('--data_tag', type=str, choices=['arxiv', 'common_crawl', 'wiki'],
                        help='Use predefined dataset pipeline')
    parser.add_argument('--download_dir', type=str, default='./downloads',
                        help='Download directory for predefined datasets')
    parser.add_argument('--config', type=str, default='configs/preprocess.json',
                        help='Path to preprocessing config file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Optional log file path')
    parser.add_argument('--num_files', type=int, default=100,
                        help='Number of files to process from HuggingFace repo')
    
    args = parser.parse_args()
    
    # Setup logging with user preferences
    global logger
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    logger.info("-" * 50)
    logger.info("NeMo Curator Preprocessing Pipeline")
    
    try:
        # Initialize executor
        executor = create_executor()
        
        # Set up paths
        input_path = get_base_dir("nemo_curator/data/sample")
        output_path = get_base_dir("nemo_curator/data/curated")
        
        # Load and validate config
        config_path = Path(__file__).parent.parent / "preprocess.json"
        if args.config:
            config_path = Path(args.config)
        
        preprocess_params = parse_config(str(config_path))
        validate_config(preprocess_params)
        
        # Determine pipeline mode
        if args.data_tag:
            # Predefined dataset pipelines
            logger.info(f"Running predefined pipeline: {args.data_tag}")
            
            if args.data_tag == "arxiv":
                arxiv_pipeline(args.download_dir, str(output_path))
            elif args.data_tag == "common_crawl":
                common_crawl_pipeline(args.download_dir, str(output_path))
            elif args.data_tag == "wiki":
                wikipedia_pipeline(args.download_dir, str(output_path))
        
        elif args.custom:
            # Custom HuggingFace dataset pipeline
            logger.info(f"Running custom pipeline with repo: {args.repo_id}")
            
            # Download data
            all_data_files = get_train_filenames(args.repo_id)
            subset_files = all_data_files[:args.num_files]
            downloaded_paths = download_raw_files(args.repo_id, subset_files, str(input_path))
            data_files = get_download_paths(str(input_path))
            
            if not data_files:
                raise DataDownloadError("No data files found after download")
            
            # Auto-detect file format
            file_format = detect_file_format(data_files)
            
            # Run curation pipeline
            if args.dedup:
                # Curate to intermediate path, then deduplicate
                intermediate_path = str(input_path) + "_curated"
                custom_pipeline(
                    file_format=file_format,
                    input_path=data_files,
                    output_path=intermediate_path,
                    columns=["text"],
                    preprocess_params=preprocess_params,
                    executor=executor
                )
                dedup_pipeline(intermediate_path, str(output_path), preprocess_params)
            else:
                # Just curate
                custom_pipeline(
                    file_format=file_format,
                    input_path=data_files,
                    output_path=str(output_path),
                    columns=["text"],
                    preprocess_params=preprocess_params,
                    executor=executor
                )
        
        logger.info("-" * 50)
        logger.info("Pipeline completed successfully!")
        
    except PreprocessingError as e:
        logger.error(f"Preprocessing error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
