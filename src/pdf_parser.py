from enum import Enum
from typing import Any
from dataclasses import dataclass

from llama_index.readers.file import PDFReader
from llama_index.readers.docling import DoclingReader
from llama_parse import LlamaParse

class ReaderType(str, Enum):
    PDF_READER = "pdf_reader"
    DOCLING = "docling"
    LLAMA_PARSE = "llama_parse"

@dataclass
class ReaderConfig:
    """Configuration for a PDF reader"""
    name: str
    description: str
    reader_instance: Any

class PDFReaderFactory:
    """
    Factory for creating and managing PDF readers.
    """
    def __init__(self):
        self._readers: dict[str, ReaderConfig] = {}
        self._default_reader = ReaderType.DOCLING.value
        self._initialize_readers()

    def _initialize_readers(self):
        """Initialize all available PDF readers with their metadata"""
        self._readers[ReaderType.PDF_READER.value] = ReaderConfig(
            name="PDF Reader",
            description="Basic PDF reader from LlamaIndex",
            reader_instance=PDFReader()
        )

        self._readers[ReaderType.DOCLING.value] = ReaderConfig(
            name="Docling",
            description="Advanced PDF reader with layout preservation",
            reader_instance=DoclingReader()
        )

        self._readers[ReaderType.LLAMA_PARSE.value] = ReaderConfig(
            name="LlamaParse",
            description="LlamaIndex's premium parsing service",
            reader_instance=LlamaParse()
        )

    def get_reader(self, reader_type: ReaderType | str) -> Any:
        key = reader_type.value if isinstance(reader_type, ReaderType) else reader_type

        if key not in self._readers:
            raise ValueError(f"Reader type '{key}' not found")

        return self._readers[key].reader_instance

    def get_all_readers_info(self) -> list[dict[str, str]]:
        return [
            {
                "value": reader_type,
                "name": config.name,
                "description": config.description
            }
            for reader_type, config in self._readers.items()
        ]

    def get_default_reader(self) -> str:
        return self._default_reader
