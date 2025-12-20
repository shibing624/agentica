# -*- coding: utf-8 -*-
"""
@author: XuMing(xuming624@qq.com)
@description: Unit tests for Knowledge system.
"""
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentica.knowledge.base import Knowledge
from agentica.document import Document


class TestKnowledgeInitialization(unittest.TestCase):
    """Test cases for Knowledge initialization."""

    def test_default_initialization(self):
        """Test Knowledge with default parameters."""
        knowledge = Knowledge()
        self.assertIsNone(knowledge.data_path)
        self.assertIsNone(knowledge.vector_db)
        self.assertEqual(knowledge.num_documents, 3)
        self.assertEqual(knowledge.chunk_size, 2000)
        self.assertTrue(knowledge.chunk)

    def test_initialization_with_data_path(self):
        """Test Knowledge with data_path."""
        knowledge = Knowledge(data_path="/path/to/data")
        self.assertEqual(knowledge.data_path, "/path/to/data")

    def test_initialization_with_list_data_path(self):
        """Test Knowledge with list of data paths."""
        paths = ["/path/to/data1", "/path/to/data2"]
        knowledge = Knowledge(data_path=paths)
        self.assertEqual(knowledge.data_path, paths)

    def test_initialization_with_custom_settings(self):
        """Test Knowledge with custom settings."""
        knowledge = Knowledge(
            num_documents=5,
            chunk_size=1000,
            chunk=False
        )
        self.assertEqual(knowledge.num_documents, 5)
        self.assertEqual(knowledge.chunk_size, 1000)
        self.assertFalse(knowledge.chunk)


class TestKnowledgeTextCleaning(unittest.TestCase):
    """Test cases for Knowledge text cleaning."""

    def test_clean_text_multiple_newlines(self):
        """Test cleaning multiple newlines."""
        text = "Hello\n\n\nWorld"
        cleaned = Knowledge._clean_text(text)
        # _clean_text replaces multiple newlines with single newline
        self.assertNotIn("\n\n\n", cleaned)

    def test_clean_text_multiple_spaces(self):
        """Test cleaning multiple spaces."""
        text = "Hello    World"
        cleaned = Knowledge._clean_text(text)
        self.assertEqual(cleaned, "Hello World")

    def test_clean_text_multiple_tabs(self):
        """Test cleaning multiple tabs."""
        text = "Hello\t\t\tWorld"
        cleaned = Knowledge._clean_text(text)
        # _clean_text replaces multiple tabs with single tab
        self.assertNotIn("\t\t\t", cleaned)

    def test_clean_text_mixed(self):
        """Test cleaning mixed whitespace."""
        text = "Hello\n\n\n  \t\t  World"
        cleaned = Knowledge._clean_text(text)
        # Multiple spaces become single space
        self.assertNotIn("  ", cleaned)


class TestKnowledgeChunking(unittest.TestCase):
    """Test cases for Knowledge document chunking."""

    def test_chunk_document_basic(self):
        """Test basic document chunking."""
        knowledge = Knowledge(chunk_size=100)
        document = Document(
            name="test_doc",
            content="This is a test document. " * 20,  # ~500 chars
            meta_data={"source": "test"}
        )
        chunks = knowledge.chunk_document(document, chunk_size=100)
        self.assertGreater(len(chunks), 1)

    def test_chunk_document_small_content(self):
        """Test chunking small content (no split needed)."""
        knowledge = Knowledge(chunk_size=1000)
        document = Document(
            name="test_doc",
            content="Short content",
            meta_data={}
        )
        chunks = knowledge.chunk_document(document, chunk_size=1000)
        self.assertEqual(len(chunks), 1)

    def test_chunk_document_preserves_metadata(self):
        """Test chunking preserves metadata."""
        knowledge = Knowledge()
        document = Document(
            name="test_doc",
            content="Test content " * 100,
            meta_data={"source": "test", "author": "tester"}
        )
        chunks = knowledge.chunk_document(document, chunk_size=50)
        for chunk in chunks:
            self.assertIn("source", chunk.meta_data)
            self.assertEqual(chunk.meta_data["source"], "test")

    def test_chunk_document_adds_chunk_number(self):
        """Test chunking adds chunk number to metadata."""
        knowledge = Knowledge()
        document = Document(
            name="test_doc",
            content="Test content " * 100,
            meta_data={}
        )
        chunks = knowledge.chunk_document(document, chunk_size=50)
        for i, chunk in enumerate(chunks, 1):
            self.assertEqual(chunk.meta_data["chunk"], i)


class TestKnowledgeWithVectorDb(unittest.TestCase):
    """Test cases for Knowledge with vector database."""

    def test_knowledge_search_without_vector_db(self):
        """Test Knowledge search returns empty list without vector_db."""
        knowledge = Knowledge()
        results = knowledge.search("test query")
        self.assertEqual(results, [])

    def test_knowledge_load_without_vector_db(self):
        """Test Knowledge load does nothing without vector_db."""
        knowledge = Knowledge()
        # Should not raise error, just return
        knowledge.load()


class TestDocument(unittest.TestCase):
    """Test cases for Document model."""

    def test_document_creation(self):
        """Test Document creation."""
        doc = Document(name="test", content="Test content")
        self.assertEqual(doc.name, "test")
        self.assertEqual(doc.content, "Test content")

    def test_document_with_metadata(self):
        """Test Document with metadata."""
        doc = Document(
            name="test",
            content="Test content",
            meta_data={"source": "file.txt", "page": 1}
        )
        self.assertEqual(doc.meta_data["source"], "file.txt")
        self.assertEqual(doc.meta_data["page"], 1)

    def test_document_with_id(self):
        """Test Document with custom ID."""
        doc = Document(id="doc-123", name="test", content="Content")
        self.assertEqual(doc.id, "doc-123")


class TestKnowledgeLoad(unittest.TestCase):
    """Test cases for Knowledge load functionality."""

    def test_load_with_no_data_path(self):
        """Test load with no data_path set."""
        knowledge = Knowledge()
        # Should not raise error, just do nothing (no vector_db)
        knowledge.load()

    def test_load_with_no_vector_db(self):
        """Test load with no vector_db."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for loading")
            temp_path = f.name

        try:
            knowledge = Knowledge(data_path=temp_path)
            # Load should just warn and return when no vector_db
            knowledge.load()
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
