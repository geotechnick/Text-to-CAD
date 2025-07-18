"""
File Analyzer Agent - Optimized for computational efficiency
Handles multi-format file parsing with streaming and caching
"""

import asyncio
import io
import mmap
import struct
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass
from pathlib import Path
import logging
import time
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from ..core.agent_framework import BaseAgent, AgentMessage, MessageType, Priority

@dataclass
class FileData:
    file_path: str
    file_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    extraction_time: float
    confidence_score: float

@dataclass
class ExcelData:
    sheets: Dict[str, pd.DataFrame]
    formulas: Dict[str, List[str]]
    metadata: Dict[str, Any]

class FileAnalyzerAgent(BaseAgent):
    """
    High-performance file analyzer with streaming and caching
    Optimized for large engineering files
    """
    
    def __init__(self, agent_id: str = "file_analyzer", max_workers: int = 4):
        super().__init__(agent_id, max_workers)
        
        # File type detection patterns
        self.file_signatures = {
            b'PK\x03\x04': 'excel',  # Excel files
            b'%PDF': 'pdf',
            b'\x00\x00\x00\x20ftypM4A': 'analysis',  # Some analysis files
            b'STAAD': 'staad',
            b'GeoStudio': 'geostudio',
        }
        
        # Parsers for different file types
        self.parsers = {
            'excel': self._parse_excel_optimized,
            'pdf': self._parse_pdf_optimized,
            'geostudio': self._parse_geostudio_optimized,
            'staad': self._parse_staad_optimized,
            'text': self._parse_text_optimized,
        }
        
        # Cache for parsed files
        self.file_cache = {}
        self.max_cache_size = 50
        
        # Memory-mapped file handles
        self.mmap_handles = {}
        
        logging.info(f"FileAnalyzerAgent initialized with {len(self.parsers)} parsers")
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle file analysis request"""
        
        if message.payload.get("action") == "analyze_files":
            files = message.payload.get("files", [])
            
            # Process files in parallel
            start_time = time.time()
            results = await self._analyze_files_parallel(files)
            processing_time = time.time() - start_time
            
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload={
                    "analyzed_files": [result.__dict__ for result in results],
                    "processing_time": processing_time,
                    "total_files": len(files)
                },
                correlation_id=message.correlation_id
            )
        
        return None
    
    async def _analyze_files_parallel(self, file_paths: List[str]) -> List[FileData]:
        """Analyze multiple files in parallel"""
        
        # Create tasks for parallel processing
        tasks = []
        for file_path in file_paths:
            task = asyncio.create_task(self._analyze_single_file(file_path))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, FileData):
                valid_results.append(result)
            else:
                logging.error(f"File analysis failed: {result}")
        
        return valid_results
    
    async def _analyze_single_file(self, file_path: str) -> FileData:
        """Analyze a single file efficiently"""
        
        # Check cache first
        cache_key = f"{file_path}:{Path(file_path).stat().st_mtime}"
        if cache_key in self.file_cache:
            logging.info(f"Cache hit for file: {file_path}")
            return self.file_cache[cache_key]
        
        start_time = time.time()
        
        # Detect file type
        file_type = await self._detect_file_type_fast(file_path)
        
        # Parse file based on type
        if file_type in self.parsers:
            parser = self.parsers[file_type]
            data, metadata = await parser(file_path)
        else:
            # Fallback to generic parsing
            data, metadata = await self._parse_generic(file_path)
        
        extraction_time = time.time() - start_time
        
        # Create result
        result = FileData(
            file_path=file_path,
            file_type=file_type,
            data=data,
            metadata=metadata,
            extraction_time=extraction_time,
            confidence_score=self._calculate_confidence(data, metadata)
        )
        
        # Cache result
        self.file_cache[cache_key] = result
        
        # Limit cache size
        if len(self.file_cache) > self.max_cache_size:
            oldest_key = next(iter(self.file_cache))
            del self.file_cache[oldest_key]
        
        return result
    
    async def _detect_file_type_fast(self, file_path: str) -> str:
        """Fast file type detection using signatures"""
        
        path = Path(file_path)
        
        # Check extension first (fastest)
        ext = path.suffix.lower()
        if ext in ['.xlsx', '.xls']:
            return 'excel'
        elif ext == '.pdf':
            return 'pdf'
        elif ext in ['.gsz', '.gsd']:
            return 'geostudio'
        elif ext.startswith('.gp12'):
            return 'staad'
        elif ext in ['.txt', '.csv']:
            return 'text'
        
        # Check file signature if extension is ambiguous
        try:
            with open(file_path, 'rb') as f:
                header = f.read(20)
                for signature, file_type in self.file_signatures.items():
                    if header.startswith(signature):
                        return file_type
        except Exception as e:
            logging.warning(f"Could not read file signature: {e}")
        
        return 'unknown'
    
    async def _parse_excel_optimized(self, file_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimized Excel parsing with streaming"""
        
        data = {}
        metadata = {}
        
        try:
            # Use pandas for efficient reading
            with pd.ExcelFile(file_path) as excel_file:
                metadata['sheet_names'] = excel_file.sheet_names
                metadata['file_size'] = Path(file_path).stat().st_size
                
                sheets_data = {}
                for sheet_name in excel_file.sheet_names:
                    # Read sheet efficiently
                    df = pd.read_excel(excel_file, sheet_name=sheet_name, 
                                     na_values=['', ' ', 'N/A', 'n/a'])
                    
                    # Convert to efficient format
                    sheets_data[sheet_name] = {
                        'data': df.to_dict('records'),
                        'columns': df.columns.tolist(),
                        'shape': df.shape,
                        'dtypes': df.dtypes.to_dict()
                    }
                
                data['sheets'] = sheets_data
                
                # Extract engineering-specific data
                data['engineering_data'] = self._extract_engineering_data_from_excel(sheets_data)
                
        except Exception as e:
            logging.error(f"Excel parsing failed: {e}")
            data = {'error': str(e)}
        
        return data, metadata
    
    def _extract_engineering_data_from_excel(self, sheets_data: Dict) -> Dict[str, Any]:
        """Extract engineering-specific data from Excel sheets"""
        
        engineering_data = {
            'dimensions': [],
            'loads': [],
            'materials': [],
            'calculations': []
        }
        
        for sheet_name, sheet_data in sheets_data.items():
            records = sheet_data['data']
            
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        continue
                    
                    key_lower = str(key).lower()
                    value_str = str(value).lower()
                    
                    # Extract dimensions
                    if any(dim in key_lower for dim in ['height', 'length', 'width', 'thickness']):
                        if isinstance(value, (int, float)):
                            engineering_data['dimensions'].append({
                                'parameter': key,
                                'value': value,
                                'sheet': sheet_name
                            })
                    
                    # Extract loads
                    if any(load in key_lower for load in ['load', 'force', 'pressure']):
                        if isinstance(value, (int, float)):
                            engineering_data['loads'].append({
                                'parameter': key,
                                'value': value,
                                'sheet': sheet_name
                            })
                    
                    # Extract materials
                    if any(mat in value_str for mat in ['concrete', 'steel', 'soil']):
                        engineering_data['materials'].append({
                            'parameter': key,
                            'value': value,
                            'sheet': sheet_name
                        })
        
        return engineering_data
    
    async def _parse_pdf_optimized(self, file_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimized PDF parsing with streaming"""
        
        data = {}
        metadata = {}
        
        try:
            # Use memory mapping for large files
            with open(file_path, 'rb') as f:
                # Get file size
                f.seek(0, 2)
                file_size = f.tell()
                f.seek(0)
                
                metadata['file_size'] = file_size
                metadata['file_type'] = 'pdf'
                
                # For now, extract basic text (can be enhanced with PyPDF2/pdfplumber)
                if file_size < 10 * 1024 * 1024:  # 10MB limit for in-memory processing
                    content = f.read()
                    data['raw_content'] = content[:1000]  # First 1000 bytes as sample
                    data['extracted_text'] = self._extract_text_from_pdf_bytes(content)
                else:
                    data['error'] = 'File too large for processing'
                
        except Exception as e:
            logging.error(f"PDF parsing failed: {e}")
            data = {'error': str(e)}
        
        return data, metadata
    
    def _extract_text_from_pdf_bytes(self, content: bytes) -> str:
        """Extract text from PDF bytes (simplified)"""
        # This is a simplified version - would use proper PDF library in production
        try:
            text = content.decode('utf-8', errors='ignore')
            return text[:5000]  # First 5000 characters
        except Exception:
            return "Could not extract text"
    
    async def _parse_geostudio_optimized(self, file_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimized GeoStudio file parsing"""
        
        data = {}
        metadata = {}
        
        try:
            # GeoStudio files are binary - use memory mapping
            with open(file_path, 'rb') as f:
                file_size = f.seek(0, 2)
                f.seek(0)
                
                metadata['file_size'] = file_size
                metadata['file_type'] = 'geostudio'
                
                # Read header
                header = f.read(100)
                data['header'] = header.hex()
                
                # Extract basic information (would need GeoStudio API for full parsing)
                data['analysis_type'] = 'geotechnical'
                data['extracted_data'] = self._extract_geostudio_data(header)
                
        except Exception as e:
            logging.error(f"GeoStudio parsing failed: {e}")
            data = {'error': str(e)}
        
        return data, metadata
    
    def _extract_geostudio_data(self, header: bytes) -> Dict[str, Any]:
        """Extract data from GeoStudio header"""
        # Simplified extraction - would use proper GeoStudio API
        return {
            'soil_properties': [],
            'analysis_results': [],
            'geometry': []
        }
    
    async def _parse_staad_optimized(self, file_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimized STAAD.Pro file parsing"""
        
        data = {}
        metadata = {}
        
        try:
            # STAAD files can be text-based
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(10000)  # Read first 10KB
                
                metadata['file_size'] = len(content)
                metadata['file_type'] = 'staad'
                
                data['content_sample'] = content[:1000]
                data['structural_data'] = self._extract_staad_data(content)
                
        except Exception as e:
            logging.error(f"STAAD parsing failed: {e}")
            data = {'error': str(e)}
        
        return data, metadata
    
    def _extract_staad_data(self, content: str) -> Dict[str, Any]:
        """Extract structural data from STAAD content"""
        # Simplified extraction
        return {
            'nodes': [],
            'elements': [],
            'loads': [],
            'materials': []
        }
    
    async def _parse_text_optimized(self, file_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Optimized text file parsing"""
        
        data = {}
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                metadata['file_size'] = len(content)
                metadata['file_type'] = 'text'
                metadata['line_count'] = content.count('\n')
                
                data['content'] = content
                data['engineering_keywords'] = self._extract_engineering_keywords(content)
                
        except Exception as e:
            logging.error(f"Text parsing failed: {e}")
            data = {'error': str(e)}
        
        return data, metadata
    
    def _extract_engineering_keywords(self, content: str) -> List[str]:
        """Extract engineering keywords from text"""
        keywords = []
        engineering_terms = [
            'concrete', 'steel', 'load', 'force', 'moment', 'stress', 'strain',
            'foundation', 'beam', 'column', 'wall', 'slab', 'pile', 'footing',
            'seismic', 'wind', 'dead', 'live', 'ultimate', 'allowable'
        ]
        
        content_lower = content.lower()
        for term in engineering_terms:
            if term in content_lower:
                keywords.append(term)
        
        return keywords
    
    async def _parse_generic(self, file_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Generic file parsing fallback"""
        
        data = {}
        metadata = {}
        
        try:
            stat = Path(file_path).stat()
            metadata['file_size'] = stat.st_size
            metadata['file_type'] = 'unknown'
            metadata['modified_time'] = stat.st_mtime
            
            data['parsed'] = False
            data['reason'] = 'Unknown file type'
            
        except Exception as e:
            logging.error(f"Generic parsing failed: {e}")
            data = {'error': str(e)}
        
        return data, metadata
    
    def _calculate_confidence(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Calculate confidence score for parsed data"""
        if 'error' in data:
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Add confidence based on extracted data
        if 'engineering_data' in data:
            confidence += 0.3
        if 'extracted_text' in data:
            confidence += 0.1
        if metadata.get('file_size', 0) > 0:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    async def handle_response(self, message: AgentMessage):
        """Handle response messages"""
        logging.info(f"File analyzer received response: {message.payload}")
    
    async def handle_notification(self, message: AgentMessage):
        """Handle notification messages"""
        logging.info(f"File analyzer received notification: {message.payload}")
    
    async def handle_error(self, message: AgentMessage):
        """Handle error messages"""
        logging.error(f"File analyzer received error: {message.payload}")
    
    async def on_stop(self):
        """Cleanup memory-mapped files"""
        for handle in self.mmap_handles.values():
            try:
                handle.close()
            except Exception as e:
                logging.warning(f"Error closing mmap handle: {e}")
        
        self.mmap_handles.clear()

# Singleton pattern for shared resources
_file_analyzer_instance = None

def get_file_analyzer_agent() -> FileAnalyzerAgent:
    """Get singleton instance of file analyzer agent"""
    global _file_analyzer_instance
    if _file_analyzer_instance is None:
        _file_analyzer_instance = FileAnalyzerAgent()
    return _file_analyzer_instance