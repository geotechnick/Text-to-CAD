# Text-to-CAD Multi-Agent System

A computationally efficient multi-agent system for converting natural language prompts and engineering files into IFC (Industry Foundation Classes) models for civil infrastructure projects.

## 🚀 Quick Start - Running the Multi-Agent Model

### Prerequisites
- Python 3.8+ (recommended: Python 3.11)
- 8GB RAM minimum (16GB recommended for large projects)
- 50GB disk space (for dependencies and model data)

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Text-to-CAD.git
cd Text-to-CAD

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models (if using pre-trained components)
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt stopwords
```

### Running the Multi-Agent System

#### Basic Usage
```bash
# Start the multi-agent system
python -m src.main

# The system will initialize all agents and be ready for requests
```

#### Processing Engineering Prompts
```python
import asyncio
from src.main import get_system

async def run_example():
    # Initialize the system
    system = get_system()
    await system.initialize()
    
    # Process a prompt with engineering files
    result = await system.process_prompt(
        prompt="Design a reinforced concrete floodwall 4.2m high and 850m long with micropile foundation for seismic zone 4",
        files=[
            "engineering files/Floodwall Bearing 101+50 to 106+00.xlsx",
            "engineering files/Micropile Capacity Weir Gate.xlsx",
            "engineering files/structural calcs.pdf"
        ]
    )
    
    # Save the generated IFC model
    with open("generated_model.ifc", "w") as f:
        f.write(result["ifc_content"])
    
    print(f"✅ Generated IFC model with {result['element_count']} elements")
    print(f"⏱️  Processing time: {result['processing_time']:.2f} seconds")
    
    # Gracefully shutdown
    await system.shutdown()

# Run the example
asyncio.run(run_example())
```

#### Command Line Interface
```bash
# Process a single prompt
python -m src.main --prompt "Design a 3m high retaining wall with strip footing" --files "data/calculations.xlsx"

# Batch process multiple prompts
python -m src.main --batch prompts.json --output-dir results/

# Monitor system performance
python -m src.main --monitor --duration 3600  # Monitor for 1 hour
```

## 🤖 Training the Multi-Agent Model

### Quick Start Training

#### Test the System First
```bash
# Test individual components
python quick_train.py --test-components

# Run quick training demo (3 epochs, ~2 minutes)
python quick_train.py
```

#### Basic Training
```bash
# Train with default settings
python train_model.py --mode full --epochs 10

# Train individual agents
python train_model.py --mode prompt_parser --epochs 5
python train_model.py --mode file_analyzer --epochs 5
python train_model.py --mode ifc_generator --epochs 5

# Validate existing system
python train_model.py --mode validate --test-cases
```

### Training Data Structure

The training system automatically creates and manages data:

```
training_data/
├── prompts/
│   ├── sample_prompts.json          # Manual training prompts
│   └── quick_train_prompts.json     # Auto-generated samples
├── files/
│   ├── excel/                       # Excel engineering files
│   ├── pdf/                         # PDF calculations
│   ├── geostudio/                   # GeoStudio analysis files
│   └── staad_pro/                   # STAAD.Pro structural files
├── ifc_models/
│   ├── reference/                   # Reference IFC models
│   └── validation/                  # Validation models
└── synthetic/
    └── generated_prompts.json       # Auto-generated synthetic data
```

### Training Configuration

#### Comprehensive Configuration (`training_config.yaml`)
```yaml
system:
  max_workers: 8
  cache_size: 100
  timeout: 300

training:
  epochs: 20                    # Number of training epochs
  batch_size: 8                # Training batch size
  learning_rate: 0.001         # Learning rate
  validation_split: 0.2        # Validation data percentage
  early_stopping: true         # Stop when no improvement
  patience: 5                  # Early stopping patience
  save_checkpoints: true       # Save training checkpoints
  checkpoint_interval: 5       # Save every N epochs

agents:
  prompt_parser:
    model_type: "engineering_bert"
    vocab_size: 50000
    enable_augmentation: true
    
  file_analyzer:
    max_workers: 4
    chunk_size: 1024
    supported_formats: ["xlsx", "pdf", "gsz", "gp12a"]
    
  ifc_generator:
    template_library: "templates/civil_engineering.json"
    geometry_cache_size: 1000
    ifc_schema: "IFC4"

data:
  generate_synthetic: true      # Auto-generate training data
  synthetic_count: 1000        # Number of synthetic examples
  augmentation_factor: 3       # Data augmentation multiplier
```

### Advanced Training Commands

#### Full Training Pipeline
```bash
# Complete training with configuration
python train_model.py \
    --config training_config.yaml \
    --mode full \
    --epochs 20 \
    --generate-synthetic \
    --synthetic-count 1000

# Large-scale training
python train_model.py \
    --mode full \
    --epochs 50 \
    --batch-size 16 \
    --synthetic-count 5000 \
    --save-checkpoints \
    --early-stopping
```

#### Specialized Training
```bash
# Focus on prompt understanding
python train_model.py \
    --mode prompt_parser \
    --epochs 15 \
    --learning-rate 0.0001 \
    --validation-split 0.3

# Optimize file processing
python train_model.py \
    --mode file_analyzer \
    --data-dir "engineering files" \
    --epochs 10

# Improve IFC generation
python train_model.py \
    --mode ifc_generator \
    --epochs 20 \
    --batch-size 4
```

### Training Process Overview

#### 1. **Data Preparation** (Automatic)
- **Load Existing Data**: Scans for engineering files and prompts
- **Generate Synthetic Data**: Creates realistic engineering scenarios
- **Data Augmentation**: Parameter variations, unit conversions
- **Train/Val/Test Split**: Automatic data splitting (70/20/10)

#### 2. **Agent Training** (Parallel)
- **Prompt Parser**: Engineering language understanding and parameter extraction
- **File Analyzer**: Multi-format engineering file processing
- **IFC Generator**: 3D model generation with proper IFC hierarchy
- **System Integration**: End-to-end workflow optimization

#### 3. **Validation & Testing** (Comprehensive)
- **Component Testing**: Individual agent performance validation
- **Integration Testing**: End-to-end system validation
- **Engineering Validation**: Code compliance and structural soundness
- **Performance Testing**: Speed and resource usage optimization

#### 4. **Results & Analysis** (Automated)
- **Training Metrics**: Accuracy, completion rates, generation quality
- **Performance Plots**: Training curves and progress visualization
- **Validation Reports**: Detailed analysis with recommendations
- **Model Checkpoints**: Best model versions saved automatically

### Training Data Sources

#### Manual Training Data
Create `training_data/prompts/engineering_prompts.json`:
```json
{
  "prompts": [
    {
      "text": "Design a reinforced concrete floodwall 4.2m high and 850m long with micropile foundation",
      "intent": "complex_infrastructure",
      "parameters": [
        {"name": "height", "value": 4.2, "unit": "m", "confidence": 0.95},
        {"name": "length", "value": 850.0, "unit": "m", "confidence": 0.95},
        {"name": "material", "value": "reinforced_concrete", "confidence": 0.9},
        {"name": "foundation_type", "value": "micropile", "confidence": 0.85}
      ],
      "constraints": [
        {"type": "flood_protection", "requirement": "500_year_protection"},
        {"type": "safety", "requirement": "factor_of_safety_2.0"}
      ]
    }
  ]
}
```

#### Automatic Synthetic Data
The system generates 1000+ synthetic examples like:
- "Design a steel beam 8m long for 150kN load capacity"
- "Create a foundation system for 25 gpm pump with concrete pad"
- "Build a retaining wall 3.5m high for earthquake zone 4"

### Training Outputs

#### Model Files
```
models/
├── checkpoints/
│   ├── best_model/              # Best performing model
│   ├── checkpoint_epoch_5/      # Regular checkpoints
│   └── checkpoint_epoch_10/
├── logs/
│   └── training_history.json    # Complete training history
└── plots/
    └── training_curves.png      # Performance visualization
```

#### Training Reports
- **Accuracy Metrics**: Component and system-wide performance
- **Engineering Validation**: Code compliance and structural soundness
- **Performance Analysis**: Speed, memory usage, scalability
- **Recommendations**: Suggestions for improvement and optimization

### Performance Expectations

#### Training Performance
- **Quick Training**: 3 epochs, ~2 minutes (demo purposes)
- **Basic Training**: 10 epochs, ~15 minutes (testing)
- **Full Training**: 20-50 epochs, 1-3 hours (production)

#### Model Performance Targets
- **Prompt Parser Accuracy**: 85%+ engineering language understanding
- **File Analysis Success**: 80%+ multi-format file processing
- **IFC Generation Quality**: 90%+ valid IFC model creation
- **End-to-End Completion**: 75%+ successful prompt-to-IFC workflows

### Monitoring & Optimization

#### Real-time Monitoring
```bash
# Monitor training progress
tail -f training.log

# View training metrics in real-time
python -c "
import json
with open('models/logs/training_history.json') as f:
    data = json.load(f)
    print(f'Latest accuracy: {data[\"validation_history\"][-1][\"accuracy\"]:.3f}')
"
```

#### Continuous Improvement
- **Incremental Training**: Add new data without retraining from scratch
- **Performance Optimization**: Automatic parameter tuning
- **Quality Assessment**: Engineering accuracy validation
- **Production Monitoring**: Real-time performance tracking

## 🚀 Features

- **High-Performance Multi-Agent Architecture**: Optimized for computational efficiency with shared memory and async processing
- **Natural Language Processing**: Parse engineering prompts with domain-specific understanding
- **Multi-Format File Analysis**: Support for Excel, PDF, GeoStudio, STAAD.Pro, and other engineering file formats
- **IFC Generation**: Generate compliant IFC4 models with proper spatial hierarchy and properties
- **Intelligent Workflow Orchestration**: Adaptive workflow management based on project complexity
- **Real-time Performance Monitoring**: Comprehensive metrics and optimization suggestions
- **Continuous Learning**: Automated model updates and performance monitoring

## 🏗️ Architecture

The system uses a **computationally efficient multi-agent architecture** with the following components:

### Core Framework (`src/core/`)

#### `agent_framework.py`
- **BaseAgent**: High-performance base class with async message processing, shared memory communication, and performance monitoring
- **MessageBroker**: Lightweight message routing with priority queues and system-wide metrics
- **PerformanceMonitor**: Real-time system optimization with bottleneck detection and suggestions
- **AgentPool**: Load balancing and scaling for agent instances
- **Shared Memory Manager**: Cross-agent communication without serialization overhead

### Specialized Agents (`src/agents/`)

#### `orchestrator_agent.py`
- **Primary Controller**: Coordinates the entire Text-to-CAD workflow
- **Workflow Management**: Creates optimized task sequences based on project complexity
- **Parallel Execution**: Manages dependency-aware task scheduling with concurrent processing
- **Performance Optimization**: Monitors system health and provides real-time workflow adjustments
- **Workflow Templates**: Pre-configured patterns for different engineering scenarios:
  - Simple structures (walls, foundations)
  - Complex infrastructure (floodwalls, pile systems)
  - Retrofit projects (existing structure modifications)
  - Analysis-only tasks (validation, checking)

#### `prompt_parser_agent.py`
- **Engineering NLP**: Specialized natural language processing for civil engineering terminology
- **Intent Classification**: Automatically determines project type and complexity
- **Parameter Extraction**: Identifies dimensions, materials, loads, and design requirements
- **Constraint Analysis**: Recognizes safety factors, code requirements, and performance criteria
- **Optimization Features**:
  - Precompiled regex patterns for fast parsing
  - Engineering vocabulary lookup tables
  - Confidence scoring for extracted parameters
  - Ambiguity detection and clarification requests

#### `file_analyzer_agent.py`
- **Multi-Format Parser**: Handles diverse engineering file formats with streaming support
- **Supported Formats**:
  - **Excel/CSV**: Structural calculations, load analysis, material properties
  - **PDF**: OCR and text extraction from technical documents
  - **GeoStudio (.gsz, .gsd)**: Geotechnical analysis data extraction
  - **STAAD.Pro (.gp12*)**: Structural analysis results parsing
  - **Text Files**: Engineering specifications and reports
- **Performance Features**:
  - Memory-mapped file processing for large files
  - Parallel file analysis with async processing
  - Intelligent caching with LRU eviction
  - Engineering-specific data extraction patterns

#### `ifc_generator_agent.py`
- **IFC Model Creation**: Generates compliant IFC4 models with proper spatial hierarchy
- **Template-Based Generation**: Efficient creation using pre-configured element templates
- **Geometry Engine**: Optimized 3D geometry creation with caching for common shapes
- **Property Management**: Comprehensive property sets for civil engineering elements
- **Specialized Elements**:
  - Structural elements (walls, beams, columns, slabs)
  - Foundation systems (footings, piles, caissons)
  - Civil infrastructure (floodwalls, culverts, bridges)
  - Custom elements via IfcBuildingElementProxy
- **Performance Optimizations**:
  - Geometry caching for repeated shapes
  - Batch processing for large element sets
  - Memory-efficient IFC file generation

### System Components (`src/`)

#### `main.py`
- **System Orchestration**: Main entry point and system lifecycle management
- **TextToCADSystem Class**: High-level API for system operations
- **Initialization**: Agent startup and message broker registration
- **Process Management**: Handles prompt processing requests and system monitoring
- **Performance Monitoring**: Continuous system health tracking and optimization
- **Graceful Shutdown**: Proper cleanup and resource management

### Configuration & Dependencies

#### `requirements.txt`
- **Performance-Optimized Dependencies**: Carefully selected packages for maximum efficiency
- **Core Libraries**:
  - `ifcopenshell`: Primary IFC processing library
  - `pandas`, `numpy`: High-performance data processing
  - `asyncio`, `uvloop`: Async processing optimization
  - `spacy`, `nltk`: Natural language processing
  - `celery`, `redis`: Distributed task processing
- **Engineering-Specific**: Libraries for CAD, structural analysis, and geotechnical data
- **Development Tools**: Testing, linting, and performance profiling tools

### Project Documentation

#### Strategy Documents
- **`strategy first pass.txt`**: Comprehensive 6-phase implementation strategy
- **`prompt to ifc thoughts.txt`**: Detailed multi-agent architecture analysis
- **`ifc summary.txt`**: Complete IFC file structure reference and civil engineering mapping

#### Engineering Assets (`engineering files/`)
- **Real Engineering Data**: Production files from actual civil infrastructure projects
- **Supported File Types**:
  - Excel spreadsheets with structural calculations
  - GeoStudio geotechnical analysis files
  - STAAD.Pro structural analysis results
  - PDF calculation documents
  - Various engineering analysis formats

### Performance Characteristics

#### Computational Efficiency
- **Memory Usage**: < 500MB for typical projects
- **Processing Speed**: 
  - Simple structures: < 30 seconds
  - Complex infrastructure: < 2 minutes
  - Large file processing: < 5 minutes
- **Scalability**: Horizontal scaling through agent pools and parallel processing
- **Optimization**: Real-time performance monitoring with automatic tuning suggestions

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Text-to-CAD.git
cd Text-to-CAD
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. For development:
```bash
pip install -r requirements-dev.txt
```

## 🚀 Quick Start

### Basic Usage

```python
import asyncio
from src.main import get_system

async def main():
    # Initialize the system
    system = get_system()
    await system.initialize()
    
    # Process a prompt
    result = await system.process_prompt(
        "Design a reinforced concrete floodwall 4.2m high and 100m long with micropile foundation",
        ["engineering_files/Floodwall_Bearing.xlsx"]
    )
    
    print(f"Generated IFC model: {result['ifc_model']}")
    
    # Shutdown
    await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Command Line Interface

```bash
# Run the system
python -m src.main

# Process a prompt
python -m src.cli --prompt "Design a 3m high retaining wall" --files "data/calculations.xlsx"

# Get system status
python -m src.cli --status
```

## 📊 Performance Optimization

The system is optimized for maximum computational efficiency:

### Memory Management
- **Shared Memory**: Cross-agent communication without serialization overhead
- **Memory Mapping**: Large file processing with minimal memory footprint
- **Intelligent Caching**: LRU caches for parsed data and geometry

### Parallel Processing
- **Async-First Design**: Non-blocking operations throughout
- **Thread/Process Pools**: Efficient resource utilization
- **Dependency-Aware Scheduling**: Optimal task execution order

### Optimization Features
- **Precompiled Regex**: Fast pattern matching for NLP
- **Template-Based Generation**: Efficient IFC model creation
- **Geometry Caching**: Reuse common geometric shapes
- **Streaming File Processing**: Handle large files efficiently

## 🏗️ Supported Engineering Files

### Excel/CSV Files
- Structural calculations
- Load analysis
- Material properties
- Quantity takeoffs

### Analysis Files
- **GeoStudio** (.gsz, .gsd): Geotechnical analysis
- **STAAD.Pro** (.gp12*): Structural analysis
- **RISA** (.r3d): 3D structural analysis
- **SAP2000** (.sdb): Structural analysis

### Documentation
- **PDF**: Structural calculations, specifications
- **CAD Drawings**: DWG, DXF (future support)
- **Images**: Scanned drawings and photos

## 🎯 Engineering Domains

### Civil Infrastructure
- Floodwalls and retaining walls
- Bridge structures
- Culverts and drainage
- Marine structures

### Foundation Systems
- Shallow foundations
- Deep foundations (piles, caissons)
- Micropile systems
- Ground improvement

### Structural Elements
- Concrete structures
- Steel structures
- Composite systems
- Precast elements

## 📈 Performance Metrics

The system provides comprehensive performance monitoring:

```python
# Get system metrics
status = await system.get_system_status()
print(f"Messages per second: {status['system_metrics']['messages_per_second']}")
print(f"Average processing time: {status['performance_analysis']['avg_processing_time']}")
```

### Typical Performance
- **Simple Structures**: < 30 seconds
- **Complex Infrastructure**: < 2 minutes
- **Large File Processing**: < 5 minutes
- **Memory Usage**: < 500MB for typical projects

## 🔧 Configuration

### Environment Variables
```bash
# System configuration
TEXT_TO_CAD_LOG_LEVEL=INFO
TEXT_TO_CAD_MAX_WORKERS=8
TEXT_TO_CAD_CACHE_SIZE=100

# Performance tuning
TEXT_TO_CAD_MEMORY_LIMIT=1GB
TEXT_TO_CAD_TIMEOUT=300
TEXT_TO_CAD_PARALLEL_AGENTS=4
```

### Configuration File
```yaml
# config.yaml
system:
  max_workers: 8
  cache_size: 100
  timeout: 300

agents:
  orchestrator:
    max_workers: 8
  prompt_parser:
    max_workers: 2
  file_analyzer:
    max_workers: 4
  ifc_generator:
    max_workers: 2

performance:
  memory_limit: "1GB"
  enable_caching: true
  enable_monitoring: true
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Performance tests
pytest tests/performance/

# Integration tests
pytest tests/integration/
```

## 📚 API Documentation

### System API
```python
class TextToCADSystem:
    async def initialize() -> None
    async def shutdown() -> None
    async def process_prompt(prompt: str, files: List[str]) -> Dict[str, Any]
    async def get_system_status() -> Dict[str, Any]
```

### Agent API
```python
class BaseAgent:
    async def start() -> None
    async def stop() -> None
    async def handle_request(message: AgentMessage) -> Optional[AgentMessage]
    def get_performance_metrics() -> Dict[str, Any]
```

## 📋 Detailed Program Descriptions

### Core Framework Programs

#### `src/core/agent_framework.py` (850+ lines)
**Purpose**: Foundation of the multi-agent system with performance-optimized communication

**Key Classes**:
- `BaseAgent`: Abstract base class for all agents with async processing, shared memory, and performance monitoring
- `MessageBroker`: Lightweight message routing system with priority queues and system metrics
- `PerformanceMonitor`: Real-time system optimization with bottleneck detection
- `AgentPool`: Load balancing and scaling for agent instances
- `AgentMessage`: Efficient message structure for inter-agent communication

**Performance Features**:
- Shared memory manager for zero-copy communication
- Priority-based message queues with non-blocking operations
- Memory-mapped caching for large data sets
- Automatic performance metrics collection and analysis

### Specialized Agent Programs

#### `src/agents/orchestrator_agent.py` (650+ lines)
**Purpose**: Master coordinator that manages the entire Text-to-CAD workflow

**Key Functionality**:
- **Workflow Creation**: Generates optimized task sequences based on prompt analysis
- **Parallel Execution**: Manages dependency-aware task scheduling with concurrent processing
- **Error Handling**: Sophisticated retry mechanisms with exponential backoff
- **Performance Monitoring**: Real-time workflow optimization and bottleneck detection

**Workflow Templates**:
- Simple structures (walls, foundations)
- Complex infrastructure (floodwalls, pile systems)
- Retrofit projects (existing structure modifications)
- Analysis-only tasks (validation, checking)

#### `src/agents/prompt_parser_agent.py` (550+ lines)
**Purpose**: Specialized NLP engine for civil engineering language processing

**Key Features**:
- **Engineering NLP**: Custom vocabulary and terminology recognition
- **Intent Classification**: Automatic project type and complexity determination
- **Parameter Extraction**: Dimensions, materials, loads, and design requirements
- **Constraint Analysis**: Safety factors, code requirements, and performance criteria

**Optimization Techniques**:
- Precompiled regex patterns for maximum parsing speed
- Engineering vocabulary lookup tables for fast matching
- Confidence scoring for extracted parameters
- Ambiguity detection with targeted clarification requests

#### `src/agents/file_analyzer_agent.py` (450+ lines)
**Purpose**: Multi-format file parser with streaming support for large engineering files

**Supported File Formats**:
- **Excel/CSV**: Structural calculations, load analysis, material properties
- **PDF**: OCR and text extraction from technical documents
- **GeoStudio (.gsz, .gsd)**: Geotechnical analysis data extraction
- **STAAD.Pro (.gp12*)**: Structural analysis results parsing
- **Text Files**: Engineering specifications and reports

**Performance Features**:
- Memory-mapped file processing for files > 10MB
- Parallel file analysis with async processing
- Intelligent caching with LRU eviction
- Engineering-specific data extraction patterns

#### `src/agents/ifc_generator_agent.py` (700+ lines)
**Purpose**: High-performance IFC model generator with template-based optimization

**Core Capabilities**:
- **IFC Model Creation**: Generates compliant IFC4 models with proper spatial hierarchy
- **Template System**: Pre-configured element templates for common structural elements
- **Geometry Engine**: Optimized 3D geometry creation with shape caching
- **Property Management**: Comprehensive property sets for civil engineering

**Specialized Elements**:
- Structural elements (walls, beams, columns, slabs)
- Foundation systems (footings, piles, caissons)
- Civil infrastructure (floodwalls, culverts, bridges)
- Custom elements via IfcBuildingElementProxy

### System Programs

#### `src/main.py` (200+ lines)
**Purpose**: System orchestration and high-level API

**Key Components**:
- `TextToCADSystem`: Main system class with lifecycle management
- Agent initialization and message broker registration
- Process management for prompt processing requests
- Continuous system health monitoring and optimization
- Graceful shutdown with proper resource cleanup

**API Methods**:
- `initialize()`: Start all agents and register with message broker
- `process_prompt()`: Main processing pipeline for user requests
- `get_system_status()`: Comprehensive system performance metrics
- `shutdown()`: Clean system shutdown with resource management

### Configuration and Dependencies

#### `requirements.txt` (80+ packages)
**Purpose**: Performance-optimized dependency management

**Core Categories**:
- **Async Processing**: `asyncio`, `uvloop`, `aiofiles`
- **Data Processing**: `pandas`, `numpy`, `openpyxl`
- **NLP**: `spacy`, `nltk`, `transformers`
- **IFC Processing**: `ifcopenshell`, `ifcopenshell-python`
- **File Processing**: `PyPDF2`, `pdfplumber`, `Pillow`
- **Performance**: `cython`, `numba`, `memory-profiler`
- **Engineering**: Custom libraries for structural analysis

### Documentation Programs

#### Strategy Documents
- **`strategy first pass.txt`** (247 lines): Comprehensive 6-phase implementation strategy
- **`prompt to ifc thoughts.txt`** (760 lines): Detailed multi-agent architecture analysis
- **`ifc summary.txt`** (329 lines): Complete IFC file structure reference

#### Engineering Assets (`engineering files/`)
**Real Production Data**: 40+ files from actual civil infrastructure projects
- Excel spreadsheets with structural calculations
- GeoStudio geotechnical analysis files (.gsz, .sczp, .scyp)
- STAAD.Pro structural analysis results (.gp12*)
- PDF calculation documents
- Various engineering analysis formats

### Performance Characteristics

#### Computational Efficiency
- **Memory Usage**: < 500MB for typical projects
- **Processing Speed**: 
  - Simple structures: < 30 seconds
  - Complex infrastructure: < 2 minutes
  - Large file processing: < 5 minutes
- **Scalability**: Horizontal scaling through agent pools
- **Optimization**: Real-time performance monitoring with automatic tuning

#### System Architecture Benefits
- **Async-First Design**: Non-blocking operations throughout
- **Shared Memory**: Zero-copy communication between agents
- **Intelligent Caching**: LRU caches for parsed data and geometry
- **Parallel Processing**: Concurrent execution with dependency awareness
- **Performance Monitoring**: Real-time optimization and bottleneck detection

## 🔍 Monitoring and Debugging

### Performance Monitoring
```python
from src.core.agent_framework import performance_monitor

# Collect metrics
metrics = performance_monitor.collect_metrics()

# Analyze performance
suggestions = performance_monitor.analyze_performance()
```

### Logging
```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# View logs
tail -f text_to_cad.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
mypy src/

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on modern async Python architecture
- Utilizes IfcOpenShell for IFC processing
- Inspired by civil engineering automation needs
- Optimized for computational efficiency

## 📞 Support

For support, questions, or contributions:
- Create an issue on GitHub
- Email: support@text-to-cad.com
- Documentation: https://docs.text-to-cad.com

---

**Text-to-CAD Multi-Agent System** - Transforming engineering design through AI-powered automation.