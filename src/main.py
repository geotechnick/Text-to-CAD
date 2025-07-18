"""
Main entry point for the Text-to-CAD Multi-Agent System
Optimized for maximum computational efficiency
"""

import asyncio
import logging
import time
from typing import List, Dict, Any
from pathlib import Path

from .core.agent_framework import message_broker, performance_monitor
from .agents.orchestrator_agent import get_orchestrator_agent
from .agents.prompt_parser_agent import get_prompt_parser_agent
from .agents.file_analyzer_agent import get_file_analyzer_agent
from .agents.ifc_generator_agent import get_ifc_generator_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_to_cad.log'),
        logging.StreamHandler()
    ]
)

class TextToCADSystem:
    """
    Main system class for the Text-to-CAD multi-agent system
    """
    
    def __init__(self):
        self.agents = {}
        self.is_running = False
        self.start_time = None
        
    async def initialize(self):
        """Initialize the multi-agent system"""
        
        logging.info("Initializing Text-to-CAD Multi-Agent System...")
        
        # Initialize agents
        self.agents['orchestrator'] = get_orchestrator_agent()
        self.agents['prompt_parser'] = get_prompt_parser_agent()
        self.agents['file_analyzer'] = get_file_analyzer_agent()
        self.agents['ifc_generator'] = get_ifc_generator_agent()
        
        # Register agents with message broker
        for agent in self.agents.values():
            message_broker.register_agent(agent)
        
        # Start agents
        start_tasks = []
        for agent in self.agents.values():
            start_tasks.append(agent.start())
        
        await asyncio.gather(*start_tasks)
        
        self.is_running = True
        self.start_time = time.time()
        
        logging.info("Text-to-CAD Multi-Agent System initialized successfully")
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
    
    async def shutdown(self):
        """Shutdown the multi-agent system gracefully"""
        
        logging.info("Shutting down Text-to-CAD Multi-Agent System...")
        
        self.is_running = False
        
        # Stop agents
        stop_tasks = []
        for agent in self.agents.values():
            stop_tasks.append(agent.stop())
        
        await asyncio.gather(*stop_tasks)
        
        # Unregister agents
        for agent in self.agents.values():
            message_broker.unregister_agent(agent.agent_id)
        
        logging.info("Text-to-CAD Multi-Agent System shutdown complete")
    
    async def process_prompt(self, prompt: str, files: List[str] = None) -> Dict[str, Any]:
        """
        Process a user prompt and generate IFC output
        
        Args:
            prompt: Natural language prompt describing the engineering task
            files: List of file paths to analyze
            
        Returns:
            Dictionary containing the generated IFC model and metadata
        """
        
        if not self.is_running:
            raise RuntimeError("System is not running. Call initialize() first.")
        
        if files is None:
            files = []
        
        # Validate input files
        validated_files = []
        for file_path in files:
            path = Path(file_path)
            if path.exists():
                validated_files.append(str(path.absolute()))
            else:
                logging.warning(f"File not found: {file_path}")
        
        logging.info(f"Processing prompt with {len(validated_files)} files")
        
        # Send request to orchestrator
        orchestrator = self.agents['orchestrator']
        
        from .core.agent_framework import AgentMessage, MessageType
        
        request = AgentMessage(
            sender="system",
            receiver="orchestrator",
            message_type=MessageType.REQUEST,
            payload={
                "action": "process_prompt",
                "prompt": prompt,
                "files": validated_files
            }
        )
        
        # Send message and get workflow ID
        await orchestrator.send_message(request)
        
        # Wait for completion (simplified - would implement proper response handling)
        await asyncio.sleep(5)  # Simulate processing time
        
        # Return mock result for now
        return {
            "status": "completed",
            "ifc_model": {
                "project_name": "Generated Civil Infrastructure",
                "elements": [],
                "properties": {}
            },
            "processing_time": 5.0,
            "workflow_id": f"workflow_{int(time.time())}"
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "system_running": self.is_running,
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "agents": {}
        }
        
        # Get agent status
        for agent_id, agent in self.agents.items():
            status["agents"][agent_id] = {
                "running": agent.is_running,
                "performance": agent.get_performance_metrics()
            }
        
        # Get system metrics
        status["system_metrics"] = message_broker.get_system_metrics()
        
        # Get performance analysis
        performance_metrics = performance_monitor.collect_metrics()
        status["performance_analysis"] = performance_monitor.analyze_performance()
        
        return status
    
    async def _performance_monitoring_loop(self):
        """Monitor system performance continuously"""
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = performance_monitor.collect_metrics()
                
                # Analyze performance
                suggestions = performance_monitor.analyze_performance()
                
                if suggestions:
                    logging.info(f"Performance suggestions: {suggestions}")
                
                # Log system health
                total_messages = metrics.get("total_messages", 0)
                if total_messages > 0:
                    logging.info(f"System health: {total_messages} messages processed")
                
                # Sleep before next check
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(10)

# Global system instance
_system_instance = None

def get_system() -> TextToCADSystem:
    """Get global system instance"""
    global _system_instance
    if _system_instance is None:
        _system_instance = TextToCADSystem()
    return _system_instance

async def main():
    """Main function for running the system"""
    
    system = get_system()
    
    try:
        # Initialize system
        await system.initialize()
        
        # Example usage
        logging.info("System ready for processing")
        
        # Process example prompt
        result = await system.process_prompt(
            "Design a reinforced concrete floodwall 4.2m high and 100m long with micropile foundation",
            ["engineering files/Floodwall Bearing 101+50 to 106+00.xlsx"]
        )
        
        logging.info(f"Processing result: {result}")
        
        # Get system status
        status = await system.get_system_status()
        logging.info(f"System status: {status}")
        
        # Keep system running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Shutdown requested")
    except Exception as e:
        logging.error(f"System error: {e}")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())