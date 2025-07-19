"""
Computationally Efficient Multi-Agent Framework for Text-to-CAD
Uses shared memory, async processing, and minimal overhead design
"""

import asyncio
import multiprocessing as mp
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
import json
from pathlib import Path

# Simple in-memory message passing for Windows compatibility
# Instead of shared memory manager which has pickling issues
_global_message_queues = {}
_global_lock = threading.Lock()

class MessageType(Enum):
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    NOTIFICATION = "NOTIFICATION"
    ERROR = "ERROR"

class Priority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class AgentMessage:
    """Lightweight message structure for inter-agent communication"""
    sender: str
    receiver: str
    message_type: MessageType
    payload: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    correlation_id: Optional[str] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class BaseAgent(ABC):
    """Base agent class optimized for computational efficiency"""
    
    def __init__(self, agent_id: str, max_workers: int = 4):
        self.agent_id = agent_id
        self.max_workers = max_workers
        self.is_running = False
        self.message_queue = queue.Queue()  # Use simple Queue instead of PriorityQueue for now
        self.response_handlers = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance monitoring
        self.start_time = time.time()
        self.messages_processed = 0
        self.total_processing_time = 0
        
        # Memory-mapped cache for large data
        self.cache_dir = Path(f"cache/{agent_id}")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Register agent in global message queues (Windows-compatible)
        with _global_lock:
            _global_message_queues[agent_id] = self.message_queue
        
        logging.info(f"Agent {agent_id} initialized with {max_workers} workers")
    
    async def start(self):
        """Start the agent with async message processing"""
        self.is_running = True
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        # Call agent-specific startup
        await self.on_start()
        
        logging.info(f"Agent {self.agent_id} started")
    
    async def stop(self):
        """Stop the agent gracefully"""
        self.is_running = False
        
        # Wait for current tasks to complete
        await self.on_stop()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logging.info(f"Agent {self.agent_id} stopped")
    
    async def _message_processing_loop(self):
        """Main message processing loop"""
        while self.is_running:
            try:
                # Non-blocking queue check
                try:
                    message = self.message_queue.get_nowait()
                    await self._process_message(message)
                except queue.Empty:
                    await asyncio.sleep(0.01)  # Minimal sleep to prevent busy waiting
                    continue
                    
            except Exception as e:
                logging.error(f"Error in message processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_message(self, message: AgentMessage):
        """Process a single message efficiently"""
        start_time = time.time()
        
        try:
            if message.message_type == MessageType.REQUEST:
                response = await self.handle_request(message)
                if response:
                    await self.send_message(response)
            elif message.message_type == MessageType.RESPONSE:
                await self.handle_response(message)
            elif message.message_type == MessageType.NOTIFICATION:
                await self.handle_notification(message)
            elif message.message_type == MessageType.ERROR:
                await self.handle_error(message)
                
        except Exception as e:
            error_msg = AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.ERROR,
                payload={"error": str(e), "original_message": message.payload}
            )
            await self.send_message(error_msg)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.messages_processed += 1
    
    async def send_message(self, message: AgentMessage):
        """Send message to another agent efficiently"""
        with _global_lock:
            if message.receiver in _global_message_queues:
                _global_message_queues[message.receiver].put(message)
            else:
                logging.warning(f"Agent {message.receiver} not found")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        uptime = time.time() - self.start_time
        avg_processing_time = self.total_processing_time / max(1, self.messages_processed)
        
        return {
            "agent_id": self.agent_id,
            "uptime": uptime,
            "messages_processed": self.messages_processed,
            "avg_processing_time": avg_processing_time,
            "messages_per_second": self.messages_processed / max(1, uptime)
        }
    
    def cache_large_data(self, key: str, data: Any) -> Path:
        """Cache large data using JSON serialization"""
        cache_path = self.cache_dir / f"{key}.cache"
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except (TypeError, ValueError):
            # Fallback to string representation for non-serializable data
            with open(cache_path, 'w') as f:
                f.write(str(data))
        return cache_path
    
    def load_cached_data(self, key: str) -> Optional[Any]:
        """Load cached data from JSON file"""
        cache_path = self.cache_dir / f"{key}.cache"
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                # Fallback to reading as string
                with open(cache_path, 'r') as f:
                    return f.read()
        return None
    
    # Abstract methods for agent-specific implementation
    @abstractmethod
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        pass
    
    @abstractmethod
    async def handle_response(self, message: AgentMessage):
        pass
    
    @abstractmethod
    async def handle_notification(self, message: AgentMessage):
        pass
    
    @abstractmethod
    async def handle_error(self, message: AgentMessage):
        pass
    
    async def on_start(self):
        """Called when agent starts - override for custom initialization"""
        pass
    
    async def on_stop(self):
        """Called when agent stops - override for custom cleanup"""
        pass

class AgentPool:
    """Manages a pool of agents for load balancing"""
    
    def __init__(self, agent_class: type, pool_size: int = 4):
        self.agent_class = agent_class
        self.pool_size = pool_size
        self.agents = []
        self.current_index = 0
        self.lock = threading.Lock()
    
    async def start_pool(self):
        """Start all agents in the pool"""
        for i in range(self.pool_size):
            agent = self.agent_class(f"{self.agent_class.__name__}_{i}")
            await agent.start()
            self.agents.append(agent)
    
    async def stop_pool(self):
        """Stop all agents in the pool"""
        for agent in self.agents:
            await agent.stop()
    
    def get_next_agent(self) -> BaseAgent:
        """Get next available agent using round-robin"""
        with self.lock:
            agent = self.agents[self.current_index]
            self.current_index = (self.current_index + 1) % self.pool_size
            return agent
    
    def get_least_loaded_agent(self) -> BaseAgent:
        """Get agent with smallest message queue"""
        return min(self.agents, key=lambda a: a.message_queue.qsize())

class MessageBroker:
    """Lightweight message broker for agent communication"""
    
    def __init__(self):
        self.agents = {}
        self.message_stats = {
            "total_messages": 0,
            "messages_per_type": {},
            "avg_latency": 0
        }
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the broker"""
        self.agents[agent.agent_id] = agent
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    async def broadcast_message(self, message: AgentMessage, exclude_sender: bool = True):
        """Broadcast message to all agents"""
        for agent_id, agent in self.agents.items():
            if exclude_sender and agent_id == message.sender:
                continue
            
            broadcast_msg = AgentMessage(
                sender=message.sender,
                receiver=agent_id,
                message_type=message.message_type,
                payload=message.payload,
                priority=message.priority
            )
            await agent.send_message(broadcast_msg)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide performance metrics"""
        agent_metrics = {}
        total_messages = 0
        
        for agent_id, agent in self.agents.items():
            metrics = agent.get_performance_metrics()
            agent_metrics[agent_id] = metrics
            total_messages += metrics["messages_processed"]
        
        return {
            "total_agents": len(self.agents),
            "total_messages": total_messages,
            "agent_metrics": agent_metrics
        }

# Global message broker instance
message_broker = MessageBroker()

class PerformanceMonitor:
    """Monitor and optimize system performance"""
    
    def __init__(self):
        self.metrics_history = []
        self.optimization_suggestions = []
    
    def collect_metrics(self):
        """Collect system performance metrics"""
        metrics = message_broker.get_system_metrics()
        metrics["timestamp"] = time.time()
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics to limit memory
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def analyze_performance(self) -> List[str]:
        """Analyze performance and suggest optimizations"""
        if len(self.metrics_history) < 2:
            return []
        
        suggestions = []
        latest = self.metrics_history[-1]
        
        # Check for slow agents
        for agent_id, metrics in latest["agent_metrics"].items():
            if metrics["avg_processing_time"] > 1.0:  # > 1 second
                suggestions.append(f"Agent {agent_id} is slow (avg: {metrics['avg_processing_time']:.2f}s)")
        
        # Check for queue buildup
        for agent_id in message_broker.agents:
            queue_size = shared_queues[agent_id].qsize()
            if queue_size > 100:
                suggestions.append(f"Agent {agent_id} has large queue ({queue_size} messages)")
        
        return suggestions

# Global performance monitor
performance_monitor = PerformanceMonitor()