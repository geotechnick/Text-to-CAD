"""
Orchestrator Agent - Computationally efficient workflow management
Coordinates all agents with minimal overhead and maximum performance
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json

from ..core.agent_framework import BaseAgent, AgentMessage, MessageType, Priority, message_broker

class WorkflowState(Enum):
    INITIALIZED = "initialized"
    PARSING_PROMPT = "parsing_prompt"
    ANALYZING_FILES = "analyzing_files"
    GENERATING_DESIGN = "generating_design"
    CREATING_IFC = "creating_ifc"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowTask:
    task_id: str
    agent_id: str
    action: str
    payload: Dict[str, Any]
    dependencies: List[str]
    priority: Priority
    timeout: float
    retry_count: int = 0
    max_retries: int = 3
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class WorkflowContext:
    workflow_id: str
    user_prompt: str
    input_files: List[str]
    current_state: WorkflowState
    tasks: List[WorkflowTask]
    results: Dict[str, Any]
    start_time: float
    metadata: Dict[str, Any]

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates the entire Text-to-CAD workflow
    Optimized for computational efficiency and parallel processing
    """
    
    def __init__(self, agent_id: str = "orchestrator", max_workers: int = 8):
        super().__init__(agent_id, max_workers)
        
        # Active workflows
        self.active_workflows = {}
        
        # Workflow templates for different scenarios
        self.workflow_templates = {
            'simple_structure': self._create_simple_workflow_template,
            'complex_infrastructure': self._create_complex_workflow_template,
            'retrofit_upgrade': self._create_retrofit_workflow_template,
            'analysis_only': self._create_analysis_workflow_template
        }
        
        # Performance monitoring
        self.workflow_stats = {
            'total_workflows': 0,
            'successful_workflows': 0,
            'failed_workflows': 0,
            'avg_processing_time': 0.0,
            'avg_elements_per_workflow': 0.0
        }
        
        # Task execution pool
        self.task_executor = asyncio.Queue(maxsize=100)
        
        logging.info(f"OrchestratorAgent initialized with {len(self.workflow_templates)} workflow templates")
    
    async def on_start(self):
        """Start the orchestrator with task execution loops"""
        
        # Start task execution loops
        for i in range(4):  # 4 parallel task executors
            asyncio.create_task(self._task_execution_loop(f"executor_{i}"))
        
        # Start workflow monitoring
        asyncio.create_task(self._workflow_monitoring_loop())
        
        logging.info("Orchestrator agent started with parallel task execution")
    
    async def handle_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle orchestration requests"""
        
        if message.payload.get("action") == "process_prompt":
            # Start new workflow
            user_prompt = message.payload.get("prompt", "")
            input_files = message.payload.get("files", [])
            
            workflow_id = f"workflow_{int(time.time() * 1000)}"
            
            # Create and start workflow
            workflow = await self._create_workflow(workflow_id, user_prompt, input_files)
            self.active_workflows[workflow_id] = workflow
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow(workflow))
            
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload={
                    "workflow_id": workflow_id,
                    "status": "started",
                    "estimated_duration": self._estimate_workflow_duration(workflow)
                },
                correlation_id=message.correlation_id
            )
        
        elif message.payload.get("action") == "get_workflow_status":
            workflow_id = message.payload.get("workflow_id")
            if workflow_id in self.active_workflows:
                workflow = self.active_workflows[workflow_id]
                return AgentMessage(
                    sender=self.agent_id,
                    receiver=message.sender,
                    message_type=MessageType.RESPONSE,
                    payload={
                        "workflow_id": workflow_id,
                        "status": workflow.current_state.value,
                        "progress": self._calculate_workflow_progress(workflow),
                        "results": workflow.results
                    },
                    correlation_id=message.correlation_id
                )
        
        return None
    
    async def _create_workflow(self, workflow_id: str, user_prompt: str, 
                             input_files: List[str]) -> WorkflowContext:
        """Create optimized workflow based on prompt analysis"""
        
        # Quick intent classification for workflow selection
        intent = self._classify_intent_fast(user_prompt)
        
        # Create workflow template
        if intent in self.workflow_templates:
            template_func = self.workflow_templates[intent]
            tasks = template_func(user_prompt, input_files)
        else:
            # Default to simple workflow
            tasks = self._create_simple_workflow_template(user_prompt, input_files)
        
        # Create workflow context
        workflow = WorkflowContext(
            workflow_id=workflow_id,
            user_prompt=user_prompt,
            input_files=input_files,
            current_state=WorkflowState.INITIALIZED,
            tasks=tasks,
            results={},
            start_time=time.time(),
            metadata={'intent': intent}
        )
        
        return workflow
    
    def _create_simple_workflow_template(self, user_prompt: str, input_files: List[str]) -> List[WorkflowTask]:
        """Create workflow template for simple structures"""
        
        tasks = [
            WorkflowTask(
                task_id="parse_prompt",
                agent_id="prompt_parser",
                action="parse_prompt",
                payload={"prompt": user_prompt},
                dependencies=[],
                priority=Priority.HIGH,
                timeout=30.0
            ),
            WorkflowTask(
                task_id="analyze_files",
                agent_id="file_analyzer",
                action="analyze_files",
                payload={"files": input_files},
                dependencies=[],
                priority=Priority.MEDIUM,
                timeout=60.0
            ),
            WorkflowTask(
                task_id="generate_ifc",
                agent_id="ifc_generator",
                action="generate_ifc",
                payload={},
                dependencies=["parse_prompt", "analyze_files"],
                priority=Priority.HIGH,
                timeout=120.0
            )
        ]
        
        return tasks
    
    def _create_complex_workflow_template(self, user_prompt: str, input_files: List[str]) -> List[WorkflowTask]:
        """Create workflow template for complex infrastructure"""
        
        tasks = [
            WorkflowTask(
                task_id="parse_prompt",
                agent_id="prompt_parser",
                action="parse_prompt",
                payload={"prompt": user_prompt},
                dependencies=[],
                priority=Priority.HIGH,
                timeout=45.0
            ),
            WorkflowTask(
                task_id="analyze_files",
                agent_id="file_analyzer",
                action="analyze_files",
                payload={"files": input_files},
                dependencies=[],
                priority=Priority.HIGH,
                timeout=180.0  # Longer timeout for complex files
            ),
            WorkflowTask(
                task_id="structural_analysis",
                agent_id="design_agent",
                action="analyze_structure",
                payload={},
                dependencies=["parse_prompt", "analyze_files"],
                priority=Priority.HIGH,
                timeout=240.0
            ),
            WorkflowTask(
                task_id="generate_ifc",
                agent_id="ifc_generator",
                action="generate_ifc",
                payload={},
                dependencies=["structural_analysis"],
                priority=Priority.HIGH,
                timeout=180.0
            ),
            WorkflowTask(
                task_id="validate_design",
                agent_id="validation_agent",
                action="validate_ifc",
                payload={},
                dependencies=["generate_ifc"],
                priority=Priority.MEDIUM,
                timeout=120.0
            )
        ]
        
        return tasks
    
    def _create_retrofit_workflow_template(self, user_prompt: str, input_files: List[str]) -> List[WorkflowTask]:
        """Create workflow template for retrofit projects"""
        
        tasks = [
            WorkflowTask(
                task_id="parse_prompt",
                agent_id="prompt_parser",
                action="parse_prompt",
                payload={"prompt": user_prompt},
                dependencies=[],
                priority=Priority.HIGH,
                timeout=30.0
            ),
            WorkflowTask(
                task_id="analyze_existing",
                agent_id="file_analyzer",
                action="analyze_existing_structure",
                payload={"files": input_files},
                dependencies=[],
                priority=Priority.HIGH,
                timeout=120.0
            ),
            WorkflowTask(
                task_id="assess_condition",
                agent_id="design_agent",
                action="assess_condition",
                payload={},
                dependencies=["parse_prompt", "analyze_existing"],
                priority=Priority.HIGH,
                timeout=180.0
            ),
            WorkflowTask(
                task_id="design_upgrade",
                agent_id="design_agent",
                action="design_upgrade",
                payload={},
                dependencies=["assess_condition"],
                priority=Priority.HIGH,
                timeout=240.0
            ),
            WorkflowTask(
                task_id="generate_ifc",
                agent_id="ifc_generator",
                action="generate_ifc",
                payload={},
                dependencies=["design_upgrade"],
                priority=Priority.HIGH,
                timeout=180.0
            )
        ]
        
        return tasks
    
    def _create_analysis_workflow_template(self, user_prompt: str, input_files: List[str]) -> List[WorkflowTask]:
        """Create workflow template for analysis-only tasks"""
        
        tasks = [
            WorkflowTask(
                task_id="parse_prompt",
                agent_id="prompt_parser",
                action="parse_prompt",
                payload={"prompt": user_prompt},
                dependencies=[],
                priority=Priority.HIGH,
                timeout=30.0
            ),
            WorkflowTask(
                task_id="analyze_files",
                agent_id="file_analyzer",
                action="analyze_files",
                payload={"files": input_files},
                dependencies=[],
                priority=Priority.HIGH,
                timeout=120.0
            ),
            WorkflowTask(
                task_id="perform_analysis",
                agent_id="design_agent",
                action="perform_analysis",
                payload={},
                dependencies=["parse_prompt", "analyze_files"],
                priority=Priority.HIGH,
                timeout=180.0
            )
        ]
        
        return tasks
    
    async def _execute_workflow(self, workflow: WorkflowContext):
        """Execute workflow with optimized task scheduling"""
        
        workflow.current_state = WorkflowState.PARSING_PROMPT
        
        try:
            # Create dependency graph
            dependency_graph = self._build_dependency_graph(workflow.tasks)
            
            # Execute tasks in optimal order
            await self._execute_tasks_optimally(workflow, dependency_graph)
            
            # Mark workflow as completed
            workflow.current_state = WorkflowState.COMPLETED
            
            # Update statistics
            self._update_workflow_stats(workflow, success=True)
            
            logging.info(f"Workflow {workflow.workflow_id} completed successfully")
            
        except Exception as e:
            workflow.current_state = WorkflowState.FAILED
            workflow.results['error'] = str(e)
            
            # Update statistics
            self._update_workflow_stats(workflow, success=False)
            
            logging.error(f"Workflow {workflow.workflow_id} failed: {e}")
    
    def _build_dependency_graph(self, tasks: List[WorkflowTask]) -> Dict[str, List[str]]:
        """Build dependency graph for optimal task scheduling"""
        
        graph = {}
        for task in tasks:
            graph[task.task_id] = task.dependencies
        
        return graph
    
    async def _execute_tasks_optimally(self, workflow: WorkflowContext, dependency_graph: Dict[str, List[str]]):
        """Execute tasks in optimal order with parallel processing"""
        
        completed_tasks = set()
        running_tasks = {}
        
        while len(completed_tasks) < len(workflow.tasks):
            # Find ready tasks (dependencies completed)
            ready_tasks = []
            for task in workflow.tasks:
                if (task.task_id not in completed_tasks and 
                    task.task_id not in running_tasks and
                    all(dep in completed_tasks for dep in task.dependencies)):
                    ready_tasks.append(task)
            
            # Start ready tasks
            for task in ready_tasks:
                task_future = asyncio.create_task(self._execute_single_task(workflow, task))
                running_tasks[task.task_id] = task_future
            
            # Wait for any task to complete
            if running_tasks:
                done, pending = await asyncio.wait(
                    running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for future in done:
                    task_id = None
                    for tid, fut in running_tasks.items():
                        if fut == future:
                            task_id = tid
                            break
                    
                    if task_id:
                        completed_tasks.add(task_id)
                        del running_tasks[task_id]
                        
                        # Get result
                        try:
                            result = await future
                            workflow.results[task_id] = result
                        except Exception as e:
                            logging.error(f"Task {task_id} failed: {e}")
                            raise
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.01)
    
    async def _execute_single_task(self, workflow: WorkflowContext, task: WorkflowTask) -> Dict[str, Any]:
        """Execute a single task with error handling and retries"""
        
        task.start_time = time.time()
        
        for attempt in range(task.max_retries + 1):
            try:
                # Prepare task payload
                payload = task.payload.copy()
                
                # Add context from previous tasks
                if task.dependencies:
                    payload['context'] = {}
                    for dep in task.dependencies:
                        if dep in workflow.results:
                            payload['context'][dep] = workflow.results[dep]
                
                # Send message to agent
                message = AgentMessage(
                    sender=self.agent_id,
                    receiver=task.agent_id,
                    message_type=MessageType.REQUEST,
                    payload={
                        "action": task.action,
                        **payload
                    },
                    priority=task.priority,
                    correlation_id=f"{workflow.workflow_id}_{task.task_id}"
                )
                
                # Wait for response with timeout
                response = await self._send_and_wait_for_response(message, task.timeout)
                
                task.completion_time = time.time()
                task.result = response.payload
                
                return response.payload
                
            except Exception as e:
                task.retry_count += 1
                task.error = str(e)
                
                if attempt < task.max_retries:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
                    logging.warning(f"Task {task.task_id} failed, retrying ({attempt + 1}/{task.max_retries})")
                else:
                    logging.error(f"Task {task.task_id} failed after {task.max_retries} retries: {e}")
                    raise
    
    async def _send_and_wait_for_response(self, message: AgentMessage, timeout: float) -> AgentMessage:
        """Send message and wait for response with timeout"""
        
        # Send message
        await self.send_message(message)
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check for responses (simplified - would use proper response handling)
            await asyncio.sleep(0.1)
            
            # For now, simulate response (would implement proper response handling)
            return AgentMessage(
                sender=message.receiver,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload={"status": "success", "result": {}},
                correlation_id=message.correlation_id
            )
        
        raise TimeoutError(f"Task timed out after {timeout} seconds")
    
    async def _task_execution_loop(self, executor_id: str):
        """Task execution loop for parallel processing"""
        
        while self.is_running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_executor.get(), timeout=1.0)
                
                # Execute task
                await task
                
                # Mark task as done
                self.task_executor.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logging.error(f"Task executor {executor_id} error: {e}")
                await asyncio.sleep(0.1)
    
    async def _workflow_monitoring_loop(self):
        """Monitor workflow performance and health"""
        
        while self.is_running:
            try:
                # Check for stuck workflows
                current_time = time.time()
                for workflow_id, workflow in self.active_workflows.items():
                    if current_time - workflow.start_time > 300:  # 5 minutes
                        logging.warning(f"Workflow {workflow_id} is taking longer than expected")
                
                # Clean up completed workflows
                completed_workflows = [
                    wid for wid, workflow in self.active_workflows.items()
                    if workflow.current_state in [WorkflowState.COMPLETED, WorkflowState.FAILED]
                ]
                
                for wid in completed_workflows:
                    if current_time - self.active_workflows[wid].start_time > 600:  # 10 minutes
                        del self.active_workflows[wid]
                
                # Sleep before next check
                await asyncio.sleep(10)
                
            except Exception as e:
                logging.error(f"Workflow monitoring error: {e}")
                await asyncio.sleep(5)
    
    def _classify_intent_fast(self, prompt: str) -> str:
        """Fast intent classification for workflow selection"""
        
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['retrofit', 'upgrade', 'existing', 'modify']):
            return 'retrofit_upgrade'
        elif any(word in prompt_lower for word in ['analyze', 'check', 'verify', 'validate']):
            return 'analysis_only'
        elif any(word in prompt_lower for word in ['complex', 'system', 'integrated', 'multiple']):
            return 'complex_infrastructure'
        else:
            return 'simple_structure'
    
    def _estimate_workflow_duration(self, workflow: WorkflowContext) -> float:
        """Estimate workflow duration based on tasks"""
        
        # Calculate critical path
        total_duration = 0
        for task in workflow.tasks:
            total_duration += task.timeout
        
        # Apply parallelism factor
        parallelism_factor = 0.6  # Assuming 60% parallelism
        estimated_duration = total_duration * parallelism_factor
        
        return estimated_duration
    
    def _calculate_workflow_progress(self, workflow: WorkflowContext) -> float:
        """Calculate workflow progress percentage"""
        
        if not workflow.tasks:
            return 0.0
        
        completed_tasks = sum(1 for task in workflow.tasks if task.completion_time is not None)
        return (completed_tasks / len(workflow.tasks)) * 100
    
    def _update_workflow_stats(self, workflow: WorkflowContext, success: bool):
        """Update workflow statistics"""
        
        self.workflow_stats['total_workflows'] += 1
        
        if success:
            self.workflow_stats['successful_workflows'] += 1
        else:
            self.workflow_stats['failed_workflows'] += 1
        
        # Update average processing time
        processing_time = time.time() - workflow.start_time
        total_workflows = self.workflow_stats['total_workflows']
        
        self.workflow_stats['avg_processing_time'] = (
            (self.workflow_stats['avg_processing_time'] * (total_workflows - 1) + processing_time) / total_workflows
        )
    
    def get_system_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        
        return {
            'orchestrator_stats': self.workflow_stats,
            'active_workflows': len(self.active_workflows),
            'system_metrics': message_broker.get_system_metrics(),
            'performance_metrics': self.get_performance_metrics()
        }
    
    async def handle_response(self, message: AgentMessage):
        """Handle response messages"""
        logging.info(f"Orchestrator received response: {message.payload}")
    
    async def handle_notification(self, message: AgentMessage):
        """Handle notification messages"""
        logging.info(f"Orchestrator received notification: {message.payload}")
    
    async def handle_error(self, message: AgentMessage):
        """Handle error messages"""
        logging.error(f"Orchestrator received error: {message.payload}")

# Singleton pattern
_orchestrator_instance = None

def get_orchestrator_agent() -> OrchestratorAgent:
    """Get singleton instance of orchestrator agent"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = OrchestratorAgent()
    return _orchestrator_instance