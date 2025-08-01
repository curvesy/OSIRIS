"""
Temporal Workflows for Liquid Neural Networks.

This module provides durable, fault-tolerant workflows for LNN training,
inference, and adaptation using Temporal's workflow-as-code approach.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import timedelta
import torch
import numpy as np
from temporalio import workflow, activity
from temporalio.common import RetryPolicy
import structlog

from .lnn import LiquidNeuralNetwork, EdgeLNN, LNNConfig, benchmark_lnn
from .lnn_consensus import (
    DistributedLNN,
    LNNConsensusOrchestrator,
    ConsensusLNNConfig,
    create_lnn_cluster
)
from ..agents.temporal import WorkflowBase

logger = structlog.get_logger()


# Data classes for workflow inputs/outputs
@dataclass
class LNNTrainingInput:
    """Input parameters for LNN training workflow."""
    config: Dict[str, Any]  # LNNConfig as dict
    dataset_path: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    checkpoint_interval: int = 10
    enable_adaptation: bool = True


@dataclass
class LNNTrainingOutput:
    """Output from LNN training workflow."""
    model_id: str
    final_loss: float
    final_accuracy: float
    training_time_seconds: float
    checkpoint_paths: List[str]
    metrics_history: Dict[str, List[float]]


@dataclass
class LNNInferenceInput:
    """Input for LNN inference workflow."""
    model_id: str
    input_data: List[float]  # Serialized tensor
    input_shape: List[int]
    use_consensus: bool = False
    consensus_nodes: int = 3
    enable_adaptation: bool = False


@dataclass
class LNNInferenceOutput:
    """Output from LNN inference workflow."""
    prediction: List[float]
    confidence: float
    inference_time_ms: float
    consensus_metadata: Optional[Dict[str, Any]] = None
    adapted: bool = False


# Activities
class LNNActivities:
    """Activities for LNN workflows."""
    
    def __init__(self):
        self.models: Dict[str, LiquidNeuralNetwork] = {}
        self.orchestrators: Dict[str, LNNConsensusOrchestrator] = {}
    
    @activity.defn(name="load_dataset")
    async def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load and preprocess dataset for training."""
        # In production, this would load from S3, database, etc.
        # For now, generate synthetic data
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Synthetic data generation
        num_samples = 1000
        input_size = 128
        output_size = 10
        
        X = torch.randn(num_samples, input_size)
        y = torch.randint(0, output_size, (num_samples,))
        
        return {
            "X_train": X[:800].tolist(),
            "y_train": y[:800].tolist(),
            "X_val": X[800:].tolist(),
            "y_val": y[800:].tolist(),
            "input_size": input_size,
            "output_size": output_size
        }
    
    @activity.defn(name="create_lnn_model")
    async def create_lnn_model(self, config_dict: Dict[str, Any]) -> str:
        """Create and initialize LNN model."""
        config = LNNConfig(**config_dict)
        model = LiquidNeuralNetwork(config)
        
        # Generate unique model ID
        model_id = f"lnn-{np.random.randint(1000000)}"
        self.models[model_id] = model
        
        logger.info(f"Created LNN model: {model_id}")
        return model_id
    
    @activity.defn(name="train_epoch")
    async def train_epoch(
        self,
        model_id: str,
        epoch: int,
        X_train: List[List[float]],
        y_train: List[int],
        batch_size: int,
        learning_rate: float
    ) -> Dict[str, float]:
        """Train model for one epoch."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Convert data
        X = torch.tensor(X_train)
        y = torch.tensor(y_train)
        
        # Simple training loop (in production, use DataLoader)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Mini-batch training
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / (len(X) / batch_size)
        accuracy = correct / total
        
        logger.info(f"Epoch {epoch}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    @activity.defn(name="validate_model")
    async def validate_model(
        self,
        model_id: str,
        X_val: List[List[float]],
        y_val: List[int]
    ) -> Dict[str, float]:
        """Validate model performance."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        X = torch.tensor(X_val)
        y = torch.tensor(y_val)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(outputs, y).item()
            
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(y).sum().item() / len(y)
        
        return {
            "val_loss": loss,
            "val_accuracy": accuracy
        }
    
    @activity.defn(name="save_checkpoint")
    async def save_checkpoint(
        self,
        model_id: str,
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """Save model checkpoint."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # In production, save to S3 or persistent storage
        checkpoint_path = f"/tmp/{model_id}_epoch_{epoch}.pt"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': model.config
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    @activity.defn(name="create_consensus_cluster")
    async def create_consensus_cluster(
        self,
        model_id: str,
        num_nodes: int,
        config_dict: Dict[str, Any]
    ) -> str:
        """Create consensus cluster for distributed inference."""
        lnn_config = LNNConfig(**config_dict)
        
        orchestrator = await create_lnn_cluster(
            num_nodes=num_nodes,
            input_size=lnn_config.input_size,
            hidden_size=lnn_config.hidden_size,
            output_size=lnn_config.output_size,
            edge_deployment=False
        )
        
        cluster_id = f"cluster-{model_id}"
        self.orchestrators[cluster_id] = orchestrator
        
        logger.info(f"Created consensus cluster: {cluster_id}")
        return cluster_id
    
    @activity.defn(name="run_inference")
    async def run_inference(
        self,
        model_id: str,
        input_data: List[float],
        input_shape: List[int],
        use_consensus: bool,
        cluster_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run inference on input data."""
        # Reshape input
        X = torch.tensor(input_data).reshape(*input_shape)
        
        if use_consensus and cluster_id:
            # Distributed inference
            orchestrator = self.orchestrators.get(cluster_id)
            if not orchestrator:
                raise ValueError(f"Cluster {cluster_id} not found")
            
            output, metadata = await orchestrator.distributed_inference(
                X,
                f"inference-{np.random.randint(1000000)}"
            )
            
            return {
                "output": output.tolist(),
                "consensus_metadata": metadata
            }
        else:
            # Single model inference
            model = self.models.get(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            model.eval()
            with torch.no_grad():
                output = model(X)
            
            return {
                "output": output.tolist(),
                "consensus_metadata": None
            }
    
    @activity.defn(name="adapt_model")
    async def adapt_model(
        self,
        model_id: str,
        feedback: List[float]
    ) -> bool:
        """Adapt model based on feedback."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        feedback_tensor = torch.tensor(feedback)
        model.adapt(feedback_tensor)
        
        logger.info(f"Adapted model {model_id}")
        return True
    
    @activity.defn(name="benchmark_model")
    async def benchmark_model(
        self,
        model_id: str,
        input_shape: List[int],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark model performance."""
        model = self.models.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        return benchmark_lnn(model, tuple(input_shape), num_iterations)


# Workflows
@workflow.defn(name="LNNTrainingWorkflow")
class LNNTrainingWorkflow(WorkflowBase):
    """Workflow for training Liquid Neural Networks."""
    
    @workflow.run
    async def run(self, input: LNNTrainingInput) -> LNNTrainingOutput:
        """Execute LNN training workflow."""
        start_time = workflow.now()
        
        # Load dataset
        dataset = await workflow.execute_activity(
            "load_dataset",
            input.dataset_path,
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Create model
        model_id = await workflow.execute_activity(
            "create_lnn_model",
            input.config,
            start_to_close_timeout=timedelta(minutes=1)
        )
        
        # Training loop
        metrics_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        checkpoint_paths = []
        
        for epoch in range(input.epochs):
            # Train epoch
            train_metrics = await workflow.execute_activity(
                "train_epoch",
                model_id,
                epoch,
                dataset["X_train"],
                dataset["y_train"],
                input.batch_size,
                input.learning_rate,
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=RetryPolicy(maximum_attempts=2)
            )
            
            # Validate
            val_metrics = await workflow.execute_activity(
                "validate_model",
                model_id,
                dataset["X_val"],
                dataset["y_val"],
                start_to_close_timeout=timedelta(minutes=5)
            )
            
            # Update history
            metrics_history["loss"].append(train_metrics["loss"])
            metrics_history["accuracy"].append(train_metrics["accuracy"])
            metrics_history["val_loss"].append(val_metrics["val_loss"])
            metrics_history["val_accuracy"].append(val_metrics["val_accuracy"])
            
            # Save checkpoint
            if epoch % input.checkpoint_interval == 0:
                checkpoint_path = await workflow.execute_activity(
                    "save_checkpoint",
                    model_id,
                    epoch,
                    {**train_metrics, **val_metrics},
                    start_to_close_timeout=timedelta(minutes=2)
                )
                checkpoint_paths.append(checkpoint_path)
            
            # Log progress
            await self.emit_progress({
                "epoch": epoch,
                "total_epochs": input.epochs,
                "train_loss": train_metrics["loss"],
                "val_accuracy": val_metrics["val_accuracy"]
            })
        
        # Final checkpoint
        final_checkpoint = await workflow.execute_activity(
            "save_checkpoint",
            model_id,
            input.epochs - 1,
            {
                "loss": metrics_history["loss"][-1],
                "accuracy": metrics_history["accuracy"][-1],
                "val_loss": metrics_history["val_loss"][-1],
                "val_accuracy": metrics_history["val_accuracy"][-1]
            },
            start_to_close_timeout=timedelta(minutes=2)
        )
        checkpoint_paths.append(final_checkpoint)
        
        # Calculate training time
        training_time = (workflow.now() - start_time).total_seconds()
        
        return LNNTrainingOutput(
            model_id=model_id,
            final_loss=metrics_history["loss"][-1],
            final_accuracy=metrics_history["val_accuracy"][-1],
            training_time_seconds=training_time,
            checkpoint_paths=checkpoint_paths,
            metrics_history=metrics_history
        )


@workflow.defn(name="LNNInferenceWorkflow")
class LNNInferenceWorkflow(WorkflowBase):
    """Workflow for LNN inference with optional consensus."""
    
    @workflow.run
    async def run(self, input: LNNInferenceInput) -> LNNInferenceOutput:
        """Execute LNN inference workflow."""
        start_time = workflow.now()
        
        # Create consensus cluster if needed
        cluster_id = None
        if input.use_consensus:
            # Get model config (in production, load from storage)
            config = {
                "input_size": input.input_shape[-1],
                "hidden_size": 256,
                "output_size": 10
            }
            
            cluster_id = await workflow.execute_activity(
                "create_consensus_cluster",
                input.model_id,
                input.consensus_nodes,
                config,
                start_to_close_timeout=timedelta(minutes=2)
            )
        
        # Run inference
        inference_result = await workflow.execute_activity(
            "run_inference",
            input.model_id,
            input.input_data,
            input.input_shape,
            input.use_consensus,
            cluster_id,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1)
            )
        )
        
        # Calculate confidence
        output = torch.tensor(inference_result["output"])
        if output.dim() > 1:
            probs = torch.softmax(output, dim=-1)
            confidence = float(probs.max().item())
        else:
            confidence = float(torch.sigmoid(output).mean().item())
        
        # Adapt if enabled
        adapted = False
        if input.enable_adaptation:
            # Simple feedback mechanism (in production, use actual feedback)
            feedback = torch.randn_like(output) * 0.1
            adapted = await workflow.execute_activity(
                "adapt_model",
                input.model_id,
                feedback.tolist(),
                start_to_close_timeout=timedelta(seconds=10)
            )
        
        # Calculate inference time
        inference_time_ms = (workflow.now() - start_time).total_seconds() * 1000
        
        return LNNInferenceOutput(
            prediction=inference_result["output"],
            confidence=confidence,
            inference_time_ms=inference_time_ms,
            consensus_metadata=inference_result.get("consensus_metadata"),
            adapted=adapted
        )


@workflow.defn(name="LNNBenchmarkWorkflow")
class LNNBenchmarkWorkflow(WorkflowBase):
    """Workflow for benchmarking LNN models."""
    
    @workflow.run
    async def run(
        self,
        model_id: str,
        input_shape: List[int],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Execute LNN benchmark workflow."""
        # Run benchmark
        benchmark_results = await workflow.execute_activity(
            "benchmark_model",
            model_id,
            input_shape,
            num_iterations,
            start_to_close_timeout=timedelta(minutes=5)
        )
        
        # Emit results
        await self.emit_progress({
            "status": "benchmark_complete",
            "results": benchmark_results
        })
        
        return benchmark_results


@workflow.defn(name="LNNEdgeDeploymentWorkflow")
class LNNEdgeDeploymentWorkflow(WorkflowBase):
    """Workflow for deploying LNN to edge devices."""
    
    @workflow.run
    async def run(
        self,
        model_id: str,
        target_devices: List[str],
        optimization_level: str = "high"
    ) -> Dict[str, Any]:
        """Deploy LNN to edge devices."""
        deployment_results = {}
        
        for device in target_devices:
            # Convert to edge model
            edge_config = await workflow.execute_activity(
                "convert_to_edge",
                model_id,
                optimization_level,
                start_to_close_timeout=timedelta(minutes=2)
            )
            
            # Deploy to device
            deploy_result = await workflow.execute_activity(
                "deploy_to_device",
                edge_config,
                device,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=RetryPolicy(maximum_attempts=3)
            )
            
            deployment_results[device] = deploy_result
            
            # Emit progress
            await self.emit_progress({
                "device": device,
                "status": "deployed",
                "metrics": deploy_result
            })
        
        return {
            "model_id": model_id,
            "deployments": deployment_results,
            "timestamp": workflow.now().isoformat()
        }