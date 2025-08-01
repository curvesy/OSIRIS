#!/usr/bin/env python3
"""
Quick System Status Check

This script checks which AURA Intelligence services are running
and identifies what needs to be started or connected.
"""

import asyncio
import subprocess
import socket
from typing import Dict, Tuple
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def check_port(host: str, port: int) -> bool:
    """Check if a port is open."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0


def check_docker_service(service_name: str) -> Tuple[bool, str]:
    """Check if a Docker service is running."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={service_name}", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip()
        if output and "Up" in output:
            return True, output
        else:
            return False, "Not running"
    except subprocess.CalledProcessError:
        return False, "Docker command failed"
    except FileNotFoundError:
        return False, "Docker not installed"


async def check_services() -> Dict[str, Dict[str, any]]:
    """Check all AURA services."""
    services = {
        "Neo4j": {
            "port": 7687,
            "docker_name": "aura-neo4j",
            "ui_port": 7474,
            "ui_url": "http://localhost:7474"
        },
        "Redis": {
            "port": 6379,
            "docker_name": "aura-redis"
        },
        "Kafka": {
            "port": 9092,
            "docker_name": "aura-kafka"
        },
        "Temporal": {
            "port": 7233,
            "docker_name": "aura-temporal",
            "ui_port": 8088,
            "ui_url": "http://localhost:8088"
        },
        "Prometheus": {
            "port": 9090,
            "docker_name": "aura-prometheus",
            "ui_url": "http://localhost:9090"
        },
        "Grafana": {
            "port": 3000,
            "docker_name": "aura-grafana",
            "ui_url": "http://localhost:3000"
        },
        "Jaeger": {
            "port": 16686,
            "docker_name": "aura-jaeger",
            "ui_url": "http://localhost:16686"
        }
    }
    
    results = {}
    
    for service_name, config in services.items():
        # Check port
        port_open = check_port("localhost", config["port"])
        
        # Check Docker
        docker_running, docker_status = check_docker_service(config["docker_name"])
        
        # Check UI port if exists
        ui_accessible = False
        if "ui_port" in config:
            ui_accessible = check_port("localhost", config["ui_port"])
        
        results[service_name] = {
            "port_open": port_open,
            "docker_running": docker_running,
            "docker_status": docker_status,
            "ui_accessible": ui_accessible,
            "ui_url": config.get("ui_url", "")
        }
    
    return results


def print_status_table(results: Dict[str, Dict[str, any]]):
    """Print a nice status table."""
    print("\n" + "="*80)
    print("AURA INTELLIGENCE - SYSTEM STATUS")
    print("="*80)
    print(f"\n{'Service':<15} {'Port':<10} {'Docker':<20} {'Status':<30}")
    print("-"*75)
    
    all_good = True
    
    for service, status in results.items():
        port_status = "✅ Open" if status["port_open"] else "❌ Closed"
        docker_status = "✅ Running" if status["docker_running"] else "❌ Stopped"
        
        if not status["port_open"] or not status["docker_running"]:
            all_good = False
        
        print(f"{service:<15} {port_status:<10} {docker_status:<20} {status['docker_status']:<30}")
        
        if status["ui_url"] and status.get("ui_accessible"):
            print(f"{'':>15} UI: {status['ui_url']}")
    
    print("\n" + "="*80)
    
    if all_good:
        print("✅ All services are running!")
    else:
        print("❌ Some services need attention")
        print("\nTo start all services:")
        print("  cd deployments/staging")
        print("  ./start.sh")
        print("\nTo start specific services:")
        print("  docker-compose up -d neo4j redis kafka temporal prometheus grafana")
    
    return all_good


def check_integration_files():
    """Check if key integration files exist."""
    print("\n" + "="*80)
    print("KEY INTEGRATION FILES")
    print("="*80)
    
    files_to_check = [
        ("GPU Allocation Workflow", "core/src/aura_intelligence/workflows/gpu_allocation.py"),
        ("LNN Council Agent", "core/src/aura_intelligence/agents/council/lnn_council.py"),
        ("Context Integration", "core/src/aura_intelligence/neural/context_integration.py"),
        ("Neo4j Adapter", "core/src/aura_intelligence/adapters/neo4j_adapter.py"),
        ("Redis Adapter", "core/src/aura_intelligence/adapters/redis_adapter.py"),
        ("Kafka Producer", "core/src/aura_intelligence/events/producers.py"),
        ("Docker Compose", "deployments/staging/docker-compose.yml"),
        ("End-to-End Test", "core/test_end_to_end_gpu_allocation.py")
    ]
    
    all_exist = True
    
    for name, path in files_to_check:
        exists = os.path.exists(path)
        status = "✅ Found" if exists else "❌ Missing"
        print(f"{name:<30} {status:<10} {path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_missing_connections():
    """Identify missing connections in the system."""
    print("\n" + "="*80)
    print("INTEGRATION GAPS")
    print("="*80)
    
    gaps = [
        {
            "gap": "LNN Agent → Real Inference",
            "current": "Council votes are mocked in gather_council_votes()",
            "needed": "Wire up actual LNN agents to process CouncilTask",
            "file": "workflows/gpu_allocation.py:236"
        },
        {
            "gap": "Consensus → Real Protocol",
            "current": "Simple majority voting",
            "needed": "Integrate actual Raft/Byzantine consensus",
            "file": "workflows/gpu_allocation.py:380"
        },
        {
            "gap": "Neo4j → Decision Storage",
            "current": "emit_allocation_event() doesn't store in Neo4j",
            "needed": "Add Neo4j adapter call to store decisions",
            "file": "workflows/gpu_allocation.py:340"
        },
        {
            "gap": "Kafka → Event Verification",
            "current": "Events published but not consumed/verified",
            "needed": "Add Kafka consumer to verify event flow",
            "file": "test_end_to_end_gpu_allocation.py:285"
        },
        {
            "gap": "Metrics → Prometheus",
            "current": "No metrics exported from workflow",
            "needed": "Add OpenTelemetry metrics to workflow steps",
            "file": "workflows/gpu_allocation.py"
        }
    ]
    
    for i, gap in enumerate(gaps, 1):
        print(f"\n{i}. {gap['gap']}")
        print(f"   Current: {gap['current']}")
        print(f"   Needed:  {gap['needed']}")
        print(f"   File:    {gap['file']}")


async def main():
    """Main entry point."""
    # Check services
    results = await check_services()
    all_services_running = print_status_table(results)
    
    # Check files
    all_files_exist = check_integration_files()
    
    # Check gaps
    check_missing_connections()
    
    # Final recommendation
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if not all_services_running:
        print("\n1. Start missing services first:")
        print("   cd deployments/staging && ./start.sh")
    
    print("\n2. Run the end-to-end test:")
    print("   python core/test_end_to_end_gpu_allocation.py")
    
    print("\n3. Fix integration gaps in priority order:")
    print("   - Wire up real LNN agents")
    print("   - Store decisions in Neo4j")
    print("   - Add metrics export")
    
    print("\n4. Once connected, only then consider:")
    print("   - Performance optimization")
    print("   - Advanced observability (Pixie)")
    print("   - GitOps deployment (ArgoCD)")


if __name__ == "__main__":
    asyncio.run(main())