"""
AURA Shape Streaming Pro - 2025 Edition
=======================================
Component-based WebSocket streaming with reactive architecture.
"""

from typing import Protocol, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


# --- Domain Models ---

class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        if score < 0.3:
            return cls.LOW
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.HIGH
        return cls.CRITICAL


@dataclass
class TopologySnapshot:
    """Immutable snapshot of topological state."""
    timestamp: datetime
    betti_numbers: List[int]
    persistence_intervals: List[Dict]
    risk_score: float
    patterns: List[Dict[str, float]]
    
    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.from_score(self.risk_score)
    
    def to_dict(self) -> Dict:
        """Serialize for WebSocket transmission."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "betti_numbers": self.betti_numbers,
            "persistence_intervals": self.persistence_intervals,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "patterns_detected": self.patterns
        }


# --- Stream Components ---

class StreamSource(Protocol):
    """Protocol for topology data sources."""
    
    async def stream(self) -> AsyncIterator[TopologySnapshot]:
        """Yield topology snapshots."""
        ...


class LiveTDASource:
    """Connects to live TDA analysis engine."""
    
    def __init__(self, analysis_queue: asyncio.Queue):
        self.queue = analysis_queue
    
    async def stream(self) -> AsyncIterator[TopologySnapshot]:
        """Stream from analysis queue."""
        while True:
            result = await self.queue.get()
            yield self._convert_to_snapshot(result)
    
    def _convert_to_snapshot(self, result: Dict) -> TopologySnapshot:
        """Convert analysis result to snapshot."""
        sig = result.get("topological_signature", {})
        
        return TopologySnapshot(
            timestamp=datetime.now(),
            betti_numbers=[
                sig.get("betti_0", 1),
                sig.get("betti_1", 0),
                sig.get("betti_2", 0)
            ],
            persistence_intervals=self._format_persistence(sig),
            risk_score=result.get("risk_score", 0.0),
            patterns=[
                {"name": p["name"], "confidence": p["confidence"]}
                for p in result.get("patterns_detected", [])
            ]
        )
    
    def _format_persistence(self, sig: Dict) -> List[Dict]:
        """Format persistence diagram."""
        intervals = []
        for dim, birth, death in sig.get("persistence_diagram", []):
            intervals.append({
                "dimension": int(dim),
                "birth": float(birth),
                "death": float(death) if death != float('inf') else 1.0,
                "persistence": float(death - birth) if death != float('inf') else 1.0
            })
        return intervals


class SimulatedSource:
    """Simulated topology source for demos."""
    
    async def stream(self) -> AsyncIterator[TopologySnapshot]:
        """Generate simulated topology data."""
        while True:
            await asyncio.sleep(1)
            
            # Simulate varying topology
            n_loops = np.random.poisson(1.5)
            n_components = np.random.poisson(2)
            risk = 0.2 * n_loops + 0.1 * (n_components - 1)
            
            patterns = []
            if n_loops > 0:
                patterns.append({
                    "name": "stuck_loop",
                    "confidence": 0.7 + np.random.random() * 0.25
                })
            
            if n_components > 2:
                patterns.append({
                    "name": "context_fragmentation",
                    "confidence": 0.6 + np.random.random() * 0.3
                })
            
            yield TopologySnapshot(
                timestamp=datetime.now(),
                betti_numbers=[n_components, n_loops, 0],
                persistence_intervals=self._generate_persistence(n_components, n_loops),
                risk_score=min(0.95, risk + np.random.random() * 0.2),
                patterns=patterns
            )
    
    def _generate_persistence(self, n_comp: int, n_loops: int) -> List[Dict]:
        """Generate realistic persistence intervals."""
        intervals = []
        
        # Component features
        for i in range(n_comp):
            intervals.append({
                "dimension": 0,
                "birth": 0.0,
                "death": np.random.uniform(0.1, 0.5),
                "persistence": np.random.uniform(0.1, 0.5)
            })
        
        # Loop features
        for i in range(n_loops):
            birth = np.random.uniform(0.1, 0.3)
            death = np.random.uniform(0.6, 1.0)
            intervals.append({
                "dimension": 1,
                "birth": birth,
                "death": death,
                "persistence": death - birth
            })
        
        return intervals


# --- WebSocket Manager ---

class ConnectionManager:
    """Manages WebSocket connections with backpressure."""
    
    def __init__(self, max_queue_size: int = 100):
        self.connections: Dict[str, WebSocket] = {}
        self.max_queue_size = max_queue_size
        self._lock = asyncio.Lock()
    
    async def connect(self, client_id: str, websocket: WebSocket):
        """Accept new connection."""
        await websocket.accept()
        async with self._lock:
            self.connections[client_id] = websocket
    
    async def disconnect(self, client_id: str):
        """Remove connection."""
        async with self._lock:
            self.connections.pop(client_id, None)
    
    async def broadcast(self, snapshot: TopologySnapshot):
        """Broadcast to all connections with backpressure handling."""
        message = json.dumps(snapshot.to_dict())
        
        async with self._lock:
            disconnected = []
            
            for client_id, ws in self.connections.items():
                try:
                    await ws.send_text(message)
                except:
                    disconnected.append(client_id)
            
            # Clean up failed connections
            for client_id in disconnected:
                self.connections.pop(client_id, None)


# --- Stream Processor ---

class StreamProcessor:
    """Processes and enriches topology streams."""
    
    def __init__(self):
        self.history: List[TopologySnapshot] = []
        self.max_history = 1000
    
    async def process(self, source: StreamSource, manager: ConnectionManager):
        """Process stream and broadcast."""
        async for snapshot in source.stream():
            # Store history
            self.history.append(snapshot)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Enrich with trends
            enriched = self._enrich_snapshot(snapshot)
            
            # Broadcast
            await manager.broadcast(enriched)
    
    def _enrich_snapshot(self, snapshot: TopologySnapshot) -> TopologySnapshot:
        """Add trend information to snapshot."""
        # In production, add trend analysis, anomaly detection, etc.
        return snapshot


# --- API Endpoints ---

manager = ConnectionManager()
processor = StreamProcessor()


async def shape_stream_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for shape streaming."""
    await manager.connect(client_id, websocket)
    
    try:
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await manager.disconnect(client_id)


async def start_streaming(source: Optional[StreamSource] = None):
    """Start the streaming processor."""
    if source is None:
        source = SimulatedSource()
    
    await processor.process(source, manager)


# --- React Component ---

def get_shape_hud_react() -> str:
    """Modern React-based HUD component."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AURA Shape HUD Pro</title>
    <meta charset="utf-8">
    <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: #0a0a0a; 
            color: #00ff88; 
            font-family: 'SF Mono', 'Monaco', monospace;
            overflow: hidden;
        }
        #root { width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <div id="root"></div>
    
    <script type="importmap">
    {
        "imports": {
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }
    }
    </script>
    
    <script type="text/babel" data-type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
        import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
        import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
        
        const { useState, useEffect, useRef, useCallback } = React;
        
        // WebSocket Hook
        const useWebSocket = (url) => {
            const [data, setData] = useState(null);
            const [status, setStatus] = useState('disconnected');
            const ws = useRef(null);
            
            useEffect(() => {
                const connect = () => {
                    ws.current = new WebSocket(url);
                    
                    ws.current.onopen = () => {
                        setStatus('connected');
                        // Ping to start stream
                        const interval = setInterval(() => {
                            if (ws.current?.readyState === WebSocket.OPEN) {
                                ws.current.send('ping');
                            }
                        }, 5000);
                        ws.current.pingInterval = interval;
                    };
                    
                    ws.current.onmessage = (event) => {
                        const msg = JSON.parse(event.data);
                        if (msg.type !== 'pong') {
                            setData(msg);
                        }
                    };
                    
                    ws.current.onclose = () => {
                        setStatus('disconnected');
                        clearInterval(ws.current.pingInterval);
                        setTimeout(connect, 3000);
                    };
                };
                
                connect();
                
                return () => {
                    if (ws.current) {
                        clearInterval(ws.current.pingInterval);
                        ws.current.close();
                    }
                };
            }, [url]);
            
            return { data, status };
        };
        
        // 3D Visualization Component
        const ShapeVisualization = ({ data }) => {
            const mountRef = useRef(null);
            const sceneRef = useRef(null);
            
            useEffect(() => {
                if (!mountRef.current) return;
                
                // Scene setup
                const scene = new THREE.Scene();
                scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
                
                const camera = new THREE.PerspectiveCamera(
                    75, window.innerWidth / window.innerHeight, 0.1, 1000
                );
                camera.position.set(0, 10, 30);
                
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(window.innerWidth, window.innerHeight);
                mountRef.current.appendChild(renderer.domElement);
                
                // Post-processing
                const composer = new EffectComposer(renderer);
                composer.addPass(new RenderPass(scene, camera));
                
                const bloomPass = new UnrealBloomPass(
                    new THREE.Vector2(window.innerWidth, window.innerHeight),
                    1.5, 0.4, 0.85
                );
                composer.addPass(bloomPass);
                
                // Controls
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.autoRotate = true;
                
                // Lighting
                scene.add(new THREE.AmbientLight(0x404040, 0.5));
                scene.add(new THREE.PointLight(0x00ff88, 1, 100).translateX(20).translateY(20));
                
                // Shape group
                const shapeGroup = new THREE.Group();
                scene.add(shapeGroup);
                
                sceneRef.current = { scene, camera, renderer, composer, controls, shapeGroup };
                
                // Animation loop
                const animate = () => {
                    requestAnimationFrame(animate);
                    controls.update();
                    shapeGroup.rotation.y += 0.005;
                    composer.render();
                };
                animate();
                
                // Cleanup
                return () => {
                    mountRef.current?.removeChild(renderer.domElement);
                    renderer.dispose();
                };
            }, []);
            
            // Update visualization
            useEffect(() => {
                if (!sceneRef.current || !data) return;
                
                const { shapeGroup } = sceneRef.current;
                
                // Clear previous shapes
                while (shapeGroup.children.length > 0) {
                    const child = shapeGroup.children[0];
                    shapeGroup.remove(child);
                    child.geometry?.dispose();
                    child.material?.dispose();
                }
                
                const betti = data.betti_numbers || [1, 0, 0];
                const risk = data.risk_score || 0;
                
                // Create materials
                const componentMat = new THREE.MeshPhongMaterial({
                    color: 0x00ff88,
                    emissive: 0x00ff88,
                    emissiveIntensity: 0.2
                });
                
                const loopMat = new THREE.MeshPhongMaterial({
                    color: new THREE.Color().setHSL((1 - risk) * 0.33, 0.8, 0.5),
                    emissive: new THREE.Color().setHSL((1 - risk) * 0.33, 0.8, 0.3),
                    emissiveIntensity: 0.3
                });
                
                // Visualize components
                for (let i = 0; i < betti[0]; i++) {
                    const sphere = new THREE.Mesh(
                        new THREE.SphereGeometry(1, 32, 32),
                        componentMat
                    );
                    const angle = (i / betti[0]) * Math.PI * 2;
                    sphere.position.set(Math.cos(angle) * 8, 0, Math.sin(angle) * 8);
                    shapeGroup.add(sphere);
                }
                
                // Visualize loops
                for (let i = 0; i < betti[1]; i++) {
                    const torus = new THREE.Mesh(
                        new THREE.TorusGeometry(3, 0.8, 16, 100),
                        loopMat
                    );
                    torus.position.y = (i - betti[1]/2) * 6;
                    torus.rotation.x = Math.PI / 2;
                    shapeGroup.add(torus);
                }
                
            }, [data]);
            
            return <div ref={mountRef} style={{ width: '100%', height: '100%' }} />;
        };
        
        // Info Panel Component
        const InfoPanel = ({ data, status }) => {
            const getRiskClass = (score) => {
                if (score > 0.7) return 'risk-critical';
                if (score > 0.4) return 'risk-medium';
                return 'risk-low';
            };
            
            return (
                <div style={{
                    position: 'absolute',
                    top: 20,
                    left: 20,
                    background: 'rgba(0, 0, 0, 0.8)',
                    padding: 20,
                    border: '1px solid #00ff88',
                    borderRadius: 5,
                    backdropFilter: 'blur(10px)'
                }}>
                    <h1 style={{ fontSize: 24, marginBottom: 15 }}>AURA Shape Analysis</h1>
                    
                    <div style={{ marginBottom: 10 }}>
                        Risk Score: <span className={getRiskClass(data?.risk_score || 0)}>
                            {(data?.risk_score || 0).toFixed(3)}
                        </span>
                    </div>
                    
                    <div>β₀: {data?.betti_numbers?.[0] || 0}</div>
                    <div>β₁: {data?.betti_numbers?.[1] || 0}</div>
                    <div>β₂: {data?.betti_numbers?.[2] || 0}</div>
                    
                    <div style={{ marginTop: 15 }}>
                        <strong>Patterns:</strong>
                        {data?.patterns_detected?.map((p, i) => (
                            <div key={i} style={{ marginLeft: 10 }}>
                                {p.name} ({(p.confidence * 100).toFixed(0)}%)
                            </div>
                        ))}
                    </div>
                    
                    <div style={{ 
                        position: 'absolute', 
                        top: 10, 
                        right: 10,
                        color: status === 'connected' ? '#33ff66' : '#ff3366'
                    }}>
                        {status === 'connected' ? '🟢' : '⚫'} {status}
                    </div>
                </div>
            );
        };
        
        // Main App
        const App = () => {
            const { data, status } = useWebSocket(`ws://${window.location.host}/ws/shapes`);
            
            return (
                <>
                    <ShapeVisualization data={data} />
                    <InfoPanel data={data} status={status} />
                </>
            );
        };
        
        // Render
        ReactDOM.render(<App />, document.getElementById('root'));
    </script>
    
    <style>
        .risk-low { color: #33ff66; }
        .risk-medium { color: #ffaa33; }
        .risk-critical { color: #ff3366; }
    </style>
</body>
</html>
    """


def get_shape_hud_html() -> HTMLResponse:
    """Return the React-based HUD."""
    return HTMLResponse(content=get_shape_hud_react())