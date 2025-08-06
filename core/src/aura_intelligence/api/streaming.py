"""
AURA Shape Streaming API
========================
Real-time WebSocket endpoint for streaming topological analysis data
to the 3D Shape HUD visualization.
"""

import asyncio
import json
import random
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np

from ..tda.algorithms import compute_persistence_diagram, compute_betti_numbers
from ..memory.shape_aware_memory import TopologicalSignature


class ShapeStreamManager:
    """Manages WebSocket connections for shape streaming"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.analysis_queue = asyncio.Queue()
        self.is_running = False
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        """Remove disconnected client"""
        self.active_connections.remove(websocket)
    
    async def broadcast_shape_data(self, data: Dict):
        """Send shape data to all connected clients"""
        message = json.dumps(data)
        
        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
    
    async def stream_analysis(self, analysis_result: Dict):
        """Stream analysis results with topological data"""
        # Extract topological features
        if 'topological_signature' in analysis_result:
            sig = analysis_result['topological_signature']
            
            shape_data = {
                'timestamp': datetime.now().isoformat(),
                'risk_score': analysis_result.get('risk_score', 0.0),
                'betti_numbers': [
                    sig.get('betti_0', 1),
                    sig.get('betti_1', 0),
                    sig.get('betti_2', 0)
                ],
                'persistence_intervals': self._format_persistence(sig),
                'patterns_detected': [
                    {
                        'name': p['name'],
                        'confidence': p['confidence']
                    } 
                    for p in analysis_result.get('patterns_detected', [])
                ],
                'domain': analysis_result.get('domain', 'unknown')
            }
            
            await self.broadcast_shape_data(shape_data)
    
    def _format_persistence(self, signature: Dict) -> List[Dict]:
        """Format persistence diagram for visualization"""
        intervals = []
        
        if 'persistence_diagram' in signature:
            for dim, birth, death in signature['persistence_diagram']:
                intervals.append({
                    'dimension': dim,
                    'birth': float(birth),
                    'death': float(death) if death != float('inf') else 1.0,
                    'persistence': float(death - birth) if death != float('inf') else 1.0
                })
        
        return intervals


# Global manager instance
shape_stream_manager = ShapeStreamManager()


async def shape_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time shape streaming
    
    Clients connect to receive:
    - Betti numbers (topological invariants)
    - Persistence diagrams
    - Risk scores
    - Detected patterns
    """
    await shape_stream_manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_text(json.dumps({
            'type': 'connection',
            'status': 'connected',
            'message': 'AURA Shape Stream Active'
        }))
        
        # Keep connection alive and handle incoming messages
        while True:
            # In production, this would receive analysis requests
            # For now, simulate periodic updates
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text(json.dumps({'type': 'pong'}))
            
            # Simulate shape analysis (in production, this connects to real TDA)
            await asyncio.sleep(1)
            await simulate_shape_update(websocket)
            
    except WebSocketDisconnect:
        shape_stream_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        shape_stream_manager.disconnect(websocket)


async def simulate_shape_update(websocket: WebSocket):
    """Simulate shape data for demo purposes"""
    # In production, this would come from actual TDA analysis
    
    # Simulate varying topological complexity
    n_loops = random.randint(0, 3)
    n_components = random.randint(1, 4)
    risk_base = 0.2 * n_loops + 0.1 * (n_components - 1)
    
    shape_data = {
        'timestamp': datetime.now().isoformat(),
        'risk_score': min(0.95, risk_base + random.random() * 0.3),
        'betti_numbers': [n_components, n_loops, 0],
        'persistence_intervals': [
            {
                'dimension': 0,
                'birth': 0.0,
                'death': random.uniform(0.1, 0.5),
                'persistence': random.uniform(0.1, 0.5)
            } for _ in range(n_components)
        ] + [
            {
                'dimension': 1,
                'birth': random.uniform(0.1, 0.3),
                'death': random.uniform(0.6, 1.0),
                'persistence': random.uniform(0.3, 0.7)
            } for _ in range(n_loops)
        ],
        'patterns_detected': []
    }
    
    # Add detected patterns based on topology
    if n_loops > 0:
        shape_data['patterns_detected'].append({
            'name': 'stuck_loop',
            'confidence': 0.7 + random.random() * 0.25
        })
    
    if n_components > 2:
        shape_data['patterns_detected'].append({
            'name': 'context_fragmentation',
            'confidence': 0.6 + random.random() * 0.3
        })
    
    await websocket.send_text(json.dumps(shape_data))


def get_shape_hud_html() -> HTMLResponse:
    """Return the 3D Shape HUD HTML page"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AURA 3D Shape HUD</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            margin: 0; 
            background: #0a0a0a; 
            color: #00ff88; 
            font-family: 'Courier New', monospace;
            overflow: hidden;
        }
        
        #info {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border: 1px solid #00ff88;
            border-radius: 5px;
            backdrop-filter: blur(10px);
        }
        
        #info h1 {
            margin: 0 0 15px 0;
            font-size: 24px;
            text-shadow: 0 0 10px #00ff88;
        }
        
        .metric {
            margin: 10px 0;
            font-size: 16px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #fff;
            text-shadow: 0 0 5px currentColor;
        }
        
        .risk-high { color: #ff3366; }
        .risk-medium { color: #ffaa33; }
        .risk-low { color: #33ff66; }
        
        #patterns {
            margin-top: 15px;
            font-size: 14px;
        }
        
        .pattern {
            margin: 5px 0;
            padding: 5px;
            background: rgba(0, 255, 136, 0.1);
            border-left: 3px solid #00ff88;
        }
        
        #connection-status {
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            border: 1px solid #00ff88;
            border-radius: 5px;
        }
        
        .connected { color: #33ff66; }
        .disconnected { color: #ff3366; }
    </style>
</head>
<body>
    <div id="info">
        <h1>AURA Shape Analysis</h1>
        <div class="metric">
            Risk Score: <span id="risk" class="metric-value risk-low">0.00</span>
        </div>
        <div class="metric">
            Components (β₀): <span id="betti0" class="metric-value">1</span>
        </div>
        <div class="metric">
            Loops (β₁): <span id="betti1" class="metric-value">0</span>
        </div>
        <div class="metric">
            Voids (β₂): <span id="betti2" class="metric-value">0</span>
        </div>
        <div id="patterns">
            <strong>Detected Patterns:</strong>
            <div id="pattern-list"></div>
        </div>
    </div>
    
    <div id="connection-status">
        <span id="status" class="disconnected">⚫ Disconnected</span>
    </div>

    <script type="importmap">
        {
            "imports": {
                "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
                "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
            }
        }
    </script>
    
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
        import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
        import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
        import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

        // Scene setup
        const scene = new THREE.Scene();
        scene.fog = new THREE.Fog(0x0a0a0a, 10, 50);
        
        const camera = new THREE.PerspectiveCamera(
            75, 
            window.innerWidth / window.innerHeight, 
            0.1, 
            1000
        );
        camera.position.set(0, 10, 30);
        
        const renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        // Post-processing for glow effects
        const composer = new EffectComposer(renderer);
        composer.addPass(new RenderPass(scene, camera));
        
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.5, // strength
            0.4, // radius
            0.85  // threshold
        );
        composer.addPass(bloomPass);
        
        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);
        
        const pointLight1 = new THREE.PointLight(0x00ff88, 1, 100);
        pointLight1.position.set(20, 20, 20);
        scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0xff3366, 0.5, 100);
        pointLight2.position.set(-20, -20, -20);
        scene.add(pointLight2);
        
        // Shape visualization group
        const shapeGroup = new THREE.Group();
        scene.add(shapeGroup);
        
        // Materials
        const materials = {
            component: new THREE.MeshPhongMaterial({
                color: 0x00ff88,
                emissive: 0x00ff88,
                emissiveIntensity: 0.2,
                transparent: true,
                opacity: 0.8
            }),
            loop: new THREE.MeshPhongMaterial({
                color: 0xff3366,
                emissive: 0xff3366,
                emissiveIntensity: 0.3,
                transparent: true,
                opacity: 0.9
            }),
            connection: new THREE.LineBasicMaterial({
                color: 0x00ff88,
                opacity: 0.3,
                transparent: true
            })
        };
        
        // WebSocket connection
        let ws;
        const wsUrl = `ws://${window.location.host}/ws/shapes`;
        
        function connect() {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                document.getElementById('status').textContent = '🟢 Connected';
                document.getElementById('status').className = 'connected';
                
                // Send ping to start receiving data
                setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send('ping');
                    }
                }, 5000);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'pong') return;
                
                updateVisualization(data);
                updateUI(data);
            };
            
            ws.onclose = () => {
                document.getElementById('status').textContent = '⚫ Disconnected';
                document.getElementById('status').className = 'disconnected';
                
                // Reconnect after 3 seconds
                setTimeout(connect, 3000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateVisualization(data) {
            // Clear existing shapes
            while (shapeGroup.children.length > 0) {
                const child = shapeGroup.children[0];
                shapeGroup.remove(child);
                child.geometry?.dispose();
            }
            
            const betti = data.betti_numbers || [1, 0, 0];
            
            // Visualize components (Betti-0) as spheres
            for (let i = 0; i < betti[0]; i++) {
                const geometry = new THREE.SphereGeometry(1, 32, 32);
                const sphere = new THREE.Mesh(geometry, materials.component);
                
                // Arrange in a circle
                const angle = (i / betti[0]) * Math.PI * 2;
                sphere.position.x = Math.cos(angle) * 8;
                sphere.position.z = Math.sin(angle) * 8;
                sphere.position.y = 0;
                
                shapeGroup.add(sphere);
            }
            
            // Visualize loops (Betti-1) as tori
            for (let i = 0; i < betti[1]; i++) {
                const geometry = new THREE.TorusGeometry(3, 0.8, 16, 100);
                const torus = new THREE.Mesh(geometry, materials.loop);
                
                // Stack vertically
                torus.position.y = (i - betti[1]/2) * 6;
                torus.rotation.x = Math.PI / 2;
                
                shapeGroup.add(torus);
            }
            
            // Add connecting lines for persistence
            if (data.persistence_intervals) {
                const points = [];
                data.persistence_intervals.forEach((interval, i) => {
                    if (interval.dimension === 1) {
                        // Create visual representation of persistence
                        const height = interval.persistence * 10;
                        points.push(
                            new THREE.Vector3(-5, -height/2, i * 2),
                            new THREE.Vector3(5, height/2, i * 2)
                        );
                    }
                });
                
                if (points.length > 0) {
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const lines = new THREE.LineSegments(geometry, materials.connection);
                    shapeGroup.add(lines);
                }
            }
            
            // Update material colors based on risk
            const risk = data.risk_score || 0;
            const hue = (1 - risk) * 0.33; // Green to red
            materials.loop.color.setHSL(hue, 0.8, 0.5);
            materials.loop.emissive.setHSL(hue, 0.8, 0.3);
        }
        
        function updateUI(data) {
            // Update risk score
            const riskElement = document.getElementById('risk');
            const risk = data.risk_score || 0;
            riskElement.textContent = risk.toFixed(3);
            
            if (risk > 0.7) {
                riskElement.className = 'metric-value risk-high';
            } else if (risk > 0.4) {
                riskElement.className = 'metric-value risk-medium';
            } else {
                riskElement.className = 'metric-value risk-low';
            }
            
            // Update Betti numbers
            const betti = data.betti_numbers || [1, 0, 0];
            document.getElementById('betti0').textContent = betti[0];
            document.getElementById('betti1').textContent = betti[1];
            document.getElementById('betti2').textContent = betti[2];
            
            // Update patterns
            const patternList = document.getElementById('pattern-list');
            patternList.innerHTML = '';
            
            if (data.patterns_detected && data.patterns_detected.length > 0) {
                data.patterns_detected.forEach(pattern => {
                    const div = document.createElement('div');
                    div.className = 'pattern';
                    div.textContent = `${pattern.name} (${(pattern.confidence * 100).toFixed(0)}%)`;
                    patternList.appendChild(div);
                });
            } else {
                patternList.innerHTML = '<div class="pattern">No patterns detected</div>';
            }
        }
        
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
            
            controls.update();
            
            // Rotate shapes
            shapeGroup.rotation.y += 0.005;
            
            // Pulse effect for high risk
            shapeGroup.children.forEach((child, i) => {
                if (child.material) {
                    const scale = 1 + Math.sin(Date.now() * 0.001 + i) * 0.1;
                    child.scale.set(scale, scale, scale);
                }
            });
            
            composer.render();
        }
        
        // Handle window resize
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            composer.setSize(window.innerWidth, window.innerHeight);
        });
        
        // Start
        connect();
        animate();
    </script>
</body>
</html>
    """
    
    return HTMLResponse(content=html_content)