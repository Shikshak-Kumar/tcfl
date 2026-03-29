import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text, Float, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

function IntersectionBar({ position, height, name, tier }) {
  const mesh = useRef();

  // Color based on tier and congestion
  const color = useMemo(() => {
    if (tier === 1) return '#ef4444'; // Hospital - Red
    if (tier === 2) return '#f59e0b'; // School - Amber
    return '#6366f1'; // Normal - Indigo
  }, [tier]);

  const targetHeight = Math.max(0.5, height * 5);

  useFrame((state) => {
    if (mesh.current) {
      // Smoothly animate height
      mesh.current.scale.y = THREE.MathUtils.lerp(mesh.current.scale.y, targetHeight, 0.1);
      mesh.current.position.y = mesh.current.scale.y / 2;
    }
  });

  return (
    <group position={position}>
      <mesh ref={mesh}>
        <boxGeometry args={[0.5, 1, 0.5]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.5}
          transparent
          opacity={0.8}
        />
      </mesh>

      {/* Label */}
      <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
        <group position={[0, targetHeight + 1.2, 0]}>
          <Text
            fontSize={0.4}
            color="white"
            anchorX="center"
            anchorY="middle"
            fontWeight="bold"
          >
            {name}
          </Text>
          <Text
            position={[0, -0.4, 0]}
            fontSize={0.3}
            color="#94a3b8"
            anchorX="center"
            anchorY="middle"
          >
            {height.toFixed(2)}
          </Text>
        </group>
      </Float>

      {/* Ground Circle */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <circleGeometry args={[0.8, 32]} />
        <meshBasicMaterial color={color} transparent opacity={0.2} />
      </mesh>
    </group>
  );
}

function TrafficFlowLines({ intersections }) {
  const lines = useMemo(() => {
    const result = [];
    for (let i = 0; i < intersections.length; i++) {
      for (let j = i + 1; j < intersections.length; j++) {
        const start = new THREE.Vector3(intersections[i].x, 0.1, intersections[i].z);
        const end = new THREE.Vector3(intersections[j].x, 0.1, intersections[j].z);
        result.push({ start, end });
      }
    }
    return result;
  }, [intersections]);

  return (
    <group>
      {lines.map((line, idx) => (
        <line key={idx}>
          <bufferGeometry attach="geometry">
            <float32BufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([...line.start.toArray(), ...line.end.toArray()])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial attach="material" color="#4338ca" transparent opacity={0.3} linewidth={1} />
        </line>
      ))}
    </group>
  );
}

export default function Map3DView({ intersections, latestMetrics }) {
  // Normalize lat/lon to a smaller coordinate system for 3D
  const coords = useMemo(() => {
    if (!intersections || intersections.length === 0) return [];

    const minLat = Math.min(...intersections.map(i => i.lat));
    const maxLat = Math.max(...intersections.map(i => i.lat));
    const minLon = Math.min(...intersections.map(i => i.lon));
    const maxLon = Math.max(...intersections.map(i => i.lon));

    const latRange = maxLat - minLat || 0.01;
    const lonRange = maxLon - minLon || 0.01;

    return intersections.map((ix, idx) => {
      const nid = `node_${idx}`;
      const metrics = latestMetrics?.intersections?.[nid] || {};

      // Scale to a 20x20 grid
      const x = ((ix.lon - minLon) / lonRange) * 20 - 10;
      const z = ((ix.lat - minLat) / latRange) * 20 - 10;

      return {
        x, z,
        name: metrics.name || ix.name || `Int ${idx + 1}`,
        tier: metrics.tier || 3,
        congestion: metrics.congestion || 1.0,
        nid
      };
    });
  }, [intersections, latestMetrics]);

  return (
    <div className="w-full h-full min-h-[500px] glass-panel rounded-2xl overflow-hidden relative">
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
        <h3 className="text-lg font-bold text-white flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          3D Traffic Density Map
        </h3>
        <p className="text-xs text-slate-400">Scale: Height = Congestion (Throughput/Queue)</p>
      </div>

      <Canvas shadows>
        <PerspectiveCamera makeDefault position={[15, 15, 15]} fov={50} />
        <OrbitControls enableDamping dampingFactor={0.05} maxPolarAngle={Math.PI / 2.1} />

        <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} castShadow />
        <spotLight position={[-10, 20, 10]} angle={0.15} penumbra={1} intensity={1} />

        {/* Ground Grid */}
        <gridHelper args={[40, 40, '#1e293b', '#0f172a']} position={[0, -0.01, 0]} />

        {/* Animated Traffic Lines */}
        <TrafficFlowLines intersections={coords} />

        {/* Intersection Bars */}
        {coords.map((c) => (
          <IntersectionBar
            key={c.nid}
            position={[c.x, 0, c.z]}
            height={c.congestion}
            name={c.name}
            tier={c.tier}
          />
        ))}

        {/* Fog for depth */}
        <fog attach="fog" args={['#0f172a', 10, 50]} />
      </Canvas>

      <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-2 glass-panel p-3 text-[10px]">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-rose-500 rounded-sm"></div>
          <span className="text-slate-300">High Priority (Hospital)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-amber-500 rounded-sm"></div>
          <span className="text-slate-300">Medium Priority (School)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-indigo-500 rounded-sm"></div>
          <span className="text-slate-300">Standard Intersection</span>
        </div>
      </div>
    </div>
  );
}
