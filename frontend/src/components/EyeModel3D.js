import React, { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { RoundedBox, Environment, ContactShadows } from "@react-three/drei";
import * as THREE from "three";
import "./Css/EyeModel.css";

function EyeRobotModel({
  perclos = 0,
  ear = 0.3,
  status = "Alert",
  eyeState = "Open",
  yaw = 0,
  pitch = 0,
}) {
  const groupRef = useRef();
  const leftEyeCoreRef = useRef();
  const rightEyeCoreRef = useRef();
  const leftEyeRingRef = useRef();
  const rightEyeRingRef = useRef();
  const browLeftRef = useRef();
  const browRightRef = useRef();
  const pulseArcRef = useRef();

  const blinkState = useRef({
    active: false,
    progress: 0,
    nextBlinkAt: 1.5 + Math.random() * 2.5,
  });

  const normalizedPerclos = Math.max(0, Math.min(1, perclos / 100));
  const normalizedEAR = Math.max(0, Math.min(1, ear / 0.4));

  const eyeColor = useMemo(() => {
    if (status === "Fatigued" || normalizedPerclos >= 0.55) return new THREE.Color("#ef4444");
    if (status === "Drowsy" || normalizedPerclos >= 0.3) return new THREE.Color("#f59e0b");
    return new THREE.Color("#7ce7d7");
  }, [status, normalizedPerclos]);

  const trimColor = useMemo(() => {
    if (status === "Fatigued" || normalizedPerclos >= 0.55) return "#fecaca";
    if (status === "Drowsy" || normalizedPerclos >= 0.3) return "#fde68a";
    return "#a7f3d0";
  }, [status, normalizedPerclos]);

  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();
    const blink = blinkState.current;

    if (!blink.active && time >= blink.nextBlinkAt) {
      blink.active = true;
      blink.progress = 0;
    }

    if (blink.active) {
      blink.progress += delta * 10;
      if (blink.progress >= 1) {
        blink.active = false;
        blink.progress = 0;
        blink.nextBlinkAt = time + 1.6 + Math.random() * 2.8;
      }
    }

    const blinkAmount = blink.active ? Math.sin(blink.progress * Math.PI) : 0;
    const fatigueSquint = THREE.MathUtils.clamp(
      normalizedPerclos * 0.45 + (1 - normalizedEAR) * 0.3,
      0,
      0.72
    );
    const eyeClosedBySignal = ["closed", "no face"].includes(String(eyeState).toLowerCase());
    const baseEyeOpen = eyeClosedBySignal ? 0.14 : 1;
    const eyeOpen = THREE.MathUtils.clamp(baseEyeOpen - blinkAmount * 0.9 - fatigueSquint, 0.1, 1);

    const clampedYaw = THREE.MathUtils.clamp(Number(yaw) || 0, -30, 30);
    const clampedPitch = THREE.MathUtils.clamp(Number(pitch) || 0, -20, 20);
    const gazeOffsetX = (clampedYaw / 30) * 0.03 + Math.sin(time * 0.85) * 0.008;
    const gazeOffsetY = (-clampedPitch / 20) * 0.018 + Math.cos(time * 0.55) * 0.006;

    if (groupRef.current) {
      groupRef.current.rotation.z = Math.sin(time * 0.33) * 0.016;
      groupRef.current.rotation.x = Math.cos(time * 0.24) * 0.014;
      groupRef.current.position.y = Math.sin(time * 0.52) * 0.018;
    }

    [leftEyeCoreRef, rightEyeCoreRef].forEach((ref, index) => {
      if (ref.current) {
        ref.current.scale.y = eyeOpen;
        ref.current.scale.x = THREE.MathUtils.lerp(0.92, 1.08, 1 - eyeOpen);
        ref.current.position.x = (index === 0 ? -0.31 : 0.31) + gazeOffsetX;
        ref.current.position.y = gazeOffsetY;
      }
    });

    [leftEyeRingRef, rightEyeRingRef].forEach((ref, index) => {
      if (ref.current) {
        ref.current.position.x = (index === 0 ? -0.31 : 0.31) + gazeOffsetX * 0.5;
        ref.current.position.y = gazeOffsetY * 0.5;
        ref.current.material.opacity = THREE.MathUtils.lerp(0.35, 0.85, eyeOpen);
      }
    });

    [browLeftRef, browRightRef].forEach((ref) => {
      if (ref.current) {
        ref.current.visible = normalizedPerclos > 0.35;
        ref.current.position.y = 0.27 - fatigueSquint * 0.12;
      }
    });

    if (pulseArcRef.current) {
      pulseArcRef.current.scale.x = THREE.MathUtils.lerp(0.4, 1, eyeOpen);
      pulseArcRef.current.material.emissiveIntensity = THREE.MathUtils.lerp(0.35, 1.3, eyeOpen);
      pulseArcRef.current.material.opacity = THREE.MathUtils.lerp(0.35, 0.82, eyeOpen);
      pulseArcRef.current.visible = !eyeClosedBySignal;
    }
  });

  return (
    <group ref={groupRef}>
      <RoundedBox args={[1.62, 1.56, 1.14]} radius={0.28} smoothness={4}>
        <meshStandardMaterial color="#d9e3ef" roughness={0.34} metalness={0.1} />
      </RoundedBox>

      <RoundedBox args={[1.32, 0.68, 0.1]} radius={0.2} smoothness={2} position={[0, 0.04, 0.6]}>
        <meshStandardMaterial color="#04101f" roughness={0.2} metalness={0.82} />
      </RoundedBox>

      <RoundedBox args={[1.2, 0.56, 0.03]} radius={0.16} smoothness={2} position={[0, 0.04, 0.665]}>
        <meshPhysicalMaterial
          color="#0f172a"
          roughness={0.04}
          transmission={0.2}
          thickness={0.2}
          opacity={0.7}
          transparent
        />
      </RoundedBox>

      <group position={[0, 0.07, 0.68]}>
        <mesh ref={leftEyeCoreRef} position={[-0.31, 0, 0]}>
          <capsuleGeometry args={[0.07, 0.16, 4, 10]} />
          <meshStandardMaterial color={eyeColor} emissive={eyeColor} emissiveIntensity={2.2} />
        </mesh>

        <mesh ref={rightEyeCoreRef} position={[0.31, 0, 0]}>
          <capsuleGeometry args={[0.07, 0.16, 4, 10]} />
          <meshStandardMaterial color={eyeColor} emissive={eyeColor} emissiveIntensity={2.2} />
        </mesh>

        <mesh ref={leftEyeRingRef} position={[-0.31, 0, 0]}>
          <torusGeometry args={[0.12, 0.012, 14, 36]} />
          <meshStandardMaterial color={eyeColor} emissive={eyeColor} emissiveIntensity={0.6} transparent opacity={0.75} />
        </mesh>

        <mesh ref={rightEyeRingRef} position={[0.31, 0, 0]}>
          <torusGeometry args={[0.12, 0.012, 14, 36]} />
          <meshStandardMaterial color={eyeColor} emissive={eyeColor} emissiveIntensity={0.6} transparent opacity={0.75} />
        </mesh>

        <mesh ref={browLeftRef} position={[-0.31, 0.23, 0.01]} rotation={[0, 0, 0.18]}>
          <boxGeometry args={[0.2, 0.016, 0.02]} />
          <meshStandardMaterial color={trimColor} emissive={trimColor} emissiveIntensity={0.45} />
        </mesh>

        <mesh ref={browRightRef} position={[0.31, 0.23, 0.01]} rotation={[0, 0, -0.18]}>
          <boxGeometry args={[0.2, 0.016, 0.02]} />
          <meshStandardMaterial color={trimColor} emissive={trimColor} emissiveIntensity={0.45} />
        </mesh>
      </group>

      <mesh ref={pulseArcRef} position={[0, -0.24, 0.675]} rotation={[0, 0, Math.PI / 2]}>
        <torusGeometry args={[0.2, 0.015, 10, 36, Math.PI]} />
        <meshStandardMaterial color={eyeColor} emissive={eyeColor} emissiveIntensity={1} transparent opacity={0.75} />
      </mesh>

      <RoundedBox args={[0.16, 0.44, 0.3]} radius={0.05} smoothness={2} position={[-0.9, 0.02, -0.02]}>
        <meshStandardMaterial color="#c4cfdd" roughness={0.42} metalness={0.08} />
      </RoundedBox>
      <RoundedBox args={[0.16, 0.44, 0.3]} radius={0.05} smoothness={2} position={[0.9, 0.02, -0.02]}>
        <meshStandardMaterial color="#c4cfdd" roughness={0.42} metalness={0.08} />
      </RoundedBox>

      <mesh position={[0, -0.74, -0.02]}>
        <cylinderGeometry args={[0.38, 0.48, 0.2, 36]} />
        <meshStandardMaterial color="#ccd6e2" roughness={0.46} metalness={0.06} />
      </mesh>

      <mesh position={[0, -0.86, -0.02]}>
        <cylinderGeometry args={[0.58, 0.58, 0.07, 40]} />
        <meshStandardMaterial color="#bdc9d7" roughness={0.5} metalness={0.05} />
      </mesh>
    </group>
  );
}

export default function EyeModel3D({
  perclos = 0,
  ear = 0.3,
  status = "Alert",
  eyeState = "Open",
  yaw = 0,
  pitch = 0,
}) {
  const safePerclos = typeof perclos === "number" ? perclos : 0;
  const safeEar = typeof ear === "number" ? ear : 0.3;
  const safeStatus = typeof status === "string" ? status : "Alert";
  const safeEyeState = typeof eyeState === "string" ? eyeState : "Open";
  const safeYaw = typeof yaw === "number" ? yaw : 0;
  const safePitch = typeof pitch === "number" ? pitch : 0;
  const riskTone = safeStatus === "Fatigued" ? "danger" : safeStatus === "Drowsy" ? "warning" : "safe";

  return (
    <div className="eye-model-container">
      <div className="eye-model-hud">
        <div className="eye-model-heading">EYE BEHAVIOUR MODEL</div>
        <div className="eye-model-badges">
          <span className={`eye-model-badge tone-${riskTone}`}>{safeStatus}</span>
          <span className="eye-model-badge tone-neutral">Eyes: {safeEyeState}</span>
          <span className="eye-model-badge tone-neutral">PERCLOS {safePerclos.toFixed(1)}%</span>
          <span className="eye-model-badge tone-neutral">EAR {safeEar.toFixed(2)}</span>
        </div>
      </div>

      <div className="eye-model-axis-label eye-model-axis-top">Pitch {safePitch.toFixed(0)}°</div>
      <div className="eye-model-axis-label eye-model-axis-right">Yaw {safeYaw.toFixed(0)}°</div>

      <Canvas camera={{ position: [0, 0, 4], fov: 50 }} style={{ width: "100%", height: "100%" }}>
        <ambientLight intensity={0.5} />
        <spotLight position={[10, 10, 10]} angle={0.2} penumbra={1} intensity={1} castShadow />
        <pointLight position={[-8, -8, -8]} intensity={0.4} />

        <EyeRobotModel
          perclos={safePerclos}
          ear={safeEar}
          status={safeStatus}
          eyeState={safeEyeState}
          yaw={safeYaw}
          pitch={safePitch}
        />

        <ContactShadows position={[0, -1.26, 0]} opacity={0.35} scale={7} blur={2.3} far={4} />
        <Environment preset="city" />
      </Canvas>
    </div>
  );
}
