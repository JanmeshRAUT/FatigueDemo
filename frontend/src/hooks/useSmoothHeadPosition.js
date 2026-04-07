import { useState, useEffect, useRef } from "react";

/**
 * Shared smooth head position animation hook
 * Used by both Standard and Vehicle modes for consistent smooth animation
 */
export const useSmoothHeadPosition = (targetAngles) => {
    const [current, setCurrent] = useState({
        x: 0, y: 0, z: 0, 
        source: "Unknown", 
        calibrated: true,
        position: "Center"
    });

    const prevTargetRef = useRef({ x: 0, y: 0, z: 0 });

    useEffect(() => {
        let animationFrameId;
        let isAnimating = true;

        const animate = () => {
            if (!isAnimating) return;

            const targetX = targetAngles?.angle_x ?? targetAngles?.x ?? 0;
            const targetY = targetAngles?.angle_y ?? targetAngles?.y ?? 0;
            const targetZ = targetAngles?.angle_z ?? targetAngles?.z ?? 0;
            const source = targetAngles?.source ?? "Unknown";
            const calibrated = targetAngles?.calibrated ?? true;
            const position = targetAngles?.position ?? "Center";

            setCurrent(prev => {
                const ALPHA = 0.1; // Smooth factor (0.1 = 10% step towards target)
                const DEADZONE = 1.0; // Degrees

                // 1. CLAMPING (Anatomical limits)
                const clampedTargetX = Math.max(-40, Math.min(40, targetX));
                const clampedTargetY = Math.max(-45, Math.min(45, targetY));
                const clampedTargetZ = Math.max(-20, Math.min(20, targetZ));

                // 2. DEADZONE (Ignore micro-movements)
                const dx = Math.abs(clampedTargetX - prevTargetRef.current.x);
                const dy = Math.abs(clampedTargetY - prevTargetRef.current.y);
                const dz = Math.abs(clampedTargetZ - prevTargetRef.current.z);
                
                let effectiveX = prevTargetRef.current.x;
                let effectiveY = prevTargetRef.current.y;
                let effectiveZ = prevTargetRef.current.z;

                if (dx > DEADZONE) effectiveX = clampedTargetX;
                if (dy > DEADZONE) effectiveY = clampedTargetY;
                if (dz > DEADZONE) effectiveZ = clampedTargetZ;

                // Update previous target
                prevTargetRef.current = { x: effectiveX, y: effectiveY, z: effectiveZ };

                // 3. ROLL REDUCTION (Visual preference)
                const visualTargetZ = effectiveZ * 0.5;

                // 4. LERP (Smooth interpolation)
                let nextX = prev.x + (effectiveX - prev.x) * ALPHA;
                let nextY = prev.y + (effectiveY - prev.y) * ALPHA;
                let nextZ = prev.z + (visualTargetZ - prev.z) * ALPHA;

                // 5. SETTLE (Stop if close)
                if (Math.abs(nextX - effectiveX) < 0.1) nextX = effectiveX;
                if (Math.abs(nextY - effectiveY) < 0.1) nextY = effectiveY;
                if (Math.abs(nextZ - visualTargetZ) < 0.1) nextZ = visualTargetZ;
                
                // Skip render if nothing changed
                if (nextX === prev.x && nextY === prev.y && nextZ === prev.z) {
                    return prev;
                }

                return {
                    x: nextX,
                    y: nextY,
                    z: nextZ,
                    source,
                    calibrated,
                    position
                };
            });

            animationFrameId = requestAnimationFrame(animate);
        };

        animationFrameId = requestAnimationFrame(animate);

        return () => {
            isAnimating = false;
            cancelAnimationFrame(animationFrameId);
        };
    }, [targetAngles]);

    return current;
};
