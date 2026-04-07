import { useState, useEffect } from "react";
import { useFatigueContext } from "../context/FatigueContext";

export const useHeadPosition = () => {
    const context = useFatigueContext();
    const fullData = context?.fullData || {};

    // Target values from Context
    const targetX = fullData?.head_position?.angle_x || 0;
    const targetY = fullData?.head_position?.angle_y || 0;
    const targetZ = fullData?.head_position?.angle_z || 0;
    const targetSource = fullData?.head_position?.source || "Unknown";
    const targetCalibrated = fullData?.head_position?.calibrated !== undefined ? fullData.head_position.calibrated : true;
    const targetPos = fullData?.head_position?.position || "Center";

    const [current, setCurrent] = useState({
        x: 0, y: 0, z: 0, 
        source: "Unknown", 
        calibrated: true,
        position: "Center",
        targetX: 0, targetY: 0, targetZ: 0
    });

    useEffect(() => {
        let animationFrameId;

        const animate = () => {
            setCurrent(prev => {
                const ALPHA = 0.1; // Smooth factor
                const DEADZONE = 1.0; // Degrees
                
                // 1. CLAMPING (Anatomical limits)
                // Pitch: -40 to 40
                const clampedTargetX = Math.max(-40, Math.min(40, targetX));
                // Yaw: -45 to 45
                const clampedTargetY = Math.max(-45, Math.min(45, targetY));
                // Roll: -20 to 20
                const clampedTargetZ = Math.max(-20, Math.min(20, targetZ));

                // 2. DEADZONE (Ignore micro-movements)
                // If the target hasn't changed enough from the LAST TARGET we acted on, ignore it.
                // Note: We compare against 'prev.targetX' to avoid getting stuck if we rely on 'prev.x' (display)
                const dx = Math.abs(clampedTargetX - prev.targetX);
                const dy = Math.abs(clampedTargetY - prev.targetY);
                const dz = Math.abs(clampedTargetZ - prev.targetZ);
                
                let effectiveX = prev.targetX;
                let effectiveY = prev.targetY;
                let effectiveZ = prev.targetZ;

                if (dx > DEADZONE) effectiveX = clampedTargetX;
                if (dy > DEADZONE) effectiveY = clampedTargetY;
                if (dz > DEADZONE) effectiveZ = clampedTargetZ;

                // 3. ROLL REDUCTION (Visual preference)
                // We dampen the roll target visually so it looks subtle
                const visualTargetZ = effectiveZ * 0.5;

                // 4. LERP (Smoothing)
                let nextX = prev.x + (effectiveX - prev.x) * ALPHA;
                let nextY = prev.y + (effectiveY - prev.y) * ALPHA;
                let nextZ = prev.z + (visualTargetZ - prev.z) * ALPHA;

                // 5. SETTLE (Stop if close)
                if (Math.abs(nextX - effectiveX) < 0.1) nextX = effectiveX;
                if (Math.abs(nextY - effectiveY) < 0.1) nextY = effectiveY;
                if (Math.abs(nextZ - visualTargetZ) < 0.1) nextZ = visualTargetZ;
                
                // Stop rendering if nothing changed
                if (nextX === prev.x && nextY === prev.y && nextZ === prev.z) {
                    return prev; 
                }

                return {
                    x: nextX,
                    y: nextY,
                    z: nextZ,
                    source: targetSource,
                    calibrated: targetCalibrated,
                    position: targetPos,
                    targetX: effectiveX, // Store the filtering state
                    targetY: effectiveY,
                    targetZ: effectiveZ
                };
            });
            animationFrameId = requestAnimationFrame(animate);
        };

        animate();

        return () => cancelAnimationFrame(animationFrameId);
    }, [targetX, targetY, targetZ, targetSource, targetCalibrated, targetPos]);

    return {
        position: current.position,
        angle_x: current.x,
        angle_y: current.y,
        angle_z: current.z,
        source: current.source,
        calibrated: current.calibrated
    };
};
