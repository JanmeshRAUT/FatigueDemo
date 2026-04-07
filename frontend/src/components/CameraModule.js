import React, { useEffect, useRef, useState } from "react";
import { Wifi, WifiOff, UserMinus } from "lucide-react";
import { useFatigueData } from "../hooks/useFatigueData";
import { useFatigueContext } from "../context/FatigueContext";
import { useModeContext } from "../context/ModeContext";
import { API_BASE } from "../api";

export default function CameraModule({ vehicleOverlayMode = "none" }) {
  // Always call hooks unconditionally at top level (React Hooks Rules)
  const contextData = useFatigueContext();
  const data = useFatigueData();
  const mode = useModeContext();
  
  // Now safely check if context is available
  const setFullData = contextData?.setFullData;
  const hasContext = !!contextData && !!contextData.setFullData;

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [cameraStatus, setCameraStatus] = useState("Idle");
  const wsRef = useRef(null);

  const noFaceDetected = data?.status === "No Face";
  const showNoSubjectOverlay = cameraStatus === "Streaming" && (noFaceDetected || vehicleOverlayMode === "no-subject");
  const showUnstableOverlay = cameraStatus === "Streaming" && vehicleOverlayMode === "unstable" && !showNoSubjectOverlay;

  useEffect(() => {
    let stream;
    let animationFrameId;

    async function startCamera() {
      try {
        setCameraStatus("Starting...");
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1280 },
            height: { ideal: 720 }
          }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        // Use the vehicle websocket path when Vehicle mode is active.
        const wsPath = mode?.isVehicleMode ? '/ws/vehicle/detect' : '/ws/detect';
        const wsUrl = API_BASE.replace(/^http/, 'ws') + wsPath;
        console.log("Connecting to WS:", wsUrl);
        
        const socket = new WebSocket(wsUrl);
        wsRef.current = socket;

        socket.onopen = () => {
          console.log("WebSocket Connected");
          setCameraStatus("Streaming");
          startSendingFrames();
        };

        socket.onmessage = (event) => {
          try {
            const result = JSON.parse(event.data);
            // Update the global context with the fresh data from backend (only if in FatigueProvider)
            if (result && hasContext && setFullData) {
                // Ensure status is active
                setFullData({ ...result, status: "Active" });
            }
          } catch (e) {
            console.error("WS Message Parse Error", e);
          }
        };

        socket.onerror = (error) => {
          console.error("WebSocket Error:", error);
          setCameraStatus("Connection Error");
        };

        socket.onclose = () => {
          console.log("WebSocket Disconnected");
          setCameraStatus("Disconnected");
        };

      } catch (err) {
        console.error("Camera access error:", err);
        setCameraStatus("Camera Error");
      }
    }

    const startSendingFrames = () => {
       const sendLoop = () => {
          if (!videoRef.current || !canvasRef.current || !wsRef.current) {
             animationFrameId = requestAnimationFrame(sendLoop);
             return;
          }

          if (wsRef.current.readyState === WebSocket.OPEN) {
             const canvas = canvasRef.current;
             const ctx = canvas.getContext("2d");

             // Downscale content for performance
             const scaleFactor = 480 / videoRef.current.videoWidth;
             canvas.width = 480;
             canvas.height = videoRef.current.videoHeight * scaleFactor;
             
             ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
             
             // Compress to JPEG 0.5
             const imageData = canvas.toDataURL("image/jpeg", 0.5);
             
             // Send
             wsRef.current.send(JSON.stringify({ image_data: imageData }));
          }

          // Throttle to ~30FPS or just use animation frame
          // Using timeout to prevent overwhelming the network if needed, but rAF is usually fine.
          // Let's stick to requestAnimationFrame for smooth 30-60fps if network allows.
          // Or throttle to 100ms? User said "Decrease Load". 30fps is high load.
          // Maybe throttle to 100ms (10fps) is enough for fatigue detection?
          // Previous interval was 800ms. 
          // Let's try 30fps (rAF) but if it's too much we can throttling.
          // Real-time head pose needs good FPS. Let's try 100ms throttle.
          
          setTimeout(() => {
              animationFrameId = requestAnimationFrame(sendLoop);
          }, 100); // 10 FPS should be sufficient and much lighter than 30fps
       };
       sendLoop();
    };

    startCamera();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, [setFullData, hasContext, mode?.isVehicleMode]);

  return (
    <>
      <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden', borderRadius: '12px' }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="camera-feed-video"
          style={{ transform: "scaleX(-1)", width: '100%', height: '100%', objectFit: 'contain' }}
        />
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* NO SUBJECT / NO FACE OVERLAY */}
        {showNoSubjectOverlay && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(239, 68, 68, 0.2)', 
            backdropFilter: 'blur(2px)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 20
          }}>
            <div className="animate-pulse" style={{ background: '#ef4444', padding: '8px 16px', borderRadius: '8px', color: 'white', fontWeight: 800, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px', boxShadow: '0 4px 12px rgba(239, 68, 68, 0.4)' }}>
              <UserMinus size={18} strokeWidth={3} />
              NO FACE DETECTED
            </div>
            <span style={{ color: 'white', fontSize: '0.7rem', fontWeight: 600, marginTop: '8px', textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>Center yourself in the frame</span>
          </div>
        )}

        {/* UNSTABLE POSITION OVERLAY */}
        {showUnstableOverlay && (
          <div style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(245, 158, 11, 0.16)',
            backdropFilter: 'blur(1px)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 19
          }}>
            <div className="animate-pulse" style={{ background: '#f59e0b', padding: '8px 16px', borderRadius: '8px', color: '#111827', fontWeight: 800, fontSize: '0.9rem', display: 'flex', alignItems: 'center', gap: '8px', boxShadow: '0 4px 12px rgba(245, 158, 11, 0.35)' }}>
              UNSTABLE HEAD POSITION
            </div>
            <span style={{ color: 'white', fontSize: '0.7rem', fontWeight: 600, marginTop: '8px', textShadow: '0 1px 2px rgba(0,0,0,0.5)' }}>Keep your head centered for reliable detection</span>
          </div>
        )}

        {/* Overlay HUD */}
        <div className="camera-overlay">
          <div className="camera-status">
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: cameraStatus === 'Streaming'
                ? (showNoSubjectOverlay ? '#ef4444' : (showUnstableOverlay ? '#f59e0b' : '#22c55e'))
                : '#ef4444'
            }}></span>
            {cameraStatus === 'Streaming'
              ? (showNoSubjectOverlay ? 'No Subject Detected' : (showUnstableOverlay ? 'Unstable Position' : 'Live Feed Active (WS)'))
              : cameraStatus}
          </div>

          <div>
            {cameraStatus === 'Streaming' ? <Wifi size={16} /> : <WifiOff size={16} />}
          </div>
        </div>
      </div>
    </>
  );
}
