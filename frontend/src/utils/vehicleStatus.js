export function getConnectionMeta(connectionStatus) {
  if (connectionStatus === "connected") {
    return { label: "Connected", tone: "safe" };
  }
  if (connectionStatus === "retrying" || connectionStatus === "connecting") {
    return { label: "Connecting", tone: "warning" };
  }
  return { label: "Offline", tone: "danger" };
}

export function getPredictionMeta(status, connectionStatus, hasData) {
  if (!hasData || connectionStatus === "offline") {
    return { label: "OFFLINE", tone: "neutral" };
  }

  if (connectionStatus === "connecting" || connectionStatus === "retrying") {
    return { label: "CONNECTING", tone: "warning" };
  }

  if (status === "Fatigued") {
    return { label: "FATIGUED", tone: "danger" };
  }
  if (status === "Drowsy") {
    return { label: "DROWSY", tone: "warning" };
  }
  if (status === "Alert") {
    return { label: "ALERT", tone: "safe" };
  }

  return { label: "UNKNOWN", tone: "neutral" };
}

export function getPerclosRiskBand(perclosPercent) {
  if (perclosPercent >= 45) return "High Risk";
  if (perclosPercent >= 25) return "Moderate Risk";
  return "Low Risk";
}
