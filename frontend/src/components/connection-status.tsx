"use client"

import { useSignal } from "@/lib/signal-context"
import { Activity, WifiOff, Loader2, AlertCircle } from "lucide-react"

export function ConnectionStatus() {
  const { connectionStatus, esp32Ip, currentHeartRate } = useSignal()

  const statusConfig = {
    connecting: {
      icon: Loader2,
      label: "Connecting",
      pillClass: "status-pill connecting",
      iconClass: "status-warning",
      animate: "animate-spin",
    },
    connected: {
      icon: Activity,
      label: "Connected",
      pillClass: "status-pill connected",
      iconClass: "status-ok",
      animate: "animate-pulse",
    },
    disconnected: {
      icon: WifiOff,
      label: "Disconnected",
      pillClass: "status-pill",
      iconClass: "text-muted-foreground",
      animate: "",
    },
    error: {
      icon: AlertCircle,
      label: "Error",
      pillClass: "status-pill error",
      iconClass: "status-error",
      animate: "",
    },
  }

  const config = statusConfig[connectionStatus]
  const Icon = config.icon

  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className={`flex items-center gap-2 ${config.pillClass}`}>
          <Icon className={`h-4 w-4 ${config.iconClass} ${config.animate}`} />
          <span className={`text-sm font-medium ${config.iconClass}`}>{config.label}</span>
        </div>

        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>ESP32</span>
          <code className="rounded bg-muted px-2 py-0.5 text-xs font-mono">{esp32Ip}</code>
        </div>
      </div>

      {connectionStatus === "connected" && currentHeartRate > 0 && (
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 status-ok" />
          <span className="text-lg font-semibold tabular-nums">{currentHeartRate}</span>
          <span className="text-sm text-muted-foreground">BPM</span>
        </div>
      )}
    </div>
  )
}
