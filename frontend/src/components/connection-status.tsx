"use client"

import { useSignal } from "@/lib/signal-context"
import { Activity, WifiOff, Loader2, AlertCircle } from "lucide-react"

export function ConnectionStatus() {
  const { connectionStatus, esp32Ip, currentHeartRate } = useSignal()

  const statusConfig = {
    connecting: {
      icon: Loader2,
      label: "Connecting",
      color: "text-yellow-500",
      bgColor: "bg-yellow-500/10",
      animate: "animate-spin",
    },
    connected: {
      icon: Activity,
      label: "Connected",
      color: "text-emerald-500",
      bgColor: "bg-emerald-500/10",
      animate: "animate-pulse",
    },
    disconnected: {
      icon: WifiOff,
      label: "Disconnected",
      color: "text-muted-foreground",
      bgColor: "bg-muted",
      animate: "",
    },
    error: {
      icon: AlertCircle,
      label: "Error",
      color: "text-red-500",
      bgColor: "bg-red-500/10",
      animate: "",
    },
  }

  const config = statusConfig[connectionStatus]
  const Icon = config.icon

  return (
    <div className="flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className={`flex items-center gap-2 rounded-lg px-3 py-1.5 ${config.bgColor}`}>
          <Icon className={`h-4 w-4 ${config.color} ${config.animate}`} />
          <span className={`text-sm font-medium ${config.color}`}>{config.label}</span>
        </div>

        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>ESP32</span>
          <code className="rounded bg-muted px-2 py-0.5 text-xs font-mono">{esp32Ip}</code>
        </div>
      </div>

      {connectionStatus === "connected" && currentHeartRate > 0 && (
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-emerald-500" />
          <span className="text-lg font-semibold tabular-nums">{currentHeartRate}</span>
          <span className="text-sm text-muted-foreground">BPM</span>
        </div>
      )}
    </div>
  )
}
