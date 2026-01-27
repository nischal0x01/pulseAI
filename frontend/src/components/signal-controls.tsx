"use client"

import { useSignal } from "@/lib/signal-context"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Play, Square, Zap, Download } from "lucide-react"
import { useState } from "react"

export function SignalControls() {
  const {
    connectionStatus,
    isAcquiring,
    esp32Ip,
    connect,
    disconnect,
    startAcquisition,
    stopAcquisition,
    calibrate,
    setEsp32Ip,
    ppgBuffer,
    ecgBuffer,
  } = useSignal()

  const [ipInput, setIpInput] = useState(esp32Ip)

  const handleConnect = () => {
    if (connectionStatus === "connected") {
      disconnect()
    } else {
      setEsp32Ip(ipInput)
      connect()
    }
  }

  const handleExport = () => {
    const ppgData = ppgBuffer.getBuffer()
    const ecgData = ecgBuffer.getBuffer()

    const exportData = {
      timestamp: new Date().toISOString(),
      ppg: ppgData,
      ecg: ecgData,
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `signal-data-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-wrap items-end gap-4">
      <div className="flex-1 min-w-[200px]">
        <Label htmlFor="esp32-ip" className="text-sm font-medium">
          ESP32 IP Address
        </Label>
        <Input
          id="esp32-ip"
          value={ipInput}
          onChange={(e) => setIpInput(e.target.value)}
          placeholder="192.168.1.100:8080"
          disabled={connectionStatus === "connected"}
          className="mt-1.5"
        />
      </div>

      <Button onClick={handleConnect} variant={connectionStatus === "connected" ? "secondary" : "default"}>
        {connectionStatus === "connected" ? "Disconnect" : "Connect"}
      </Button>

      <div className="h-8 w-px bg-border" />

      <Button onClick={startAcquisition} disabled={connectionStatus !== "connected" || isAcquiring} variant="default">
        <Play className="h-4 w-4 mr-2" />
        Start
      </Button>

      <Button onClick={stopAcquisition} disabled={connectionStatus !== "connected" || !isAcquiring} variant="secondary">
        <Square className="h-4 w-4 mr-2" />
        Stop
      </Button>

      <Button onClick={calibrate} disabled={connectionStatus !== "connected"} variant="outline">
        <Zap className="h-4 w-4 mr-2" />
        Calibrate
      </Button>

      <Button onClick={handleExport} variant="outline">
        <Download className="h-4 w-4 mr-2" />
        Export
      </Button>
    </div>
  )
}
