"use client"

import { SignalProvider } from "@/lib/signal-context"
import { ConnectionStatus } from "@/components/connection-status"
import { SignalChart } from "@/components/signal-chart"
import { SignalControls } from "@/components/signal-controls"
import { SignalMetrics } from "@/components/signal-metrics"
import { useSignal } from "@/lib/signal-context"

function MonitorDashboard() {
  const { ppgBuffer, ecgBuffer } = useSignal()

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-3xl font-bold tracking-tight">Physiological Signal Monitor</h1>
          <p className="text-muted-foreground">
            Real-time PPG and ECG waveform visualization for cuffless blood pressure monitoring
          </p>
        </div>

        {/* Connection Status */}
        <div className="rounded-lg border bg-card p-4">
          <ConnectionStatus />
        </div>

        {/* Metrics */}
        <SignalMetrics />

        {/* PPG Chart */}
        <div className="rounded-lg border bg-card overflow-hidden">
          <div className="border-b bg-muted/30 px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">PPG Signal</h2>
                <p className="text-sm text-muted-foreground">Photoplethysmography • 250Hz sampling rate</p>
              </div>
              <div className="h-2 w-2 rounded-full bg-red-500 animate-pulse" />
            </div>
          </div>
          <div className="h-64 p-4">
            <SignalChart buffer={ppgBuffer} color="hsl(0, 84%, 60%)" label="PPG" yMin={-2} yMax={2} />
          </div>
        </div>

        {/* ECG Chart */}
        <div className="rounded-lg border bg-card overflow-hidden">
          <div className="border-b bg-muted/30 px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">ECG Signal</h2>
                <p className="text-sm text-muted-foreground">Electrocardiogram • 500Hz sampling rate</p>
              </div>
              <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
            </div>
          </div>
          <div className="h-64 p-4">
            <SignalChart buffer={ecgBuffer} color="hsl(217, 91%, 60%)" label="ECG" yMin={-3} yMax={3} />
          </div>
        </div>

        {/* Controls */}
        <div className="rounded-lg border bg-card p-4">
          <SignalControls />
        </div>
      </div>
    </div>
  )
}

export default function Page() {
  return (
    <SignalProvider>
      <MonitorDashboard />
    </SignalProvider>
  )
}
