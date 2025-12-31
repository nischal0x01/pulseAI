"use client"

import { SignalProvider } from "@/lib/signal-context"
import { ConnectionStatus } from "@/components/connection-status"
import { SignalChart } from "@/components/signal-chart"
import { SignalControls } from "@/components/signal-controls"
import { SignalMetrics } from "@/components/signal-metrics"
import { ProjectOverviewSection } from "@/components/project-overview-section"
import { TerminologiesSection } from "@/components/terminologies-section"
import { AboutSection } from "@/components/about-section"
import { useSignal } from "@/lib/signal-context"

function MonitorDashboard() {
  const { ppgBuffer, ecgBuffer } = useSignal()

  return (
    <div className="min-h-screen bg-background">
      {/* Home Section */}
      <section id="home" className="pt-24 pb-12 px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl space-y-8">
          {/* Header */}
          <div className="text-center space-y-4 mb-8">
            <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
              <span className="bg-gradient-to-r from-blue-600 via-purple-600 to-blue-600 bg-clip-text text-transparent">
                Pulse
              </span>
              <span className="bg-gradient-to-r from-pink-600 via-red-500 to-orange-500 bg-clip-text text-transparent">
                AI
              </span>
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
              Real-time cuffless blood pressure monitoring using PPG and ECG signals
            </p>
            <p className="text-base text-muted-foreground">
              Advanced machine learning for continuous, non-invasive blood pressure estimation
            </p>
          </div>

          {/* Connection Status */}
          <div className="rounded-lg border bg-card p-4 shadow-sm">
            <ConnectionStatus />
          </div>

          {/* Metrics */}
          <SignalMetrics />

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* PPG Chart */}
            <div className="rounded-lg border bg-card overflow-hidden shadow-sm">
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
            <div className="rounded-lg border bg-card overflow-hidden shadow-sm">
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
          </div>

          {/* Controls */}
          <div className="rounded-lg border bg-card p-4 shadow-sm">
            <SignalControls />
          </div>
        </div>
      </section>

      {/* Project Overview Section */}
      <ProjectOverviewSection />

      {/* Terminologies Section */}
      <TerminologiesSection />

      {/* About Section (includes Team) */}
      <AboutSection />
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
