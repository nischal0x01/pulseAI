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
    <div className="min-h-screen bg-gradient-to-b from-background via-background to-muted/20">
      {/* Home Section */}
      <section id="home" className="pt-32 pb-16 px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-7xl space-y-10">
          
          {/* Header with Modified Branding */}
          <div className="text-center space-y-6 mb-4 animate-fade-in-up">
            <h1 className="text-6xl md:text-7xl lg:text-8xl font-extrabold font-display tracking-tighter">
              <span className="bg-gradient-to-r from-black via-green-700 to-green-700 bg-clip-text text-transparent pr-2">Pulse</span>
              <span className="bg-gradient-to-r from-green-700 via-green-700 to-black bg-clip-text text-transparent ml-2">
                AI
              </span>
            </h1>
            <p className="text-2xl md:text-2xl text-muted-foreground max-w-4xl mx-auto font-medium leading-relaxed font-ui">
              Real-time cuffless blood pressure monitoring using PPG and ECG signals
            </p>
          </div>

          {/* --- CENTERED ANIMATED ECG PATTERN --- */}
          <div className="flex justify-center items-center h-24 w-full overflow-hidden relative">
            <svg width="400" height="100" viewBox="0 0 400 100" className="drop-shadow-[0_0_8px_rgba(220,38,38,0.5)]">
              <path
                d="M0,50 L150,50 L160,30 L175,70 L190,50 L210,50 L220,10 L235,90 L250,50 L400,50"
                fill="none"
                stroke="#dc2626"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="ecg-line"
              />
            </svg>
            <style jsx>{`
              .ecg-line {
                stroke-dasharray: 1000;
                stroke-dashoffset: 1000;
                animation: draw 3s linear infinite;
              }
              @keyframes draw {
                to {
                  stroke-dashoffset: 0;
                }
              }
            `}</style>
          </div>

          <p className="text-lg md:text-xl text-muted-foreground/80 text-center max-w-3xl mx-auto font-ui">
            Advanced machine learning for continuous, non-invasive blood pressure estimation
          </p>

          {/* Connection Status */}
          <div className="rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm p-6 shadow-xl shadow-primary/5 hover:shadow-2xl hover:shadow-primary/10 transition-all duration-500 animate-fade-in-up">
            <ConnectionStatus />
          </div>

          {/* Metrics */}
          <div className="animate-fade-in-up delay-200">
            <SignalMetrics />
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* PPG Chart */}
            <div className="rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden shadow-xl shadow-primary/5 hover:shadow-2xl hover:shadow-primary/10 transition-all duration-500 animate-slide-in-left">
              <div className="border-b border-border/50 bg-gradient-to-r from-muted/50 to-muted/30 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold font-display">PPG Signal</h2>
                    <p className="text-sm text-muted-foreground mt-1">Photoplethysmography • 250Hz sampling rate</p>
                  </div>
                  <div className="h-3 w-3 rounded-full bg-red-600 animate-pulse" />
                </div>
              </div>
              <div className="h-72 p-6">
                <SignalChart buffer={ppgBuffer} color="#dc2626" label="PPG" yMin={-2} yMax={2} />
              </div>
            </div>

            {/* ECG Chart */}
            <div className="rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm overflow-hidden shadow-xl shadow-primary/5 hover:shadow-2xl hover:shadow-primary/10 transition-all duration-500 animate-slide-in-right">
              <div className="border-b border-border/50 bg-gradient-to-r from-muted/50 to-muted/30 px-6 py-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-bold font-display">ECG Signal</h2>
                    <p className="text-sm text-muted-foreground mt-1">Electrocardiogram • 500Hz sampling rate</p>
                  </div>
                  <div className="h-3 w-3 rounded-full bg-blue-600 animate-pulse" />
                </div>
              </div>
              <div className="h-72 p-6">
                <SignalChart buffer={ecgBuffer} color="#2563eb" label="ECG" yMin={-3} yMax={3} />
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="rounded-2xl border border-border/50 bg-card/50 backdrop-blur-sm p-6 shadow-xl shadow-primary/5 hover:shadow-2xl hover:shadow-primary/10 transition-all duration-500 animate-fade-in-up delay-400">
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