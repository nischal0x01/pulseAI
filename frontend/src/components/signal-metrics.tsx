"use client"

import type React from "react"

import { useSignal } from "@/lib/signal-context"
import { useEffect, useState } from "react"
import { Activity, Signal } from "lucide-react"

export function SignalMetrics() {
  const { ppgBuffer, ecgBuffer, currentHeartRate, signalQuality } = useSignal()
  const [ppgMetrics, setPpgMetrics] = useState({ estimatedSBP: 0, estimatedDBP: 0, lastRRInterval: 0 })
  const [ecgMetrics, setEcgMetrics] = useState({ estimatedSBP: 0, estimatedDBP: 0, lastRRInterval: 0 })

  useEffect(() => {
    const interval = setInterval(() => {
      // Both buffers return SignalMetrics with estimatedSBP/estimatedDBP now
      setPpgMetrics(ppgBuffer.calculateMetrics())
      setEcgMetrics(ecgBuffer.calculateMetrics())
    }, 1000)

    return () => clearInterval(interval)
  }, [ppgBuffer, ecgBuffer])

  const getQualityColor = (quality: number) => {
    if (quality >= 70) return "text-emerald-500"
    if (quality >= 40) return "text-yellow-500"
    return "text-red-500"
  }

  const getQualityBg = (quality: number) => {
    if (quality >= 70) return "bg-emerald-500"
    if (quality >= 40) return "bg-yellow-500"
    return "bg-red-500"
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard
        icon={Activity}
        label="Heart Rate"
        value={currentHeartRate > 0 ? `${currentHeartRate}` : "--"}
        unit="BPM"
        color="text-emerald-500"
      />

      <MetricCard
        icon={Signal}
        label="Signal Quality"
        value={signalQuality > 0 ? Math.round(signalQuality).toString() : "--"}
        unit="%"
        color={getQualityColor(signalQuality)}
        progress={signalQuality}
        progressColor={getQualityBg(signalQuality)}
      />

      {/* Estimated BP from PPG */}
      <MetricCard
        label="Estimated SBP (PPG)"
        value={ppgMetrics.estimatedSBP > 0 ? Math.round(ppgMetrics.estimatedSBP).toString() : "--"}
        unit="mmHg"
        color="text-red-400"
      />

      <MetricCard
        label="Estimated DBP (PPG)"
        value={ppgMetrics.estimatedDBP > 0 ? Math.round(ppgMetrics.estimatedDBP).toString() : "--"}
        unit="mmHg"
        color="text-red-400"
      />

      {/* Optional: show RR interval from ECG if desired */}
      {/*
      <MetricCard
        label="R-R Interval"
        value={ecgMetrics.lastRRInterval > 0 ? Math.round(ecgMetrics.lastRRInterval).toString() : "--"}
        unit="ms"
        color="text-blue-400"
      />
      */}
    </div>
  )
}

interface MetricCardProps {
  icon?: React.ComponentType<{ className?: string }>
  label: string
  value: string
  unit: string
  color: string
  progress?: number
  progressColor?: string
}

function MetricCard({ icon: Icon, label, value, unit, color, progress, progressColor }: MetricCardProps) {
  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-center gap-2 mb-2">
        {Icon && <Icon className={`h-4 w-4 ${color}`} />}
        <span className="text-sm text-muted-foreground">{label}</span>
      </div>

      <div className="flex items-baseline gap-2">
        <span className={`text-3xl font-bold tabular-nums ${color}`}>{value}</span>
        <span className="text-sm text-muted-foreground">{unit}</span>
      </div>

      {progress !== undefined && (
        <div className="mt-3 h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className={`h-full ${progressColor} transition-all duration-300`}
            style={{ width: `${Math.min(100, progress)}%` }}
          />
        </div>
      )}
    </div>
  )
}
