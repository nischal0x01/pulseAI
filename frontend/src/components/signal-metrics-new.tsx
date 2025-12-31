"use client"

import type React from "react"

import { useSignal } from "@/lib/signal-context"
import { useEffect, useState } from "react"
import { Activity, Signal, Heart, TrendingUp } from "lucide-react"

export function SignalMetrics() {
  const { ppgBuffer, ecgBuffer, currentHeartRate, signalQuality } = useSignal()
  const [ppgMetrics, setPpgMetrics] = useState({ 
    amplitude: 0, 
    lastRRInterval: 0, 
    estimatedSBP: 0, 
    estimatedDBP: 0 
  })
  const [ecgMetrics, setEcgMetrics] = useState({ 
    amplitude: 0, 
    lastRRInterval: 0, 
    estimatedSBP: 0, 
    estimatedDBP: 0 
  })

  useEffect(() => {
    const interval = setInterval(() => {
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

  const getBPColor = (sbp: number, dbp: number) => {
    // Blood pressure classification colors
    if (sbp >= 140 || dbp >= 90) return "text-red-500" // High
    if (sbp >= 130 || dbp >= 80) return "text-yellow-500" // Elevated
    if (sbp < 90 || dbp < 60) return "text-blue-500" // Low
    return "text-emerald-500" // Normal
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <MetricCard
        icon={Heart}
        label="Blood Pressure"
        value={ppgMetrics.estimatedSBP > 0 ? `${ppgMetrics.estimatedSBP}/${ppgMetrics.estimatedDBP}` : "--/--"}
        unit="mmHg"
        color={getBPColor(ppgMetrics.estimatedSBP, ppgMetrics.estimatedDBP)}
      />

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

      <MetricCard
        icon={TrendingUp}
        label="Pulse Amplitude"
        value={ppgMetrics.amplitude > 0 ? ppgMetrics.amplitude.toFixed(2) : "--"}
        unit="V"
        color="text-blue-500"
      />
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

function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  color,
  progress,
  progressColor,
}: MetricCardProps) {
  return (
    <div className="relative overflow-hidden rounded-lg border bg-card p-6 shadow">
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-sm font-medium text-muted-foreground">{label}</p>
          <div className="flex items-baseline space-x-1">
            <p className={`text-2xl font-bold ${color}`}>{value}</p>
            <p className="text-sm text-muted-foreground">{unit}</p>
          </div>
        </div>
        {Icon && (
          <div className="rounded-full bg-muted/20 p-2">
            <Icon className={`h-4 w-4 ${color}`} />
          </div>
        )}
      </div>
      
      {progress !== undefined && progressColor && (
        <div className="mt-4">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Signal Strength</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="mt-1 h-2 w-full bg-muted/20 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-1000 ${progressColor}`}
              style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
