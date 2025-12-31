"use client"

import { useEffect, useRef, useState } from "react"
import type { SignalBuffer } from "@/lib/signal-buffer"

interface BloodPressureChartProps {
  buffer: SignalBuffer
  label: string
}

interface BPReading {
  timestamp: number
  systolic: number
  diastolic: number
}

export function BloodPressureChart({ buffer, label }: BloodPressureChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | undefined>(undefined)
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })
  const [bpReadings, setBpReadings] = useState<BPReading[]>([])

  // Handle canvas resize
  useEffect(() => {
    const updateDimensions = () => {
      if (canvasRef.current?.parentElement) {
        const { width, height } = canvasRef.current.parentElement.getBoundingClientRect()
        setDimensions({ width, height })
      }
    }

    updateDimensions()
    window.addEventListener("resize", updateDimensions)
    return () => window.removeEventListener("resize", updateDimensions)
  }, [])

  // Update blood pressure readings
  useEffect(() => {
    const updateInterval = setInterval(() => {
      const metrics = buffer.calculateMetrics()
      if (metrics.estimatedSBP > 0 && metrics.estimatedDBP > 0) {
        const newReading: BPReading = {
          timestamp: Date.now(),
          systolic: metrics.estimatedSBP,
          diastolic: metrics.estimatedDBP,
        }

        setBpReadings(prev => {
          const updated = [...prev, newReading]
          // Keep only last 60 readings (1 minute at 1 reading/second)
          return updated.slice(-60)
        })
      }
    }, 1000) // Update every second

    return () => clearInterval(updateInterval)
  }, [buffer])

  // Render chart
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const render = () => {
      // Clear canvas
      ctx.fillStyle = "hsl(var(--background))"
      ctx.fillRect(0, 0, dimensions.width, dimensions.height)

      if (bpReadings.length === 0) {
        // Show "No Data" message
        ctx.fillStyle = "hsl(var(--foreground))"
        ctx.font = "16px Arial"
        ctx.textAlign = "center"
        ctx.fillText("No Blood Pressure Data", dimensions.width / 2, dimensions.height / 2)
        animationRef.current = requestAnimationFrame(render)
        return
      }

      // Safety check for canvas dimensions
      if (dimensions.width === 0 || dimensions.height === 0) {
        animationRef.current = requestAnimationFrame(render)
        return
      }

      // Draw grid
      ctx.strokeStyle = "hsl(var(--border))"
      ctx.lineWidth = 1
      ctx.setLineDash([2, 4])

      // Horizontal grid lines (BP ranges)
      const bpRanges = [60, 80, 100, 120, 140, 160, 180]
      bpRanges.forEach(bp => {
        const y = dimensions.height - ((bp - 50) / 130) * dimensions.height
        if (y >= 0 && y <= dimensions.height) {
          ctx.beginPath()
          ctx.moveTo(0, y)
          ctx.lineTo(dimensions.width, y)
          ctx.stroke()
          
          // Label
          ctx.fillStyle = "hsl(var(--muted-foreground))"
          ctx.font = "12px Arial"
          ctx.textAlign = "left"
          ctx.fillText(`${bp}`, 5, y - 2)
        }
      })

      // Vertical grid lines (time)
      const timeGridLines = 6
      for (let i = 0; i <= timeGridLines; i++) {
        const x = (dimensions.width / timeGridLines) * i
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, dimensions.height)
        ctx.stroke()
      }

      ctx.setLineDash([])

      // Use a rolling time window for consistent X-axis scaling
      const now = Date.now()
      const windowStart = now - 60000 // 60 seconds ago
      const windowEnd = now
      const timeRange = 60000

      // Filter data to only include points within our time window
      const filteredReadings = bpReadings.filter(reading => 
        reading.timestamp >= windowStart && reading.timestamp <= windowEnd
      )

      if (filteredReadings.length > 1) {
        // Draw systolic line (red)
        ctx.strokeStyle = "hsl(0, 84%, 60%)"
        ctx.lineWidth = 3
        ctx.beginPath()

        filteredReadings.forEach((reading, index) => {
          const x = ((reading.timestamp - windowStart) / timeRange) * dimensions.width
          const y = dimensions.height - ((reading.systolic - 50) / 130) * dimensions.height

          if (index === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        })

        ctx.stroke()

        // Draw diastolic line (blue)
        ctx.strokeStyle = "hsl(217, 91%, 60%)"
        ctx.lineWidth = 3
        ctx.beginPath()

        filteredReadings.forEach((reading, index) => {
          const x = ((reading.timestamp - windowStart) / timeRange) * dimensions.width
          const y = dimensions.height - ((reading.diastolic - 50) / 130) * dimensions.height

          if (index === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        })

        ctx.stroke()

        // Draw current values as points
        const latest = filteredReadings[filteredReadings.length - 1]
        if (latest) {
          const latestX = ((latest.timestamp - windowStart) / timeRange) * dimensions.width
          
          // Systolic point
          const sysY = dimensions.height - ((latest.systolic - 50) / 130) * dimensions.height
          ctx.fillStyle = "hsl(0, 84%, 60%)"
          ctx.beginPath()
          ctx.arc(latestX, sysY, 6, 0, 2 * Math.PI)
          ctx.fill()

          // Diastolic point
          const diaY = dimensions.height - ((latest.diastolic - 50) / 130) * dimensions.height
          ctx.fillStyle = "hsl(217, 91%, 60%)"
          ctx.beginPath()
          ctx.arc(latestX, diaY, 6, 0, 2 * Math.PI)
          ctx.fill()
        }
      }

      // Draw current reading text
      const currentMetrics = buffer.calculateMetrics()
      if (currentMetrics.estimatedSBP > 0 && currentMetrics.estimatedDBP > 0) {
        ctx.fillStyle = "hsl(var(--foreground))"
        ctx.font = "24px Arial"
        ctx.textAlign = "right"
        ctx.fillText(
          `${currentMetrics.estimatedSBP}/${currentMetrics.estimatedDBP} mmHg`,
          dimensions.width - 10,
          40
        )
      }

      animationRef.current = requestAnimationFrame(render)
    }

    animationRef.current = requestAnimationFrame(render)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [buffer, dimensions, bpReadings])

  return (
    <div className="relative h-full w-full">
      <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="absolute inset-0" />
      
      {/* Legend */}
      <div className="absolute top-2 left-2 space-y-1">
        <div className="flex items-center gap-2 text-sm">
          <div className="w-4 h-1 bg-red-500 rounded"></div>
          <span className="text-muted-foreground">Systolic</span>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <div className="w-4 h-1 bg-blue-500 rounded"></div>
          <span className="text-muted-foreground">Diastolic</span>
        </div>
      </div>
    </div>
  )
}
