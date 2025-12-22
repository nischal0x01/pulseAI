"use client"

import { useEffect, useRef, useState } from "react"
import type { SignalBuffer } from "@/lib/signal-buffer"

interface SignalChartProps {
  buffer: SignalBuffer
  color: string
  label: string
  yMin: number
  yMax: number
}

export function SignalChart({ buffer, color, label, yMin, yMax }: SignalChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 })

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

  // Render chart
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const render = () => {
      const data = buffer.getWindowedData(10000) // 10 second window

      // Clear canvas
      ctx.fillStyle = "hsl(var(--background))"
      ctx.fillRect(0, 0, dimensions.width, dimensions.height)

      if (data.length === 0) {
        // Show "No Data" message
        ctx.fillStyle = color
        ctx.font = "16px Arial"
        ctx.textAlign = "center"
        ctx.fillText("No Signal Data", dimensions.width / 2, dimensions.height / 2)
        animationRef.current = requestAnimationFrame(render)
        return
      }

      // Draw grid
      ctx.strokeStyle = "hsl(var(--border))"
      ctx.lineWidth = 1
      ctx.setLineDash([2, 4])

      // Horizontal grid lines
      for (let i = 0; i <= 4; i++) {
        const y = (dimensions.height / 4) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(dimensions.width, y)
        ctx.stroke()
      }

      // Vertical grid lines (every second)
      for (let i = 0; i <= 10; i++) {
        const x = (dimensions.width / 10) * i
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, dimensions.height)
        ctx.stroke()
      }

      ctx.setLineDash([])

      // Draw signal
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()

      if (data.length > 1) {
        // Safety check for canvas dimensions
        if (dimensions.width === 0 || dimensions.height === 0) {
          animationRef.current = requestAnimationFrame(render)
          return
        }

        // Use a rolling 10-second window for consistent X-axis scaling
        const now = Date.now()
        const windowStart = now - 10000 // 10 seconds ago
        const windowEnd = now
        const timeRange = 10000 // Fixed 10 second window
        const valueRange = yMax - yMin

        // Filter data to only include points within our time window
        const filteredData = data.filter(point => 
          point.timestamp >= windowStart && point.timestamp <= windowEnd
        )

        // Debug logging - show every 60th frame (roughly once per second)
        if (Math.random() < 0.016) { 
          console.log(`[${label}] Total points: ${data.length}, Filtered points: ${filteredData.length}`)
          console.log(`[${label}] Value range in data: ${Math.min(...data.map(d => d.value)).toFixed(3)} to ${Math.max(...data.map(d => d.value)).toFixed(3)}`)
          console.log(`[${label}] Expected Y range: ${yMin} to ${yMax}`)
          console.log(`[${label}] Canvas dimensions: ${dimensions.width}x${dimensions.height}`)
          console.log(`[${label}] Time window: ${windowStart} to ${windowEnd} (${timeRange}ms)`)
          
          // Show first few coordinates with new calculation
          if (filteredData.length > 0) {
            const samplePoint = filteredData[0]
            const sampleX = ((samplePoint.timestamp - windowStart) / timeRange) * dimensions.width
            const sampleY = dimensions.height - ((samplePoint.value - yMin) / valueRange) * dimensions.height
            console.log(`[${label}] Sample coordinate: (${sampleX.toFixed(1)}, ${sampleY.toFixed(1)}) [timestamp: ${samplePoint.timestamp}]`)
          }
        }

        filteredData.forEach((point, index) => {
          const x = ((point.timestamp - windowStart) / timeRange) * dimensions.width
          const y = dimensions.height - ((point.value - yMin) / valueRange) * dimensions.height

          if (index === 0) {
            ctx.moveTo(x, y)
          } else {
            ctx.lineTo(x, y)
          }
        })

        ctx.stroke()

        // Reset stroke style
        ctx.strokeStyle = color

        // Draw peaks
        const peaks = buffer.getPeaks()
        ctx.fillStyle = color

        peaks.forEach((peakTime) => {
          const peakData = filteredData.find((d) => d.timestamp === peakTime)
          if (peakData) {
            const x = ((peakData.timestamp - windowStart) / timeRange) * dimensions.width
            const y = dimensions.height - ((peakData.value - yMin) / valueRange) * dimensions.height

            ctx.beginPath()
            ctx.arc(x, y, 4, 0, 2 * Math.PI)
            ctx.fill()
          }
        })
      }

      animationRef.current = requestAnimationFrame(render)
    }

    animationRef.current = requestAnimationFrame(render)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [buffer, dimensions, color, yMin, yMax])

  return (
    <div className="relative h-full w-full">
      <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="absolute inset-0" />
    </div>
  )
}
