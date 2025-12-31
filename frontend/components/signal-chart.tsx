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
  const dimensionsRef = useRef(dimensions)

  // Keep ref in sync with state
  useEffect(() => {
    dimensionsRef.current = dimensions
  }, [dimensions])

  // Handle canvas resize
  useEffect(() => {
    const updateDimensions = () => {
      if (canvasRef.current?.parentElement) {
        const { width, height } = canvasRef.current.parentElement.getBoundingClientRect()
        setDimensions({
          width: Math.max(1, width),
          height: Math.max(1, height),
        })
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
      const { width, height } = dimensionsRef.current
      // Resolve CSS color variables from the document so canvas gets actual colors
      const rootStyle = getComputedStyle(document.documentElement)
      const bgVar = rootStyle.getPropertyValue("--color-background") || rootStyle.getPropertyValue("--background")
      const borderVar = rootStyle.getPropertyValue("--color-border") || rootStyle.getPropertyValue("--border")
      const bgColor = bgVar ? bgVar.trim() : "#ffffff"
      const borderColor = borderVar ? borderVar.trim() : "#e5e7eb"

      // Handle device pixel ratio for crisp rendering
      const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1
      const displayWidth = Math.max(1, width)
      const displayHeight = Math.max(1, height)
      const resWidth = Math.max(1, Math.floor(displayWidth * dpr))
      const resHeight = Math.max(1, Math.floor(displayHeight * dpr))

      // Resize backing store if needed and scale context to device pixels
      if (canvas.width !== resWidth || canvas.height !== resHeight) {
        canvas.width = resWidth
        canvas.height = resHeight
        canvas.style.width = `${displayWidth}px`
        canvas.style.height = `${displayHeight}px`
        ctx.setTransform(1, 0, 0, 1, 0, 0)
        ctx.scale(dpr, dpr)
      }
      const data = buffer.getWindowedData(10000) // 10 second logical window based on buffer timestamps

      // If we don't have a valid canvas size yet, skip drawing this frame
      if (width <= 0 || height <= 0) {
        animationRef.current = requestAnimationFrame(render)
        return
      }

      // Clear canvas using resolved CSS variable
      ctx.fillStyle = bgColor
      ctx.fillRect(0, 0, displayWidth, displayHeight)

      if (data.length === 0) {
        ctx.fillStyle = color
        ctx.font = "16px Arial"
        ctx.textAlign = "center"
        ctx.fillText("No Signal Data", width / 2, height / 2)
        animationRef.current = requestAnimationFrame(render)
        return
      }

      // Draw grid
      ctx.strokeStyle = borderColor
      ctx.lineWidth = 1
      ctx.setLineDash([2, 4])

      for (let i = 0; i <= 4; i++) {
        const y = (height / 4) * i
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(width, y)
        ctx.stroke()
      }

      for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, height)
        ctx.stroke()
      }

      ctx.setLineDash([])

      // Draw signal
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.beginPath()

      // Derive window from actual data timestamps
      const firstTs = data[0].timestamp
      const lastTs = data[data.length - 1].timestamp
      const timeRange = Math.max(1, lastTs - firstTs)

      // Auto-scale Y based on data, but respect provided yMin/yMax as soft bounds
      const values = data.map((p) => p.value)
      const dataMin = Math.min(...values)
      const dataMax = Math.max(...values)
      const effectiveMin = Math.min(yMin, dataMin)
      const effectiveMax = Math.max(yMax, dataMax)
      const valueRange = Math.max(1e-6, effectiveMax - effectiveMin)

      if (Math.random() < 0.02) {
        console.log(
          `[${label}] size=${width}x${height}, points=${data.length}, t=[${firstTs}, ${lastTs}], v=[${dataMin}, ${dataMax}]`,
        )
      }

      data.forEach((point, index) => {
        const x = ((point.timestamp - firstTs) / timeRange) * displayWidth
        const y = displayHeight - ((point.value - effectiveMin) / valueRange) * displayHeight

        if (index === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }
      })

      ctx.stroke()

      // Draw peaks (if timestamps are comparable)
      const peaks = buffer.getPeaks()
      ctx.fillStyle = color
      peaks.forEach((peakTime) => {
        const peakData = data.find((d) => d.timestamp === peakTime)
        if (peakData) {
          const x = ((peakData.timestamp - firstTs) / timeRange) * displayWidth
          const y = displayHeight - ((peakData.value - effectiveMin) / valueRange) * displayHeight
          ctx.beginPath()
          ctx.arc(x, y, 4, 0, 2 * Math.PI)
          ctx.fill()
        }
      })

      animationRef.current = requestAnimationFrame(render)
    }

    animationRef.current = requestAnimationFrame(render)

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [buffer, color, yMin, yMax])

  return (
    <div className="relative h-full w-full">
      <canvas ref={canvasRef} width={dimensions.width} height={dimensions.height} className="absolute inset-0" />
    </div>
  )
}
