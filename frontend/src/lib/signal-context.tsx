"use client"

import type React from "react"
import { createContext, useContext, useEffect, useState, useCallback } from "react"
import { WebSocketManager, type ConnectionStatus, type SignalData } from "./websocket-manager"
import { SignalBuffer } from "./signal-buffer"

interface SignalContextType {
  connectionStatus: ConnectionStatus
  connectionError: string | null
  ppgBuffer: SignalBuffer
  ecgBuffer: SignalBuffer
  isAcquiring: boolean
  currentHeartRate: number
  signalQuality: number
  esp32Ip: string
  connect: () => void
  disconnect: () => void
  startAcquisition: () => void
  stopAcquisition: () => void
  calibrate: () => void
  setEsp32Ip: (ip: string) => void
  clearError: () => void
}

const SignalContext = createContext<SignalContextType | undefined>(undefined)

export function useSignal() {
  const context = useContext(SignalContext)
  if (!context) {
    throw new Error("useSignal must be used within SignalProvider")
  }
  return context
}

export function SignalProvider({ children }: { children: React.ReactNode }) {
  const [esp32Ip, setEsp32Ip] = useState("localhost:8080")
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>("disconnected")
  const [connectionError, setConnectionError] = useState<string | null>(null)
  const [isAcquiring, setIsAcquiring] = useState(false)
  const [currentHeartRate, setCurrentHeartRate] = useState(0)
  const [signalQuality, setSignalQuality] = useState(0)
  const [wsManager, setWsManager] = useState<WebSocketManager | null>(null)

  // Initialize buffers
  const [ppgBuffer] = useState(
    () =>
      new SignalBuffer({
        maxSize: 2500, // 10 seconds at 250Hz
        samplingRate: 250,
        windowDuration: 10000,
      }),
  )

  const [ecgBuffer] = useState(
    () =>
      new SignalBuffer({
        maxSize: 5000, // 10 seconds at 500Hz
        samplingRate: 500,
        windowDuration: 10000,
      }),
  )

  // Create WebSocket manager when IP changes
  useEffect(() => {
    const manager = new WebSocketManager(`ws://${esp32Ip}/signals`)
    setWsManager(manager)

    return () => {
      manager.disconnect()
    }
  }, [esp32Ip])

  // Setup WebSocket listeners
  useEffect(() => {
    if (!wsManager) return

    const unsubSignal = wsManager.onSignal((data: SignalData) => {
      if (data.signalType === "PPG") {
        ppgBuffer.addSample(data.timestamp, data.value)
      } else if (data.signalType === "ECG") {
        ecgBuffer.addSample(data.timestamp, data.value)
      }
    })

    const unsubStatus = wsManager.onStatusChange((status: ConnectionStatus) => {
      setConnectionStatus(status)
      
      // Set error messages based on status
      if (status === "error") {
        setConnectionError(`Failed to connect to ESP32/Server at ${esp32Ip}. 
        
For development: Run the mock server with 'python mock_esp32_server.py'
For production: Check the ESP32 device connection and IP address.`)
      } else if (status === "connected") {
        setConnectionError(null) // Clear error on successful connection
      }
    })

    return () => {
      unsubSignal()
      unsubStatus()
    }
  }, [wsManager, ppgBuffer, ecgBuffer])

  // Update metrics periodically
  useEffect(() => {
    const interval = setInterval(() => {
      const ppgMetrics = ppgBuffer.calculateMetrics()
      setCurrentHeartRate(ppgMetrics.heartRate)
      setSignalQuality(ppgMetrics.signalQuality)
    }, 1000)

    return () => clearInterval(interval)
  }, [ppgBuffer])

  const connect = useCallback(() => {
    wsManager?.connect()
  }, [wsManager])

  const disconnect = useCallback(() => {
    wsManager?.disconnect()
    setIsAcquiring(false)
  }, [wsManager])

  const startAcquisition = useCallback(() => {
    wsManager?.sendCommand({ type: "control", command: "start_acquisition" })
    setIsAcquiring(true)
  }, [wsManager])

  const stopAcquisition = useCallback(() => {
    wsManager?.sendCommand({ type: "control", command: "stop_acquisition" })
    setIsAcquiring(false)
  }, [wsManager])

  const calibrate = useCallback(() => {
    wsManager?.sendCommand({ type: "control", command: "calibrate" })
  }, [wsManager])

  const clearError = useCallback(() => {
    setConnectionError(null)
  }, [])

  return (
    <SignalContext.Provider
      value={{
        connectionStatus,
        connectionError,
        ppgBuffer,
        ecgBuffer,
        isAcquiring,
        currentHeartRate,
        signalQuality,
        esp32Ip,
        connect,
        disconnect,
        startAcquisition,
        stopAcquisition,
        calibrate,
        setEsp32Ip,
        clearError,
      }}
    >
      {children}
    </SignalContext.Provider>
  )
}
