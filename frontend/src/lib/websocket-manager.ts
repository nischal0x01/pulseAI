export type SignalType = "PPG" | "ECG"

export interface SignalData {
  timestamp: number
  signalType: SignalType
  value: number
  quality: number
}

export interface SignalPacket {
  type: "signal_data"
  payload: {
    timestamp: number
    ppg_value?: number
    ecg_value?: number
    sample_rate: number
    quality: number
    heart_rate?: number
  }
}

export interface BPPrediction {
  type: "bp_prediction"
  payload: {
    sbp: number | null
    dbp: number | null
    timestamp: number
    confidence: number
    prediction_count: number
    error?: string
  }
}

export interface StatusMessage {
  type: "status"
  message: string
  timestamp: number
}

export interface ControlCommand {
  type: "control"
  command: "start_acquisition" | "stop_acquisition" | "calibrate"
  parameters?: Record<string, unknown>
}

export type WebSocketMessage = SignalPacket | BPPrediction | StatusMessage

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error"

type SignalCallback = (data: SignalData) => void
type StatusCallback = (status: ConnectionStatus) => void
type BPCallback = (prediction: BPPrediction['payload']) => void

// WebSocket configuration constants
const MAX_RECONNECT_ATTEMPTS = 5
const INITIAL_RECONNECT_DELAY_MS = 1000
const CONNECTION_TIMEOUT_MS = 10000
const HEARTBEAT_INTERVAL_MS = 30000
const HEARTBEAT_TIMEOUT_MS = 35000

export class WebSocketManager {
  private ws: WebSocket | null = null
  private reconnectTimer: NodeJS.Timeout | null = null
  private connectionTimer: NodeJS.Timeout | null = null
  private reconnectAttempts = 0
  private readonly maxReconnectAttempts = MAX_RECONNECT_ATTEMPTS
  private readonly reconnectDelay = INITIAL_RECONNECT_DELAY_MS
  private readonly connectionTimeout = CONNECTION_TIMEOUT_MS
  private heartbeatInterval: NodeJS.Timeout | null = null
  private lastHeartbeat = 0

  private signalCallbacks: SignalCallback[] = []
  private statusCallbacks: StatusCallback[] = []
  private bpCallbacks: BPCallback[] = []

  constructor(private url: string) {}

  private getReadyStateText(readyState: number | undefined): string {
    switch (readyState) {
      case WebSocket.CONNECTING:
        return "CONNECTING"
      case WebSocket.OPEN:
        return "OPEN"
      case WebSocket.CLOSING:
        return "CLOSING"
      case WebSocket.CLOSED:
        return "CLOSED"
      default:
        return "UNKNOWN"
    }
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log("[v0] WebSocket already connected")
      return
    }

    console.log("[v0] Attempting to connect to:", this.url)
    this.updateStatus("connecting")

    // Set connection timeout
    this.connectionTimer = setTimeout(() => {
      if (this.ws?.readyState === WebSocket.CONNECTING) {
        console.error("[v0] Connection timeout")
        this.ws.close()
        this.updateStatus("error")
      }
    }, this.connectionTimeout)

    try {
      this.ws = new WebSocket(this.url)

      this.ws.onopen = () => {
        console.log("[v0] WebSocket connected to ESP32")
        this.clearConnectionTimer()
        this.reconnectAttempts = 0
        this.updateStatus("connected")
        this.startHeartbeat()
      }

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage
          
          // Handle different message types
          if (message.type === "signal_data") {
            this.handleSignalPacket(message)
            this.lastHeartbeat = Date.now()
          } else if (message.type === "bp_prediction") {
            this.handleBPPrediction(message)
            this.lastHeartbeat = Date.now()
          } else if (message.type === "status") {
            console.log("[v0] Status message:", message.message)
          } else {
            console.warn("[v0] Unknown message type:", message)
          }
        } catch (error) {
          console.error("[v0] Error parsing WebSocket message:", error, event.data)
        }
      }

      this.ws.onerror = (error) => {
        // Only log meaningful errors, suppress connection attempts during startup
        if (this.ws?.readyState === WebSocket.CONNECTING) {
          console.log("[v0] Attempting to connect to bridge server...")
          return
        }
        
        console.warn("[v0] WebSocket error:", {
          message: 'Connection issue - check if bridge server is running',
          url: this.url,
          readyState: this.getReadyStateText(this.ws?.readyState)
        })
        this.updateStatus("error")
      }

      this.ws.onclose = (event) => {
        console.log("[v0] WebSocket disconnected:", {
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
          url: this.url,
          timestamp: new Date().toISOString()
        })
        this.clearConnectionTimer()
        this.updateStatus("disconnected")
        this.stopHeartbeat()
        this.attemptReconnect()
      }
    } catch (error) {
      console.error("[v0] Error creating WebSocket:", {
        error,
        url: this.url,
        timestamp: new Date().toISOString()
      })
      this.updateStatus("error")
    }
  }

  disconnect(): void {
    this.stopReconnect()
    this.stopHeartbeat()
    this.clearConnectionTimer()

    if (this.ws) {
      this.ws.close()
      this.ws = null
    }

    this.updateStatus("disconnected")
  }

  sendCommand(command: ControlCommand): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(command))
    } else {
      console.warn("Cannot send command: WebSocket not connected")
    }
  }

  onSignal(callback: SignalCallback): () => void {
    this.signalCallbacks.push(callback)
    return () => {
      this.signalCallbacks = this.signalCallbacks.filter((cb) => cb !== callback)
    }
  }

  onStatusChange(callback: StatusCallback): () => void {
    this.statusCallbacks.push(callback)
    return () => {
      this.statusCallbacks = this.statusCallbacks.filter((cb) => cb !== callback)
    }
  }

  onBPPrediction(callback: BPCallback): () => void {
    this.bpCallbacks.push(callback)
    return () => {
      this.bpCallbacks = this.bpCallbacks.filter((cb) => cb !== callback)
    }
  }

  private handleSignalPacket(packet: SignalPacket): void {
    if (!packet.payload) {
      console.warn("[v0] Signal packet missing payload:", packet)
      return
    }

    const { timestamp, ppg_value, ecg_value, quality, heart_rate } = packet.payload

    if (ppg_value !== undefined) {
      this.emitSignal({
        timestamp,
        signalType: "PPG",
        value: ppg_value,
        quality,
        ...(heart_rate && { heart_rate })
      } as any)
    }

    if (ecg_value !== undefined) {
      this.emitSignal({
        timestamp,
        signalType: "ECG", 
        value: ecg_value,
        quality,
      })
    }
  }

  private emitSignal(data: SignalData): void {
    this.signalCallbacks.forEach((callback) => callback(data))
  }

  private handleBPPrediction(prediction: BPPrediction): void {
    if (!prediction.payload) {
      console.warn("[v0] BP prediction missing payload:", prediction)
      return
    }
    
    // Check for error state (low signal quality)
    if (prediction.payload.error) {
      console.warn("[v0] BP prediction error:", prediction.payload.error)
      console.warn("[v0] Signal quality too low:", prediction.payload.confidence)
      // Still emit the prediction so UI can show the error state
      this.bpCallbacks.forEach((callback) => callback(prediction.payload))
      return
    }
    
    console.log("[v0] BP Prediction received:", prediction.payload)
    this.bpCallbacks.forEach((callback) => callback(prediction.payload))
  }

  private updateStatus(status: ConnectionStatus): void {
    this.statusCallbacks.forEach((callback) => callback(status))
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log("  Max reconnection attempts reached")
      return
    }

    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts)
    this.reconnectAttempts++

    console.log(`  Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)

    this.reconnectTimer = setTimeout(() => {
      this.connect()
    }, delay)
  }

  private stopReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.reconnectAttempts = 0
  }

  private clearConnectionTimer(): void {
    if (this.connectionTimer) {
      clearTimeout(this.connectionTimer)
      this.connectionTimer = null
    }
  }

  private startHeartbeat(): void {
    this.lastHeartbeat = Date.now()
    this.heartbeatInterval = setInterval(() => {
      const timeSinceLastHeartbeat = Date.now() - this.lastHeartbeat
      if (timeSinceLastHeartbeat > 60000) {  // 60 seconds
        console.warn("  No heartbeat received for 60s, connection may be stale")
        // Don't disconnect, just warn - let the WebSocket handle actual disconnects
      }
    }, 10000)  // Check every 10 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }
}
