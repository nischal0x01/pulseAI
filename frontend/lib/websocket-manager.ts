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

export type WebSocketMessage = SignalPacket | StatusMessage

export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "error"

type SignalCallback = (data: SignalData) => void
type StatusCallback = (status: ConnectionStatus) => void

export class WebSocketManager {
  private ws: WebSocket | null = null
  private reconnectTimer: NodeJS.Timeout | null = null
  private connectionTimer: NodeJS.Timeout | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private connectionTimeout = 10000 // 10 seconds
  private heartbeatInterval: NodeJS.Timeout | null = null
  private lastHeartbeat = 0

  private signalCallbacks: SignalCallback[] = []
  private statusCallbacks: StatusCallback[] = []

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
        console.error("[v0] WebSocket error:", {
          error: error,
          message: error instanceof Event ? 'Connection failed - check if server is running' : String(error),
          url: this.url,
          readyState: this.ws?.readyState,
          readyStateText: this.getReadyStateText(this.ws?.readyState),
          timestamp: new Date().toISOString(),
          errorType: error.constructor.name,
          troubleshooting: {
            possibleCauses: [
              'ESP32 device not connected or powered on',
              'Incorrect IP address or port',
              'Network connectivity issues',
              'WebSocket server not running on ESP32',
              'Firewall blocking connection'
            ],
            recommendations: [
              'Check ESP32 device connection and power',
              'Verify IP address and port in settings',
              'Ensure ESP32 and client are on same network',
              'Check ESP32 WebSocket server code is running'
            ]
          }
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

  private handleSignalPacket(packet: SignalPacket): void {
    if (!packet.payload) {
      console.warn("[v0] Signal packet missing payload:", packet)
      return
    }

    const { timestamp, ppg_value, ecg_value, quality } = packet.payload

    if (ppg_value !== undefined) {
      this.emitSignal({
        timestamp,
        signalType: "PPG",
        value: ppg_value,
        quality,
      })
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
      if (timeSinceLastHeartbeat > 5000) {
        console.warn("  No heartbeat received, connection may be stale")
      }
    }, 2000)
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }
}
