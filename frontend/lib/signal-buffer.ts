export interface BufferConfig {
  maxSize: number
  samplingRate: number
  windowDuration: number
}

export interface SignalMetrics {
  heartRate: number
  signalQuality: number
  lastRRInterval: number
  estimatedSBP: number
  estimatedDBP: number
}

export class SignalBuffer {
  private buffer: Array<{ timestamp: number; value: number }> = []
  private readonly maxSize: number
  private readonly samplingRate: number
  private peaks: number[] = []

  constructor(config: BufferConfig) {
    this.maxSize = config.maxSize
    this.samplingRate = config.samplingRate
  }

  addSample(timestamp: number, value: number): void {
    this.buffer.push({ timestamp, value })

    // Maintain circular buffer
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift()
    }
  }

  getBuffer(): Array<{ timestamp: number; value: number }> {
    return [...this.buffer]
  }

  getWindowedData(duration: number): Array<{ timestamp: number; value: number }> {
    if (this.buffer.length === 0) return []

    const now = this.buffer[this.buffer.length - 1].timestamp
    const cutoff = now - duration

    return this.buffer.filter((sample) => sample.timestamp >= cutoff)
  }

  clear(): void {
    this.buffer = []
    this.peaks = []
  }

  calculateMetrics(): SignalMetrics {
    if (this.buffer.length < 2) {
      return {
        heartRate: 0,
        signalQuality: 0,
        lastRRInterval: 0,
        estimatedSBP: 0,
        estimatedDBP: 0,
      }
    }

    // Calculate heart rate from peaks
    this.detectPeaks()
    const heartRate = this.calculateHeartRate()

    // Calculate signal quality (simplified)
    const values = this.buffer.map((s) => s.value)
    const mean = values.reduce((a, b) => a + b, 0) / values.length
    const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length
    const signalQuality = Math.min(100, (variance / 100) * 100)

    // --- Estimated BP placeholder logic ---
    // NOTE: This is a stub using simple scaling of the signal.
    // Replace with  trained model or a proper estimation function.
    const max = Math.max(...values)
    const min = Math.min(...values)
    const amplitude = max - min

    // Very rough heuristic mapping amplitude -> BP (for demo only)
    const estimatedSBP = 100 + amplitude * 5
    const estimatedDBP = 60 + amplitude * 3

    // Last R-R interval
    const lastRRInterval =
      this.peaks.length >= 2 ? this.peaks[this.peaks.length - 1] - this.peaks[this.peaks.length - 2] : 0

    return {
      heartRate,
      signalQuality,
      lastRRInterval,
      estimatedSBP,
      estimatedDBP,
    }
  }

  private detectPeaks(): void {
    if (this.buffer.length < 3) return

    this.peaks = []
    const values = this.buffer.map((s) => s.value)
    const threshold = (Math.max(...values) + Math.min(...values)) / 2

    for (let i = 1; i < values.length - 1; i++) {
      if (values[i] > threshold && values[i] > values[i - 1] && values[i] > values[i + 1]) {
        this.peaks.push(this.buffer[i].timestamp)
      }
    }
  }

  private calculateHeartRate(): number {
    if (this.peaks.length < 2) return 0

    // Calculate average interval between peaks
    const intervals: number[] = []
    for (let i = 1; i < this.peaks.length; i++) {
      intervals.push(this.peaks[i] - this.peaks[i - 1])
    }

    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length

    // Convert to BPM (beats per minute)
    return Math.round(60000 / avgInterval)
  }

  getPeaks(): number[] {
    return [...this.peaks]
  }
}
