"use client"

export interface ModelPrediction {
  sbp: number
  dbp: number
  confidence: number
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export class BloodPressureModel {
  private apiUrl: string
  private isHealthy: boolean = false
  private loadError: string | null = null

  constructor() {
    this.apiUrl = API_BASE_URL
  }

  async loadModel(): Promise<boolean> {
    try {
      console.log('Checking API health...')
      
      const response = await fetch(`${this.apiUrl}/health`)
      
      if (!response.ok) {
        throw new Error(`API health check failed: ${response.status}`)
      }
      
      const health = await response.json()
      this.isHealthy = health.model_loaded
      
      if (this.isHealthy) {
        console.log('✓ Backend API is healthy')
        console.log('  - Model loaded:', health.model_loaded)
        console.log('  - Input shape:', health.input_shape)
      } else {
        console.warn('⚠ API is running but model not loaded')
      }
      
      return this.isHealthy
    } catch (error) {
      console.error('✗ Failed to connect to backend API:', error)
      this.loadError = error instanceof Error ? error.message : 'Unknown error'
      this.isHealthy = false
      return false
    }
  }

  isModelReady(): boolean {
    return this.isHealthy
  }

  getLoadError(): string | null {
    return this.loadError
  }

  /**
   * Predict blood pressure from signal data
   * Expected input: ECG and PPG signals (at least 875 samples each)
   * 
   * @param ecgSignal - ECG signal array
   * @param ppgSignal - PPG signal array
   * @param patSignal - Optional PAT signal array
   * @param hrSignal - Optional heart rate signal array
   * @returns Prediction with SBP and DBP values
   */
  async predict(
    ecgSignal: number[],
    ppgSignal: number[],
    patSignal?: number[],
    hrSignal?: number[]
  ): Promise<ModelPrediction | null> {
    if (!this.isHealthy) {
      console.error('Backend API not healthy. Call loadModel() first.')
      return null
    }

    try {
      const requestBody = {
        ecg: ecgSignal,
        ppg: ppgSignal,
        ...(patSignal && { pat: patSignal }),
        ...(hrSignal && { hr: hrSignal })
      }

      console.log('Sending prediction request to backend...')
      
      const response = await fetch(`${this.apiUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(error.detail || `Prediction failed: ${response.status}`)
      }

      const result = await response.json()
      
      console.log('✓ Prediction received:', result)

      return {
        sbp: result.sbp,
        dbp: result.dbp,
        confidence: result.confidence
      }
    } catch (error) {
      console.error('Prediction error:', error)
      return null
    }
  }

  /**
   * Clean up resources (no-op for API-based model)
   */
  dispose(): void {
    // Nothing to dispose when using API
    console.log('Model service disposed')
  }
}

// Singleton instance
let modelInstance: BloodPressureModel | null = null

export function getModelInstance(): BloodPressureModel {
  if (!modelInstance) {
    modelInstance = new BloodPressureModel()
  }
  return modelInstance
}
