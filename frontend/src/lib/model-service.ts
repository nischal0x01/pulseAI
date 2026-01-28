"use client"

import * as tf from '@tensorflow/tfjs'

export interface ModelPrediction {
  sbp: number
  dbp: number
  confidence: number
}

export class BloodPressureModel {
  private model: tf.LayersModel | null = null
  private isLoading: boolean = false
  private loadError: string | null = null

  async loadModel(): Promise<boolean> {
    if (this.model) return true
    if (this.isLoading) return false

    this.isLoading = true
    this.loadError = null

    try {
      console.log('Loading TensorFlow.js model...')
      this.model = await tf.loadLayersModel('/models/model.json')
      console.log('✓ Model loaded successfully')
      console.log('  - Input shape:', this.model.inputs[0].shape)
      console.log('  - Output shape:', this.model.outputs.map(o => o.shape))
      this.isLoading = false
      return true
    } catch (error) {
      console.error('✗ Failed to load model:', error)
      this.loadError = error instanceof Error ? error.message : 'Unknown error'
      this.isLoading = false
      return false
    }
  }

  isModelReady(): boolean {
    return this.model !== null
  }

  getLoadError(): string | null {
    return this.loadError
  }

  /**
   * Predict blood pressure from signal data
   * Expected input shape: [1, 875, 4]
   * - 875 samples (3.5 seconds at 250 Hz)
   * - 4 features: [ECG, PPG, PAT, HR]
   * 
   * @param ecgSignal - ECG signal array (875 samples)
   * @param ppgSignal - PPG signal array (875 samples)
   * @returns Prediction with SBP and DBP values
   */
  async predict(
    ecgSignal: number[],
    ppgSignal: number[],
    patSignal?: number[],
    hrSignal?: number[]
  ): Promise<ModelPrediction | null> {
    if (!this.model) {
      console.error('Model not loaded. Call loadModel() first.')
      return null
    }

    try {
      // Ensure signals have correct length (875 samples for 3.5s at 250Hz)
      const targetLength = 875
      const ecg = this.resampleOrPad(ecgSignal, targetLength)
      const ppg = this.resampleOrPad(ppgSignal, targetLength)
      
      // For PAT and HR, use defaults if not provided
      const pat = patSignal ? this.resampleOrPad(patSignal, targetLength) : new Array(targetLength).fill(0.2)
      const hr = hrSignal ? this.resampleOrPad(hrSignal, targetLength) : new Array(targetLength).fill(75)

      // Normalize signals
      const ecgNorm = this.normalize(ecg)
      const ppgNorm = this.normalize(ppg)
      const patNorm = this.normalize(pat)
      const hrNorm = this.normalizeHeartRate(hr)

      // Create input tensor [1, 875, 4]
      const inputData: number[][][] = [[]]
      for (let i = 0; i < targetLength; i++) {
        inputData[0].push([ecgNorm[i], ppgNorm[i], patNorm[i], hrNorm[i]])
      }

      const inputTensor = tf.tensor3d(inputData)

      // Make prediction
      const prediction = this.model.predict(inputTensor) as tf.Tensor | tf.Tensor[]
      
      // Handle multiple outputs [SBP, DBP]
      let sbp: number, dbp: number
      
      if (Array.isArray(prediction)) {
        // Model has separate outputs for SBP and DBP
        const sbpData = await prediction[0].data()
        const dbpData = await prediction[1].data()
        sbp = sbpData[0]
        dbp = dbpData[0]
        
        // Clean up tensors
        prediction.forEach(t => t.dispose())
      } else {
        // Model has single output [SBP, DBP]
        const predData = await prediction.data()
        sbp = predData[0]
        dbp = predData[1]
        prediction.dispose()
      }

      // Clean up input tensor
      inputTensor.dispose()

      // Ensure values are in valid range
      sbp = Math.max(80, Math.min(200, sbp))
      dbp = Math.max(40, Math.min(130, dbp))

      return {
        sbp: Math.round(sbp),
        dbp: Math.round(dbp),
        confidence: 0.85 // TODO: Calculate actual confidence score
      }
    } catch (error) {
      console.error('Prediction error:', error)
      return null
    }
  }

  /**
   * Resample or pad signal to target length
   */
  private resampleOrPad(signal: number[], targetLength: number): number[] {
    if (signal.length === targetLength) return signal
    
    if (signal.length > targetLength) {
      // Downsample by taking evenly spaced samples
      const step = signal.length / targetLength
      return Array.from({ length: targetLength }, (_, i) => 
        signal[Math.floor(i * step)]
      )
    } else {
      // Pad with last value
      return [...signal, ...new Array(targetLength - signal.length).fill(signal[signal.length - 1] || 0)]
    }
  }

  /**
   * Z-score normalization
   */
  private normalize(signal: number[]): number[] {
    const mean = signal.reduce((a, b) => a + b, 0) / signal.length
    const std = Math.sqrt(
      signal.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / signal.length
    )
    
    if (std === 0) return signal.map(() => 0)
    return signal.map(v => (v - mean) / std)
  }

  /**
   * Normalize heart rate to 0-1 range (assuming 40-200 bpm)
   */
  private normalizeHeartRate(hr: number[]): number[] {
    return hr.map(v => (v - 40) / 160)
  }

  /**
   * Clean up model and free memory
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose()
      this.model = null
    }
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
