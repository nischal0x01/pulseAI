"use client"

import { useEffect, useState } from "react"
import { useSignal } from "@/lib/signal-context"
import { getModelInstance, type ModelPrediction } from "@/lib/model-service"
import { Activity, AlertCircle, Brain, CheckCircle2, Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"

// Prediction configuration constants
const REQUIRED_SAMPLES = 875 // Number of samples needed for model input
const PREDICTION_INTERVAL_MS = 2000 // Update predictions every 2 seconds

export function BPPredictor() {
  const { ppgBuffer, ecgBuffer } = useSignal()
  const [modelLoaded, setModelLoaded] = useState(false)
  const [modelError, setModelError] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<ModelPrediction | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null)

  // Load model on mount
  useEffect(() => {
    const loadModel = async () => {
      const model = getModelInstance()
      const success = await model.loadModel()
      
      if (success) {
        setModelLoaded(true)
        setModelError(null)
      } else {
        setModelError(model.getLoadError() || 'Failed to load model')
      }
    }

    loadModel()
  }, [])

  // Make predictions periodically when model is ready
  useEffect(() => {
    if (!modelLoaded) return

    const predictInterval = setInterval(async () => {
      // Get latest signal data
      const ppgData = ppgBuffer.getBuffer()
      const ecgData = ecgBuffer.getBuffer()

      // Need minimum samples for prediction
      if (ppgData.length < REQUIRED_SAMPLES || ecgData.length < REQUIRED_SAMPLES) {
        console.log('Waiting for sufficient data...', {
          ppg: ppgData.length,
          ecg: ecgData.length,
          required: REQUIRED_SAMPLES
        })
        return
      }

      setIsProcessing(true)

      try {
        const model = getModelInstance()
        
        // Extract signal values (take last N samples)
        const ppgValues = ppgData.slice(-REQUIRED_SAMPLES).map(s => s.value)
        const ecgValues = ecgData.slice(-REQUIRED_SAMPLES).map(s => s.value)

        // Make prediction
        const result = await model.predict(ecgValues, ppgValues)
        
        if (result) {
          setPrediction(result)
          setLastUpdateTime(new Date())
        }
      } catch (error) {
        console.error('Prediction failed:', error)
      } finally {
        setIsProcessing(false)
      }
    }, PREDICTION_INTERVAL_MS)

    return () => clearInterval(predictInterval)
  }, [modelLoaded, ppgBuffer, ecgBuffer])

  const getBPCategory = (sbp: number, dbp: number) => {
    if (sbp >= 180 || dbp >= 120) {
      return { label: 'Hypertensive Crisis', color: 'bg-red-600', textColor: 'text-red-600' }
    }
    if (sbp >= 140 || dbp >= 90) {
      return { label: 'High', color: 'bg-red-500', textColor: 'text-red-500' }
    }
    if (sbp >= 130 || dbp >= 80) {
      return { label: 'Elevated', color: 'bg-yellow-500', textColor: 'text-yellow-500' }
    }
    if (sbp < 90 || dbp < 60) {
      return { label: 'Low', color: 'bg-blue-500', textColor: 'text-blue-500' }
    }
    return { label: 'Normal', color: 'bg-emerald-500', textColor: 'text-emerald-500' }
  }

  if (modelError) {
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          <strong>API Connection Error:</strong> {modelError}
          <br />
          <span className="text-xs mt-1 block">
            Make sure the Python backend API is running on port 8000
          </span>
          <code className="text-xs block bg-destructive/10 p-2 rounded mt-2">
            python api_server.py
          </code>
        </AlertDescription>
      </Alert>
    )
  }

  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur-sm shadow-xl">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            <CardTitle className="text-xl font-display">AI Blood Pressure Estimation</CardTitle>
          </div>
          {modelLoaded ? (
            <Badge variant="outline" className="gap-1">
              <CheckCircle2 className="h-3 w-3 text-emerald-500" />
              Model Ready
            </Badge>
          ) : (
            <Badge variant="outline" className="gap-1">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading...
            </Badge>
          )}
        </div>
        <CardDescription>
          Real-time predictions using deep learning model
        </CardDescription>
      </CardHeader>

      <CardContent>
        {!modelLoaded && !modelError && (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            Loading model...
          </div>
        )}

        {modelLoaded && !prediction && (
          <div className="flex items-center justify-center py-8 text-muted-foreground">
            <Activity className="h-6 w-6 mr-2 animate-pulse" />
            Collecting signal data...
          </div>
        )}

        {prediction && (
          <div className="space-y-6">
            {/* BP Values */}
            <div className="grid grid-cols-2 gap-4">
              {/* SBP */}
              <div className="rounded-lg border border-border/50 bg-gradient-to-br from-red-500/10 to-transparent p-6">
                <div className="text-sm font-medium text-muted-foreground mb-2">
                  Systolic (SBP)
                </div>
                <div className="text-4xl font-bold font-display">
                  {prediction.sbp}
                  <span className="text-lg text-muted-foreground ml-1">mmHg</span>
                </div>
              </div>

              {/* DBP */}
              <div className="rounded-lg border border-border/50 bg-gradient-to-br from-blue-500/10 to-transparent p-6">
                <div className="text-sm font-medium text-muted-foreground mb-2">
                  Diastolic (DBP)
                </div>
                <div className="text-4xl font-bold font-display">
                  {prediction.dbp}
                  <span className="text-lg text-muted-foreground ml-1">mmHg</span>
                </div>
              </div>
            </div>

            {/* Category and Status */}
            <div className="flex items-center justify-between p-4 rounded-lg bg-muted/30">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${getBPCategory(prediction.sbp, prediction.dbp).color} animate-pulse`} />
                <div>
                  <div className="text-sm font-medium">
                    Status: <span className={getBPCategory(prediction.sbp, prediction.dbp).textColor}>
                      {getBPCategory(prediction.sbp, prediction.dbp).label}
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {lastUpdateTime && `Last updated: ${lastUpdateTime.toLocaleTimeString()}`}
                  </div>
                </div>
              </div>
              
              {isProcessing && (
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              )}
            </div>

            {/* Confidence Score */}
            {prediction.confidence && (
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">Prediction Confidence</span>
                <div className="flex items-center gap-2">
                  <div className="w-32 h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-yellow-500 to-emerald-500 transition-all duration-500"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                  <span className="font-medium min-w-[3rem] text-right">
                    {Math.round(prediction.confidence * 100)}%
                  </span>
                </div>
              </div>
            )}

            {/* Disclaimer */}
            <Alert>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-xs">
                <strong>Medical Disclaimer:</strong> This is an experimental AI estimation and should not replace professional medical devices or advice. Consult healthcare professionals for accurate measurements.
              </AlertDescription>
            </Alert>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
