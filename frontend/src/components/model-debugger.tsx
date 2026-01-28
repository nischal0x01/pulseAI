"use client"

import * as tf from '@tensorflow/tfjs'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Brain, Download, CheckCircle2, AlertCircle } from 'lucide-react'

/**
 * Development component to test model loading and display model info
 * Only use this for debugging - remove from production
 */
export function ModelDebugger() {
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const testModelLoad = async () => {
    setLoading(true)
    setError(null)
    setModelInfo(null)

    try {
      console.log('Attempting to load model from /models/model.json...')
      const model = await tf.loadLayersModel('/models/model.json')
      
      const info = {
        inputShape: model.inputs[0].shape,
        outputShapes: model.outputs.map(o => o.shape),
        layerCount: model.layers.length,
        trainable: model.trainable,
        backend: tf.getBackend(),
        memory: tf.memory(),
      }

      setModelInfo(info)
      console.log('Model loaded successfully:', info)
      
      // Clean up
      model.dispose()
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMsg)
      console.error('Model load failed:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="border-dashed">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-amber-500" />
          <CardTitle className="text-lg">Model Debugger</CardTitle>
        </div>
        <CardDescription>
          Development tool to test TensorFlow.js model loading
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Button 
          onClick={testModelLoad} 
          disabled={loading}
          className="w-full"
        >
          <Download className="h-4 w-4 mr-2" />
          {loading ? 'Loading Model...' : 'Test Model Load'}
        </Button>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-xs">
              <strong>Error:</strong> {error}
              <br />
              <span className="text-xs mt-2 block">
                Make sure model files exist at <code>/public/models/model.json</code>
              </span>
            </AlertDescription>
          </Alert>
        )}

        {modelInfo && (
          <Alert>
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
            <AlertDescription>
              <div className="text-xs space-y-1">
                <div><strong>✓ Model loaded successfully!</strong></div>
                <div>Input Shape: {JSON.stringify(modelInfo.inputShape)}</div>
                <div>Output Shapes: {JSON.stringify(modelInfo.outputShapes)}</div>
                <div>Layers: {modelInfo.layerCount}</div>
                <div>Backend: {modelInfo.backend}</div>
                <div>Memory: {Math.round(modelInfo.memory.numBytes / 1024 / 1024)}MB</div>
              </div>
            </AlertDescription>
          </Alert>
        )}

        <div className="text-xs text-muted-foreground p-3 bg-muted/30 rounded">
          <strong>Quick Test:</strong> This component attempts to load your converted model.
          Remove this component before production deployment.
        </div>
      </CardContent>
    </Card>
  )
}
