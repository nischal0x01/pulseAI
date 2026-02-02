"use client"

import { useSignal } from "@/lib/signal-context"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Activity, Heart } from "lucide-react"

export function BloodPressureDisplay() {
  const { latestBP, connectionStatus } = useSignal()

  const getBPCategory = (sbp: number, dbp: number) => {
    if (sbp >= 180 || dbp >= 120) {
      return { label: 'Hypertensive Crisis', color: 'bg-red-600 text-white', severity: 'critical' }
    }
    if (sbp >= 140 || dbp >= 90) {
      return { label: 'High', color: 'bg-red-500 text-white', severity: 'high' }
    }
    if (sbp >= 130 || dbp >= 80) {
      return { label: 'Elevated', color: 'bg-yellow-500 text-white', severity: 'elevated' }
    }
    if (sbp < 90 || dbp < 60) {
      return { label: 'Low', color: 'bg-blue-500 text-white', severity: 'low' }
    }
    return { label: 'Normal', color: 'bg-emerald-500 text-white', severity: 'normal' }
  }

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString()
  }

  if (connectionStatus !== 'connected') {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Heart className="h-5 w-5" />
            Blood Pressure
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            Connect to ESP32 to see predictions
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!latestBP) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Heart className="h-5 w-5" />
            Blood Pressure
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground py-8">
            <Activity className="h-12 w-12 mx-auto mb-2 animate-pulse" />
            <p>Collecting data...</p>
            <p className="text-sm">Waiting for sufficient signal</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const category = getBPCategory(latestBP.sbp, latestBP.dbp)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 justify-between">
          <div className="flex items-center gap-2">
            <Heart className="h-5 w-5" />
            Blood Pressure
          </div>
          <Badge className={category.color}>
            {category.label}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Main BP Reading */}
          <div className="text-center">
            <div className="text-6xl font-bold tracking-tight">
              {Math.round(latestBP.sbp)}
              <span className="text-3xl text-muted-foreground mx-2">/</span>
              {Math.round(latestBP.dbp)}
            </div>
            <div className="text-sm text-muted-foreground mt-2">
              mmHg
            </div>
          </div>

          {/* Details */}
          <div className="grid grid-cols-2 gap-4 pt-4 border-t">
            <div>
              <div className="text-sm text-muted-foreground">Systolic (SBP)</div>
              <div className="text-2xl font-semibold">{Math.round(latestBP.sbp)}</div>
            </div>
            <div>
              <div className="text-sm text-muted-foreground">Diastolic (DBP)</div>
              <div className="text-2xl font-semibold">{Math.round(latestBP.dbp)}</div>
            </div>
          </div>

          {/* Metadata */}
          <div className="flex items-center justify-between text-xs text-muted-foreground pt-4 border-t">
            <div>
              Confidence: {(latestBP.confidence * 100).toFixed(0)}%
            </div>
            <div>
              Updated: {formatTimestamp(latestBP.timestamp)}
            </div>
          </div>

          {/* Prediction Count */}
          <div className="text-center text-xs text-muted-foreground">
            Prediction #{latestBP.prediction_count}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
