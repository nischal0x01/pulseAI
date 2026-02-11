"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Loader2, Activity, TrendingUp, TrendingDown, CheckCircle2, XCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface PatientMetrics {
  mae: number
  rmse: number
  r2: number
  pearson: number
  mean_error: number
  std_error: number
}

interface SamplePrediction {
  index: number
  actual_sbp: number
  pred_sbp: number
  sbp_error: number
  actual_dbp: number
  pred_dbp: number
  dbp_error: number
}

interface PatientAnalysisData {
  patient_id: string
  n_samples: number
  sbp_metrics: PatientMetrics
  dbp_metrics: PatientMetrics
  sample_predictions: SamplePrediction[]
  calibrated: boolean
  summary_stats: {
    actual_sbp_mean: number
    actual_sbp_std: number
    pred_sbp_mean: number
    pred_sbp_std: number
    actual_dbp_mean: number
    actual_dbp_std: number
    pred_dbp_mean: number
    pred_dbp_std: number
  }
}

export function PatientAnalysis() {
  const [patients, setPatients] = useState<string[]>([])
  const [selectedPatient, setSelectedPatient] = useState<string>("")
  const [analysisData, setAnalysisData] = useState<PatientAnalysisData | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingPatients, setLoadingPatients] = useState(true)
  const [error, setError] = useState<string>("")

  // Fetch available patients on mount
  useEffect(() => {
    fetchPatients()
  }, [])

  const fetchPatients = async () => {
    try {
      setLoadingPatients(true)
      const response = await fetch("http://localhost:8000/patients")
      if (!response.ok) throw new Error("Failed to fetch patients")
      const data = await response.json()
      setPatients(data.patients)
      if (data.patients.length > 0) {
        setSelectedPatient(data.patients[0])
      }
    } catch (err) {
      setError("Failed to load patients. Make sure the API server is running.")
      console.error(err)
    } finally {
      setLoadingPatients(false)
    }
  }

  const analyzePatient = async () => {
    if (!selectedPatient) return

    try {
      setLoading(true)
      setError("")
      
      const response = await fetch("http://localhost:8000/analyze-patient", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          patient_id: selectedPatient,
          n_samples: 10,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Analysis failed")
      }

      const data = await response.json()
      setAnalysisData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze patient")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const MetricCard = ({ title, value, subtitle, icon: Icon, colorClass }: any) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        {Icon && <Icon className={`h-4 w-4 ${colorClass}`} />}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
      </CardContent>
    </Card>
  )

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h2 className="text-3xl font-bold tracking-tight">Patient Analysis</h2>
        <p className="text-muted-foreground">
          Analyze blood pressure predictions for individual patients using trained model
        </p>
      </div>

      {/* Patient Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Patient</CardTitle>
          <CardDescription>
            Choose a patient from the processed data to analyze predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <Select
              value={selectedPatient}
              onValueChange={setSelectedPatient}
              disabled={loadingPatients || loading}
            >
              <SelectTrigger className="flex-1">
                <SelectValue placeholder="Select a patient" />
              </SelectTrigger>
              <SelectContent>
                {patients.map((patient) => (
                  <SelectItem key={patient} value={patient}>
                    {patient}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              onClick={analyzePatient}
              disabled={!selectedPatient || loading || loadingPatients}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="mr-2 h-4 w-4" />
                  Analyze Patient
                </>
              )}
            </Button>
          </div>

          {loadingPatients && (
            <p className="text-sm text-muted-foreground">Loading patients...</p>
          )}
          {!loadingPatients && patients.length === 0 && (
            <Alert>
              <AlertDescription>
                No patients found. Make sure patient data exists in data/processed/
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert variant="destructive">
          <XCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Analysis Results */}
      {analysisData && (
        <div className="space-y-6">
          {/* Summary Header */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Analysis: {analysisData.patient_id}</CardTitle>
                  <CardDescription>
                    {analysisData.n_samples} samples analyzed
                  </CardDescription>
                </div>
                {analysisData.calibrated ? (
                  <Badge variant="default" className="bg-green-600">
                    <CheckCircle2 className="mr-1 h-3 w-3" />
                    Calibrated
                  </Badge>
                ) : (
                  <Badge variant="secondary">
                    Uncalibrated
                  </Badge>
                )}
              </div>
            </CardHeader>
          </Card>

          {/* Summary Statistics */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Summary Statistics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                title="Actual SBP"
                value={`${analysisData.summary_stats.actual_sbp_mean.toFixed(1)} mmHg`}
                subtitle={`± ${analysisData.summary_stats.actual_sbp_std.toFixed(1)}`}
                icon={Activity}
                colorClass="text-red-600"
              />
              <MetricCard
                title="Predicted SBP"
                value={`${analysisData.summary_stats.pred_sbp_mean.toFixed(1)} mmHg`}
                subtitle={`± ${analysisData.summary_stats.pred_sbp_std.toFixed(1)}`}
                icon={TrendingUp}
                colorClass="text-orange-600"
              />
              <MetricCard
                title="Actual DBP"
                value={`${analysisData.summary_stats.actual_dbp_mean.toFixed(1)} mmHg`}
                subtitle={`± ${analysisData.summary_stats.actual_dbp_std.toFixed(1)}`}
                icon={Activity}
                colorClass="text-blue-600"
              />
              <MetricCard
                title="Predicted DBP"
                value={`${analysisData.summary_stats.pred_dbp_mean.toFixed(1)} mmHg`}
                subtitle={`± ${analysisData.summary_stats.pred_dbp_std.toFixed(1)}`}
                icon={TrendingDown}
                colorClass="text-cyan-600"
              />
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* SBP Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="text-red-600">SBP Metrics (Systolic)</CardTitle>
                <CardDescription>Model performance on systolic blood pressure</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">MAE (Mean Absolute Error)</span>
                    <span className="text-lg font-bold">{analysisData.sbp_metrics.mae.toFixed(2)} mmHg</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">RMSE (Root Mean Square Error)</span>
                    <span className="text-lg font-bold">{analysisData.sbp_metrics.rmse.toFixed(2)} mmHg</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">R² Score</span>
                    <span className="text-lg font-bold">{analysisData.sbp_metrics.r2.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Pearson Correlation</span>
                    <span className="text-lg font-bold">{analysisData.sbp_metrics.pearson.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Mean Error</span>
                    <span className={`text-lg font-bold ${analysisData.sbp_metrics.mean_error > 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {analysisData.sbp_metrics.mean_error > 0 ? '+' : ''}{analysisData.sbp_metrics.mean_error.toFixed(2)} mmHg
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Std Error</span>
                    <span className="text-lg font-bold">{analysisData.sbp_metrics.std_error.toFixed(2)} mmHg</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* DBP Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="text-blue-600">DBP Metrics (Diastolic)</CardTitle>
                <CardDescription>Model performance on diastolic blood pressure</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">MAE (Mean Absolute Error)</span>
                    <span className="text-lg font-bold">{analysisData.dbp_metrics.mae.toFixed(2)} mmHg</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">RMSE (Root Mean Square Error)</span>
                    <span className="text-lg font-bold">{analysisData.dbp_metrics.rmse.toFixed(2)} mmHg</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">R² Score</span>
                    <span className="text-lg font-bold">{analysisData.dbp_metrics.r2.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Pearson Correlation</span>
                    <span className="text-lg font-bold">{analysisData.dbp_metrics.pearson.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Mean Error</span>
                    <span className={`text-lg font-bold ${analysisData.dbp_metrics.mean_error > 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {analysisData.dbp_metrics.mean_error > 0 ? '+' : ''}{analysisData.dbp_metrics.mean_error.toFixed(2)} mmHg
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Std Error</span>
                    <span className="text-lg font-bold">{analysisData.dbp_metrics.std_error.toFixed(2)} mmHg</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sample Predictions */}
          <Card>
            <CardHeader>
              <CardTitle>Sample Predictions</CardTitle>
              <CardDescription>
                Showing {analysisData.sample_predictions.length} random samples
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2 font-medium">Index</th>
                      <th className="text-right p-2 font-medium text-red-600">Actual SBP</th>
                      <th className="text-right p-2 font-medium text-orange-600">Pred SBP</th>
                      <th className="text-right p-2 font-medium">Error</th>
                      <th className="text-right p-2 font-medium text-blue-600">Actual DBP</th>
                      <th className="text-right p-2 font-medium text-cyan-600">Pred DBP</th>
                      <th className="text-right p-2 font-medium">Error</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysisData.sample_predictions.map((sample) => (
                      <tr key={sample.index} className="border-b hover:bg-muted/50">
                        <td className="p-2">{sample.index}</td>
                        <td className="text-right p-2 font-medium">{sample.actual_sbp.toFixed(1)}</td>
                        <td className="text-right p-2 font-medium">{sample.pred_sbp.toFixed(1)}</td>
                        <td className={`text-right p-2 font-medium ${Math.abs(sample.sbp_error) > 10 ? 'text-red-600' : 'text-green-600'}`}>
                          {sample.sbp_error > 0 ? '+' : ''}{sample.sbp_error.toFixed(1)}
                        </td>
                        <td className="text-right p-2 font-medium">{sample.actual_dbp.toFixed(1)}</td>
                        <td className="text-right p-2 font-medium">{sample.pred_dbp.toFixed(1)}</td>
                        <td className={`text-right p-2 font-medium ${Math.abs(sample.dbp_error) > 10 ? 'text-red-600' : 'text-green-600'}`}>
                          {sample.dbp_error > 0 ? '+' : ''}{sample.dbp_error.toFixed(1)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Info Box */}
          <Alert>
            <Activity className="h-4 w-4" />
            <AlertDescription>
              {analysisData.calibrated ? (
                <>
                  <strong>Calibrated predictions:</strong> Patient-specific calibration has been applied to improve accuracy.
                </>
              ) : (
                <>
                  <strong>Uncalibrated predictions:</strong> Run calibration to improve accuracy: 
                  <code className="ml-2 px-2 py-1 bg-muted rounded text-xs">
                    python calibration.py --patient {analysisData.patient_id} --n-samples 50
                  </code>
                </>
              )}
            </AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  )
}
