"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { GraduationCap, Sparkles, Target, Zap } from "lucide-react"

export function ProjectOverviewSection() {
  return (
    <section className="py-20 bg-gradient-to-b from-background via-muted/20 to-background">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            Project Overview
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Learn about PulseAI and how we're revolutionizing blood pressure monitoring
          </p>
        </div>

        <Card className="relative overflow-hidden border-2 bg-gradient-to-br from-background via-background to-muted/30 hover:shadow-2xl transition-all duration-300">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-pink-500/5" />
          <CardHeader className="relative z-10">
            <CardTitle className="flex items-center gap-3 text-2xl">
              <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500/20 to-purple-500/20">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              About PulseAI
            </CardTitle>
            <CardDescription className="text-base">Innovative Healthcare Technology</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6 text-muted-foreground relative z-10">
            <p className="text-base leading-relaxed">
              PulseAI represents an innovative approach to continuous blood pressure monitoring without
              the need for traditional cuff-based methods. Our system utilizes Photoplethysmography (PPG)
              and Electrocardiography (ECG) signals to estimate blood pressure in real-time.
            </p>
            <p className="text-base leading-relaxed">
              Through the application of deep learning and signal processing techniques, we aim to provide
              a convenient, non-invasive solution for blood pressure monitoring that can be integrated
              into wearable devices and healthcare systems.
            </p>
            <p className="text-base leading-relaxed">
              This research project focuses on advancing the field of healthcare technology and improving
              patient care through innovative solutions that make health monitoring more accessible and
              comfortable.
            </p>

            {/* Key Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-6 border-t border-border">
              <div className="flex gap-3 items-start">
                <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400 mt-1">
                  <Target className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Non-Invasive</h4>
                  <p className="text-sm">No cuff required, comfortable monitoring</p>
                </div>
              </div>
              <div className="flex gap-3 items-start">
                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-600 dark:text-purple-400 mt-1">
                  <Zap className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Real-Time</h4>
                  <p className="text-sm">Continuous monitoring and instant results</p>
                </div>
              </div>
              <div className="flex gap-3 items-start">
                <div className="p-2 rounded-lg bg-pink-500/10 text-pink-600 dark:text-pink-400 mt-1">
                  <GraduationCap className="h-5 w-5" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Research-Based</h4>
                  <p className="text-sm">Advanced ML and signal processing</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}

