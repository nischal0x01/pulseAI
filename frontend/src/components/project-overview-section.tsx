"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { GraduationCap, Sparkles, Target, Zap } from "lucide-react"

export function ProjectOverviewSection() {
  return (
    <section className="py-24 bg-gradient-to-b from-background via-muted/30 to-background">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 animate-fade-in-up">
          <h2 
            className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-primary via-purple-600 to-pink-600 bg-clip-text text-transparent"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            Project Overview
          </h2>
          <p 
            className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto font-medium"
            style={{ fontFamily: "var(--font-inter)" }}
          >
            Learn about PulseAI and how we're revolutionizing blood pressure monitoring
          </p>
        </div>

        <Card className="relative overflow-hidden border-2 border-primary/20 bg-gradient-to-br from-card via-card to-primary/5 hover:shadow-2xl hover:shadow-primary/20 transition-all duration-500 animate-fade-in-up">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-purple-500/5 to-pink-500/5 opacity-50" />
          <CardHeader className="relative z-10 pb-6">
            <CardTitle 
              className="flex items-center gap-4 text-3xl md:text-4xl font-extrabold"
              style={{ fontFamily: "var(--font-poppins)" }}
            >
              <div className="p-3 rounded-xl bg-gradient-to-br from-primary/20 to-purple-500/20 shadow-lg shadow-primary/20">
                <Sparkles className="h-7 w-7 text-primary" />
              </div>
              About PulseAI
            </CardTitle>
            <CardDescription className="text-lg font-medium mt-2">Innovative Healthcare Technology</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6 text-foreground/80 relative z-10">
            <p className="text-lg md:text-xl leading-relaxed font-medium" style={{ fontFamily: "var(--font-inter)" }}>
              PulseAI represents an innovative approach to continuous blood pressure monitoring without
              the need for traditional cuff-based methods. Our system utilizes Photoplethysmography (PPG)
              and Electrocardiography (ECG) signals to estimate blood pressure in real-time.
            </p>
            <p className="text-lg md:text-xl leading-relaxed font-medium" style={{ fontFamily: "var(--font-inter)" }}>
              Through the application of deep learning and signal processing techniques, we aim to provide
              a convenient, non-invasive solution for blood pressure monitoring that can be integrated
              into wearable devices and healthcare systems.
            </p>
            <p className="text-lg md:text-xl leading-relaxed font-medium" style={{ fontFamily: "var(--font-inter)" }}>
              This research project focuses on advancing the field of healthcare technology and improving
              patient care through innovative solutions that make health monitoring more accessible and
              comfortable.
            </p>

            {/* Key Features */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-8 border-t border-border/50">
              <div className="flex gap-4 items-start p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-primary/10 hover:from-blue-500/20 hover:to-primary/20 transition-all duration-300 group">
                <div className="p-3 rounded-lg bg-gradient-to-br from-blue-500 to-primary text-white shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <Target className="h-6 w-6" />
                </div>
                <div>
                  <h4 className="font-bold text-xl mb-2" style={{ fontFamily: "var(--font-poppins)" }}>Non-Invasive</h4>
                  <p className="text-base" style={{ fontFamily: "var(--font-inter)" }}>No cuff required, comfortable monitoring</p>
                </div>
              </div>
              <div className="flex gap-4 items-start p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-pink-500/10 hover:from-purple-500/20 hover:to-pink-500/20 transition-all duration-300 group">
                <div className="p-3 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 text-white shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <Zap className="h-6 w-6" />
                </div>
                <div>
                  <h4 className="font-bold text-xl mb-2" style={{ fontFamily: "var(--font-poppins)" }}>Real-Time</h4>
                  <p className="text-base" style={{ fontFamily: "var(--font-inter)" }}>Continuous monitoring and instant results</p>
                </div>
              </div>
              <div className="flex gap-4 items-start p-4 rounded-xl bg-gradient-to-br from-primary/10 to-blue-500/10 hover:from-primary/20 hover:to-blue-500/20 transition-all duration-300 group">
                <div className="p-3 rounded-lg bg-gradient-to-br from-primary to-blue-500 text-white shadow-lg group-hover:scale-110 transition-transform duration-300">
                  <GraduationCap className="h-6 w-6" />
                </div>
                <div>
                  <h4 className="font-bold text-xl mb-2" style={{ fontFamily: "var(--font-poppins)" }}>Research-Based</h4>
                  <p className="text-base" style={{ fontFamily: "var(--font-inter)" }}>Advanced ML and signal processing</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </section>
  )
}
