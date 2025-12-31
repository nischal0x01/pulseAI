"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Linkedin, Github, GraduationCap } from "lucide-react"
import { cn } from "@/lib/utils"

interface TeamMember {
  name: string
  role: string
  linkedin?: string
  github?: string
  googleScholar?: string
  initials: string
  image?: string
}

// Helper function to get image path
const getImagePath = (name: string): string => {
  // Convert name to kebab-case for filename
  let filename = name
    .toLowerCase()
    .replace(/,/g, "")
    .replace(/\s+/g, "-")
    .replace(/phd/g, "")
    .replace(/-+/g, "-") // Replace multiple dashes with single dash
    .replace(/^-|-$/g, "") // Remove leading/trailing dashes
    .trim()
  
  // Return image path - try jpg first, component will fallback if needed
  return `/team/${filename}.jpg`
}

const teamMembers: TeamMember[] = [
  {
    name: "Suvesh Gurung",
    role: "Team Member",
    linkedin: "https://www.linkedin.com/in/suvesh-gurung-471998294/",
    github: "https://github.com/suveshgurung",
    initials: "SG",
    image: getImagePath("Suvesh Gurung"),
  },
  {
    name: "Pramisha Sapkota",
    role: "Team Member",
    linkedin: "https://www.linkedin.com/in/pramisha-sapkota-b4549629a/",
    github: "https://github.com/pramisha56",
    initials: "PS",
    image: "/team/pramisha-sapkota.png", // Direct path since it's PNG
  },
  {
    name: "Nischal Subedi",
    role: "Team Member",
    linkedin: "https://linkedin.com/in/nischal0x01",
    github: "https://github.com/nischal0x01",
    initials: "NS",
    image: getImagePath("Nischal Subedi"),
  },
  {
    name: "Arwin Shrestha",
    role: "Team Member",
    linkedin: "https://www.linkedin.com/in/arwin-shrestha-7532463a1/",
    github: "https://github.com/sthaarwin",
    initials: "AS",
    image: getImagePath("Arwin Shrestha"),
  },
]

const supervisor: TeamMember = {
  name: "Rabindra Bista, Phd",
  role: "Associate Professor",
  linkedin: "https://www.linkedin.com/in/rabindra-bista-phd-16696157/",
  googleScholar: "https://scholar.google.com.tr/citations?user=yToyV6kAAAAJ&hl=tr",
  initials: "RB",
  image: getImagePath("Rabindra Bista, Phd"),
}

export function AboutSection() {
  return (
    <section id="about" className="py-24 bg-gradient-to-b from-background via-muted/30 to-background">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-20 animate-fade-in-up">
          <h2 
            className="text-5xl md:text-6xl lg:text-7xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-primary via-purple-600 to-pink-600 bg-clip-text text-transparent"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            About Us
          </h2>
          <p 
            className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto font-medium"
            style={{ fontFamily: "var(--font-inter)" }}
          >
            PulseAI is a research project focused on developing non-invasive blood pressure monitoring
            using advanced signal processing and machine learning techniques.
          </p>
        </div>

        {/* Supervisor */}
        <div className="mb-20 animate-fade-in-up">
          <h3 
            className="text-3xl md:text-4xl font-extrabold mb-12 text-center text-foreground/90"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            Supervisor
          </h3>
          <div className="flex justify-center">
            <SupervisorCard member={supervisor} />
          </div>
        </div>

        {/* Team Members */}
        <div id="team" className="animate-fade-in-up">
          <h3 
            className="text-3xl md:text-4xl font-extrabold mb-12 text-center text-foreground/90"
            style={{ fontFamily: "var(--font-poppins)" }}
          >
            Team Members
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamMembers.map((member, index) => (
              <TeamMemberCard key={member.name} member={member} index={index} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function SupervisorCard({ member }: { member: TeamMember }) {
  return (
    <Card className="relative overflow-hidden border-2 border-primary/20 bg-gradient-to-br from-card via-card to-primary/5 hover:shadow-2xl hover:shadow-primary/20 hover:scale-105 transition-all duration-500 max-w-md w-full group">
      <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-purple-500/10 to-pink-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
      <CardHeader className="text-center pb-6 relative z-10 pt-8">
        <div className="flex justify-center mb-8">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/30 to-purple-500/30 rounded-full blur-2xl group-hover:blur-3xl transition-all duration-500 animate-pulse" />
            <Avatar className="h-36 w-36 relative border-4 border-primary/20 shadow-2xl shadow-primary/20 group-hover:border-primary/40 transition-all duration-500">
              <AvatarImage src={member.image} alt={member.name} />
              <AvatarFallback className="text-3xl font-extrabold bg-gradient-to-br from-primary to-purple-600 text-white" style={{ fontFamily: "var(--font-poppins)" }}>
                {member.initials}
              </AvatarFallback>
            </Avatar>
          </div>
        </div>
        <CardTitle 
          className="text-3xl mb-3 font-extrabold"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          {member.name}
        </CardTitle>
        <CardDescription className="text-xl font-semibold">{member.role}</CardDescription>
      </CardHeader>
      <CardContent className="pt-0 pb-8 relative z-10">
        <div className="flex justify-center gap-5">
          {member.linkedin && (
            <a
              href={member.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 rounded-xl bg-gradient-to-br from-blue-500/10 to-primary/10 hover:from-blue-500/20 hover:to-primary/20 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-all duration-300 hover:scale-110 hover:rotate-6 hover:shadow-xl hover:shadow-blue-500/30"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-6 w-6" />
            </a>
          )}
          {member.googleScholar && (
            <a
              href={member.googleScholar}
              target="_blank"
              rel="noopener noreferrer"
              className="p-4 rounded-xl bg-gradient-to-br from-amber-500/10 to-orange-500/10 hover:from-amber-500/20 hover:to-orange-500/20 text-amber-600 dark:text-amber-400 hover:text-amber-700 dark:hover:text-amber-300 transition-all duration-300 hover:scale-110 hover:rotate-6 hover:shadow-xl hover:shadow-amber-500/30"
              aria-label="Google Scholar"
            >
              <GraduationCap className="h-6 w-6" />
            </a>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function TeamMemberCard({ member, index }: { member: TeamMember; index: number }) {
  return (
    <Card 
      className="relative overflow-hidden border-2 border-primary/10 bg-gradient-to-br from-card via-card to-primary/5 hover:shadow-2xl hover:shadow-primary/20 hover:border-primary/30 hover:scale-105 transition-all duration-500 group"
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className="absolute inset-0 bg-gradient-to-br from-primary/0 via-purple-500/0 to-pink-500/0 group-hover:from-primary/10 group-hover:via-purple-500/10 group-hover:to-pink-500/10 transition-all duration-500" />
      <CardHeader className="text-center pb-5 relative z-10 pt-8">
        <div className="flex justify-center mb-5">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-purple-500/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-500" />
            <Avatar className="h-28 w-28 relative border-[3px] border-primary/20 group-hover:border-primary/40 transition-all duration-500 shadow-xl shadow-primary/10">
              <AvatarImage src={member.image} alt={member.name} />
              <AvatarFallback className="text-2xl font-extrabold bg-gradient-to-br from-primary to-purple-600 text-white" style={{ fontFamily: "var(--font-poppins)" }}>
                {member.initials}
              </AvatarFallback>
            </Avatar>
          </div>
        </div>
        <CardTitle 
          className="text-xl font-extrabold mb-2"
          style={{ fontFamily: "var(--font-poppins)" }}
        >
          {member.name}
        </CardTitle>
        <CardDescription className="text-base font-semibold">{member.role}</CardDescription>
      </CardHeader>
      <CardContent className="pt-0 pb-7 relative z-10">
        <div className="flex justify-center gap-4">
          {member.linkedin && (
            <a
              href={member.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-xl bg-gradient-to-br from-blue-500/10 to-primary/10 hover:from-blue-500/20 hover:to-primary/20 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-all duration-300 hover:scale-110 hover:rotate-6 hover:shadow-lg hover:shadow-blue-500/30"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-5 w-5" />
            </a>
          )}
          {member.github && (
            <a
              href={member.github}
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-xl bg-gradient-to-br from-foreground/10 to-muted hover:from-foreground/20 hover:to-muted/80 text-foreground hover:text-foreground transition-all duration-300 hover:scale-110 hover:rotate-6 hover:shadow-lg hover:shadow-primary/20"
              aria-label="GitHub"
            >
              <Github className="h-5 w-5" />
            </a>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
