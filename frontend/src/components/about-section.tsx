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
  const filename = name
    .toLowerCase()
    .replace(/,/g, "")
    .replace(/\s+/g, "-")
    .replace(/phd/g, "")
    .trim()
  
  // Return image path - images should be placed in public/team/ directory
  // Expected formats: jpg, jpeg, png, webp
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
    image: getImagePath("Pramisha Sapkota"),
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
  image: getImagePath("Rabindra Bista Phd"),
}

export function AboutSection() {
  return (
    <section id="about" className="py-20 bg-background">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
            About Us
          </h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            PulseAI is a research project focused on developing non-invasive blood pressure monitoring
            using advanced signal processing and machine learning techniques.
          </p>
        </div>

        {/* Supervisor */}
        <div className="mb-16">
          <h3 className="text-2xl font-semibold mb-8 text-center text-foreground/90">Supervisor</h3>
          <div className="flex justify-center">
            <SupervisorCard member={supervisor} />
          </div>
        </div>

        {/* Team Members */}
        <div id="team">
          <h3 className="text-2xl font-semibold mb-8 text-center text-foreground/90">Team Members</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {teamMembers.map((member) => (
              <TeamMemberCard key={member.name} member={member} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function SupervisorCard({ member }: { member: TeamMember }) {
  return (
    <Card className="relative overflow-hidden border-2 bg-gradient-to-br from-background via-background to-muted/20 hover:shadow-2xl transition-all duration-300 max-w-sm w-full group">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 via-purple-500/5 to-pink-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      <CardHeader className="text-center pb-4 relative z-10">
        <div className="flex justify-center mb-6">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-300" />
            <Avatar className="h-28 w-28 relative border-4 border-background shadow-lg">
              <AvatarImage src={member.image} alt={member.name} />
              <AvatarFallback className="text-2xl font-bold bg-gradient-to-br from-blue-500 to-purple-600 text-white">
                {member.initials}
              </AvatarFallback>
            </Avatar>
          </div>
        </div>
        <CardTitle className="text-2xl mb-2">{member.name}</CardTitle>
        <CardDescription className="text-base font-medium">{member.role}</CardDescription>
      </CardHeader>
      <CardContent className="pt-0 pb-6 relative z-10">
        <div className="flex justify-center gap-4">
          {member.linkedin && (
            <a
              href={member.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-full bg-muted hover:bg-blue-500/10 hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 hover:scale-110 hover:shadow-lg"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-5 w-5" />
            </a>
          )}
          {member.googleScholar && (
            <a
              href={member.googleScholar}
              target="_blank"
              rel="noopener noreferrer"
              className="p-3 rounded-full bg-muted hover:bg-amber-500/10 hover:text-amber-600 dark:hover:text-amber-400 transition-all duration-300 hover:scale-110 hover:shadow-lg"
              aria-label="Google Scholar"
            >
              <GraduationCap className="h-5 w-5" />
            </a>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function TeamMemberCard({ member }: { member: TeamMember }) {
  return (
    <Card className="relative overflow-hidden border bg-gradient-to-br from-background via-background to-muted/10 hover:shadow-xl hover:border-primary/20 transition-all duration-300 group">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/0 via-purple-500/0 to-pink-500/0 group-hover:from-blue-500/5 group-hover:via-purple-500/5 group-hover:to-pink-500/5 transition-all duration-300" />
      <CardHeader className="text-center pb-4 relative z-10 pt-6">
        <div className="flex justify-center mb-4">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-full blur-lg group-hover:blur-xl transition-all duration-300" />
            <Avatar className="h-24 w-24 relative border-2 border-muted group-hover:border-primary/30 transition-colors duration-300">
              <AvatarImage src={member.image} alt={member.name} />
              <AvatarFallback className="text-xl font-semibold bg-gradient-to-br from-blue-600/90 to-purple-600/90 text-white">
                {member.initials}
              </AvatarFallback>
            </Avatar>
          </div>
        </div>
        <CardTitle className="text-lg font-semibold mb-1">{member.name}</CardTitle>
        <CardDescription className="text-sm">{member.role}</CardDescription>
      </CardHeader>
      <CardContent className="pt-0 pb-6 relative z-10">
        <div className="flex justify-center gap-3">
          {member.linkedin && (
            <a
              href={member.linkedin}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2.5 rounded-full bg-muted hover:bg-blue-500/10 hover:text-blue-600 dark:hover:text-blue-400 transition-all duration-300 hover:scale-110 hover:shadow-md"
              aria-label="LinkedIn"
            >
              <Linkedin className="h-4 w-4" />
            </a>
          )}
          {member.github && (
            <a
              href={member.github}
              target="_blank"
              rel="noopener noreferrer"
              className="p-2.5 rounded-full bg-muted hover:bg-foreground/10 hover:text-foreground transition-all duration-300 hover:scale-110 hover:shadow-md"
              aria-label="GitHub"
            >
              <Github className="h-4 w-4" />
            </a>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
