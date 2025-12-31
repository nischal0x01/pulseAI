# Frontend Directory Structure

This document describes the organization of the PulseAI frontend codebase.

## Directory Structure

```
frontend/
├── src/                    # Source code directory
│   ├── app/               # Next.js App Router
│   │   ├── globals.css    # Global styles
│   │   ├── layout.tsx     # Root layout
│   │   └── page.tsx       # Home page
│   ├── components/        # React components
│   │   ├── ui/           # Shadcn UI components
│   │   ├── about-section.tsx
│   │   ├── navbar.tsx
│   │   ├── footer.tsx
│   │   └── ...
│   ├── lib/              # Utility functions
│   │   ├── utils.ts
│   │   ├── signal-context.tsx
│   │   └── ...
│   └── hooks/            # Custom React hooks
│       └── ...
├── public/                # Static assets
│   └── team/             # Team member photos
│       ├── README.md
│       └── [member-photos]
├── components.json        # Shadcn configuration
├── tsconfig.json          # TypeScript configuration
├── next.config.mjs        # Next.js configuration
└── package.json           # Dependencies

```

## Key Features

### Image Support for Team Members

Team member photos should be placed in `public/team/` with the following naming convention:

- `rabindra-bista-phd.jpg` - Supervisor
- `suvesh-gurung.jpg` - Team Member
- `pramisha-sapkota.jpg` - Team Member
- `nischal-subedi.jpg` - Team Member
- `arwin-shrestha.jpg` - Team Member

The `about-section.tsx` component automatically generates image paths based on team member names. If an image is not found, the component will fallback to displaying initials with a gradient background.

### Path Aliases

All imports use the `@/` alias which points to `src/`:
- `@/components` → `src/components`
- `@/lib` → `src/lib`
- `@/hooks` → `src/hooks`

## Adding Team Photos

1. Place photos in `public/team/` directory
2. Name files according to the convention above
3. Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`
4. Recommended size: 400x400px or larger (square format)
5. The component will automatically use the photos if they exist

