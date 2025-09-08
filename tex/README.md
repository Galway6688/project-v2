# Interactive Multimodal GPT System - LaTeX Presentation

This folder contains the LaTeX source files for creating a professional presentation about the Interactive Multimodal GPT System project.

## Files Included

- **`main.tex`** - Main presentation file with all slides
- **`beamerthemeSimpleDarkBlue.sty`** - Custom Beamer theme file
- **`README.md`** - This file with usage instructions

## How to Use

### Option 1: Overleaf (Recommended)
1. Go to [Overleaf.com](https://www.overleaf.com)
2. Create a new project
3. Upload all files from this folder to your Overleaf project
4. Compile the `main.tex` file

### Option 2: Local LaTeX Installation
1. Ensure you have a LaTeX distribution installed (TeX Live, MiKTeX, etc.)
2. Install required packages:
   ```bash
   tlmgr install beamer tikz xcolor fontawesome
   ```
3. Compile the presentation:
   ```bash
   pdflatex main.tex
   ```

## Presentation Structure

The presentation includes:

1. **Title Page** - Project title and team information
2. **Overview** - Table of contents
3. **Core Features & Workflow** - Key capabilities and process flow
4. **Technical Architecture** - System design and code structure
5. **Implementation Status & Roadmap** - Current progress and future plans
6. **Project Summary** - Key achievements and technical highlights

## Customization

### Changing Colors
Edit the color definitions in `beamerthemeSimpleDarkBlue.sty`:
```latex
\definecolor{primary}{HTML}{2C3E50}
\definecolor{secondary}{HTML}{3498DB}
\definecolor{accent}{HTML}{E74C3C}
```

### Adding Content
Modify `main.tex` to add or modify slides. The structure follows standard Beamer conventions.

### Changing Theme
Replace `\usetheme{SimpleDarkBlue}` with another Beamer theme if desired.

## Features

- **Professional Design** - Clean, modern appearance
- **Responsive Layout** - Works well on different screen sizes
- **Custom Theme** - Consistent branding throughout
- **TikZ Diagrams** - Professional flowcharts and diagrams
- **Color-coded Sections** - Easy navigation and visual hierarchy

## Requirements

- LaTeX distribution with Beamer class
- TikZ package for diagrams
- FontAwesome package for icons
- XColor package for color management

## Output

The presentation will generate a PDF file with:
- 16:9 aspect ratio (widescreen)
- Professional color scheme
- Interactive navigation
- Print-friendly design 