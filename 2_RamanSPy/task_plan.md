# Task: Rebuild RamanSPy as a Modern Web App (Real UI MAX)

## 1. Objective
Refactor the existing Streamlit application into a high-performance, aesthetically premium web application using **Next.js (Frontend)** and **FastAPI (Backend)**. The goal is to achieve "Real UI MAX" - a modern, fluid, and highly interactive user experience.

## 2. Architecture
- **Frontend**: 
  - Framework: Next.js 14+ (App Router).
  - Styling: TailwindCSS (with custom design system for "Rich Aesthetics").
  - Icons: Lucide React.
  - Visualization: Recharts (for performance) or Plotly.js (for scientific detail).
  - State Management: React Context / Zustand.
  - Animations: Framer Motion.
- **Backend**:
  - Framework: FastAPI.
  - Processing: RamanSPy (existing logic).
  - Storage: In-memory session storage (to mimic `st.session_state` behavior for local use).

## 3. Implementation Plan

### Phase 1: Backend Foundation (FastAPI)
- [ ] Initialize FastAPI project structure (`/backend`).
- [ ] Create Session Manager to handle loaded spectral data.
- [ ] **Endpoint: Upload**: Handle file uploads (proprietary formats, CSV, etc.) and load into memory.
- [ ] **Endpoint: Data Retrieval**: Get spectral data arrays for visualization.
- [ ] **Endpoint: Preprocessing**: Expose `ramanspy` preprocessing methods (denoising, baseline correction, etc.).
- [ ] **Endpoint: Analysis**: Expose basic analysis features.

### Phase 2: Frontend "MAX" Design System (Next.js)
- [ ] Initialize Next.js project (`/frontend`).
- [ ] Configure TailwindCSS with a premium color palette (Dark mode focus, vibrant accents).
- [ ] Create **Layout Shell**:
  - Glassmorphic Sidebar.
  - Dynamic Header.
  - Smooth page transitions.
- [ ] Build **UI Components**:
  - `GradientButton`: Interactive buttons with glow effects.
  - `Card`: Glass panels for content.
  - `SpectralViewer`: High-performance interactive chart component.
  - `PipelineBuilder`: Drag-and-drop or step-based preprocessing UI.

### Phase 3: Feature Integration
- [ ] **Dashboard**: Overview of loaded data/projects.
- [ ] **Visualize Page**: Interactive exploring of spectra (zoom, pan, select).
- [ ] **Process Page**: Apply preprocessing steps and see real-time previews (Before/After).
- [ ] **Analysis Page**: Run analysis and show results.

### Phase 4: Polish & "Wow" Factors
- [ ] Add micro-animations (hover states, loading skeletons).
- [ ] Implement command palette (Cmd+K) for quick actions.
- [ ] Optimize chart performance for large datasets.
- [ ] SEO & Accessibility checks.

## 4. Current Status
- [x] Requirement Analysis.
- [ ] Environment Setup.
