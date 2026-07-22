## 1. Architecture Design
```mermaid
flowchart TD
    A["Browser UI"] --> B["FastAPI Web Layer"]
    B --> C["Script Analysis Pipeline"]
    B --> D["Plan Storage"]
    B --> E["Render Process Manager"]
    C --> F["Scene Splitter and Timing"]
    C --> G["Query Generator"]
    C --> H["Clip Search and Ranking"]
    H --> I["External Stock APIs"]
    H --> J["Vision Verification"]
    E --> K["Scene Builder"]
    E --> L["Video Assembler"]
    K --> M["Temp Scene Files"]
    L --> N["Output Video Files"]
    J --> O["Local ML Models"]
```

## 2. Technology Description
- Frontend: Server-rendered HTML/CSS/JavaScript first, with a path to migrate to React if a richer timeline editor is needed later
- Initialization Tool: Existing Python application structure, no separate JS build system required for MVP
- Backend: FastAPI + Uvicorn
- Processing Engine: Existing local Python pipeline modules for analysis, alignment, search, scene building, and export
- Media Stack: ffmpeg via `imageio-ffmpeg`
- ML / Matching: Faster Whisper for voice alignment, CLIP-based vision verification for image/video ranking
- Storage: Local filesystem JSON and generated files; no database required for MVP
- External Services: Pexels API and Pixabay API for stock footage discovery

## 3. Route Definitions
| Route | Purpose |
|-------|---------|
| `/` | Main AI editing workspace |
| `/api/analyze` | Accept script and optional voiceover, generate timed beat plan and editable queries |
| `/api/plan` | Save user overrides to generated beat queries or locked matches |
| `/api/render` | Start local rendering using current approved plan |
| `/api/stop` | Stop an active render process |
| `/api/progress` | Return current processing state, completed beats, and total beats |
| `/api/outputs` | List recent output videos and audit assets |
| `/outputs/{name}` | Download a finished render |

## 4. API Definitions
```ts
type BeatPlanItem = {
  scene: number;
  start: number;
  end: number;
  text: string;
  keywords: string[];
  confidence?: number;
  locked?: boolean;
  candidates?: VisualCandidate[];
};

type VisualCandidate = {
  source: "pexels" | "pixabay";
  video_id: string;
  query: string;
  desc: string;
  sim: number;
  thumb?: string;
};

type AnalyzeResponse = {
  scenes: BeatPlanItem[];
  duration: number;
};

type RenderRequest = {
  scene_duration: number;
  cards: boolean;
  sfx: boolean;
  vision: boolean;
  motion: boolean;
};

type ProgressResponse = {
  state: "idle" | "rendering" | "done" | "error";
  done: number;
  total: number;
};
```

## 5. Server Architecture Diagram
```mermaid
flowchart TD
    A["FastAPI Route"] --> B["Request Validation"]
    B --> C["Pipeline Service Layer"]
    C --> D["Plan File Manager"]
    C --> E["Render Process Manager"]
    C --> F["Search and Matching Modules"]
    E --> G["Checkpoint State"]
    F --> H["External API Clients"]
    F --> I["Vision Verification"]
```

## 6. Data Model
### 6.1 Data Model Definition
```mermaid
erDiagram
    SCRIPT_RUN ||--o{ BEAT : "produces"
    BEAT ||--o{ VISUAL_CANDIDATE : "matches"
    SCRIPT_RUN ||--o{ OUTPUT_ASSET : "exports"

    SCRIPT_RUN {
        string run_id
        string script_path
        string voice_path
        float beat_duration
        string state
        datetime created_at
    }

    BEAT {
        int scene
        float start
        float end
        string text
        string primary_query
        boolean locked
        float confidence
    }

    VISUAL_CANDIDATE {
        string source
        string asset_id
        string query
        string description
        float similarity
        string thumbnail_url
    }

    OUTPUT_ASSET {
        string file_name
        string file_type
        string path
        int size_mb
    }
```

### 6.2 Data Definition Language
No SQL schema is required for MVP because the tool is local-first and file-backed.

Planned filesystem artifacts:

```text
my_scripts/ui_script.txt
my_scripts/ui_voiceover.mp3
my_scripts/ui_plan.json
temp/progress.json
output/<script>_<timestamp>.mp4
output/audit/*
```

Recommended implementation notes:
- Change the current analysis default from 6-second scene export to 3-second beat export for this product mode.
- Extend the plan JSON schema to carry confidence score, candidate previews, and a locked-selection marker.
- Keep all long-running render work in a subprocess so the FastAPI server remains responsive.
- Preserve checkpoint-based resume behavior so failed or interrupted renders continue cleanly.
