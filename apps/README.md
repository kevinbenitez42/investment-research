# Apps

Application runtime surfaces live here.

Current layout:

- [`desktop/`](desktop/README.md): active Electron shell prototype
- [`api/`](api/README.md): reserved for the future FastAPI service
- [`web/`](web/README.md): reserved for the future browser-based frontend
- [`worker/`](worker/README.md): reserved for background jobs and scheduled tasks

The long-term direction is to move reusable logic out of notebooks and into `Quantapp`, then expose that functionality through these app entry points.
