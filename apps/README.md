# Apps

Application runtime surfaces live here.

Current layout:

- [`api/`](api/README.md): reserved for the future FastAPI service
- [`web/`](web/README.md): reserved for the future browser-based frontend
- [`worker/`](worker/README.md): reserved for background jobs and scheduled tasks

The long-term direction is to move reusable logic out of notebooks and into `Quantapp`, expose it through the API, and consume it from the browser frontend and background workers.
