# Desktop App

This folder contains the current Electron prototype for the investment research platform.

Key files:

- `main.js`: Electron main process entry
- `preload.js`: preload bridge
- `renderer.js`: renderer logic
- `index.html`: renderer shell
- `forge.config.js`: Electron Forge packaging configuration

Useful commands:

- `npm start`: run the Electron app in development
- `npm run package`: package the app locally
- `npm run make`: build distributable artifacts through Electron Forge

This app is still an early shell. Most domain logic currently lives in `Quantapp` and the notebooks.
