# Desktop installer drop folder

Put the prebuilt CoreCS desktop installer here, for example `CoreCS_0.1.0_x64-setup.exe`.
The server uses the newest `*-setup.exe` in this folder for:

- the "Download student app (.zip)" button on the deploy page (installer + `corecs-model.json`);
- the `runtime/desktop/` folder inside every lecturer bundle.

The installer must be built from `GLOBAL/runtime/desktop` (`npm run tauri build -- --bundles nsis`).
Only that build has the first-launch model download and the NSIS hook that picks up
`corecs-model.json`. A build of the standalone `corecs/` app does neither.

`*.exe` files here are git-ignored. No restart is needed after replacing the file.
`GLOBAL_DESKTOP_INSTALLER` (a full path) overrides this folder.
