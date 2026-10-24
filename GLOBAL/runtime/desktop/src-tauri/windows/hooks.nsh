; NSIS installer hooks (bundle.windows.nsis.installerHooks in tauri.conf.json).
;
; CoreCS Global hands lecturers a "student app" ZIP: this generic installer plus a
; corecs-model.json naming the lecturer's model on the Hugging Face Hub. Copying the manifest
; from the installer's folder into the install folder lets the app pull that model on first
; launch (src/provision.rs). Without a manifest the app pulls the default model instead.

!macro NSIS_HOOK_POSTINSTALL
  IfFileExists "$EXEDIR\corecs-model.json" 0 +2
    CopyFiles /SILENT "$EXEDIR\corecs-model.json" "$INSTDIR\corecs-model.json"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  Delete "$INSTDIR\corecs-model.json"
!macroend
