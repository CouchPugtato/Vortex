const { contextBridge, ipcRenderer } = require("electron");

function makeSubscriber(channel) {
  return (handler) => {
    const listener = (_evt, payload) => handler(payload);
    ipcRenderer.on(channel, listener);
    return () => ipcRenderer.removeListener(channel, listener);
  };
}

contextBridge.exposeInMainWorld("vortexApi", {
  bootstrap: () => ipcRenderer.invoke("bootstrap"),
  chooseFolder: () => ipcRenderer.invoke("choose-folder"),
  saveAppConfig: (config) => ipcRenderer.invoke("save-app-config", config),
  loadRuntimeConfig: (runtimePath) => ipcRenderer.invoke("load-runtime-config", runtimePath),
  saveRuntimeConfig: (runtimePath, config) =>
    ipcRenderer.invoke("save-runtime-config", runtimePath, config),
  deployStart: (settings) => ipcRenderer.invoke("deploy-start", settings),
  monitorStart: (settings) => ipcRenderer.invoke("monitor-start", settings),
  monitorStop: () => ipcRenderer.invoke("monitor-stop"),
  previewStart: (settings) => ipcRenderer.invoke("preview-start", settings),
  previewStop: () => ipcRenderer.invoke("preview-stop"),
  listRemoteCameras: (settings) => ipcRenderer.invoke("list-remote-cameras", settings),
  applyMonitorPreset: (appConfig, preset) =>
    ipcRenderer.invoke("apply-monitor-preset", appConfig, preset),
  onLog: makeSubscriber("log"),
  onMonitorLine: makeSubscriber("monitor-line"),
  onMonitorState: makeSubscriber("monitor-state"),
  onPreviewState: makeSubscriber("preview-state"),
  onPreviewFrame: makeSubscriber("preview-frame"),
  onDeployProgress: makeSubscriber("deploy-progress"),
  onMonitorStartProgress: makeSubscriber("monitor-start-progress")
});
