// Copyright 2026 Cisco Systems, Inc. and its affiliates
//
// SPDX-License-Identifier: Apache-2.0

import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

const enableMocking = async () => {
  // In Vite, use import.meta.env.DEV instead of process.env.NODE_ENV
  if (!import.meta.env.DEV) {
    console.log("Not in development mode, skipping MSW");
    return;
  }
  console.log("Starting MSW");
  const { worker } = await import("./mocks/browser");
  console.log("MSW started successfully");
  return await worker.start();
};

// Enable mocking in the background

enableMocking().then(() => {
  const root = createRoot(document.getElementById("root")!);
  root.render(<App />);
});
