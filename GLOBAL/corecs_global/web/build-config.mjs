// Vercel build step: writes config.js with the backend URL from the GLOBAL_API_URL env var.
// Unset -> null, so app.js falls back to the local dev backend (http://127.0.0.1:8000).
import { writeFileSync } from "node:fs";

const url = (process.env.GLOBAL_API_URL || "").trim().replace(/\/+$/, "") || null;
writeFileSync(new URL("./config.js", import.meta.url), `window.GLOBAL_API_URL = ${JSON.stringify(url)};\n`);
console.log(`config.js: GLOBAL_API_URL = ${url ?? "(unset, using http://127.0.0.1:8000)"}`);
