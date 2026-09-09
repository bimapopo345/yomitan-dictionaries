/**
 * build.cjs — JMdict (JP→EN gloss) → translate to Indonesian via LibreTranslate → export Yomitan ZIP
 *
 * Requirements:
 * - JMdict.gz in same folder
 * - LibreTranslate running locally (127.0.0.1) on chosen port
 * - npm i fast-xml-parser yomichan-dict-builder
 *
 * Notes:
 * - Uses global fetch (Node >=18). Node 22 OK.
 * - Attribution is required by yomichan-dict-builder index builder.
 */

const fs = require("node:fs");
const zlib = require("node:zlib");
const path = require("node:path");
const { XMLParser } = require("fast-xml-parser");
const { Dictionary, DictionaryIndex, TermEntry } = require("yomichan-dict-builder");

const INPUT_GZ = "./JMdict.gz";
const OUTPUT_ZIP = "./jmdict-jp-id.zip";

// LibreTranslate config (custom port)
const LT_HOST = "127.0.0.1";
const LT_PORT = 13218; // <-- GANTI PORT DI SINI
const LT_URL = `http://${LT_HOST}:${LT_PORT}/translate`;

// Optional: if your LibreTranslate requires an API key, set it here; otherwise leave null.
const LT_API_KEY = null;

// Performance controls
const MAX_ENTRIES_WITH_OUTPUT = 5000; // mulai kecil dulu. Naikin pelan-pelan.
const SAVE_CACHE_EVERY = 200;         // simpan cache tiap N translate unik
const REQUEST_TIMEOUT_MS = 60_000;    // 60s per request

// Cache to avoid translating the same definition again
const CACHE_PATH = "./translate-cache.json";
let cache = {};
if (fs.existsSync(CACHE_PATH)) {
  try {
    cache = JSON.parse(fs.readFileSync(CACHE_PATH, "utf8"));
  } catch {
    cache = {};
  }
}

function gunzipToString(filePath) {
  const gz = fs.readFileSync(filePath);
  return zlib.gunzipSync(gz).toString("utf8");
}

function asArray(x) {
  if (!x) return [];
  return Array.isArray(x) ? x : [x];
}

function saveCache() {
  fs.writeFileSync(CACHE_PATH, JSON.stringify(cache, null, 2));
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    return res;
  } finally {
    clearTimeout(t);
  }
}

async function translateEnToId(text) {
  const q = (text || "").trim();
  if (!q) return "";

  // cache hit
  if (cache[q]) return cache[q];

  const payload = {
    q,
    source: "en",
    target: "id",
    format: "text",
  };
  if (LT_API_KEY) payload.api_key = LT_API_KEY;

  const res = await fetchWithTimeout(
    LT_URL,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    REQUEST_TIMEOUT_MS
  );

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`LibreTranslate HTTP ${res.status}: ${body}`);
  }

  const data = await res.json();
  const out = (data && data.translatedText) ? String(data.translatedText) : "";

  cache[q] = out;

  // periodic cache save
  if (Object.keys(cache).length % SAVE_CACHE_EVERY === 0) {
    saveCache();
  }

  return out;
}

(async () => {
  if (!fs.existsSync(INPUT_GZ)) {
    console.error(`❌ File tidak ketemu: ${INPUT_GZ}`);
    process.exit(1);
  }

  // Quick ping to LibreTranslate
  console.log(`Checking LibreTranslate at ${LT_URL} ...`);
  try {
    const test = await translateEnToId("hello");
    console.log("LibreTranslate OK. Test:", test);
  } catch (e) {
    console.error("❌ LibreTranslate gak bisa diakses / error:", e.message);
    console.error(`Pastikan kamu jalanin: libretranslate --host 127.0.0.1 --port ${LT_PORT} --load-only en,id`);
    process.exit(1);
  }

  console.log("Reading & gunzip:", INPUT_GZ);
  const xml = gunzipToString(INPUT_GZ);

  console.log("Parsing JMdict XML...");
  const parser = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "@_" });
  const data = parser.parse(xml);
  const entries = asArray(data?.JMdict?.entry);
  console.log("Total JMdict entries:", entries.length);

  // Build dictionary
  const dictionary = new Dictionary({ fileName: OUTPUT_ZIP });

  // Attribution REQUIRED by builder
  const index = new DictionaryIndex()
    .setTitle("JMdict JP→ID (machine translated)")
    .setRevision("1.0")
    .setAuthor("samit")
    .setDescription("JMdict English glosses translated to Indonesian using local LibreTranslate.")
    .setAttribution("JMdict/EDICT (EDRDG) — CC BY-SA 4.0 — https://www.edrdg.org/edrdg/licence.html")
    .build();

  await dictionary.setIndex(index);

  let keptEntries = 0;
  let addedTerms = 0;
  let skippedNoEnglish = 0;
  let failedTranslate = 0;

  // Main loop
  for (const e of entries) {
    if (keptEntries >= MAX_ENTRIES_WITH_OUTPUT) break;

    const kebs = asArray(e.k_ele).map((k) => k.keb).filter(Boolean);
    const rebs = asArray(e.r_ele).map((r) => r.reb).filter(Boolean);
    const senses = asArray(e.sense);

    // Extract English glosses:
    // - If gloss is string (no xml:lang), treat as English (common in JMdict)
    // - If gloss is object and xml:lang is eng/en, include it
    const enGlosses = [];
    for (const s of senses) {
      for (const g of asArray(s.gloss)) {
        if (typeof g === "string") {
          enGlosses.push(g);
        } else {
          const lang = g?.["@_xml:lang"] || "eng";
          const text = g?.["#text"];
          if ((lang === "eng" || lang === "en") && text) enGlosses.push(text);
        }
      }
    }

    if (enGlosses.length === 0) {
      skippedNoEnglish++;
      continue;
    }

    const headwords = kebs.length ? kebs : rebs;
    if (headwords.length === 0) continue;

    const englishDef = enGlosses.join("; ");

    let indoDef = "";
    try {
      indoDef = await translateEnToId(englishDef);
    } catch (err) {
      failedTranslate++;
      // fallback: store English if translate fails (optional)
      indoDef = englishDef;
    }

    if (!indoDef.trim()) continue;

    for (const hw of headwords) {
      const term = new TermEntry(hw)
        .setReading(rebs[0] || "")
        .addDefinition(indoDef)
        .build();

      await dictionary.addTerm(term);
      addedTerms++;
    }

    keptEntries++;
    if (keptEntries % 200 === 0) {
      console.log(`Progress: keptEntries=${keptEntries}, terms=${addedTerms}, cache=${Object.keys(cache).length}`);
    }
  }

  // final cache save
  saveCache();

  console.log("Entries kept:", keptEntries);
  console.log("Terms added:", addedTerms);
  console.log("Skipped (no English gloss):", skippedNoEnglish);
  console.log("Translate failures (fallback to English):", failedTranslate);

  console.log("Exporting zip...");
  const stats = await dictionary.export("./");
  console.table(stats);

  console.log("✅ DONE. Output:", path.resolve(OUTPUT_ZIP));
})();
