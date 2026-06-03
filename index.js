import 'dotenv/config';
import { GoogleGenerativeAI } from "@google/generative-ai";
import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import path from "path";
import { fileURLToPath } from 'url';
import rateLimit from 'express-rate-limit';

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({
  model: "gemini-2.5-flash",
  systemInstruction: "You are a master of haiku in the tradition of Bashō, Buson, and Issa. You will receive the name of a place on Earth — a city, region, and country. Do not use these names in the poem. Instead, consider what this place means to those who belong to it: someone born here, someone who left and still carries it, someone visiting for the first time and feeling something stir without knowing why. Consider the people who have lived on this ground — those born here, those who left and still carry it, those who arrived and never quite left. Let the haiku begin from a human moment: a gesture, a gathering, a departure, something passed between people. The poem should feel like something a person from this place would read and quietly recognise. Strict 5-7-5 syllable structure. Draw on wabi-sabi, mono no aware, and ma. Return only the three lines, separated by newlines. No titles, no commentary."
});

const app = express();
const port = 3000;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Rate limiting — keep within Gemini free tier (15 req/min, 1500 req/day)
const apiLimiter = rateLimit({
  windowMs: 60 * 1000,   // 1 minute
  max: 12,               // 12 requests per IP per minute (some headroom under the 15/min limit)
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "too many requests — please wait a moment" },
});

app.use(bodyParser.json({ limit: '2mb' }));
app.use(cors());
app.use('/', express.static(path.join(__dirname, 'public')));

app.post("/api/haiku", apiLimiter, async (req, res) => {
  try {
    const { message } = req.body;
    if (!message || typeof message !== 'string' || message.length > 200) {
      return res.status(400).json({ error: "invalid request" });
    }

    const result = await model.generateContent(message);
    const text = result.response.text();

    res.json({
      completion: {
        content: text
      }
    });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "could not generate haiku" });
  }
});

app.post("/api/emotion", apiLimiter, async (req, res) => {
  try {
    const { frame } = req.body;
    if (!frame || typeof frame !== 'string') return res.status(400).json({ error: "no frame" });

    const visionModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
    const result = await visionModel.generateContent([
      {
        inlineData: { mimeType: "image/jpeg", data: frame }
      },
      `Look at the face in this image. Return ONLY a valid JSON object with exactly these two fields:
{"emotion":"neutral","intensity":0.5}
The emotion must be one of: happy, sad, angry, fearful, surprised, disgusted, neutral, calm, excited, anxious
Intensity is a float 0.0–1.0 reflecting how strongly the emotion reads.
If no face is visible or the image is unclear, return {"emotion":"neutral","intensity":0.5}
Return only the JSON object. No explanation, no markdown.`
    ]);

    const text  = result.response.text().trim();
    const match = text.match(/\{[\s\S]*?\}/);
    if (!match) throw new Error("no JSON in response");
    res.json(JSON.parse(match[0]));
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "could not analyse emotion" });
  }
});

// ── ambulant ──────────────────────────────────────────────────────────────────
const ambulantLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 4,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "too many requests — please wait a moment" },
});

const sceneModel = genAI.getGenerativeModel({
  model: "gemini-2.5-flash",
  systemInstruction: `You are a psychogeographer and landscape artist working in the tradition of the Situationist International. You receive movement biometrics extracted from a 10-second walk and must divine what landscape the body was unconsciously enacting.

The landscape IS the body. The body does not walk through a landscape — it performs one. Your role is to name that landscape precisely and render it in language and image.

Each metric is a float 0–1:
- cadence:   0 = very slow, meditative pace   /  1 = fast urban stride
- vertOsc:   0 = gliding, flat terrain         /  1 = high bounce, rough/uneven ground
- symmetry:  0 = wild, irregular, organic      /  1 = formal, ordered, bilateral
- armSwing:  0 = constrained, enclosed space   /  1 = open, unimpeded, vast
- lean:      0 = upright, contemplative        /  1 = forward, purposeful, transitional
- energy:    0 = still, minimal movement       /  1 = vigorous, dynamic

Return ONLY valid JSON with these three fields:
{
  "landscape": "concise place name, 2–5 words",
  "prose": "2–3 sentences of psychogeographic prose written in present tense. Describe only the landscape — its textures, light quality, ecological or geological character, atmosphere. No mention of walking, bodies, or people.",
  "imagePrompt": "A detailed prompt for Imagen 3. Describe a cinematic landscape: subject, specific light (time of day, quality), ecological/geological detail, atmosphere, mood. Reference a specific painter, photographer or visual tradition. No people, no figures. End with: photorealistic, cinematic, 4:3 aspect ratio, high detail."
}`
});

app.post("/api/ambulant/scene", ambulantLimiter, async (req, res) => {
  try {
    const { metrics } = req.body;
    if (!metrics || typeof metrics !== 'object') return res.status(400).json({ error: "invalid metrics" });

    const { cadence=0, vertOsc=0, symmetry=0, armSwing=0, lean=0, energy=0 } = metrics;

    // Step 1: generate scene description + image prompt via Gemini
    const sceneRes = await sceneModel.generateContent(
      `cadence:${cadence.toFixed(2)} vertOsc:${vertOsc.toFixed(2)} symmetry:${symmetry.toFixed(2)} armSwing:${armSwing.toFixed(2)} lean:${lean.toFixed(2)} energy:${energy.toFixed(2)}`
    );
    const raw = sceneRes.response.text().trim();
    let scene;
    try {
      scene = JSON.parse(raw);
    } catch {
      const cb = raw.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      const block = cb ? cb[1] : raw.match(/\{[\s\S]*\}/)?.[0];
      if (!block) { console.error("raw Gemini response:", raw); throw new Error("bad scene JSON"); }
      scene = JSON.parse(block);
    }

    // Step 2: generate image via Imagen 3
    const imgRes = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${process.env.GEMINI_API_KEY}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          instances: [{ prompt: scene.imagePrompt }],
          parameters: { sampleCount: 1, aspectRatio: "4:3" }
        })
      }
    );
    const imgData = await imgRes.json();
    if (!imgRes.ok || imgData.error) {
      console.error("Imagen error:", JSON.stringify(imgData));
      throw new Error(`Imagen: ${imgData.error?.message || imgRes.status}`);
    }
    const prediction = imgData.predictions?.[0];
    if (!prediction?.bytesBase64Encoded) {
      console.error("Imagen no prediction:", JSON.stringify(imgData));
      throw new Error("image generation failed");
    }

    res.json({
      landscape: scene.landscape,
      prose:     scene.prose,
      image:     prediction.bytesBase64Encoded,
      mimeType:  prediction.mimeType || "image/png"
    });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ error: "could not generate scene" });
  }
});

// ── caterwaul ─────────────────────────────────────────────────────────────────
const catLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "too many requests — please wait a moment" },
});

const catModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });

app.post("/api/caterwaul", catLimiter, async (req, res) => {
  try {
    const { audio } = req.body;
    if (!audio || typeof audio !== 'string' || audio.length > 1_500_000) {
      return res.status(400).json({ error: "invalid" });
    }

    const result = await catModel.generateContent([
      { inlineData: { mimeType: "audio/wav", data: audio } },
      `You are an expert in cat vocalisation science, familiar with the research of Susanne Schötz (2013, 2016), John Bradshaw (2013), Karen McComb et al. (2009), and Mildred Moelk (1944).

Listen carefully to this audio clip. Be strict: only identify genuine cat vocalisations — not human speech, music, background noise, other animals, or silence.

If a cat vocalisation IS present, return this JSON:
{"sound":"meow"|"purr"|"hiss"|"caterwaul"|"chirp"|"general","meaning":"plain English meaning in a short phrase","science":"one sentence of scientific context with author citation","confidence":0.0-1.0}

If no cat vocalisation is present, return exactly:
{"sound":"none"}

Return only valid JSON, no other text.`
    ]);

    const text  = result.response.text().trim();
    const match = text.match(/\{[\s\S]*?\}/);
    if (!match) return res.json({ sound: "none" });
    try {
      res.json(JSON.parse(match[0]));
    } catch {
      res.json({ sound: "none" });
    }
  } catch (err) {
    console.error("/api/caterwaul:", err);
    res.status(500).json({ sound: "none", error: err.message });
  }
});

if (process.env.NODE_ENV !== 'production') {
  app.listen(port, () => {
    console.log(`Haiku app listening at http://localhost:${port}`);
  });
}

export default app;
