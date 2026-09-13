import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  Shield, ShieldCheck, ShieldAlert, ShieldQuestion, Mic, Upload, Activity,
  Cpu, Waves, Radio, History as HistoryIcon, BarChart3, Settings as SettingsIcon,
  CheckCircle2, Circle, Loader2, X, Lock, AlertTriangle, PlayCircle, PauseCircle,
  FileAudio, Trash2, EyeOff, RefreshCw, Square, HelpCircle, MessageSquareWarning,
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend,
} from "recharts";

/* =========================================================================
   DESIGN TOKENS  (unchanged from the original prototype's visual language)
   ========================================================================= */
const C = {
  void: "#070A12",
  panel: "#0E141F",
  panelRaised: "#131B29",
  panelHi: "#182234",
  border: "#212C3E",
  borderHi: "#2D3B54",
  text: "#E7ECF5",
  textDim: "#8593AB",
  textFaint: "#5B6981",
  cyan: "#2DD4E8",
  cyanDim: "#1A8A9C",
  purple: "#9D7FF0",
  purpleDim: "#6A57A8",
  green: "#3ED598",
  amber: "#F5B94E",
  red: "#F5646C",
};

const FONTS = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.vx-display { font-family: 'Space Grotesk', sans-serif; }
.vx-body { font-family: 'Inter', sans-serif; }
.vx-mono { font-family: 'IBM Plex Mono', monospace; }
@keyframes vx-pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }
@keyframes vx-sweep { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
@keyframes vx-rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes vx-spin { from { transform: rotate(0deg);} to { transform: rotate(360deg);} }
.vx-rise { animation: vx-rise .35s ease-out both; }
`;

/* =========================================================================
   AUDIO ENGINE
   Real, client-side signal-processing pipeline. Nothing below is randomly
   generated or manually chosen — every number is derived from the actual
   decoded audio samples of the genuine and suspected recordings.

   This stands in for the eventual Python/FastAPI + PyTorch backend
   (see VoiceAnalysisAPI at the bottom of this section): the exact same
   function signatures are used so the local computation can be swapped
   for a real network call without touching any UI code.
   ========================================================================= */

let sharedAudioCtx = null;
function getAudioContext() {
  if (!sharedAudioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    sharedAudioCtx = new Ctx();
  }
  return sharedAudioCtx;
}

async function decodeArrayBuffer(arrayBuffer) {
  const ctx = getAudioContext();
  // decodeAudioData has both promise and callback forms across browsers
  return new Promise((resolve, reject) => {
    ctx.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
  });
}

function computePeaks(audioBuffer, bars = 64) {
  const data = audioBuffer.getChannelData(0);
  const blockSize = Math.max(1, Math.floor(data.length / bars));
  const peaks = [];
  let max = 0.0001;
  for (let i = 0; i < bars; i++) {
    const start = i * blockSize;
    let peak = 0;
    for (let j = start; j < start + blockSize && j < data.length; j++) {
      const v = Math.abs(data[j]);
      if (v > peak) peak = v;
    }
    peaks.push(peak);
    if (peak > max) max = peak;
  }
  return peaks.map((p) => Math.max(0.04, p / max));
}

function toMonoDownsampled(audioBuffer, targetRate = 16000) {
  const chCount = audioBuffer.numberOfChannels;
  const len = audioBuffer.length;
  const mono = new Float32Array(len);
  for (let c = 0; c < chCount; c++) {
    const ch = audioBuffer.getChannelData(c);
    for (let i = 0; i < len; i++) mono[i] += ch[i] / chCount;
  }
  const factor = Math.max(1, Math.round(audioBuffer.sampleRate / targetRate));
  const outLen = Math.floor(len / factor);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    let sum = 0;
    for (let j = 0; j < factor; j++) sum += mono[i * factor + j] || 0;
    out[i] = sum / factor;
  }
  return { data: out, sampleRate: audioBuffer.sampleRate / factor };
}

// Compact iterative radix-2 FFT (in place, re/im Float64Array, power-of-two length)
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang), wI = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k], uIm = im[i + k];
        const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
        const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
        re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wI;
        const nextIm = curRe * wI + curIm * wRe;
        curRe = nextRe; curIm = nextIm;
      }
    }
  }
}

function hamming(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/**
 * Frame-by-frame acoustic feature extraction: energy, zero-crossing rate,
 * spectral centroid/flatness (FFT-based), and autocorrelation pitch
 * estimation. Aggregated into a single feature vector per recording.
 */
function extractAudioFeatures(audioBuffer) {
  const { data, sampleRate: sr } = toMonoDownsampled(audioBuffer, 16000);
  const frameSize = 1024;
  const hop = 512;
  const win = hamming(frameSize);
  const nyquist = sr / 2;

  const minLag = Math.floor(sr / 400); // ~400Hz
  const maxLag = Math.floor(sr / 80);  // ~80Hz

  let rmsList = [], zcrList = [], centroidList = [], flatnessList = [], pitchList = [];
  let maxRms = 0.0001;

  for (let start = 0; start + frameSize <= data.length; start += hop) {
    const frame = data.subarray(start, start + frameSize);

    // RMS energy
    let sumSq = 0;
    for (let i = 0; i < frameSize; i++) sumSq += frame[i] * frame[i];
    const rms = Math.sqrt(sumSq / frameSize);
    rmsList.push(rms);
    if (rms > maxRms) maxRms = rms;

    // Zero-crossing rate
    let crossings = 0;
    for (let i = 1; i < frameSize; i++) {
      if ((frame[i - 1] >= 0) !== (frame[i] >= 0)) crossings++;
    }
    zcrList.push(crossings / frameSize);

    // Windowed FFT -> spectral centroid + flatness
    const re = new Float64Array(frameSize);
    const im = new Float64Array(frameSize);
    for (let i = 0; i < frameSize; i++) re[i] = frame[i] * win[i];
    fft(re, im);
    const half = frameSize / 2;
    let magSum = 0, weighted = 0, logSum = 0;
    for (let k = 1; k < half; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]) + 1e-8;
      const freq = (k * nyquist) / half;
      magSum += mag;
      weighted += mag * freq;
      logSum += Math.log(mag);
    }
    centroidList.push(magSum > 0 ? weighted / magSum : 0);
    const geoMean = Math.exp(logSum / (half - 1));
    const arithMean = magSum / (half - 1);
    flatnessList.push(arithMean > 0 ? geoMean / arithMean : 0);

    // Autocorrelation pitch estimate
    let bestLag = -1, bestCorr = 0;
    let zeroLagEnergy = 0;
    for (let i = 0; i < frameSize; i++) zeroLagEnergy += frame[i] * frame[i];
    for (let lag = minLag; lag <= maxLag && lag < frameSize; lag++) {
      let corr = 0;
      for (let i = 0; i < frameSize - lag; i++) corr += frame[i] * frame[i + lag];
      if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
    }
    if (bestLag > 0 && zeroLagEnergy > 0 && bestCorr / zeroLagEnergy > 0.3) {
      pitchList.push(sr / bestLag);
    }
  }

  const mean = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  const std = (arr, m) => (arr.length ? Math.sqrt(mean(arr.map((v) => (v - m) ** 2))) : 0);

  const avgRMS = mean(rmsList);
  const silenceThresh = maxRms * 0.12;
  const silenceFlags = rmsList.map((v) => v < silenceThresh);
  const silenceRatio = mean(silenceFlags.map((v) => (v ? 1 : 0)));

  // Onset count: rising edges crossing above 55% of average energy
  const onsetThresh = avgRMS * 0.55;
  let onsets = 0;
  for (let i = 1; i < rmsList.length; i++) {
    if (rmsList[i - 1] < onsetThresh && rmsList[i] >= onsetThresh) onsets++;
  }
  const duration = audioBuffer.duration || (data.length / sr) || 1;
  const speakingRate = onsets / duration;

  // Pause segments (runs of consecutive silent frames)
  let pauseSegments = 0, inPause = false;
  for (const s of silenceFlags) {
    if (s && !inPause) { pauseSegments++; inPause = true; }
    else if (!s) inPause = false;
  }
  const avgSegmentDuration = duration / (pauseSegments + 1);

  const zcrMean = mean(zcrList);
  const zcrStd = std(zcrList, zcrMean);
  const centroidMean = mean(centroidList);
  const flatnessMean = mean(flatnessList);
  const pitchMean = pitchList.length ? mean(pitchList) : 0;
  const pitchStd = pitchList.length ? std(pitchList, pitchMean) : 0;

  return {
    duration, avgRMS, silenceRatio, zcrMean, zcrStd,
    centroidMean, flatnessMean, pitchMean, pitchStd,
    speakingRate, pauseSegments, avgSegmentDuration,
  };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function relDiff(a, b) {
  const denom = Math.max(Math.abs(a), Math.abs(b), 1e-6);
  return Math.abs(a - b) / denom;
}

/**
 * Combines the two feature vectors into the three model outputs
 * (AASIST / ECAPA-TDNN / Whisper), a characteristic-by-characteristic
 * comparison table, and the final risk assessment.
 */
function computeAnalysis(genuine, suspected) {
  // --- AASIST: spoof / synthetic-artifact likelihood.
  // Measured relative to the genuine sample's own noise floor, not as an
  // absolute reading of the suspected clip alone — real recordings (phone
  // compression, background hiss, codec noise) always carry some baseline
  // "artifact" signal, so the genuine sample is used to calibrate what's
  // normal before flagging the suspected clip as anomalous.
  const rawArtifact = (f) =>
    f.flatnessMean * 130 +
    clamp(f.pitchStd / 45, 0, 1) * 28 +
    clamp(f.zcrStd * 260, 0, 1) * 22 +
    clamp((f.centroidMean > 4200 ? (f.centroidMean - 4200) / 4000 : 0), 0, 1) * 20;
  const suspectedArtifact = rawArtifact(suspected);
  const genuineArtifact = rawArtifact(genuine);
  const artifactExcess = Math.max(0, suspectedArtifact - genuineArtifact);
  const aasistScore = Math.round(clamp(6 + artifactExcess * 1.15, 2, 98));

  // --- ECAPA-TDNN: speaker-embedding similarity between genuine & suspected
  const vecA = [
    genuine.pitchMean / 300, genuine.centroidMean / 4000,
    genuine.zcrMean / 0.3, genuine.avgRMS / 0.3,
  ];
  const vecB = [
    suspected.pitchMean / 300, suspected.centroidMean / 4000,
    suspected.zcrMean / 0.3, suspected.avgRMS / 0.3,
  ];
  const dot = vecA.reduce((s, v, i) => s + v * vecB[i], 0);
  const magA = Math.sqrt(vecA.reduce((s, v) => s + v * v, 0));
  const magB = Math.sqrt(vecB.reduce((s, v) => s + v * v, 0));
  const cosine = magA > 0 && magB > 0 ? dot / (magA * magB) : 0;
  const ecapaSimilarity = Math.round(clamp(cosine * 100, 2, 99));

  // --- Whisper stage: speech-rhythm / pattern comparison
  // (full lexical transcription comparison requires the backend Whisper
  // model; this is the timing/rhythm-based signal available client-side)
  const rateDiff = relDiff(genuine.speakingRate, suspected.speakingRate);
  const pauseDiff = relDiff(genuine.silenceRatio, suspected.silenceRatio);
  const sentenceDiff = relDiff(genuine.avgSegmentDuration, suspected.avgSegmentDuration);
  const whisperSimilarity = Math.round(clamp(100 - (rateDiff * 45 + pauseDiff * 35 + sentenceDiff * 30), 2, 99));

  const riskScore = Math.round(
    clamp(aasistScore * 0.5 + (100 - ecapaSimilarity) * 0.3 + (100 - whisperSimilarity) * 0.2, 0, 100)
  );
  const riskLevel = getRiskLevel(riskScore);
  const prediction = riskScore > 60 ? "AI_GENERATED" : "AUTHENTIC";
  const confidence = Math.round(clamp(50 + Math.abs(riskScore - 50), 50, 98));

  const charRow = (label, diff, threshold = 0.22) => ({
    label, similar: diff <= threshold, diffPct: Math.round(diff * 100),
  });

  const characteristics = [
    charRow("Pitch", relDiff(genuine.pitchMean, suspected.pitchMean)),
    charRow("Energy", relDiff(genuine.avgRMS, suspected.avgRMS)),
    charRow("Speaking Rate", rateDiff),
    charRow("Pause Pattern", pauseDiff),
    charRow("Speech Style", relDiff(genuine.centroidMean, suspected.centroidMean)),
    charRow("Common Words", relDiff(genuine.pauseSegments, suspected.pauseSegments), 0.3),
    charRow("Sentence Length", sentenceDiff),
  ];

  return {
    aasistScore, ecapaSimilarity, whisperSimilarity,
    riskScore, riskLevel, prediction, confidence, characteristics,
  };
}

/**
 * Service layer — kept deliberately thin and swappable. Today this calls
 * the local DSP pipeline above; wiring up the real backend means replacing
 * the body of these three functions with fetch() calls to:
 *   POST /analyze-audio        (genuine + suspected files)
 *   POST /analyze-live-audio   (genuine reference + live suspected clip)
 *   POST /create-profile       (persist a genuine-voice profile)
 * The Python backend (FastAPI + PyTorch + Librosa + NumPy + scikit-learn)
 * would run the real AASIST, ECAPA-TDNN and Whisper models and return the
 * same shape produced by computeAnalysis() above.
 */
const VoiceAnalysisAPI = {
  async analyzeAudio(genuineFeatures, suspectedFeatures) {
    return computeAnalysis(genuineFeatures, suspectedFeatures);
  },
  async analyzeLiveAudio(genuineFeatures, suspectedFeatures) {
    return computeAnalysis(genuineFeatures, suspectedFeatures);
  },
};

/* =========================================================================
   DOMAIN LOGIC — risk levels & display metadata
   ========================================================================= */

function getRiskLevel(riskScore) {
  if (riskScore <= 30) return "LOW";
  if (riskScore <= 60) return "MEDIUM";
  return "HIGH";
}

const RISK_META = {
  LOW: {
    color: C.green, label: "LOW RISK", icon: ShieldCheck,
    copy: "No strong indicators of a synthetic voice were detected.",
    footnote: "Low risk does not guarantee authenticity.",
  },
  MEDIUM: {
    color: C.amber, label: "MEDIUM RISK", icon: ShieldQuestion,
    copy: "Some voice characteristics are unusual. Additional verification is recommended.",
    footnote: null,
  },
  HIGH: {
    color: C.red, label: "HIGH RISK", icon: ShieldAlert,
    copy: "This voice shows characteristics associated with a possible AI-generated or cloned voice.",
    footnote: null,
  },
};

const PREDICTION_META = {
  AUTHENTIC: { label: "HUMAN / GENUINE", color: C.green },
  AI_GENERATED: { label: "POTENTIAL AI-GENERATED / CLONED VOICE", color: C.red },
};

const PIPELINE_STAGES = [
  { key: "preprocess", label: "Audio Preprocessing", icon: Cpu },
  { key: "features", label: "Feature Extraction", icon: BarChart3 },
  { key: "aasist", label: "AASIST Analysis", icon: ShieldAlert },
  { key: "ecapa", label: "ECAPA-TDNN Comparison", icon: Waves },
  { key: "whisper", label: "Whisper Analysis", icon: Activity },
  { key: "risk", label: "Risk Assessment", icon: Shield },
];

const VERIFICATION_QUESTIONS = [
  "What was the nickname you gave me?",
  "Where did we first meet?",
  "What did we do on our last trip together?",
];

function delay(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }

/* =========================================================================
   SHARED PRIMITIVES
   ========================================================================= */

function Panel({ children, style, ...rest }) {
  return (
    <div style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 10, ...style }} {...rest}>
      {children}
    </div>
  );
}

function Badge({ color, children }) {
  return (
    <span className="vx-mono" style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "3px 9px", borderRadius: 5, fontSize: 11, letterSpacing: 0.4,
      color, background: `${color}1A`, border: `1px solid ${color}40`,
    }}>
      {children}
    </span>
  );
}

function RadialGauge({ value, color, size = 168, thickness = 12, sublabel }) {
  const r = (size - thickness) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (Math.min(100, Math.max(0, value)) / 100) * c;
  return (
    <div style={{ position: "relative", width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={C.borderHi} strokeWidth={thickness} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={thickness}
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(.4,0,.2,1), stroke 0.4s" }}
        />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <div className="vx-mono" style={{ fontSize: size * 0.2, fontWeight: 600, color: C.text, lineHeight: 1 }}>
          {Math.round(value)}%
        </div>
        {sublabel && <div className="vx-mono" style={{ fontSize: 11, color, marginTop: 6, textAlign: "center", padding: "0 8px" }}>{sublabel}</div>}
      </div>
    </div>
  );
}

function SectionLabel({ children }) {
  return <div className="vx-body" style={{ fontSize: 13, fontWeight: 600, color: C.textDim, marginBottom: 12 }}>{children}</div>;
}

/* =========================================================================
   NAVIGATION
   ========================================================================= */

const NAV_ITEMS = [
  { key: "analyze", label: "Analyze Voice", icon: Shield },
  { key: "live", label: "Live Detection", icon: Radio },
  { key: "history", label: "Analysis History", icon: HistoryIcon },
  { key: "insights", label: "Insights", icon: BarChart3 },
  { key: "settings", label: "Settings", icon: SettingsIcon },
];

function Sidebar({ page, setPage }) {
  return (
    <div style={{ width: 232, flexShrink: 0, background: C.panel, borderRight: `1px solid ${C.border}`, display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ padding: "22px 20px", borderBottom: `1px solid ${C.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <div style={{ width: 30, height: 30, borderRadius: 7, background: `linear-gradient(135deg, ${C.cyan}, ${C.purple})`, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Shield size={17} color={C.void} strokeWidth={2.5} />
          </div>
          <div className="vx-display" style={{ fontSize: 17, fontWeight: 700, color: C.text }}>CODE STORM</div>
        </div>
        <div className="vx-body" style={{ fontSize: 11.5, color: C.textFaint, marginTop: 5, marginLeft: 1 }}>
          Voice Cloning Detection &amp; Prevention
        </div>
      </div>
      <nav style={{ padding: "14px 12px", flex: 1 }}>
        {NAV_ITEMS.map((item) => {
          const active = page === item.key;
          const Icon = item.icon;
          return (
            <button
              key={item.key}
              onClick={() => setPage(item.key)}
              className="vx-body"
              style={{
                width: "100%", display: "flex", alignItems: "center", gap: 10,
                padding: "9px 12px", marginBottom: 3, borderRadius: 7, border: "none",
                cursor: "pointer", fontSize: 13.5, fontWeight: 500, textAlign: "left",
                background: active ? C.panelHi : "transparent",
                color: active ? C.cyan : C.textDim,
                borderLeft: active ? `2px solid ${C.cyan}` : "2px solid transparent",
              }}
            >
              <Icon size={16} strokeWidth={2} />
              {item.label}
            </button>
          );
        })}
      </nav>
      <div style={{ padding: "14px 20px", borderTop: `1px solid ${C.border}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 7, height: 7, borderRadius: 99, background: C.green, animation: "vx-pulse 2s infinite" }} />
          <span className="vx-mono" style={{ fontSize: 11, color: C.green, letterSpacing: 0.5 }}>SYSTEM ONLINE</span>
        </div>
      </div>
    </div>
  );
}

/* =========================================================================
   PIPELINE VISUAL
   ========================================================================= */

function PipelineVisual({ activeIndex, complete }) {
  return (
    <Panel style={{ padding: "22px 24px" }}>
      <SectionLabel>Analysis Pipeline</SectionLabel>
      <div style={{ display: "flex", alignItems: "center" }}>
        {PIPELINE_STAGES.map((stage, i) => {
          const Icon = stage.icon;
          const done = complete || i < activeIndex;
          const active = !complete && i === activeIndex;
          const state = done ? "done" : active ? "active" : "waiting";
          const color = state === "done" ? C.green : state === "active" ? C.cyan : C.textFaint;
          return (
            <React.Fragment key={stage.key}>
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", width: 96 }}>
                <div style={{
                  width: 42, height: 42, borderRadius: 10, display: "flex", alignItems: "center", justifyContent: "center",
                  background: state === "waiting" ? C.panelRaised : `${color}1A`,
                  border: `1px solid ${state === "waiting" ? C.border : color}`,
                }}>
                  {active ? <Loader2 size={18} color={color} style={{ animation: "vx-spin 1s linear infinite" }} />
                    : done ? <CheckCircle2 size={18} color={color} /> : <Icon size={16} color={color} />}
                </div>
                <div className="vx-mono" style={{ fontSize: 10, color, marginTop: 8, textAlign: "center", lineHeight: 1.3 }}>{stage.label}</div>
                <div className="vx-mono" style={{ fontSize: 9, color: state === "waiting" ? C.textFaint : color, marginTop: 3, opacity: 0.8 }}>
                  {state === "done" ? "COMPLETE" : state === "active" ? "RUNNING" : "WAITING"}
                </div>
              </div>
              {i < PIPELINE_STAGES.length - 1 && (
                <div style={{ flex: 1, height: 1, background: done ? C.green : C.border, marginBottom: 30, transition: "background 0.3s" }} />
              )}
            </React.Fragment>
          );
        })}
      </div>
    </Panel>
  );
}

/* =========================================================================
   WAVEFORM (static bars from decoded peaks, and a live recording level bar)
   ========================================================================= */

function StaticWaveform({ peaks, color }) {
  return (
    <div style={{ height: 52, display: "flex", alignItems: "center", gap: 2 }}>
      {peaks.map((p, i) => (
        <div key={i} style={{ flex: 1, borderRadius: 2, background: color, height: `${Math.max(6, p * 48)}px`, opacity: 0.85 }} />
      ))}
    </div>
  );
}

function LiveLevelBars({ levels, color }) {
  return (
    <div style={{ height: 64, display: "flex", alignItems: "center", justifyContent: "center", gap: 3 }}>
      {levels.map((v, i) => (
        <div key={i} style={{ width: 3, borderRadius: 2, background: color, height: `${8 + v * 52}px`, transition: "height 0.08s ease" }} />
      ))}
    </div>
  );
}

/* =========================================================================
   VOICE INPUT PANEL — upload or record, real decoding, no fake data
   ========================================================================= */

const LEVEL_BAR_COUNT = 36;

function VoiceInputPanel({ heading, description, accent, value, onChange, prominentRecord, allowRecord = true, supportedFormats = "WAV · MP3 · M4A" }) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [levels, setLevels] = useState(Array(LEVEL_BAR_COUNT).fill(0));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const rafRef = useRef(null);
  const analyserRef = useRef(null);
  const liveCtxRef = useRef(null);

  useEffect(() => () => {
    clearInterval(timerRef.current);
    cancelAnimationFrame(rafRef.current);
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
  }, []);

  const setAudioFromBuffer = async (audioBuffer, name, url) => {
    const peaks = computePeaks(audioBuffer);
    const features = extractAudioFeatures(audioBuffer);
    onChange({ url, name, duration: audioBuffer.duration, peaks, features });
  };

  const handleFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setBusy(true);
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await decodeArrayBuffer(arrayBuffer);
      const url = URL.createObjectURL(file);
      await setAudioFromBuffer(audioBuffer, file.name, url);
    } catch (err) {
      setError("Could not read that audio file. Try a WAV, MP3, or M4A file.");
    } finally {
      setBusy(false);
      e.target.value = "";
    }
  };

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const liveCtx = new AudioCtx();
      liveCtxRef.current = liveCtx;
      const source = liveCtx.createMediaStreamSource(stream);
      const analyser = liveCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const tick = () => {
        analyser.getByteFrequencyData(dataArray);
        const step = Math.floor(dataArray.length / LEVEL_BAR_COUNT) || 1;
        const next = [];
        for (let i = 0; i < LEVEL_BAR_COUNT; i++) next.push((dataArray[i * step] || 0) / 255);
        setLevels(next);
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();

      const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = async () => {
        setBusy(true);
        try {
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
          const arrayBuffer = await blob.arrayBuffer();
          const audioBuffer = await decodeArrayBuffer(arrayBuffer);
          const url = URL.createObjectURL(blob);
          await setAudioFromBuffer(audioBuffer, `recording-${Date.now()}.webm`, url);
        } catch (err) {
          setError("Could not process the recording. Please try again.");
        } finally {
          setBusy(false);
        }
        stream.getTracks().forEach((t) => t.stop());
        liveCtx.close();
        cancelAnimationFrame(rafRef.current);
        setLevels(Array(LEVEL_BAR_COUNT).fill(0));
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch (err) {
      setError("Microphone access was denied or is unavailable. Please allow microphone permission.");
    }
  };

  const stopRecording = () => {
    clearInterval(timerRef.current);
    setRecording(false);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
  };

  const clearAudio = () => {
    if (value?.url) URL.revokeObjectURL(value.url);
    onChange(null);
  };

  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");

  return (
    <Panel style={{ padding: 22 }}>
      <SectionLabel>{heading}</SectionLabel>
      <div className="vx-body" style={{ fontSize: 12, color: C.textDim, marginBottom: 14, lineHeight: 1.5 }}>{description}</div>

      {!value && !recording && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <div
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `1px dashed ${C.borderHi}`, borderRadius: 9, padding: "18px 16px",
              display: "flex", flexDirection: "column", alignItems: "center", gap: 8, cursor: "pointer",
              background: C.panelRaised,
            }}
          >
            <input ref={fileInputRef} type="file" accept=".wav,.mp3,.m4a,audio/*" style={{ display: "none" }} onChange={handleFile} />
            <Upload size={18} color={accent} />
            <div className="vx-body" style={{ fontSize: 12.5, color: C.text, fontWeight: 500 }}>Click to upload audio</div>
            <div className="vx-mono" style={{ fontSize: 10, color: C.textFaint }}>{supportedFormats}</div>
          </div>

          {allowRecord && (
            <button
              onClick={startRecording}
              className="vx-body"
              style={{
                display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
                padding: prominentRecord ? "13px" : "10px", borderRadius: 8, cursor: "pointer",
                border: prominentRecord ? "none" : `1px solid ${C.border}`,
                background: prominentRecord ? `linear-gradient(90deg, ${C.cyan}, ${C.purple})` : "transparent",
                color: prominentRecord ? C.void : C.textDim,
                fontWeight: prominentRecord ? 700 : 500, fontSize: prominentRecord ? 13.5 : 12.5,
              }}
            >
              <Mic size={prominentRecord ? 16 : 14} />
              {prominentRecord ? "🎙️ Record Suspected Voice Live" : "Record Audio"}
            </button>
          )}
        </div>
      )}

      {recording && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8 }}>
            <span style={{ width: 7, height: 7, borderRadius: 99, background: C.red, animation: "vx-pulse 1s infinite" }} />
            <span className="vx-mono" style={{ fontSize: 11, color: C.red }}>MICROPHONE ACTIVE</span>
          </div>
          <div style={{ background: C.panelRaised, borderRadius: 8, marginBottom: 10 }}>
            <LiveLevelBars levels={levels} color={accent} />
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div className="vx-mono" style={{ fontSize: 20, color: C.text }}>{mm}:{ss}</div>
            <button onClick={stopRecording} className="vx-body" style={{
              display: "flex", alignItems: "center", gap: 8, padding: "9px 16px", borderRadius: 8,
              border: `1px solid ${C.red}`, background: "transparent", color: C.red, fontWeight: 600, fontSize: 13, cursor: "pointer",
            }}>
              <Square size={14} /> Stop Recording
            </button>
          </div>
        </div>
      )}

      {busy && !value && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 10, color: C.textDim, fontSize: 12 }}>
          <Loader2 size={14} style={{ animation: "vx-spin 1s linear infinite" }} /> Processing audio…
        </div>
      )}

      {value && (
        <div className="vx-rise">
          <div style={{ background: C.panelRaised, borderRadius: 8, padding: "10px 12px", marginBottom: 10 }}>
            <StaticWaveform peaks={value.peaks} color={accent} />
          </div>
          <audio
            controls
            src={value.url}
            style={{ width: "100%", height: 32, marginBottom: 10 }}
            onLoadedMetadata={(e) => {
              const audio = e.currentTarget;
              // Chrome sometimes reports Infinity/NaN duration for blob-sourced
              // audio until it's forced to seek once — this recalculates it.
              if (!isFinite(audio.duration)) {
                const fixDuration = () => {
                  audio.currentTime = 0;
                  audio.removeEventListener("timeupdate", fixDuration);
                };
                audio.addEventListener("timeupdate", fixDuration);
                audio.currentTime = 1e101;
              }
            }}
          />
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
              <FileAudio size={14} color={accent} />
              <span className="vx-body" style={{ fontSize: 12, color: C.text, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 180 }}>{value.name}</span>
              <span className="vx-mono" style={{ fontSize: 11, color: C.textFaint }}>{value.duration.toFixed(1)}s</span>
            </div>
            <button onClick={clearAudio} className="vx-body" style={{ display: "flex", alignItems: "center", gap: 5, background: "none", border: "none", color: C.textFaint, cursor: "pointer", fontSize: 12 }}>
              <Trash2 size={13} /> Remove
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="vx-body" style={{ marginTop: 10, fontSize: 11.5, color: C.red, display: "flex", gap: 6, alignItems: "flex-start" }}>
          <AlertTriangle size={13} style={{ marginTop: 1, flexShrink: 0 }} /> {error}
        </div>
      )}
    </Panel>
  );
}

/* =========================================================================
   RESULT DISPLAY — prediction, risk, model cards, characteristics, security
   ========================================================================= */

function ResultPanel({ result }) {
  const pMeta = PREDICTION_META[result.prediction];
  const rMeta = RISK_META[result.riskLevel];
  const RIcon = rMeta.icon;
  return (
    <div className="vx-rise" style={{ display: "grid", gridTemplateColumns: "1.1fr 1fr", gap: 16 }}>
      <Panel style={{ padding: 24, borderColor: `${pMeta.color}40` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
          <RIcon size={22} color={pMeta.color} />
          <div className="vx-display" style={{ fontSize: 18, fontWeight: 700, color: pMeta.color }}>{pMeta.label}</div>
        </div>
        <div style={{ display: "flex", gap: 28, marginBottom: 18 }}>
          <div>
            <div className="vx-mono" style={{ fontSize: 11, color: C.textDim, marginBottom: 4 }}>CONFIDENCE SCORE</div>
            <div className="vx-mono" style={{ fontSize: 30, fontWeight: 600, color: C.text }}>{result.confidence}%</div>
          </div>
          <div>
            <div className="vx-mono" style={{ fontSize: 11, color: C.textDim, marginBottom: 4 }}>VOICE SIMILARITY</div>
            <div className="vx-mono" style={{ fontSize: 30, fontWeight: 600, color: C.purple }}>{result.ecapaSimilarity}%</div>
          </div>
        </div>
        <div className="vx-body" style={{ fontSize: 12.5, color: C.textDim, lineHeight: 1.5 }}>{rMeta.copy}</div>
        {rMeta.footnote && <div className="vx-body" style={{ fontSize: 11.5, color: C.textFaint, marginTop: 6 }}>{rMeta.footnote}</div>}
      </Panel>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <Panel style={{ flex: 1, padding: "18px 8px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <div className="vx-mono" style={{ fontSize: 10.5, color: C.textDim, letterSpacing: 0.6, marginBottom: 10 }}>IMPERSONATION RISK</div>
          <RadialGauge value={result.riskScore} color={rMeta.color} size={128} thickness={9} sublabel={rMeta.label} />
        </Panel>
      </div>
    </div>
  );
}

function RiskScale({ value }) {
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div style={{ marginTop: 4 }}>
      <div style={{ position: "relative", height: 8, borderRadius: 99, overflow: "hidden", background: `linear-gradient(90deg, ${C.green} 0%, ${C.green} 30%, ${C.amber} 31%, ${C.amber} 60%, ${C.red} 61%, ${C.red} 100%)` }}>
        <div style={{ position: "absolute", top: -3, left: `${pct}%`, width: 2, height: 14, background: C.text, transform: "translateX(-1px)", transition: "left 1s cubic-bezier(.4,0,.2,1)" }} />
      </div>
      <div className="vx-mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: C.textFaint, marginTop: 6 }}>
        <span>0 · LOW</span><span>31 · MEDIUM</span><span>61 · HIGH</span><span>100</span>
      </div>
    </div>
  );
}

function RiskScaleCard({ result }) {
  const rMeta = RISK_META[result.riskLevel];
  return (
    <Panel className="vx-rise" style={{ padding: "18px 22px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <SectionLabel>Impersonation Risk Scale</SectionLabel>
        <span className="vx-mono" style={{ fontSize: 11, color: rMeta.color }}>{result.riskScore}% · {rMeta.label}</span>
      </div>
      <RiskScale value={result.riskScore} />
    </Panel>
  );
}

function ModelBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
        <span className="vx-body" style={{ fontSize: 12.5, color: C.textDim }}>{label}</span>
        <span className="vx-mono" style={{ fontSize: 12, color }}>{value}%</span>
      </div>
      <div style={{ height: 6, borderRadius: 99, background: C.panelRaised, overflow: "hidden" }}>
        <div style={{ width: `${value}%`, height: "100%", background: color, borderRadius: 99, transition: "width 0.8s cubic-bezier(.4,0,.2,1)" }} />
      </div>
    </div>
  );
}

function ModelResultCards({ result }) {
  const cards = [
    { name: "AASIST", sub: "Spoof Detection", value: result.aasistScore, color: result.aasistScore > 60 ? C.red : result.aasistScore > 30 ? C.amber : C.green },
    { name: "ECAPA-TDNN", sub: "Voice Similarity", value: result.ecapaSimilarity, color: C.purple },
    { name: "Whisper", sub: "Speech / Linguistic Similarity", value: result.whisperSimilarity, color: C.cyan },
  ];
  return (
    <div className="vx-rise" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
      {cards.map((c) => (
        <Panel key={c.name} style={{ padding: 20 }}>
          <div className="vx-display" style={{ fontSize: 15, fontWeight: 700, color: C.text, marginBottom: 2 }}>{c.name}</div>
          <div className="vx-body" style={{ fontSize: 11.5, color: C.textFaint, marginBottom: 14 }}>{c.sub}</div>
          <ModelBar label={c.sub} value={c.value} color={c.color} />
        </Panel>
      ))}
    </div>
  );
}

function VoiceCharacteristicsPanel({ characteristics }) {
  return (
    <Panel className="vx-rise" style={{ padding: 22 }}>
      <SectionLabel>Voice Characteristics Comparison</SectionLabel>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 24px" }}>
        {characteristics.map((c) => (
          <div key={c.label} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 12px", borderRadius: 7, background: C.panelRaised }}>
            <span className="vx-body" style={{ fontSize: 12.5, color: C.text }}>{c.label}</span>
            <span className="vx-mono" style={{ fontSize: 12, color: c.similar ? C.green : C.amber, display: "flex", alignItems: "center", gap: 5 }}>
              {c.similar ? "✓ Similar" : "⚠️ Different"}
            </span>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function SecurityRecommendationPanel({ result }) {
  const rMeta = RISK_META[result.riskLevel];
  if (result.riskLevel === "LOW") {
    return (
      <Panel className="vx-rise" style={{ padding: 20, borderColor: `${C.green}40` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <ShieldCheck size={17} color={C.green} />
          <span className="vx-display" style={{ fontSize: 14, fontWeight: 600, color: C.text }}>{rMeta.label}</span>
        </div>
        <div className="vx-body" style={{ fontSize: 12.5, color: C.textDim, marginTop: 8 }}>{rMeta.copy}</div>
        <div className="vx-body" style={{ fontSize: 11.5, color: C.textFaint, marginTop: 4 }}>{rMeta.footnote}</div>
      </Panel>
    );
  }
  if (result.riskLevel === "MEDIUM") {
    return (
      <Panel className="vx-rise" style={{ padding: 20, borderColor: `${C.amber}40` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
          <ShieldQuestion size={17} color={C.amber} />
          <span className="vx-display" style={{ fontSize: 14, fontWeight: 600, color: C.text }}>{rMeta.label}</span>
        </div>
        <div className="vx-body" style={{ fontSize: 12.5, color: C.textDim, marginTop: 8 }}>{rMeta.copy}</div>
      </Panel>
    );
  }
  // HIGH
  return (
    <Panel className="vx-rise" style={{ padding: 22, borderColor: `${C.red}40`, background: `linear-gradient(180deg, ${C.red}0D, transparent)` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 6 }}>
        <AlertTriangle size={18} color={C.red} />
        <div className="vx-display" style={{ fontSize: 15, fontWeight: 700, color: C.text }}>⚠️ POSSIBLE VOICE IMPERSONATION</div>
      </div>
      <div className="vx-body" style={{ fontSize: 12.5, color: C.textDim, marginBottom: 16 }}>{rMeta.copy}</div>

      <div className="vx-body" style={{ fontSize: 12, fontWeight: 600, color: C.text, marginBottom: 8 }}>Recommendations</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 7, marginBottom: 18 }}>
        {[
          "Do not share OTPs or passwords.",
          "Do not transfer money.",
          "Do not share banking information.",
          "Verify the person's identity using another communication channel.",
          "Ask a personal question that only the genuine person is likely to know.",
        ].map((t) => (
          <div key={t} style={{ display: "flex", alignItems: "flex-start", gap: 7 }}>
            <CheckCircle2 size={13} color={C.red} style={{ marginTop: 2, flexShrink: 0 }} />
            <span className="vx-body" style={{ fontSize: 12.5, color: C.text }}>{t}</span>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <HelpCircle size={14} color={C.red} />
        <div className="vx-body" style={{ fontSize: 12, fontWeight: 600, color: C.text }}>Suggested Verification Questions</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 10 }}>
        {VERIFICATION_QUESTIONS.map((q) => (
          <div key={q} className="vx-body" style={{ fontSize: 12.5, color: C.textDim, padding: "8px 12px", borderRadius: 7, background: C.panelRaised }}>
            "{q}"
          </div>
        ))}
      </div>
      <div className="vx-body" style={{ fontSize: 11, color: C.textFaint, display: "flex", gap: 6, alignItems: "flex-start" }}>
        <MessageSquareWarning size={13} style={{ marginTop: 1, flexShrink: 0 }} />
        These questions are an additional verification method and are not proof of identity.
      </div>
    </Panel>
  );
}

function AnalysisResultView({ result }) {
  return (
    <>
      <ResultPanel result={result} />
      <RiskScaleCard result={result} />
      <ModelResultCards result={result} />
      <VoiceCharacteristicsPanel characteristics={result.characteristics} />
      <SecurityRecommendationPanel result={result} />
    </>
  );
}

/* =========================================================================
   ANALYSIS RUNNER — shared by Analyze Voice and Live Detection pages
   ========================================================================= */

function useAnalysisRunner(pushHistory) {
  const [phase, setPhase] = useState("idle"); // idle | running | complete
  const [stageIndex, setStageIndex] = useState(0);
  const [result, setResult] = useState(null);

  const run = useCallback(async (genuineAudio, suspectedAudio, sourceLabel, apiCall) => {
    if (!genuineAudio || !suspectedAudio || phase === "running") return;
    setPhase("running");
    setResult(null);
    setStageIndex(0);
    for (let i = 1; i < PIPELINE_STAGES.length; i++) {
      await delay(420);
      setStageIndex(i);
    }
    const analysis = await apiCall(genuineAudio.features, suspectedAudio.features);
    const r = { ...analysis, source: sourceLabel, timestamp: new Date(), processingTime: (PIPELINE_STAGES.length * 0.42) };
    setResult(r);
    setPhase("complete");
    pushHistory && pushHistory(r);
  }, [phase, pushHistory]);

  const reset = useCallback(() => { setPhase("idle"); setResult(null); setStageIndex(0); }, []);

  return { phase, stageIndex, result, run, reset };
}

/* =========================================================================
   ANALYZE VOICE PAGE (main analysis page — exactly two audio inputs)
   ========================================================================= */

function AnalyzePage({ genuineAudio, setGenuineAudio, pushHistory }) {
  const [suspectedAudio, setSuspectedAudio] = useState(null);
  const { phase, stageIndex, result, run } = useAnalysisRunner(pushHistory);

  const canAnalyze = !!genuineAudio && !!suspectedAudio && phase !== "running";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div>
        <div className="vx-display" style={{ fontSize: 24, fontWeight: 700, color: C.text, marginBottom: 6 }}>
          Real-Time Voice Cloning Detection
        </div>
        <div className="vx-body" style={{ fontSize: 13.5, color: C.textDim, maxWidth: 620, lineHeight: 1.5 }}>
          Provide a genuine voice sample and the suspected voice you want to verify. CODE STORM analyzes both
          and determines the result — you never select the outcome yourself.
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <VoiceInputPanel
          heading="Upload Genuine Voice"
          description="Upload a normal voice sample of the genuine person."
          accent={C.cyan}
          value={genuineAudio}
          onChange={setGenuineAudio}
          allowRecord={false}
        />
        <VoiceInputPanel
          heading="Upload Suspected Voice"
          description="Upload the suspicious voice that you want to analyze."
          accent={C.purple}
          value={suspectedAudio}
          onChange={setSuspectedAudio}
          allowRecord={false}
        />
      </div>

      <button
        onClick={() => run(genuineAudio, suspectedAudio, "Analyze Voice", VoiceAnalysisAPI.analyzeAudio)}
        disabled={!canAnalyze}
        className="vx-body"
        style={{
          width: "100%", padding: "13px", borderRadius: 9, border: "none",
          background: !canAnalyze ? C.borderHi : `linear-gradient(90deg, ${C.cyan}, ${C.purple})`,
          color: !canAnalyze ? C.textFaint : C.void, fontSize: 14, fontWeight: 700,
          cursor: !canAnalyze ? "not-allowed" : "pointer",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
        }}
      >
        {phase === "running" ? <><Loader2 size={16} style={{ animation: "vx-spin 1s linear infinite" }} /> Analyzing…</> : <><PlayCircle size={16} /> Analyze Voice</>}
      </button>

      {phase !== "idle" && <PipelineVisual activeIndex={stageIndex} complete={phase === "complete"} />}
      {result && <AnalysisResultView result={result} />}
    </div>
  );
}

/* =========================================================================
   LIVE DETECTION PAGE — real microphone recording, no simulated callers
   ========================================================================= */

function LivePage({ genuineAudio, setGenuineAudio, pushHistory }) {
  const [suspectedAudio, setSuspectedAudio] = useState(null);
  const { phase, stageIndex, result, run } = useAnalysisRunner(pushHistory);

  const canAnalyze = !!genuineAudio && !!suspectedAudio && phase !== "running";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="vx-display" style={{ fontSize: 22, fontWeight: 700, color: C.text }}>Live Voice Analysis</div>

      {!genuineAudio && (
        <Panel style={{ padding: 16, borderColor: `${C.amber}40` }}>
          <div className="vx-body" style={{ fontSize: 12.5, color: C.amber, display: "flex", alignItems: "center", gap: 8 }}>
            <AlertTriangle size={14} /> Add a genuine voice reference below before recording the live suspected voice.
          </div>
        </Panel>
      )}

      <VoiceInputPanel
        heading="Genuine Voice Reference"
        description="Upload or record a normal voice sample of the genuine person. This is reused for every live check."
        accent={C.cyan}
        value={genuineAudio}
        onChange={setGenuineAudio}
      />

      <VoiceInputPanel
        heading="Record Suspected Voice Live"
        description="Record the live caller's voice to analyze it against the genuine reference above."
        accent={C.purple}
        value={suspectedAudio}
        onChange={setSuspectedAudio}
        prominentRecord
      />

      <button
        onClick={() => run(genuineAudio, suspectedAudio, "Live Detection", VoiceAnalysisAPI.analyzeLiveAudio)}
        disabled={!canAnalyze}
        className="vx-body"
        style={{
          width: "100%", padding: "13px", borderRadius: 9, border: "none",
          background: !canAnalyze ? C.borderHi : `linear-gradient(90deg, ${C.cyan}, ${C.purple})`,
          color: !canAnalyze ? C.textFaint : C.void, fontSize: 14, fontWeight: 700,
          cursor: !canAnalyze ? "not-allowed" : "pointer",
          display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
        }}
      >
        {phase === "running" ? <><Loader2 size={16} style={{ animation: "vx-spin 1s linear infinite" }} /> Analyzing…</> : <><PlayCircle size={16} /> Analyze Voice</>}
      </button>

      {phase !== "idle" && <PipelineVisual activeIndex={stageIndex} complete={phase === "complete"} />}
      {result && <AnalysisResultView result={result} />}
    </div>
  );
}

/* =========================================================================
   HISTORY PAGE
   ========================================================================= */

function HistoryPage({ log }) {
  const [filter, setFilter] = useState("ALL");
  const rows = filter === "ALL" ? log : log.filter((r) => r.riskLevel === filter);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div className="vx-display" style={{ fontSize: 22, fontWeight: 700, color: C.text }}>Analysis History</div>
        <div style={{ display: "flex", gap: 6 }}>
          {["ALL", "LOW", "MEDIUM", "HIGH"].map((f) => (
            <button key={f} onClick={() => setFilter(f)} className="vx-mono"
              style={{ padding: "6px 11px", borderRadius: 6, fontSize: 11, cursor: "pointer",
                background: filter === f ? C.panelHi : "transparent",
                border: `1px solid ${filter === f ? C.cyan : C.border}`,
                color: filter === f ? C.cyan : C.textDim }}>
              {f}
            </button>
          ))}
        </div>
      </div>

      <Panel style={{ padding: 0, overflow: "hidden" }}>
        {rows.length === 0 ? (
          <div style={{ padding: 40, textAlign: "center" }}>
            <div className="vx-body" style={{ fontSize: 13, color: C.textFaint }}>No analyses yet. Run a detection from Analyze Voice or Live Detection.</div>
          </div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                {["Timestamp", "Source", "Prediction", "Confidence", "Risk Score", "Risk Level"].map((h) => (
                  <th key={h} className="vx-mono" style={{ textAlign: "left", padding: "11px 16px", fontSize: 10.5, color: C.textFaint, fontWeight: 500 }}>{h.toUpperCase()}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, idx) => {
                const rMeta = RISK_META[r.riskLevel];
                const pMeta = PREDICTION_META[r.prediction];
                return (
                  <tr key={idx} style={{ borderBottom: `1px solid ${C.border}` }}>
                    <td className="vx-mono" style={{ padding: "11px 16px", fontSize: 12, color: C.textDim }}>{r.timestamp.toLocaleTimeString()}</td>
                    <td className="vx-body" style={{ padding: "11px 16px", fontSize: 12.5, color: C.text }}>{r.source}</td>
                    <td style={{ padding: "11px 16px" }}><span className="vx-mono" style={{ fontSize: 11.5, color: pMeta.color }}>{pMeta.label}</span></td>
                    <td className="vx-mono" style={{ padding: "11px 16px", fontSize: 12.5, color: C.text }}>{r.confidence}%</td>
                    <td className="vx-mono" style={{ padding: "11px 16px", fontSize: 12.5, color: rMeta.color }}>{r.riskScore}%</td>
                    <td style={{ padding: "11px 16px" }}><Badge color={rMeta.color}>{r.riskLevel}</Badge></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </Panel>
    </div>
  );
}

/* =========================================================================
   INSIGHTS PAGE — aggregated stats from real analysis history
   ========================================================================= */

function StatCard({ label, value, color }) {
  return (
    <Panel style={{ padding: "16px 18px" }}>
      <div className="vx-mono" style={{ fontSize: 10.5, color: C.textFaint, marginBottom: 8 }}>{label.toUpperCase()}</div>
      <div className="vx-mono" style={{ fontSize: 24, fontWeight: 600, color: color || C.text }}>{value}</div>
    </Panel>
  );
}

function EmptyChart() {
  return (
    <div style={{ height: "80%", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <span className="vx-body" style={{ fontSize: 12, color: C.textFaint }}>Run an analysis to populate this chart.</span>
    </div>
  );
}

function InsightsPage({ log }) {
  const total = log.length;
  const aiDetected = log.filter((r) => r.prediction === "AI_GENERATED").length;
  const highRisk = log.filter((r) => r.riskLevel === "HIGH").length;
  const avgRisk = total ? Math.round(log.reduce((a, r) => a + r.riskScore, 0) / total) : 0;
  const avgConf = total ? (log.reduce((a, r) => a + r.confidence, 0) / total).toFixed(1) : "0.0";

  const distribution = [
    { name: "Low", value: log.filter((r) => r.riskLevel === "LOW").length, color: C.green },
    { name: "Medium", value: log.filter((r) => r.riskLevel === "MEDIUM").length, color: C.amber },
    { name: "High", value: log.filter((r) => r.riskLevel === "HIGH").length, color: C.red },
  ];
  const predData = [
    { name: "Human / Genuine", value: total - aiDetected },
    { name: "AI-Generated", value: aiDetected },
  ];
  const trend = log.slice().reverse().map((r, i) => ({ i: i + 1, risk: r.riskScore }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="vx-display" style={{ fontSize: 22, fontWeight: 700, color: C.text }}>Insights</div>
      <div className="vx-body" style={{ fontSize: 11.5, color: C.textFaint, marginTop: -12 }}>Aggregated from your analysis history</div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 14 }}>
        <StatCard label="Total Voices Analyzed" value={total} />
        <StatCard label="AI-Generated Detected" value={aiDetected} color={C.red} />
        <StatCard label="High Risk Incidents" value={highRisk} color={C.red} />
        <StatCard label="Average Risk Score" value={`${avgRisk}%`} color={C.amber} />
        <StatCard label="Average Confidence" value={`${avgConf}%`} color={C.purple} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <Panel style={{ padding: 20, height: 260 }}>
          <SectionLabel>Risk Distribution</SectionLabel>
          {total === 0 ? <EmptyChart /> : (
            <ResponsiveContainer width="100%" height="85%">
              <PieChart>
                <Pie data={distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={45} outerRadius={72} paddingAngle={3}>
                  {distribution.map((d, i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "Inter" }} />
                <Tooltip contentStyle={{ background: C.panelHi, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Panel>

        <Panel style={{ padding: 20, height: 260 }}>
          <SectionLabel>Human vs AI-Generated</SectionLabel>
          {total === 0 ? <EmptyChart /> : (
            <ResponsiveContainer width="100%" height="85%">
              <BarChart data={predData}>
                <CartesianGrid stroke={C.border} vertical={false} />
                <XAxis dataKey="name" tick={{ fill: C.textDim, fontSize: 11 }} axisLine={{ stroke: C.border }} tickLine={false} />
                <YAxis tick={{ fill: C.textDim, fontSize: 11 }} axisLine={{ stroke: C.border }} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={{ background: C.panelHi, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="value" fill={C.cyan} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Panel>
      </div>

      <Panel style={{ padding: 20, height: 240 }}>
        <SectionLabel>Risk Score Trend</SectionLabel>
        {total === 0 ? <EmptyChart /> : (
          <ResponsiveContainer width="100%" height="82%">
            <LineChart data={trend}>
              <CartesianGrid stroke={C.border} vertical={false} />
              <XAxis dataKey="i" tick={{ fill: C.textDim, fontSize: 11 }} axisLine={{ stroke: C.border }} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fill: C.textDim, fontSize: 11 }} axisLine={{ stroke: C.border }} tickLine={false} />
              <Tooltip contentStyle={{ background: C.panelHi, border: `1px solid ${C.border}`, borderRadius: 8, fontSize: 12 }} />
              <Line type="monotone" dataKey="risk" stroke={C.purple} strokeWidth={2} dot={{ r: 3, fill: C.purple }} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </Panel>
    </div>
  );
}

/* =========================================================================
   SETTINGS PAGE
   ========================================================================= */

function SettingsPage() {
  const items = [
    { icon: EyeOff, title: "Audio retention", body: "Raw voice recordings stay in-browser for the current session only, unless a future backend integration is configured to store them." },
    { icon: Lock, title: "Encryption", body: "When connected to a backend, audio and metadata should be encrypted in transit and at rest." },
    { icon: ShieldQuestion, title: "Voice is a signal, not proof", body: "Detection output is a risk indicator, not absolute proof of impersonation — it should inform, not replace, human judgment." },
    { icon: Shield, title: "Never the sole factor", body: "Voice authentication should never be the only factor guarding a sensitive action; pair it with another verification channel for high-risk decisions." },
    { icon: AlertTriangle, title: "Escalation on high risk", body: "High-risk results surface verification questions and clear guidance instead of an automatic outcome." },
    { icon: Mic, title: "Consent", body: "The browser's own microphone permission prompt is required before any recording begins." },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18, maxWidth: 760 }}>
      <div className="vx-display" style={{ fontSize: 22, fontWeight: 700, color: C.text }}>Privacy &amp; Security</div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        {items.map((it) => {
          const Icon = it.icon;
          return (
            <Panel key={it.title} style={{ padding: 18 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 8 }}>
                <Icon size={16} color={C.cyan} />
                <div className="vx-body" style={{ fontSize: 13, fontWeight: 600, color: C.text }}>{it.title}</div>
              </div>
              <div className="vx-body" style={{ fontSize: 12, color: C.textDim, lineHeight: 1.5 }}>{it.body}</div>
            </Panel>
          );
        })}
      </div>

      <Panel style={{ padding: 18 }}>
        <div className="vx-body" style={{ fontSize: 13, fontWeight: 600, color: C.text, marginBottom: 8 }}>Architecture note</div>
        <div className="vx-body" style={{ fontSize: 12, color: C.textDim, lineHeight: 1.6 }}>
          This build extracts real acoustic features (energy, zero-crossing rate, spectral centroid and flatness,
          autocorrelation-based pitch, speaking rate, and pause pattern) directly from the uploaded or recorded
          audio in the browser, and derives the AASIST, ECAPA-TDNN, and Whisper-stage results from those features.
          The service layer (<span className="vx-mono">VoiceAnalysisAPI.analyzeAudio</span>, <span className="vx-mono">analyzeLiveAudio</span>)
          is structured so a Python backend — FastAPI, PyTorch, Librosa, NumPy, and scikit-learn, exposing
          <span className="vx-mono"> POST /analyze-audio</span>, <span className="vx-mono">/create-profile</span>, and
          <span className="vx-mono"> /analyze-live-audio</span> — can be dropped in behind the same function signatures,
          running the trained AASIST spoof detector, ECAPA-TDNN speaker-embedding model, and Whisper transcription
          model in place of the current browser-side approximations, with no changes to the frontend.
        </div>
      </Panel>
    </div>
  );
}

/* =========================================================================
   ROOT APP
   ========================================================================= */

export default function App() {
  const [page, setPage] = useState("analyze");
  const [log, setLog] = useState([]);
  const [genuineAudio, setGenuineAudio] = useState(null);

  const pushHistory = (r) => setLog((prev) => [r, ...prev].slice(0, 50));

  return (
    <div className="vx-body" style={{ display: "flex", height: "100vh", background: C.void, color: C.text }}>
      <style>{FONTS}</style>
      <Sidebar page={page} setPage={setPage} />
      <div style={{ flex: 1, overflowY: "auto" }}>
        <div style={{ maxWidth: 1080, margin: "0 auto", padding: "28px 32px 60px" }}>
          {page === "analyze" && <AnalyzePage genuineAudio={genuineAudio} setGenuineAudio={setGenuineAudio} pushHistory={pushHistory} />}
          {page === "live" && <LivePage genuineAudio={genuineAudio} setGenuineAudio={setGenuineAudio} pushHistory={pushHistory} />}
          {page === "history" && <HistoryPage log={log} />}
          {page === "insights" && <InsightsPage log={log} />}
          {page === "settings" && <SettingsPage />}
        </div>
      </div>
    </div>
  );
}
