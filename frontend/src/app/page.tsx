"use client";

import React, { useState, useEffect } from "react";
import { Mic, Send, ShieldAlert, Database, RefreshCw, HelpCircle, AudioLines, Sparkles } from "lucide-react";
import VoiceRecorder from "../components/VoiceRecorder";
import TranscriptView from "../components/TranscriptView";
import SourcesPanel from "../components/SourcesPanel";

interface CorpusStats {
  last_refreshed: string | null;
  document_count: number;
  chunk_count: number;
}

export default function Home() {
  const [appState, setAppState] = useState<"idle" | "recording" | "processing" | "playing">("idle");
  const [textInput, setTextInput] = useState("");
  
  // Results State
  const [transcript, setTranscript] = useState<string | null>(null);
  const [answer, setAnswer] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [sources, setSources] = useState<any[]>([]);
  const [sttConfidence, setSttConfidence] = useState<number | null>(null);
  const [lowConfidenceWarning, setLowConfidenceWarning] = useState(false);
  
  // General State
  const [error, setError] = useState<string | null>(null);
  const [corpusStats, setCorpusStats] = useState<CorpusStats | null>(null);
  const [refreshingCorpus, setRefreshingCorpus] = useState(false);
  const [corpusMessage, setCorpusMessage] = useState<string | null>(null);

  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Curated demo questions for easy testing
  const SUGGESTED_QUESTIONS = [
    { text: "ශ්‍රී ලංකාවේ ජනාධිපති කවුද?", label: "ජනාධිපති කවුද?" },
    { text: "සිංහල භාෂාව යනු කුමක්ද?", label: "සිංහල භාෂාව" },
    { text: "ශ්‍රී ලංකාව පිහිටා ඇත්තේ කොහේද?", label: "ලංකාවේ පිහිටීම" },
    { text: "අද කාලගුණය හොඳද?", label: "කාලගුණය" },
  ];

  // Load corpus stats on mount
  useEffect(() => {
    fetchCorpusStats();
  }, []);

  const fetchCorpusStats = async () => {
    try {
      const response = await fetch(`${apiUrl}/api/corpus/status`);
      if (response.ok) {
        const data = await response.json();
        setCorpusStats(data);
      }
    } catch (err) {
      console.warn("Failed to load corpus status:", err);
    }
  };

  const triggerCorpusRefresh = async () => {
    setRefreshingCorpus(true);
    setCorpusMessage("මූලාශ්‍ර දත්ත සමුදාය යාවත්කාලීන වෙමින් පවතී (මෙයට විනාඩි කිහිපයක් ගතවිය හැක)...");
    try {
      const response = await fetch(`${apiUrl}/api/corpus/refresh`, {
        method: "POST",
      });
      if (response.ok) {
        const data = await response.json();
        setCorpusMessage(data.message || "යාවත්කාලීන කිරීම සාර්ථකයි!");
        fetchCorpusStats();
      } else {
        throw new Error("යාවත්කාලීන කිරීම අසාර්ථක විය.");
      }
    } catch (err: any) {
      setError(err.message || "දත්ත සමුදාය යාවත්කාලීන කිරීමේ දෝෂයකි.");
    } finally {
      setRefreshingCorpus(false);
      setTimeout(() => setCorpusMessage(null), 8000);
    }
  };

  const handleVoiceResponse = (data: any) => {
    setError(null);
    setTranscript(data.transcript);
    setAnswer(data.answer_text);
    setAudioUrl(data.answer_audio_url);
    setSources(data.sources || []);
    setSttConfidence(data.stt_confidence ?? null);
    setLowConfidenceWarning(data.low_confidence_warning ?? false);
    setAppState("playing"); // Auto-play starts playing immediately
  };

  const handleTextQuerySubmit = async (e?: React.FormEvent, customQuery?: string) => {
    if (e) e.preventDefault();
    const queryToSend = customQuery || textInput;
    if (!queryToSend.trim()) return;

    setAppState("processing");
    setError(null);
    setTranscript(queryToSend);
    setAnswer(null);
    setAudioUrl(null);
    setSources([]);
    setSttConfidence(null);
    setLowConfidenceWarning(false);

    try {
      const response = await fetch(`${apiUrl}/api/text-query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: queryToSend }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ detail: "Server error" }));
        throw new Error(errData.detail || "පිළිතුරක් ලබා ගැනීමට අපොහොසත් විය.");
      }

      const data = await response.json();
      if (data.transliterated_question) {
        setTranscript(data.transliterated_question);
      }
      setAnswer(data.answer_text);
      setAudioUrl(data.answer_audio_url);
      setSources(data.sources || []);
      setAppState("playing");
      if (!customQuery) setTextInput("");
    } catch (err: any) {
      console.error("Text query error:", err);
      setError(err.message || "සේවාදායකය සමඟ සම්බන්ධ වීමට නොහැක.");
      setAppState("idle");
    }
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans selection:bg-violet-500/30">
      {/* Background glow effects */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl -z-10" />
      <div className="absolute top-1/3 right-1/4 w-96 h-96 bg-emerald-600/5 rounded-full blur-3xl -z-10" />

      {/* Main Container */}
      <main className="flex-1 max-w-4xl w-full mx-auto px-4 py-12 flex flex-col space-y-10 items-center">
        {/* Premium Header */}
        <div className="text-center space-y-4 max-w-2xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-violet-500/10 border border-violet-500/20 text-violet-400 text-xs font-semibold uppercase tracking-wider">
            <Sparkles className="w-3.5 h-3.5" /> RAG Powered Voice AI
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400">
            සිංහල හඬ සහකාරයා
          </h1>
          <p className="text-zinc-400 text-base md:text-lg">
            සිංහල භාෂාවෙන් ප්‍රශ්න අසන්න, ඔබේ හඬ හෝ ටයිප් කිරීම මඟින් පිළිතුරු සහ හඬ මඟින් විස්තර ලබාගන්න.
          </p>
        </div>

        {/* Honest Limitations Panel */}
        <div className="w-full max-w-2xl bg-zinc-900/30 border border-zinc-800/40 rounded-2xl p-4 flex gap-3 text-xs md:text-sm text-zinc-400">
          <ShieldAlert className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold text-zinc-300">ප්‍රධාන සීමාවන් (Limitations Notice):</span>
            <p>
              සිංහල යනු තාක්ෂණිකව අඩු සම්පත් සහිත භාෂාවකි (Low-resource language). මේ නිසා, හඬ හඳුනාගැනීම (Whisper STT) සමහරවිට 
              අපැහැදිලි විය හැක. නිරවද්‍යතාවය තහවුරු කිරීමට මූලාශ්‍ර සහ විශ්වාසනීයත්ව දර්ශකය (Confidence score) සපයා ඇත.
            </p>
          </div>
        </div>

        {/* Audio Recorder Control Center */}
        <div className="w-full flex flex-col items-center justify-center space-y-4">
          <VoiceRecorder
            onResponse={handleVoiceResponse}
            onError={(err) => setError(err)}
            onStateChange={(state) => setAppState(state)}
            currentState={appState}
          />
        </div>

        {/* Text Fallback Input Form (FR-8) */}
        <div className="w-full max-w-2xl space-y-3">
          <form
            onSubmit={(e) => handleTextQuerySubmit(e)}
            className="relative flex items-center p-1 rounded-2xl bg-zinc-900/60 border border-zinc-800/80 shadow-2xl focus-within:border-violet-500/50 focus-within:ring-2 focus-within:ring-violet-500/10 transition-all duration-300 backdrop-blur-xl"
          >
            <input
              type="text"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              placeholder="නැතහොත් ප්‍රශ්නය මෙතැන ටයිප් කරන්න..."
              className="flex-1 bg-transparent px-4 py-3 text-zinc-200 placeholder-zinc-500 focus:outline-none text-base"
              disabled={appState === "processing"}
            />
            <button
              type="submit"
              disabled={!textInput.trim() || appState === "processing"}
              className="p-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white disabled:opacity-40 disabled:hover:bg-violet-600 hover:scale-105 active:scale-95 transition-all duration-200 flex-shrink-0"
              title="ප්‍රශ්නය යවන්න"
            >
              {appState === "processing" ? (
                <RefreshCw className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </form>

          {/* Suggested Sample Questions (SDLC Risk Item 2 adaptation) */}
          <div className="flex flex-wrap gap-2 items-center justify-center pt-1 text-xs text-zinc-500">
            <span className="inline-flex items-center gap-1 font-medium">
              <HelpCircle className="w-3.5 h-3.5" /> උදාහරණ ප්‍රශ්න:
            </span>
            {SUGGESTED_QUESTIONS.map((q, idx) => (
              <button
                key={idx}
                onClick={() => handleTextQuerySubmit(undefined, q.text)}
                disabled={appState === "processing"}
                className="px-3 py-1.5 rounded-xl bg-zinc-900/40 hover:bg-zinc-800/60 border border-zinc-800/50 hover:border-zinc-700/60 text-zinc-400 hover:text-zinc-200 transition-all duration-200"
              >
                {q.label}
              </button>
            ))}
          </div>
        </div>

        {/* Global Error Notice */}
        {error && (
          <div className="w-full max-w-2xl p-4 rounded-2xl bg-red-950/20 border border-red-800/30 text-red-400 text-sm text-center font-medium animate-pulse">
            {error}
          </div>
        )}

        {/* Results Card (Transcript & Answer) */}
        <TranscriptView
          transcript={transcript}
          answer={answer}
          audioUrl={audioUrl}
          sttConfidence={sttConfidence}
          lowConfidenceWarning={lowConfidenceWarning}
          onPlaybackStateChange={(isPlaying) => setAppState(isPlaying ? "playing" : "idle")}
          onRetry={() => setAppState("idle")}
        />

        {/* Grounded Citations */}
        <SourcesPanel sources={sources} />

        {/* Database Stats & Corpus Management Panel */}
        <div className="w-full max-w-2xl border-t border-zinc-900 pt-8 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-zinc-500">
          <div className="flex items-center gap-2">
            <Database className="w-4 h-4 text-zinc-600" />
            <span>
              {corpusStats ? (
                <>
                  මූලාශ්‍ර ගොනු ගණන: <strong className="text-zinc-400">{corpusStats.document_count}</strong> | 
                  කොටස් (Chunks): <strong className="text-zinc-400">{corpusStats.chunk_count}</strong>
                </>
              ) : (
                "මූලාශ්‍ර තොරතුරු සොයමින්..."
              )}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {corpusStats?.last_refreshed && (
              <span className="font-mono">
                යාවත්කාලීන වූයේ: {new Date(corpusStats.last_refreshed).toLocaleDateString()}
              </span>
            )}

            <button
              onClick={triggerCorpusRefresh}
              disabled={refreshingCorpus}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-zinc-900/60 hover:bg-zinc-800/80 border border-zinc-800/50 hover:border-zinc-700/60 text-zinc-400 hover:text-zinc-300 disabled:opacity-50 transition-all duration-200"
              title="මූලාශ්‍ර යාවත්කාලීන කරන්න"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${refreshingCorpus ? "animate-spin" : ""}`} />
              යාවත්කාලීන කරන්න
            </button>
          </div>
        </div>

        {/* Corpus refresh status banners */}
        {corpusMessage && (
          <div className="fixed bottom-4 right-4 max-w-sm p-4 rounded-xl bg-zinc-900 border border-zinc-800 shadow-2xl text-xs text-violet-400 font-medium z-50 animate-bounce">
            {corpusMessage}
          </div>
        )}
      </main>
    </div>
  );
}
