"use client";

import { useEffect, useState } from "react";
import {
  disposeEmo,
  suggestEmoji,
} from "./services/emoService";

const MAX_EMOJI_SUGGESTIONS = 5;
const MIN_EMOJI_CONFIDENCE = 0.35;

interface TimedCaption {
  start: number;
  end: number;
  text: string;
}

interface CaptionEmoji {
  text: string;
  emoji: string;
  confidence: number;
  start: number;
  end: number;
}

interface EmojiOverlay {
  emoji: string;
  start: number;
  end: number;
  caption: string;
  asset_id: string;
}

interface EmojiAsset {
  id: string;
  image_base64: string;
}

function parseSelectedCaptions(srt: unknown): TimedCaption[] {
  if (typeof srt !== "string" || !srt.trim()) {
    return [];
  }

  return srt
    .trim()
    .split(/\r?\n\s*\r?\n/)
    .flatMap((block): TimedCaption[] => {
      const lines = block.split(/\r?\n/).map((line) => line.trim());
      const timestampIndex = lines.findIndex((line) =>
        /^\d+:\d{2}:\d{2}[,.]\d{3}\s+-->/.test(line)
      );
      if (timestampIndex < 0) {
        return [];
      }

      const timestampMatch = lines[timestampIndex].match(
        /^(\d+):(\d{2}):(\d{2})[,.](\d{3})\s+-->\s+(\d+):(\d{2}):(\d{2})[,.](\d{3})$/
      );
      const text = lines.slice(timestampIndex + 1).filter(Boolean).join(" ");
      if (!timestampMatch || !text) {
        return [];
      }

      const toSeconds = (
        hours: string,
        minutes: string,
        seconds: string,
        milliseconds: string
      ) =>
        Number(hours) * 3600 +
        Number(minutes) * 60 +
        Number(seconds) +
        Number(milliseconds) / 1000;

      const start = toSeconds(
        timestampMatch[1],
        timestampMatch[2],
        timestampMatch[3],
        timestampMatch[4]
      );
      const end = toSeconds(
        timestampMatch[5],
        timestampMatch[6],
        timestampMatch[7],
        timestampMatch[8]
      );

      return Number.isFinite(start) && Number.isFinite(end) && end > start
        ? [{ start, end, text }]
        : [];
    });
}

function createEmojiPng(emoji: string): string {
  const canvas = document.createElement("canvas");
  canvas.width = 192;
  canvas.height = 192;

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not create an emoji canvas.");
  }

  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font =
    '128px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif';
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(emoji, canvas.width / 2, canvas.height / 2, 168);

  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let hasVisiblePixel = false;
  for (let index = 3; index < pixels.length; index += 4) {
    if (pixels[index] > 0) {
      hasVisiblePixel = true;
      break;
    }
  }
  if (!hasVisiblePixel) {
    throw new Error(`The browser could not render emoji ${emoji}.`);
  }

  const dataUrl = canvas.toDataURL("image/png");
  const prefix = "data:image/png;base64,";
  if (!dataUrl.startsWith(prefix)) {
    throw new Error("Could not encode the emoji canvas as a PNG.");
  }
  return dataUrl.slice(prefix.length);
}

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [topic, setTopic] = useState("");
  const [progress, setProgress] = useState(0);
  const [progressStage, setProgressStage] = useState("Idle");
  const [status, setStatus] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [videoUrl, setVideoUrl] = useState("");
  const [captionEmojis, setCaptionEmojis] = useState<CaptionEmoji[]>([]);
  const [emojiStatus, setEmojiStatus] = useState("");
  const [emojiApplied, setEmojiApplied] = useState(false);
  const [generatedTitle, setGeneratedTitle] = useState("");
  const [generatedHashtags, setGeneratedHashtags] = useState<string[]>([]);
  const [hashtagsCopied, setHashtagsCopied] = useState(false);
  const [titleCopied, setTitleCopied] = useState(false);
  // =========================
  // PROGRESS POLLING
  // =========================

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(
          "http://localhost:8001/api/video/progress"
        );

        if (!response.ok) {
          return;
        }

        const data = await response.json();

        setProgress(data.progress);
        setProgressStage(data.stage);
      } catch (error) {
        console.error("Progress check failed:", error);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // =========================
  // GENERATE VIDEO
  // =========================

  const handleGenerate = async () => {
    console.log("GENERATE BUTTON CLICKED");

    if (!videoFile || !topic.trim()) {
      alert("Please select a video file and enter a topic.");
      return;
    }

    const formData = new FormData();

    formData.append("file", videoFile);
    formData.append("topic", topic.trim());

    try {
      setIsProcessing(true);
      setStatus("Uploading video & running AI repurposing pipeline...");

      setProgress(0);
      setProgressStage("Starting");

      setVideoUrl("");
      setCaptionEmojis([]);
      setEmojiStatus("");
      setEmojiApplied(false);
      setGeneratedTitle(""); 
      setGeneratedHashtags([]);
      setHashtagsCopied(false);
      setTitleCopied(false);

      const response = await fetch(
        "http://localhost:8001/api/video/process",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Video processing failed"
        );
      }

      setProgress(100);
      setProgressStage("Completed");

      setStatus(
        "🎉 Short-form video generated & validated successfully!"
      );
      setGeneratedTitle(
  typeof data.title === "string" && data.title.trim()
    ? data.title.trim()
    : "AI title unavailable"
);

setGeneratedHashtags(
  Array.isArray(data.hashtags)
    ? data.hashtags.filter(
        (tag: unknown): tag is string =>
          typeof tag === "string" && tag.trim().length > 0
      )
    : []
);

      const baseVideoUrl =
        `http://localhost:8001/api/video/result?t=${Date.now()}`;
      setVideoUrl(baseVideoUrl);
      setIsProcessing(false);
      const captions = parseSelectedCaptions(data.captions);

      if (captions.length === 0) {
        setEmojiStatus("No captions available for AI emoji suggestions.");
      } else {
        setEmojiStatus("Finding relevant emoji suggestions...");

        try {
          const results: CaptionEmoji[] = [];

          for (const caption of captions) {
            const result = await suggestEmoji(caption.text);
            const bestSuggestion = result.suggestions[0];

            if (
              bestSuggestion &&
              bestSuggestion.confidence >= MIN_EMOJI_CONFIDENCE
            ) {
              results.push({
                text: result.text,
                emoji: bestSuggestion.emoji,
                confidence: bestSuggestion.confidence,
                start: caption.start,
                end: caption.end,
              });
            }
          }

          setCaptionEmojis(results);
          const overlays: EmojiOverlay[] = [];
          const assets: EmojiAsset[] = [];
          const assetIdsByEmoji = new Map<string, string>();
          for (const suggestion of results) {
            let assetId = assetIdsByEmoji.get(suggestion.emoji);
            if (!assetId) {
              assetId = `emoji-${assets.length}`;
              assetIdsByEmoji.set(suggestion.emoji, assetId);
              assets.push({
                id: assetId,
                image_base64: createEmojiPng(suggestion.emoji),
              });
            }

            overlays.push({
              emoji: suggestion.emoji,
              start: suggestion.start,
              end: suggestion.end,
              caption: suggestion.text,
              asset_id: assetId,
            });
          }

          if (overlays.length === 0) {
            setEmojiStatus("No relevant emoji suggestions found.");
          } else {
            setEmojiStatus("Applying AI emojis...");
            const emojiResponse = await fetch(
              "http://localhost:8001/api/video/add-emojis",
              {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ overlays, assets }),
              }
            );
            const emojiData = await emojiResponse.json();
            if (!emojiResponse.ok) {
              throw new Error(emojiData.detail || "Emoji rendering failed.");
            }

            if (emojiData.status === "success") {
              setEmojiApplied(true);
              setVideoUrl(
                `http://localhost:8001/api/video/result?t=${Date.now()}`
              );
              setEmojiStatus("AI emojis added to video ✓");
            } else {
              setEmojiStatus(
                "Emoji rendering unavailable; using the original video."
              );
            }
          }
        } catch (error) {
          console.error("AI emoji processing unavailable:", error);
          setEmojiStatus("AI emojis unavailable; using the original video.");
          setVideoUrl(baseVideoUrl);
        } finally {
          await disposeEmo();
        }
      }

    } catch (error: unknown) {
      console.error("Processing error:", error);

      const message =
        error instanceof Error
          ? error.message
          : String(error);

      setStatus(`❌ Processing failed: ${message}`);

    } finally {
      setIsProcessing(false);
    }
  };

  const handleCopyTitle = async () => {
    if (!generatedTitle) {
      return;
    }

    try {
      await navigator.clipboard.writeText(generatedTitle);
      setTitleCopied(true);
      window.setTimeout(() => setTitleCopied(false), 1800);
    } catch (error) {
      console.error("Could not copy generated title:", error);
    }
  };

  const handleCopyHashtags = async () => {
    if (generatedHashtags.length === 0) {
      return;
    }

    try {
      await navigator.clipboard.writeText(generatedHashtags.join(" "));
      setHashtagsCopied(true);
      window.setTimeout(() => setHashtagsCopied(false), 1800);
    } catch (error) {
      console.error("Could not copy generated hashtags:", error);
    }
  };

  // =========================
  // UI
  // =========================

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center p-6 font-sans">

      <div className="w-full max-w-2xl">

        {/* TITLE */}

        <h1 className="text-4xl font-extrabold text-center mb-3 tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
          AI Video Repurposer
        </h1>

        <p className="text-center text-gray-400 mb-8">
          Turn long videos into viral, content-aware short-form videos (9:16)
        </p>

        {/* MAIN CARD */}

        <div className="border border-gray-800 rounded-2xl p-8 bg-zinc-950 shadow-2xl backdrop-blur-sm">

          {/* VIDEO UPLOAD */}

          <label className="block mb-2 font-semibold text-gray-200">
            Upload Source Video
          </label>

          <input
            type="file"
            accept="video/*"
            disabled={isProcessing}
            onChange={(e) =>
              setVideoFile(e.target.files?.[0] || null)
            }
            className="w-full mb-6 text-sm text-gray-400 file:mr-4 file:py-2.5 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-zinc-800 file:text-white hover:file:bg-zinc-700 cursor-pointer"
          />

          {/* TOPIC */}

          <label className="block mb-2 font-semibold text-gray-200">
            Target Topic
          </label>

          <input
            type="text"
            placeholder="Example: Python functions"
            value={topic}
            disabled={isProcessing}
            onChange={(e) => setTopic(e.target.value)}
            className="w-full rounded-lg bg-zinc-900 border border-gray-800 p-3 mb-6 text-white focus:outline-none focus:ring-2 focus:ring-white transition-all"
          />

          {/* GENERATE BUTTON */}

          <button
            type="button"
            onClick={handleGenerate}
            disabled={isProcessing}
            className="w-full py-3.5 rounded-lg font-bold text-black bg-white hover:bg-gray-200 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isProcessing
              ? "Processing Video Pipeline..."
              : "Generate Short-Form Video"}
          </button>

          {/* STATUS */}

          {status && (
            <p className="text-center text-sm font-medium text-gray-300 mt-5">
              {status}
            </p>
          )}

          {/* PROGRESS */}

          {(isProcessing || progress > 0) && (
            <div className="mt-6">

              <div className="flex justify-between text-sm text-gray-400 mb-2">

                <span>
                  {progressStage}
                </span>

                <span>
                  {progress}%
                </span>

              </div>

              <div className="w-full h-3 bg-zinc-800 rounded-full overflow-hidden">

                <div
                  className="h-full bg-white transition-all duration-500"
                  style={{
                    width: `${progress}%`,
                  }}
                />

              </div>

            </div>
          )}

          {/* GENERATED VIDEO */}

          {videoUrl && (
            <div className="mt-8 border-t border-gray-800 pt-6">

              <h2 className="text-xl font-bold mb-4 text-center">
                Generated Short Video
              </h2>

              <div className="flex justify-center bg-black/50 p-2 rounded-xl">

                <video
                  src={videoUrl}
                  controls
                  autoPlay
                  playsInline
                  className="max-h-[500px] rounded-lg shadow-lg border border-gray-800"
                />

              </div>

              <section className="mt-6 border-t border-gray-800 pt-5">
                <h3 className="text-lg font-bold mb-3 text-center">
                  🏷️ AI Generated Title
                </h3>
                <p className="text-center text-gray-200 mb-4">
                  {generatedTitle || "AI title unavailable"}
                </p>
                <button
                  type="button"
                  onClick={handleCopyTitle}
                  disabled={!generatedTitle}
                  className="mx-auto block rounded-lg bg-zinc-800 px-4 py-2 text-sm font-semibold text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {titleCopied ? "Copied!" : "Copy Title"}
                </button>
                {generatedHashtags.length > 0 && (
                  <div className="mt-5">
                    <div className="flex flex-wrap justify-center gap-2">
                      {generatedHashtags.map((hashtag, index) => (
                        <span
                          key={`${hashtag}-${index}`}
                          className="rounded-full border border-gray-700 bg-zinc-900 px-3 py-1 text-sm text-gray-300"
                        >
                          {hashtag}
                        </span>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={handleCopyHashtags}
                      className="mx-auto mt-4 block rounded-lg bg-zinc-800 px-4 py-2 text-sm font-semibold text-white hover:bg-zinc-700"
                    >
                      {hashtagsCopied ? "Copied!" : "Copy Hashtags"}
                    </button>
                  </div>
                )}
              </section>
              {/* DOWNLOAD */}

              <a
                href={videoUrl}
                download={
                  emojiApplied
                    ? "final_pipeline_emoji.mp4"
                    : "final_pipeline.mp4"
                }
                className="block text-center mt-6 bg-emerald-500 text-black py-3 rounded-lg font-bold hover:bg-emerald-400 transition-all shadow-md"
              >
                Download Video (.mp4)
              </a>

              <section className="mt-6 border-t border-gray-800 pt-5">
                <h3 className="text-lg font-bold mb-3 text-center">
                  AI Emoji Suggestions
                </h3>

                {emojiStatus && (
                  <p className="text-center text-sm text-gray-400">
                    {emojiStatus}
                  </p>
                )}

                {captionEmojis.length > 0 && (
                  <ul className="space-y-2">
                    {captionEmojis
                      .slice(0, MAX_EMOJI_SUGGESTIONS)
                      .map(({ text, emoji, start, end }) => (
                      <li
                        key={`${start}-${end}-${emoji}`}
                        className="flex items-start gap-3 rounded-lg bg-zinc-900 p-3"
                      >
                        <span aria-hidden="true" className="text-xl">
                          {emoji}
                        </span>
                        <span className="text-sm text-gray-300">{text}</span>
                      </li>
                      ))}
                  </ul>
                )}
              </section>

            </div>
          )}

        </div>
      </div>

    </main>
  );
} 