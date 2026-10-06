import { Emo, type EmoSuggestion } from "@desert-ant-labs/emo";

export interface EmojiSuggestionResult {
  text: string;
  suggestions: EmoSuggestion[];
}

let emoPromise: Promise<Emo> | null = null;

function loadEmo(): Promise<Emo> {
  if (!emoPromise) {
    emoPromise = Emo.load().catch((error: unknown) => {
      emoPromise = null;
      throw error;
    });
  }

  return emoPromise;
}

export async function suggestEmoji(text: string): Promise<EmojiSuggestionResult> {
  const caption = text.trim();
  if (!caption) {
    return { text: caption, suggestions: [] };
  }

  const emo = await loadEmo();
  const suggestions = await emo.suggestions(caption, { limit: 3 });

  return { text: caption, suggestions };
}

export async function disposeEmo(): Promise<void> {
  const modelPromise = emoPromise;
  emoPromise = null;

  if (!modelPromise) {
    return;
  }

  try {
    const emo = await modelPromise;
    emo.dispose();
  } catch (error) {
    console.error("Failed to dispose Emo:", error);
  }
}
