"use client";

import { cn } from "@/lib/utils";

interface GenerationWaveformProps {
  isGenerating: boolean;
  className?: string;
}

const BAR_COUNT = 10;
const ANIMATION_DELAY_MS = 100;

export function GenerationWaveform({
  isGenerating,
  className,
}: GenerationWaveformProps) {
  if (!isGenerating) return null;

  return (
    <div className={cn("flex flex-col items-center gap-3 py-4", className)}>
      <style>{`
        @keyframes waveBarPulse {
          0%, 100% { transform: scaleY(0.2); }
          50% { transform: scaleY(1); }
        }
      `}</style>

      <div className="flex items-center justify-center gap-1 h-12">
        {Array.from({ length: BAR_COUNT }).map((_, i) => (
          <div
            key={i}
            className="w-1.5 h-12 rounded-full bg-primary/70 origin-center"
            style={{
              animation: "waveBarPulse 0.9s ease-in-out infinite",
              animationDelay: `${i * ANIMATION_DELAY_MS}ms`,
              transform: "scaleY(0.2)",
            }}
          />
        ))}
      </div>

      <p className="text-sm text-muted-foreground animate-pulse">
        Generating audio…
      </p>
    </div>
  );
}
