"use client";

import { useRef, useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, Volume2 } from "lucide-react";

interface AudioComparisonPlayerProps {
  srcA: string;
  srcB: string;
  labelA?: string;
  labelB?: string;
}

export function AudioComparisonPlayer({
  srcA,
  srcB,
  labelA = "Track A",
  labelB = "Track B",
}: AudioComparisonPlayerProps) {
  const audioRefA = useRef<HTMLAudioElement>(null);
  const audioRefB = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);

  useEffect(() => {
    const audioA = audioRefA.current;
    const audioB = audioRefB.current;
    if (!audioA || !audioB) return;

    const handleTimeUpdate = () => setCurrentTime(audioA.currentTime);
    const handleLoadedMetadata = () => setDuration(audioA.duration);
    const handleEnded = () => setIsPlaying(false);

    audioA.addEventListener("timeupdate", handleTimeUpdate);
    audioA.addEventListener("loadedmetadata", handleLoadedMetadata);
    audioA.addEventListener("ended", handleEnded);

    return () => {
      audioA.removeEventListener("timeupdate", handleTimeUpdate);
      audioA.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audioA.removeEventListener("ended", handleEnded);
    };
  }, [srcA, srcB]);

  const togglePlay = () => {
    const audioA = audioRefA.current;
    const audioB = audioRefB.current;
    if (!audioA || !audioB) return;

    if (isPlaying) {
      audioA.pause();
      audioB.pause();
    } else {
      audioA.play();
      audioB.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (value: number[]) => {
    const audioA = audioRefA.current;
    const audioB = audioRefB.current;
    if (!audioA || !audioB) return;

    audioA.currentTime = value[0];
    audioB.currentTime = value[0];
    setCurrentTime(value[0]);
  };

  const handleVolumeChange = (value: number[]) => {
    const audioA = audioRefA.current;
    const audioB = audioRefB.current;
    if (!audioA || !audioB) return;

    audioA.volume = value[0];
    audioB.volume = value[0];
    setVolume(value[0]);
  };

  const formatTime = (time: number) => {
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="space-y-4">
      <audio ref={audioRefA} src={srcA} className="hidden" />
      <audio ref={audioRefB} src={srcB} className="hidden" />

      {/* Track display areas */}
      <div className="grid grid-cols-2 gap-4">
        {/* Track A */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-center">{labelA}</p>
          <div className="h-16 bg-muted rounded-lg overflow-hidden relative flex items-end gap-px px-2 pb-2">
            {Array.from({ length: 40 }).map((_, i) => (
              <div
                key={i}
                className="flex-1 rounded-sm transition-all duration-75"
                style={{
                  height: isPlaying
                    ? `${20 + Math.abs(Math.sin((i + currentTime * 10) * 0.5)) * 70}%`
                    : "20%",
                  backgroundColor:
                    (i / 40) * 100 <= progress
                      ? "hsl(var(--primary))"
                      : "hsl(var(--muted-foreground) / 0.3)",
                }}
              />
            ))}
          </div>
        </div>

        {/* Track B */}
        <div className="space-y-2">
          <p className="text-sm font-medium text-center">{labelB}</p>
          <div className="h-16 bg-muted rounded-lg overflow-hidden relative flex items-end gap-px px-2 pb-2">
            {Array.from({ length: 40 }).map((_, i) => (
              <div
                key={i}
                className="flex-1 rounded-sm transition-all duration-75"
                style={{
                  height: isPlaying
                    ? `${20 + Math.abs(Math.sin((i + currentTime * 10) * 0.7 + 1)) * 70}%`
                    : "20%",
                  backgroundColor:
                    (i / 40) * 100 <= progress
                      ? "hsl(var(--primary))"
                      : "hsl(var(--muted-foreground) / 0.3)",
                }}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Shared playback controls */}
      <div className="flex items-center gap-3">
        <Button
          variant="outline"
          size="icon"
          onClick={togglePlay}
          className="h-10 w-10"
        >
          {isPlaying ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </Button>

        <div className="flex-1">
          <Slider
            value={[currentTime]}
            max={duration || 100}
            step={0.1}
            onValueChange={handleSeek}
          />
        </div>

        <span className="text-sm text-muted-foreground w-20 text-right">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>

      {/* Shared volume control */}
      <div className="flex items-center gap-2">
        <Volume2 className="h-4 w-4 text-muted-foreground" />
        <Slider
          value={[volume]}
          max={1}
          step={0.1}
          onValueChange={handleVolumeChange}
          className="w-24"
        />
      </div>
    </div>
  );
}
