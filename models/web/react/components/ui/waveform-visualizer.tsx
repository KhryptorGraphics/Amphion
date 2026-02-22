"use client";

import { useRef, useEffect, useCallback } from "react";
import { useAudioAnalyser } from "@/hooks/use-audio-analyser";
import { cn } from "@/lib/utils";

interface WaveformVisualizerProps {
  audioElement: HTMLAudioElement | null;
  className?: string;
  mode?: "waveform" | "frequency";
  barColor?: string;
  backgroundColor?: string;
  showGrid?: boolean;
}

export function WaveformVisualizer({
  audioElement,
  className,
  mode = "waveform",
  barColor = "hsl(var(--primary))",
  backgroundColor = "transparent",
  showGrid = false,
}: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);

  const {
    isReady,
    dataArray,
    bufferLength,
    connectToElement,
    getByteFrequencyData,
    getByteTimeDomainData,
  } = useAudioAnalyser({
    fftSize: 2048,
    smoothingTimeConstant: 0.8,
  });

  // Connect to audio element when it changes
  useEffect(() => {
    connectToElement(audioElement);
  }, [audioElement, connectToElement]);

  // Draw waveform (time domain data)
  const drawWaveform = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      data: Uint8Array
    ) => {
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);

      if (showGrid) {
        ctx.strokeStyle = "hsl(var(--border))";
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        // Horizontal center line
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        // Vertical lines
        for (let i = 0; i < width; i += 40) {
          ctx.moveTo(i, 0);
          ctx.lineTo(i, height);
        }
        ctx.stroke();
      }

      ctx.lineWidth = 2;
      ctx.strokeStyle = barColor;
      ctx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = data[i] / 128.0;
        const y = (v * height) / 2;

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }

        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();
    },
    [backgroundColor, barColor, bufferLength, showGrid]
  );

  // Draw frequency bars
  const drawFrequencyBars = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      data: Uint8Array
    ) => {
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);

      if (showGrid) {
        ctx.strokeStyle = "hsl(var(--border))";
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        // Horizontal lines
        for (let i = 0; i < height; i += 20) {
          ctx.moveTo(0, i);
          ctx.lineTo(width, i);
        }
        // Vertical lines
        for (let i = 0; i < width; i += 40) {
          ctx.moveTo(i, 0);
          ctx.lineTo(i, height);
        }
        ctx.stroke();
      }

      const barWidth = (width / bufferLength) * 2.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (data[i] / 255) * height;

        // Create gradient effect
        const gradient = ctx.createLinearGradient(0, height - barHeight, 0, height);
        gradient.addColorStop(0, barColor);
        gradient.addColorStop(1, `${barColor}80`);

        ctx.fillStyle = gradient;
        ctx.fillRect(x, height - barHeight, barWidth, barHeight);

        x += barWidth + 1;
      }
    },
    [backgroundColor, barColor, bufferLength, showGrid]
  );

  // Animation loop
  useEffect(() => {
    if (!isReady || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size to match display size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const width = rect.width;
    const height = rect.height;

    const data = new Uint8Array(bufferLength);

    const animate = () => {
      if (mode === "frequency") {
        getByteFrequencyData(data);
        drawFrequencyBars(ctx, width, height, data);
      } else {
        getByteTimeDomainData(data);
        drawWaveform(ctx, width, height, data);
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [
    isReady,
    bufferLength,
    mode,
    getByteFrequencyData,
    getByteTimeDomainData,
    drawWaveform,
    drawFrequencyBars,
  ]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (!canvasRef.current) return;

      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className={cn("w-full h-32 rounded-md", className)}
      style={{ backgroundColor }}
    />
  );
}
