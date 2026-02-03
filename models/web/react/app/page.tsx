"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Mic, Volume2, Music, FileAudio, BarChart3, Wand2, AudioWaveform, Waves } from "lucide-react";

const FEATURES = [
  {
    title: "Text to Speech",
    description: "Generate natural speech from text using state-of-the-art models",
    icon: Mic,
    href: "/tts",
    color: "bg-blue-500",
  },
  {
    title: "Voice Conversion",
    description: "Convert voices while preserving content and style",
    icon: Volume2,
    href: "/vc",
    color: "bg-purple-500",
  },
  {
    title: "Singing Voice",
    description: "Singing voice conversion and synthesis",
    icon: Music,
    href: "/svc",
    color: "bg-pink-500",
  },
  {
    title: "Text to Audio",
    description: "Generate sound effects and audio from text descriptions",
    icon: AudioWaveform,
    href: "/tta",
    color: "bg-indigo-500",
  },
  {
    title: "Audio Codecs",
    description: "Neural audio codec encoding and decoding",
    icon: FileAudio,
    href: "/codec",
    color: "bg-green-500",
  },
  {
    title: "Vocoders",
    description: "Convert spectrograms to high-fidelity audio waveforms",
    icon: Waves,
    href: "/vocoder",
    color: "bg-cyan-500",
  },
  {
    title: "Evaluation",
    description: "Analyze and compare audio quality metrics",
    icon: BarChart3,
    href: "/tools/evaluation",
    color: "bg-orange-500",
  },
  {
    title: "Batch Processing",
    description: "Process multiple files efficiently",
    icon: Wand2,
    href: "/batch",
    color: "bg-teal-500",
  },
];

export default function HomePage() {
  return (
    <div className="container mx-auto py-8">
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-4">Amphion Studio</h1>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          A comprehensive toolkit for audio, music, and speech generation
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {FEATURES.map((feature) => (
          <Link key={feature.title} href={feature.href}>
            <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer">
              <CardHeader>
                <div className={`w-12 h-12 rounded-lg ${feature.color} flex items-center justify-center mb-4`}>
                  <feature.icon className="h-6 w-6 text-white" />
                </div>
                <CardTitle>{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground">{feature.description}</p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      <div className="mt-12 text-center">
        <h2 className="text-2xl font-semibold mb-4">Getting Started</h2>
        <p className="text-muted-foreground mb-6">
          Choose a tool above to start generating audio, or explore the documentation to learn more.
        </p>
        <div className="flex gap-4 justify-center">
          <Button size="lg" asChild>
            <Link href="/tts">Start with TTS</Link>
          </Button>
          <Button size="lg" variant="outline" asChild>
            <Link href="/tts/compare">Compare Models</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
