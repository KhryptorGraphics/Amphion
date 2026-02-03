"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Music, AudioWaveform } from "lucide-react";
import Link from "next/link";

const ttaModels = [
  {
    id: "audioldm",
    name: "AudioLDM",
    description: "Text-to-audio generation using latent diffusion - create sound effects and music from text descriptions",
    icon: AudioWaveform,
    href: "/tta/audioldm",
  },
  {
    id: "picoaudio",
    name: "PicoAudio",
    description: "Lightweight text-to-audio generation for real-time applications and edge devices",
    icon: Music,
    href: "/tta/picoaudio",
  },
];

export default function TTAPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Text-to-Audio</h1>
        <p className="text-muted-foreground">
          Generate audio from text descriptions - sound effects, ambience, and more
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {ttaModels.map((model) => {
          const Icon = model.icon;
          return (
            <Link key={model.id} href={model.href}>
              <Card className="hover:border-primary transition-colors cursor-pointer h-full">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Icon className="h-5 w-5" />
                    {model.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">
                    {model.description}
                  </p>
                  <div className="flex items-center gap-1 mt-4 text-sm text-primary">
                    Try it <ArrowRight className="h-4 w-4" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
