"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Waves, Volume2, Sparkles } from "lucide-react";
import Link from "next/link";

const vocoderModels = [
  {
    id: "bigvgan",
    name: "BigVGAN",
    description: "Large-scale GAN-based vocoder for high-fidelity audio synthesis from spectrograms",
    icon: Waves,
    href: "/vocoder/bigvgan",
  },
  {
    id: "hifigan",
    name: "HiFiGAN",
    description: "High-fidelity GAN vocoder with fast inference - convert mel-spectrograms to audio",
    icon: Volume2,
    href: "/vocoder/hifigan",
  },
  {
    id: "generic",
    name: "Generic Vocoder",
    description: "Universal vocoder interface supporting multiple vocoder backends",
    icon: Sparkles,
    href: "/vocoder/generic",
  },
];

export default function VocoderPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Vocoders</h1>
        <p className="text-muted-foreground">
          Convert spectrograms and acoustic features to high-quality audio waveforms
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {vocoderModels.map((model) => {
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
