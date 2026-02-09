"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Music, Wand2, Layers, Sparkles, FileAudio, BookOpen, Lightbulb } from "lucide-react";
import Link from "next/link";

const svcModels = [
  {
    id: "vevosing",
    name: "VevoSing",
    description: "Singing voice conversion using flow matching - convert singer voice while preserving melody",
    icon: Music,
    href: "/svc/vevosing",
    experimental: false,
  },
  {
    id: "diffcomosvc",
    name: "DiffComoSVC",
    description: "Diffusion-based SVC using consistency model for fast, high-quality singing voice conversion",
    icon: Wand2,
    href: "/svc/diffcomosvc",
    experimental: true,
  },
  {
    id: "transformersvc",
    name: "TransformerSVC",
    description: "Transformer-based SVC with attention mechanisms for high-fidelity timbre transfer",
    icon: Layers,
    href: "/svc/transformersvc",
    experimental: true,
  },
  {
    id: "vitssvc",
    name: "VitsSVC",
    description: "VITS-based end-to-end SVC with variational inference and adversarial training",
    icon: Sparkles,
    href: "/svc/vitssvc",
    experimental: true,
  },
  {
    id: "multiplecontentssvc",
    name: "MultipleContentsSVC",
    description: "Multi-content SVC leveraging multiple content features for enhanced quality",
    icon: FileAudio,
    href: "/svc/multiplecontentssvc",
    experimental: true,
  },
];

export default function SVCPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Singing Voice Conversion</h1>
        <p className="text-muted-foreground">
          Convert singing voices while preserving melody and pitch
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 mb-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              How Singing Voice Conversion Works
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-sm text-muted-foreground">
              SVC lets you re-sing a song in a completely different voice while keeping the
              original melody, lyrics, and timing intact.
            </p>
            <ol className="text-sm text-muted-foreground space-y-2 list-decimal list-inside">
              <li><strong>Upload Content Audio</strong> &mdash; the song you want to convert. The melody, pitch, and rhythm from this file are preserved.</li>
              <li><strong>Upload Reference Audio</strong> &mdash; a sample of the target singer&apos;s voice. This provides the vocal timbre (tone color) to apply.</li>
              <li><strong>Adjust Parameters</strong> &mdash; if the model supports them (e.g., VevoSing has mode and step controls).</li>
              <li><strong>Click Convert</strong> &mdash; the model will generate a new audio file singing the same song in the target voice.</li>
            </ol>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="h-5 w-5" />
              Tips for Best Results
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm text-muted-foreground space-y-2 list-disc list-inside">
              <li><strong>Use clean audio</strong> &mdash; background music or noise in either file reduces quality. Isolated vocals work best.</li>
              <li><strong>WAV over MP3</strong> &mdash; lossless formats preserve more detail for the model to work with.</li>
              <li><strong>Similar pitch range</strong> &mdash; conversion works best when source and target singers have a similar vocal range.</li>
              <li><strong>Reference length</strong> &mdash; 5-15 seconds of reference audio is usually enough. Longer isn&apos;t always better.</li>
              <li><strong>VevoSing is recommended</strong> &mdash; it&apos;s the most capable model with tunable parameters. The experimental models require pre-trained checkpoints.</li>
              <li><strong>Full Song Mode</strong> &mdash; all models support Full Song Mode. Toggle it on to upload a complete song with instrumentals &mdash; Demucs will separate vocals automatically, convert them, and remix with the original accompaniment.</li>
            </ul>
          </CardContent>
        </Card>
      </div>

      <h2 className="text-xl font-semibold tracking-tight mb-4">Choose a Model</h2>

      <div className="grid gap-4 md:grid-cols-2">
        {svcModels.map((model) => {
          const Icon = model.icon;
          return (
            <Link key={model.id} href={model.href}>
              <Card className="hover:border-primary transition-colors cursor-pointer h-full">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Icon className="h-5 w-5" />
                    {model.name}
                    {model.experimental && (
                      <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-amber-500/10 text-amber-500 border border-amber-500/20">
                        Experimental
                      </span>
                    )}
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
