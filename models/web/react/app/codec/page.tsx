"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Disc3, FileAudio } from "lucide-react";
import Link from "next/link";

const codecModels = [
  {
    id: "dualcodec",
    name: "DualCodec",
    description: "Neural audio codec with dual-stream encoding for high-quality audio compression and reconstruction",
    icon: Disc3,
    href: "/codec/dualcodec",
  },
  {
    id: "facodec",
    name: "FACodec",
    description: "Factorized neural audio codec - decompose audio into different attributes (content, timbre, prosody)",
    icon: FileAudio,
    href: "/codec/facodec",
  },
];

export default function CodecPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Neural Audio Codecs</h1>
        <p className="text-muted-foreground">
          Encode and decode audio using neural codecs - compress, reconstruct, and manipulate audio
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {codecModels.map((model) => {
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
