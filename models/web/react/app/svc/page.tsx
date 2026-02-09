"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ArrowRight, Music, Wand2, Layers, Sparkles, FileAudio } from "lucide-react";
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
