"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { FileUpload } from "@/components/ui/file-upload";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowRight, Mic2, Music, Volume2, Wand2, Loader2, FileAudio } from "lucide-react";
import Link from "next/link";

const vcModels = [
  {
    id: "vevo-voice",
    name: "Vevo Voice",
    description: "Full voice conversion - transfer voice characteristics",
    icon: Mic2,
    href: "/vc/vevo-voice",
  },
  {
    id: "vevo-timbre",
    name: "Vevo Timbre",
    description: "Timbre-only conversion - preserve style, change voice",
    icon: Volume2,
    href: "/vc/vevo-timbre",
  },
  {
    id: "vevo-style",
    name: "Vevo Style",
    description: "Style/accent conversion - change speaking style",
    icon: Music,
    href: "/vc/vevo-style",
  },
  {
    id: "noro",
    name: "Noro",
    description: "Noise-robust voice conversion for challenging audio",
    icon: Wand2,
    href: "/vc/noro",
  },
];

const svcModels = [
  {
    id: "vevosing",
    name: "VevoSing",
    description: "Singing voice conversion - convert singer voice",
    icon: Music,
    href: "/svc/vevosing",
  },
];

export default function VCPage() {
  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Voice Conversion</h1>
        <p className="text-muted-foreground">
          Convert voice characteristics while preserving content
        </p>
      </div>

      <Tabs defaultValue="vc" className="space-y-6">
        <TabsList className="grid w-full grid-cols-2 lg:max-w-md">
          <TabsTrigger value="vc">Voice Conversion</TabsTrigger>
          <TabsTrigger value="svc">Singing Voice</TabsTrigger>
        </TabsList>

        <TabsContent value="vc" className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            {vcModels.map((model) => {
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
        </TabsContent>

        <TabsContent value="svc" className="space-y-6">
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

          <Card className="border-dashed">
            <CardContent className="py-8 text-center">
              <Music className="mx-auto h-12 w-12 text-muted-foreground/50 mb-4" />
              <p className="text-muted-foreground">
                More SVC models coming soon: DiffComoSVC, TransformerSVC, VitsSVC
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
