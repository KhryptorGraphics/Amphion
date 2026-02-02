"use client";

import { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { TTSInterface } from "./components/TTSInterface";
import { ModelSelector } from "./components/ModelSelector";
import { TTSHistory } from "./components/TTSHistory";
import { BatchProcessor } from "./components/BatchProcessor";
import { ArrowRight, Sparkles } from "lucide-react";

const ttsModels = [
  { id: "maskgct", name: "MaskGCT", description: "Masked Generative Codec Transformer" },
  { id: "dualcodec-valle", name: "DualCodec-VALLE", description: "Neural Codec Language Model" },
  { id: "vevo-tts", name: "Vevo TTS", description: "Versatile Voice Conversion" },
  { id: "metis", name: "Metis", description: "Multi-Task Audio Model" },
];

const individualModels = [
  { id: "maskgct", name: "MaskGCT", description: "Masked Generative Codec Transformer for zero-shot TTS", href: "/tts/maskgct" },
  { id: "vits", name: "VITS", description: "Variational Inference with adversarial learning for end-to-end TTS", href: "/tts/vits" },
  { id: "fastspeech2", name: "FastSpeech 2", description: "Fast and controllable speech synthesis", href: "/tts/fastspeech2" },
  { id: "jets", name: "JETS", description: "Joint End-to-End Text-to-Speech model", href: "/tts/jets" },
  { id: "naturalspeech2", name: "NaturalSpeech 2", description: "Latent diffusion for speech synthesis", href: "/tts/naturalspeech2" },
  { id: "valle", name: "VALL-E", description: "Neural codec language model for TTS", href: "/tts/valle" },
  { id: "dualcodec-valle", name: "DualCodec-VALLE", description: "Enhanced codec with VALL-E architecture", href: "/tts/dualcodec-valle" },
  { id: "vevo-tts", name: "Vevo TTS", description: "Versatile voice conversion for text-to-speech", href: "/tts/vevo-tts" },
  { id: "metis", name: "Metis", description: "Multi-task audio generation model", href: "/tts/metis" },
  { id: "debatts", name: "DebaTTS", description: "Debate-based text-to-speech synthesis", href: "/tts/debatts" },
];

export default function TTSPage() {
  const [selectedModel, setSelectedModel] = useState(ttsModels[0]);

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold tracking-tight">Text-to-Speech</h1>
        <p className="text-muted-foreground">
          Convert text to natural-sounding speech using state-of-the-art models
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-4">
        <div className="lg:col-span-1">
          <ModelSelector
            models={ttsModels}
            selected={selectedModel}
            onSelect={setSelectedModel}
          />
        </div>

        <div className="lg:col-span-3">
          <Tabs defaultValue="generate" className="space-y-4">
            <TabsList className="grid w-full grid-cols-3 lg:max-w-md">
              <TabsTrigger value="generate">Generate</TabsTrigger>
              <TabsTrigger value="batch">Batch</TabsTrigger>
              <TabsTrigger value="history">History</TabsTrigger>
            </TabsList>

            <TabsContent value="generate" className="space-y-4">
              <TTSInterface model={selectedModel} />
            </TabsContent>

            <TabsContent value="batch">
              <BatchProcessor model={selectedModel} />
            </TabsContent>

            <TabsContent value="history">
              <TTSHistory />
            </TabsContent>
          </Tabs>
        </div>
      </div>

      <div className="mt-12">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-2xl font-bold flex items-center gap-2">
              <Sparkles className="h-6 w-6" />
              Browse All Models
            </h2>
            <p className="text-muted-foreground mt-1">
              Explore individual model interfaces with specialized controls
            </p>
          </div>
          <Button variant="outline" asChild>
            <Link href="/tts/compare">
              Compare Models
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {individualModels.map((model) => (
            <Link key={model.id} href={model.href}>
              <Card className="h-full hover:shadow-lg hover:border-primary/50 transition-all cursor-pointer">
                <CardHeader>
                  <CardTitle className="text-lg">{model.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{model.description}</p>
                  <div className="flex items-center text-xs text-primary mt-3">
                    Try it now
                    <ArrowRight className="ml-1 h-3 w-3" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
