"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TTSInterface } from "./components/TTSInterface";
import { ModelSelector } from "./components/ModelSelector";
import { TTSHistory } from "./components/TTSHistory";
import { BatchProcessor } from "./components/BatchProcessor";

const ttsModels = [
  { id: "maskgct", name: "MaskGCT", description: "Masked Generative Codec Transformer" },
  { id: "dualcodec-valle", name: "DualCodec-VALLE", description: "Neural Codec Language Model" },
  { id: "vevo-tts", name: "Vevo TTS", description: "Versatile Voice Conversion" },
  { id: "metis", name: "Metis", description: "Multi-Task Audio Model" },
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
    </div>
  );
}
