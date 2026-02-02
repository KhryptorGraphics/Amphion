"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, BarChart3, Loader2, Check, X } from "lucide-react";
import Link from "next/link";

interface ModelResult {
  model: string;
  audioUrl: string | null;
  isGenerating: boolean;
  progress: number;
  error?: string;
}

const TTS_MODELS = [
  { id: "maskgct", name: "MaskGCT", description: "Zero-shot TTS with masking" },
  { id: "dualcodec-valle", name: "DualCodec-VALLE", description: "Fast neural codec TTS" },
  { id: "vevo-tts", name: "Vevo TTS", description: "Style-controllable TTS" },
  { id: "valle", name: "VALLE", description: "Neural Codec LM" },
  { id: "vits", name: "VITS", description: "Variational inference TTS" },
  { id: "fastspeech2", name: "FastSpeech2", description: "Fast non-autoregressive" },
  { id: "naturalspeech2", name: "NaturalSpeech2", description: "Latent diffusion TTS" },
  { id: "jets", name: "Jets", description: "Joint TTS + vocoder" },
  { id: "metis", name: "Metis", description: "Foundation model TTS" },
  { id: "debatts", name: "DebaTTS", description: "Mandarin Chinese TTS" },
];

export default function TTSComparePage() {
  const { toast } = useToast();
  const [text, setText] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>(["maskgct", "vits"]);
  const [results, setResults] = useState<ModelResult[]>([]);
  const [isComparing, setIsComparing] = useState(false);

  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId) ? prev.filter((m) => m !== modelId) : [...prev, modelId]
    );
  };

  const handleCompare = async () => {
    if (!text.trim()) {
      toast({ title: "Error", description: "Please enter text", variant: "destructive" });
      return;
    }
    if (selectedModels.length < 2) {
      toast({ title: "Error", description: "Please select at least 2 models to compare", variant: "destructive" });
      return;
    }

    setIsComparing(true);
    const initialResults: ModelResult[] = selectedModels.map((model) => ({
      model,
      audioUrl: null,
      isGenerating: true,
      progress: 0,
    }));
    setResults(initialResults);

    // Generate with all models in parallel
    await Promise.all(
      selectedModels.map(async (modelId, index) => {
        try {
          const formData = new FormData();
          formData.append("text", text);

          // Simulate progress updates
          const progressInterval = setInterval(() => {
            setResults((prev) =>
              prev.map((r, i) =>
                i === index ? { ...r, progress: Math.min(r.progress + Math.random() * 15, 90) } : r
              )
            );
          }, 500);

          const response = await fetch(`/api/tts/${modelId}`, {
            method: "POST",
            body: formData,
          });

          clearInterval(progressInterval);

          if (!response.ok) throw new Error("Generation failed");

          const blob = await response.blob();
          const audioUrl = URL.createObjectURL(blob);

          setResults((prev) =>
            prev.map((r, i) =>
              i === index ? { ...r, audioUrl, isGenerating: false, progress: 100 } : r
            )
          );
        } catch (error) {
          setResults((prev) =>
            prev.map((r, i) =>
              i === index
                ? { ...r, isGenerating: false, error: String(error), progress: 0 }
                : r
            )
          );
        }
      })
    );

    setIsComparing(false);
    toast({ title: "Complete", description: "Comparison finished" });
  };

  const clearAll = () => {
    setResults([]);
    setText("");
    setSelectedModels([]);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link href="/tts" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />Back to TTS
        </Link>
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold">Model Comparison</h1>
          <BarChart3 className="h-8 w-8 text-primary" />
        </div>
        <p className="text-muted-foreground">Compare TTS models side-by-side with the same input text</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Text Input</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={4}
                placeholder="Enter text to synthesize..."
                disabled={isComparing}
              />
              <p className="text-sm text-muted-foreground">{text.length} characters</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Select Models ({selectedModels.length} selected)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {TTS_MODELS.map((model) => (
                  <div key={model.id} className="flex items-start space-x-3">
                    <Checkbox
                      id={model.id}
                      checked={selectedModels.includes(model.id)}
                      onCheckedChange={() => toggleModel(model.id)}
                      disabled={isComparing}
                    />
                    <div className="space-y-1">
                      <Label htmlFor={model.id} className="font-medium cursor-pointer">
                        {model.name}
                      </Label>
                      <p className="text-xs text-muted-foreground">{model.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <div className="flex gap-2">
            <Button
              onClick={handleCompare}
              disabled={isComparing || !text.trim() || selectedModels.length < 2}
              className="flex-1 h-12"
            >
              {isComparing ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Comparing...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5" />
                  Compare Models
                </>
              )}
            </Button>
            <Button variant="outline" onClick={clearAll} disabled={isComparing}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="lg:col-span-2">
          <Tabs defaultValue="grid" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="grid">Grid View</TabsTrigger>
              <TabsTrigger value="list">List View</TabsTrigger>
            </TabsList>

            <TabsContent value="grid" className="mt-0">
              <div className="grid gap-4 md:grid-cols-2">
                {results.map((result, index) => {
                  const modelInfo = TTS_MODELS.find((m) => m.id === result.model);
                  return (
                    <Card key={result.model} className={result.error ? "border-red-500" : ""}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-lg">{modelInfo?.name}</CardTitle>
                          {result.isGenerating ? (
                            <Loader2 className="h-5 w-5 animate-spin text-primary" />
                          ) : result.audioUrl ? (
                            <Check className="h-5 w-5 text-green-500" />
                          ) : result.error ? (
                            <X className="h-5 w-5 text-red-500" />
                          ) : null}
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        {result.isGenerating && (
                          <Progress value={result.progress} className="h-2" />
                        )}
                        {result.error ? (
                          <p className="text-sm text-red-500">{result.error}</p>
                        ) : result.audioUrl ? (
                          <>
                            <AudioPlayer src={result.audioUrl} />
                            <Button
                              variant="outline"
                              size="sm"
                              className="w-full"
                              onClick={() => {
                                const a = document.createElement("a");
                                a.href = result.audioUrl!;
                                a.download = `${result.model}-compare-${Date.now()}.wav`;
                                a.click();
                              }}
                            >
                              <Download className="mr-2 h-4 w-4" />
                              Download
                            </Button>
                          </>
                        ) : (
                          <p className="text-sm text-muted-foreground">Waiting to generate...</p>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </TabsContent>

            <TabsContent value="list" className="mt-0">
              <div className="space-y-4">
                {results.map((result) => {
                  const modelInfo = TTS_MODELS.find((m) => m.id === result.model);
                  return (
                    <Card key={result.model} className={result.error ? "border-red-500" : ""}>
                      <CardContent className="p-4">
                        <div className="flex items-center gap-4">
                          <div className="w-32 shrink-0">
                            <p className="font-medium">{modelInfo?.name}</p>
                            <p className="text-xs text-muted-foreground">{modelInfo?.description}</p>
                          </div>
                          <div className="flex-1">
                            {result.isGenerating ? (
                              <Progress value={result.progress} className="h-2" />
                            ) : result.error ? (
                              <p className="text-sm text-red-500">{result.error}</p>
                            ) : result.audioUrl ? (
                              <AudioPlayer src={result.audioUrl} compact />
                            ) : (
                              <p className="text-sm text-muted-foreground">Waiting...</p>
                            )}
                          </div>
                          <div className="shrink-0">
                            {result.audioUrl && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => {
                                  const a = document.createElement("a");
                                  a.href = result.audioUrl!;
                                  a.download = `${result.model}-compare-${Date.now()}.wav`;
                                  a.click();
                                }}
                              >
                                <Download className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </TabsContent>
          </Tabs>

          {results.length === 0 && !isComparing && (
            <Card className="border-dashed">
              <CardContent className="py-12 text-center">
                <BarChart3 className="mx-auto h-12 w-12 text-muted-foreground/50 mb-4" />
                <p className="text-muted-foreground">
                  Enter text and select models to start comparison
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
