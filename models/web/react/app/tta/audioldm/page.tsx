"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, Loader2, Wand2, Clock, SlidersHorizontal } from "lucide-react";
import Link from "next/link";

const PARAMETER_HELP = {
  text: "Describe the sound you want to generate - be specific about the type of audio, environment, and qualities",
  numInferenceSteps: "Number of diffusion steps (higher = better quality but slower)",
  audioLength: "Duration of generated audio in seconds (1-30 seconds)",
};

export default function AudioLDMPage() {
  const { toast } = useToast();
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);

  const [params, setParams] = useState({
    text: "A dog barking in a park with birds chirping in the background",
    numInferenceSteps: 50,
    audioLength: 5.0,
  });

  const handleGenerate = async () => {
    if (!params.text.trim()) {
      toast({
        title: "Error",
        description: "Please enter a text description",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("text", params.text);
      formData.append("num_inference_steps", params.numInferenceSteps.toString());
      formData.append("audio_length_in_s", params.audioLength.toString());

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 8;
        });
      }, 1000);

      const response = await fetch("/api/tta/audioldm", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Generation failed");
      }

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      setGeneratedAudio(audioUrl);
      setProgress(100);

      toast({
        title: "Success",
        description: "Audio generated successfully with AudioLDM",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to generate audio",
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!generatedAudio) return;
    const a = document.createElement("a");
    a.href = generatedAudio;
    a.download = `audioldm-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link
          href="/tta"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Text-to-Audio
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">AudioLDM</h1>
        <p className="text-muted-foreground">
          Text-to-audio generation using latent diffusion - create sound effects and music from text
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="h-5 w-5" />
                Text Prompt
                <HelpTooltip content={PARAMETER_HELP.text} />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                value={params.text}
                onChange={(e) => setParams((p) => ({ ...p, text: e.target.value }))}
                placeholder="Describe the sound you want to generate..."
                className="min-h-[120px]"
              />
              <p className="text-xs text-muted-foreground mt-2">
                Examples: "A dog barking in a park", "Thunderstorm with rain", "Piano playing a gentle melody"
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <SlidersHorizontal className="h-5 w-5" />
                Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    Inference Steps
                    <HelpTooltip content={PARAMETER_HELP.numInferenceSteps} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.numInferenceSteps}
                  </span>
                </div>
                <Slider
                  value={[params.numInferenceSteps]}
                  onValueChange={([v]) => setParams((p) => ({ ...p, numInferenceSteps: v }))}
                  min={10}
                  max={200}
                  step={10}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    <Clock className="h-4 w-4" />
                    Audio Length (seconds)
                    <HelpTooltip content={PARAMETER_HELP.audioLength} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.audioLength}s
                  </span>
                </div>
                <Slider
                  value={[params.audioLength]}
                  onValueChange={([v]) => setParams((p) => ({ ...p, audioLength: v }))}
                  min={1}
                  max={30}
                  step={1}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Generation</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={handleGenerate}
                disabled={isGenerating || !params.text.trim()}
                className="w-full h-12"
                size="lg"
              >
                {isGenerating ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Generate Audio
                  </>
                )}
              </Button>

              {isGenerating && (
                <div className="space-y-2">
                  <Progress value={progress} />
                  <p className="text-sm text-center text-muted-foreground">
                    {progress < 30 && "Initializing diffusion model..."}
                    {progress >= 30 && progress < 70 && "Generating audio through diffusion..."}
                    {progress >= 70 && progress < 100 && "Finalizing audio..."}
                    {progress === 100 && "Complete!"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {generatedAudio && (
            <Card>
              <CardHeader>
                <CardTitle>Generated Audio</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <AudioPlayer src={generatedAudio} />
                <Button variant="outline" className="w-full" onClick={handleDownload}>
                  <Download className="mr-2 h-4 w-4" />
                  Download WAV
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
