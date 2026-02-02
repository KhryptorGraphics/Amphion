"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { FileUpload } from "@/components/ui/file-upload";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, Loader2, FileAudio, Volume2, Wand2 } from "lucide-react";
import Link from "next/link";

const PARAMETER_HELP = {
  inferenceSteps: "Number of diffusion steps (150-300). Higher = better quality but slower.",
  sigma: "Sigma parameter (0.95-1.5). Controls the noise level in the diffusion process.",
};

export default function NoroPage() {
  const { toast } = useToast();
  const [sourceAudio, setSourceAudio] = useState<File | null>(null);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [convertedAudio, setConvertedAudio] = useState<string | null>(null);

  const [params, setParams] = useState({
    inferenceSteps: 200,
    sigma: 1.2,
  });

  const handleConvert = async () => {
    if (!sourceAudio || !referenceAudio) {
      toast({
        title: "Error",
        description: "Please upload both source and reference audio",
        variant: "destructive",
      });
      return;
    }

    setIsConverting(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("source_audio", sourceAudio);
      formData.append("reference_audio", referenceAudio);
      formData.append("inference_steps", params.inferenceSteps.toString());
      formData.append("sigma", params.sigma.toString());

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 500);

      const response = await fetch("/api/vc/noro", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Conversion failed");
      }

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      setConvertedAudio(audioUrl);
      setProgress(100);

      toast({
        title: "Success",
        description: "Voice converted successfully with Noro",
      });
    } catch (error) {
      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to convert voice",
        variant: "destructive",
      });
    } finally {
      setIsConverting(false);
    }
  };

  const handleDownload = () => {
    if (!convertedAudio) return;
    const a = document.createElement("a");
    a.href = convertedAudio;
    a.download = `noro-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link
          href="/vc"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Voice Conversion
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">Noro</h1>
        <p className="text-muted-foreground">
          Noise-robust voice conversion using diffusion - handles challenging audio conditions
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileAudio className="h-5 w-5" />
                Source Audio
                <HelpTooltip content="The audio you want to convert (can be noisy or low quality)" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setSourceAudio}
                selectedFile={sourceAudio}
                label="Upload source audio (person to convert)"
                description="Works well with noisy/challenging audio"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Volume2 className="h-5 w-5" />
                Reference Audio
                <HelpTooltip content="The voice you want to convert TO (target voice characteristics)" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setReferenceAudio}
                selectedFile={referenceAudio}
                label="Upload reference audio (target voice)"
                description="WAV or MP3 format recommended"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="h-5 w-5" />
                Diffusion Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    Inference Steps
                    <HelpTooltip content={PARAMETER_HELP.inferenceSteps} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.inferenceSteps}
                  </span>
                </div>
                <Slider
                  value={[params.inferenceSteps]}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, inferenceSteps: v }))
                  }
                  min={150}
                  max={300}
                  step={10}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    Sigma
                    <HelpTooltip content={PARAMETER_HELP.sigma} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.sigma.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[params.sigma]}
                  onValueChange={([v]) => setParams((p) => ({ ...p, sigma: v }))}
                  min={0.95}
                  max={1.5}
                  step={0.05}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Conversion</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={handleConvert}
                disabled={isConverting || !sourceAudio || !referenceAudio}
                className="w-full h-12"
                size="lg"
              >
                {isConverting ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Converting...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Convert with Noro
                  </>
                )}
              </Button>

              {isConverting && (
                <div className="space-y-2">
                  <Progress value={progress} />
                  <p className="text-sm text-center text-muted-foreground">
                    {progress < 30 && "Analyzing audio..."}
                    {progress >= 30 && progress < 70 && "Running diffusion process..."}
                    {progress >= 70 && progress < 100 && "Finalizing..."}
                    {progress === 100 && "Complete!"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {convertedAudio && (
            <Card>
              <CardHeader>
                <CardTitle>Converted Audio</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <AudioPlayer src={convertedAudio} />
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={handleDownload}
                >
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
