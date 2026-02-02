"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FileUpload } from "@/components/ui/file-upload";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, Loader2, FileAudio, Volume2, Music } from "lucide-react";
import Link from "next/link";

const PARAMETER_HELP = {
  mode: "FM (Flow Matching) for timbre-only control, AR (Autoregressive) for full control including prosody",
  useShiftedSrc: "Use pitch-shifted source for better prosody extraction",
  flowMatchingSteps: "Number of flow matching steps (higher = better quality but slower)",
};

export default function VevoSingPage() {
  const { toast } = useToast();
  const [contentAudio, setContentAudio] = useState<File | null>(null);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [convertedAudio, setConvertedAudio] = useState<string | null>(null);

  const [params, setParams] = useState({
    mode: "fm",
    useShiftedSrc: true,
    flowMatchingSteps: 32,
  });

  const handleConvert = async () => {
    if (!contentAudio || !referenceAudio) {
      toast({
        title: "Error",
        description: "Please upload both content and reference audio",
        variant: "destructive",
      });
      return;
    }

    setIsConverting(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("content_audio", contentAudio);
      formData.append("reference_audio", referenceAudio);
      formData.append("mode", params.mode);
      formData.append("use_shifted_src", params.useShiftedSrc.toString());
      formData.append("flow_matching_steps", params.flowMatchingSteps.toString());

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 500);

      const response = await fetch("/api/svc/vevosing", {
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
        description: "Singing voice converted successfully with VevoSing",
      });
    } catch (error) {
      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to convert singing voice",
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
    a.download = `vevosing-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link
          href="/svc"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Singing Voice Conversion
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">VevoSing</h1>
        <p className="text-muted-foreground">
          Singing voice conversion using flow matching - preserve melody, change singer
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Music className="h-5 w-5" />
                Content Audio (Melody)
                <HelpTooltip content="The singing voice you want to convert - the melody/content is preserved" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setContentAudio}
                selectedFile={contentAudio}
                label="Upload content audio (song to convert)"
                description="WAV or MP3 format recommended"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Volume2 className="h-5 w-5" />
                Reference Audio (Timbre)
                <HelpTooltip content="The voice timbre you want to apply to the singing" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setReferenceAudio}
                selectedFile={referenceAudio}
                label="Upload reference audio (target singer voice)"
                description="WAV or MP3 format recommended"
              />
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Parameters</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  Mode
                  <HelpTooltip content={PARAMETER_HELP.mode} />
                </Label>
                <Select
                  value={params.mode}
                  onValueChange={(v) => setParams((p) => ({ ...p, mode: v }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fm">
                      Flow Matching (Timbre only)
                    </SelectItem>
                    <SelectItem value="ar">
                      Autoregressive (Full control)
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-2 cursor-pointer">
                  Use Shifted Source
                  <HelpTooltip content={PARAMETER_HELP.useShiftedSrc} />
                </Label>
                <Switch
                  checked={params.useShiftedSrc}
                  onCheckedChange={(v) =>
                    setParams((p) => ({ ...p, useShiftedSrc: v }))
                  }
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    Flow Matching Steps
                    <HelpTooltip content={PARAMETER_HELP.flowMatchingSteps} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.flowMatchingSteps}
                  </span>
                </div>
                <Slider
                  value={[params.flowMatchingSteps]}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, flowMatchingSteps: v }))
                  }
                  min={10}
                  max={100}
                  step={2}
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Conversion</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={handleConvert}
                disabled={isConverting || !contentAudio || !referenceAudio}
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
                    Convert Singing Voice
                  </>
                )}
              </Button>

              {isConverting && (
                <div className="space-y-2">
                  <Progress value={progress} />
                  <p className="text-sm text-center text-muted-foreground">
                    {progress < 30 && "Analyzing audio..."}
                    {progress >= 30 && progress < 70 && "Converting voice..."}
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
