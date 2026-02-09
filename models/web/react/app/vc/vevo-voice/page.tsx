"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { FileUpload } from "@/components/ui/file-upload";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, Loader2, FileAudio, Volume2, Wand2, FlaskConical } from "lucide-react";
import Link from "next/link";

const PARAMETER_HELP = {
  flowMatchingSteps: "Number of flow matching steps (higher = better quality but slower)",
  separateTimbre: "Use a separate audio file for timbre reference instead of the same reference for both style and timbre",
};

export default function VevoVoicePage() {
  const { toast } = useToast();
  const [sourceAudio, setSourceAudio] = useState<File | null>(null);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);
  const [timbreAudio, setTimbreAudio] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [convertedAudio, setConvertedAudio] = useState<string | null>(null);

  const [params, setParams] = useState({
    flowMatchingSteps: 32,
    separateTimbre: false,
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

    if (params.separateTimbre && !timbreAudio) {
      toast({
        title: "Error",
        description: "Please upload timbre reference audio or disable separate timbre",
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
      formData.append("flow_matching_steps", params.flowMatchingSteps.toString());
      if (params.separateTimbre && timbreAudio) {
        formData.append("timbre_audio", timbreAudio);
      }

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 15;
        });
      }, 500);

      const response = await fetch("/api/vc/vevo-voice", {
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
        description: "Voice converted successfully with Vevo Voice",
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
    a.download = `vevo-voice-${Date.now()}.wav`;
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
        <h1 className="text-3xl font-bold tracking-tight">Vevo Voice</h1>
        <p className="text-muted-foreground">
          Full voice conversion - transfer voice characteristics from reference to source
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileAudio className="h-5 w-5" />
                Source Audio
                <HelpTooltip content="The audio you want to convert (the person speaking/singing)" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setSourceAudio}
                selectedFile={sourceAudio}
                label="Upload source audio (person to convert)"
                description="WAV or MP3 format recommended"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Volume2 className="h-5 w-5" />
                {params.separateTimbre ? "Style Reference Audio" : "Reference Audio"}
                <HelpTooltip content={params.separateTimbre
                  ? "The speaking style/prosody you want to apply"
                  : "The voice you want to convert TO (target voice characteristics)"
                } />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setReferenceAudio}
                selectedFile={referenceAudio}
                label={params.separateTimbre ? "Upload style reference audio" : "Upload reference audio (target voice)"}
                description="WAV or MP3 format recommended"
              />
            </CardContent>
          </Card>

          {params.separateTimbre && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Volume2 className="h-5 w-5" />
                  Timbre Reference Audio
                  <HelpTooltip content="Separate audio for voice timbre/color (different from the style reference)" />
                </CardTitle>
              </CardHeader>
              <CardContent>
                <FileUpload
                  accept="audio/*"
                  onFileSelect={setTimbreAudio}
                  selectedFile={timbreAudio}
                  label="Upload timbre reference audio"
                  description="WAV or MP3 format recommended"
                />
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="h-5 w-5" />
                Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-2 cursor-pointer">
                  Separate Timbre Reference
                  <HelpTooltip content={PARAMETER_HELP.separateTimbre} />
                </Label>
                <Switch
                  checked={params.separateTimbre}
                  onCheckedChange={(v) =>
                    setParams((p) => ({ ...p, separateTimbre: v }))
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
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FlaskConical className="h-5 w-5" />
                About Vevo Voice
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Vevo Voice performs full voice conversion, transferring both speaking style (prosody,
                accent, rhythm) and voice timbre (tone color) from the reference to the source audio.
                It uses an autoregressive model for semantic token prediction followed by flow matching
                for acoustic generation. Optionally, you can provide separate style and timbre references
                for more fine-grained control over the conversion.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Conversion</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={handleConvert}
                disabled={isConverting || !sourceAudio || !referenceAudio || (params.separateTimbre && !timbreAudio)}
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
                    Convert Voice
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
