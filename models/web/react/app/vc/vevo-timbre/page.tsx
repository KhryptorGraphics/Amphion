"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { FileUpload } from "@/components/ui/file-upload";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, Loader2, FileAudio, Volume2 } from "lucide-react";
import Link from "next/link";

export default function VevoTimbrePage() {
  const { toast } = useToast();
  const [sourceAudio, setSourceAudio] = useState<File | null>(null);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [convertedAudio, setConvertedAudio] = useState<string | null>(null);

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

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 15;
        });
      }, 500);

      const response = await fetch("/api/vc/vevo-timbre", {
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
        description: "Timbre converted successfully with Vevo Timbre",
      });
    } catch (error) {
      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to convert timbre",
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
    a.download = `vevo-timbre-${Date.now()}.wav`;
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
        <h1 className="text-3xl font-bold tracking-tight">Vevo Timbre</h1>
        <p className="text-muted-foreground">
          Timbre-only conversion - preserve speaking style, change only voice characteristics
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileAudio className="h-5 w-5" />
                Source Audio
                <HelpTooltip content="The audio you want to convert - preserves the speaking style/content" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setSourceAudio}
                selectedFile={sourceAudio}
                label="Upload source audio"
                description="WAV or MP3 format recommended"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Volume2 className="h-5 w-5" />
                Reference Audio
                <HelpTooltip content="The timbre/voice color you want to apply (only timbre is transferred)" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setReferenceAudio}
                selectedFile={referenceAudio}
                label="Upload reference audio (target timbre)"
                description="WAV or MP3 format recommended"
              />
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
                    Convert Timbre
                  </>
                )}
              </Button>

              {isConverting && (
                <div className="space-y-2">
                  <Progress value={progress} />
                  <p className="text-sm text-center text-muted-foreground">
                    {progress < 30 && "Analyzing timbre..."}
                    {progress >= 30 && progress < 70 && "Applying timbre conversion..."}
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
