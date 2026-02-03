"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileUpload } from "@/components/ui/file-upload";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, Loader2, Upload, Waves } from "lucide-react";
import Link from "next/link";

export default function BigVGANPage() {
  const { toast } = useToast();
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [resultAudio, setResultAudio] = useState<string | null>(null);

  const handleProcess = async () => {
    if (!audioFile) {
      toast({
        title: "Error",
        description: "Please upload an audio file or spectrogram",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("audio", audioFile);

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 15;
        });
      }, 500);

      const response = await fetch("/api/vocoder/bigvgan", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Processing failed");
      }

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      setResultAudio(audioUrl);
      setProgress(100);

      toast({
        title: "Success",
        description: "Audio processed successfully with BigVGAN",
      });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process audio",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDownload = () => {
    if (!resultAudio) return;
    const a = document.createElement("a");
    a.href = resultAudio;
    a.download = `bigvgan-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link
          href="/vocoder"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Vocoders
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">BigVGAN</h1>
        <p className="text-muted-foreground">
          Large-scale GAN-based vocoder for high-fidelity audio synthesis from spectrograms
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Input Audio / Spectrogram
                <HelpTooltip content="Upload audio or mel-spectrogram to vocode into high-quality audio" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setAudioFile}
                selectedFile={audioFile}
                label="Upload audio or spectrogram"
                description="WAV format recommended"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Processing</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Button
                onClick={handleProcess}
                disabled={isProcessing || !audioFile}
                className="w-full h-12"
                size="lg"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Processing...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Vocode Audio
                  </>
                )}
              </Button>

              {isProcessing && (
                <div className="space-y-2">
                  <Progress value={progress} />
                  <p className="text-sm text-center text-muted-foreground">
                    {progress < 50 && "Analyzing input..."}
                    {progress >= 50 && progress < 100 && "Generating high-fidelity audio..."}
                    {progress === 100 && "Complete!"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Waves className="h-5 w-5" />
                Output Audio
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {resultAudio ? (
                <>
                  <AudioPlayer src={resultAudio} />
                  <Button variant="outline" className="w-full" onClick={handleDownload}>
                    <Download className="mr-2 h-4 w-4" />
                    Download WAV
                  </Button>
                </>
              ) : (
                <p className="text-muted-foreground text-center py-8">
                  Processed audio will appear here
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
