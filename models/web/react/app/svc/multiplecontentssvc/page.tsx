"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileUpload } from "@/components/ui/file-upload";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, ArrowLeft, Loader2, Music, Volume2, FlaskConical, Scissors } from "lucide-react";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import Link from "next/link";

export default function MultipleContentsSVCPage() {
  const { toast } = useToast();
  const [contentAudio, setContentAudio] = useState<File | null>(null);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);
  const [isConverting, setIsConverting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [convertedAudio, setConvertedAudio] = useState<string | null>(null);
  const [fullSongMode, setFullSongMode] = useState(false);
  const [vocalsVolumeDb, setVocalsVolumeDb] = useState(0);

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
      formData.append("full_song_mode", fullSongMode.toString());
      formData.append("vocals_volume_db", vocalsVolumeDb.toString());

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 500);

      const response = await fetch("/api/svc/multiplecontentssvc", {
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
        description: fullSongMode
          ? "Full song converted successfully with MultipleContentsSVC"
          : "Singing voice converted successfully with MultipleContentsSVC",
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
    a.download = `multiplecontentssvc-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const getProgressMessage = () => {
    if (fullSongMode) {
      if (progress < 20) return "Separating vocals from instrumentals...";
      if (progress < 60) return "Converting singing voice...";
      if (progress < 90) return "Remixing with instrumentals...";
      if (progress < 100) return "Finalizing...";
      return "Complete!";
    }
    if (progress < 30) return "Analyzing audio...";
    if (progress < 70) return "Converting voice...";
    if (progress < 100) return "Finalizing...";
    return "Complete!";
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
        <div className="flex items-center gap-2 mb-1">
          <h1 className="text-3xl font-bold tracking-tight">MultipleContentsSVC</h1>
          <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-amber-500/10 text-amber-500 border border-amber-500/20">
            Experimental
          </span>
        </div>
        <p className="text-muted-foreground">
          Multi-content singing voice conversion that leverages multiple content features for enhanced quality
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Music className="h-5 w-5" />
                {fullSongMode ? "Content Audio (Full Song)" : "Content Audio (Melody)"}
                <HelpTooltip
                  content={
                    fullSongMode
                      ? "Upload a complete song with vocals and instrumentals - will be auto-separated"
                      : "The singing voice you want to convert - the melody/content is preserved"
                  }
                />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept="audio/*"
                onFileSelect={setContentAudio}
                selectedFile={contentAudio}
                label={
                  fullSongMode
                    ? "Upload complete song (vocals + instrumentals)"
                    : "Upload content audio (song to convert)"
                }
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
              <CardTitle className="flex items-center gap-2">
                <FlaskConical className="h-5 w-5" />
                About MultipleContentsSVC
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <p className="text-sm text-muted-foreground">
                MultipleContentsSVC extracts and combines multiple content representations (phonetic,
                melodic, rhythmic) from the source audio to achieve more complete content preservation
                during voice conversion. This multi-feature approach reduces artifacts and improves
                naturalness compared to single-content methods.
              </p>
              <div className="text-sm text-muted-foreground">
                <p className="font-medium text-foreground mb-1">How to use:</p>
                <ol className="space-y-1 list-decimal list-inside">
                  <li>Upload the <strong>song you want to convert</strong> as Content Audio &mdash; isolated vocals without background music work best.</li>
                  <li>Upload a <strong>sample of the target singer&apos;s voice</strong> as Reference Audio &mdash; 5-15 seconds is enough.</li>
                  <li>Click <strong>Convert Singing Voice</strong> and wait for the result.</li>
                </ol>
                <p className="mt-2 text-xs italic">Requires a pre-trained MultipleContentsSVC checkpoint. Contact the project maintainer if the model is not loaded.</p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Scissors className="h-5 w-5" />
                Full Song Mode
                <HelpTooltip content="Automatically separate vocals from instrumentals, convert the vocals, and remix with the original accompaniment" />
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label className="flex items-center gap-2 cursor-pointer">
                  Enable Full Song Mode
                </Label>
                <Switch
                  checked={fullSongMode}
                  onCheckedChange={setFullSongMode}
                />
              </div>
              <p className="text-xs text-muted-foreground">
                Upload a complete song with vocals and instrumentals. Demucs will automatically
                separate the vocals, convert them, and remix with the original accompaniment.
              </p>

              {fullSongMode && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="flex items-center gap-2">
                      Vocals Volume
                      <HelpTooltip content="Adjust the volume of the converted vocals relative to the accompaniment" />
                    </Label>
                    <span className="text-sm text-muted-foreground w-16 text-right">
                      {vocalsVolumeDb > 0 ? "+" : ""}{vocalsVolumeDb} dB
                    </span>
                  </div>
                  <Slider
                    value={[vocalsVolumeDb]}
                    onValueChange={([v]) => setVocalsVolumeDb(v)}
                    min={-6}
                    max={6}
                    step={0.5}
                  />
                </div>
              )}
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
                    {getProgressMessage()}
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
