"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
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
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { FileUpload } from "@/components/ui/file-upload";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import {
  Play,
  Download,
  Wand2,
  Loader2,
  Mic,
  FileAudio,
  ArrowLeft,
  Music2,
} from "lucide-react";
import Link from "next/link";

interface VevoTTSParams {
  temperature: number;
  topK: number;
  topP: number;
  maxLength: number;
  styleStrength: number;
  voiceId?: string;
  timbreId?: string;
}

const PARAMETER_HELP = {
  temperature:
    "Controls randomness in generation. Higher values produce more varied output.",
  topK: "Limits token selection to top K most likely tokens.",
  topP: "Nucleus sampling threshold.",
  maxLength: "Maximum length of generated audio in tokens.",
  styleStrength: "Strength of style transfer when using reference audio (0.0 - 1.0).",
};

const VOICE_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "female-1", label: "Female 1" },
  { value: "female-2", label: "Female 2" },
  { value: "male-1", label: "Male 1" },
  { value: "male-2", label: "Male 2" },
  { value: "singing", label: "Singing Voice" },
];

const TIMBRE_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "warm", label: "Warm" },
  { value: "bright", label: "Bright" },
  { value: "dark", label: "Dark" },
  { value: "breathy", label: "Breathy" },
  { value: "nasal", label: "Nasal" },
];

export default function VevoTTSPage() {
  const { toast } = useToast();
  const [text, setText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [useVoiceClone, setUseVoiceClone] = useState(false);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);

  const [params, setParams] = useState<VevoTTSParams>({
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
    maxLength: 2048,
    styleStrength: 0.5,
    voiceId: "default",
    timbreId: "default",
  });

  const handleGenerate = async () => {
    if (!text.trim()) {
      toast({
        title: "Error",
        description: "Please enter text to synthesize",
        variant: "destructive",
      });
      return;
    }

    setIsGenerating(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("text", text);
      formData.append("temperature", params.temperature.toString());
      formData.append("top_k", params.topK.toString());
      formData.append("top_p", params.topP.toString());
      formData.append("max_length", params.maxLength.toString());
      formData.append("style_strength", params.styleStrength.toString());

      if (useVoiceClone && referenceAudio) {
        formData.append("reference_audio", referenceAudio);
      } else {
        if (params.voiceId && params.voiceId !== "default") {
          formData.append("voice_id", params.voiceId);
        }
        if (params.timbreId && params.timbreId !== "default") {
          formData.append("timbre_id", params.timbreId);
        }
      }

      const progressInterval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 500);

      const response = await fetch("/api/tts/vevo", {
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
        description: "Audio generated successfully with Vevo TTS",
      });
    } catch (error) {
      toast({
        title: "Error",
        description:
          error instanceof Error ? error.message : "Failed to generate audio",
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
    a.download = `vevo-tts-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link
          href="/tts"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to TTS
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">Vevo TTS</h1>
        <p className="text-muted-foreground">
          Versatile Voice Conversion with style and timbre control
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mic className="h-5 w-5" />
                Text Input
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="Enter text to convert to speech..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={6}
                className="resize-none"
              />
              <div className="flex justify-between text-sm text-muted-foreground">
                <span>{text.length} characters</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileAudio className="h-5 w-5" />
                Voice Selection
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="voice-clone" className="cursor-pointer">
                  Use Voice Clone
                </Label>
                <Switch
                  id="voice-clone"
                  checked={useVoiceClone}
                  onCheckedChange={setUseVoiceClone}
                />
              </div>

              {useVoiceClone ? (
                <FileUpload
                  accept="audio/*"
                  onFileSelect={setReferenceAudio}
                  selectedFile={referenceAudio}
                  label="Upload reference audio (3-10 seconds)"
                />
              ) : (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Voice</Label>
                    <Select
                      value={params.voiceId}
                      onValueChange={(value) =>
                        setParams((p) => ({ ...p, voiceId: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select voice" />
                      </SelectTrigger>
                      <SelectContent>
                        {VOICE_OPTIONS.map((voice) => (
                          <SelectItem key={voice.value} value={voice.value}>
                            {voice.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <Music2 className="h-4 w-4" />
                      Timbre
                    </Label>
                    <Select
                      value={params.timbreId}
                      onValueChange={(value) =>
                        setParams((p) => ({ ...p, timbreId: value }))
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select timbre" />
                      </SelectTrigger>
                      <SelectContent>
                        {TIMBRE_OPTIONS.map((timbre) => (
                          <SelectItem key={timbre.value} value={timbre.value}>
                            {timbre.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="h-5 w-5" />
                Generation Parameters
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    Temperature
                    <HelpTooltip content={PARAMETER_HELP.temperature} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.temperature.toFixed(1)}
                  </span>
                </div>
                <Slider
                  value={[params.temperature]}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, temperature: v }))
                  }
                  min={0.1}
                  max={1.5}
                  step={0.1}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    Top K
                    <HelpTooltip content={PARAMETER_HELP.topK} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.topK}
                  </span>
                </div>
                <Slider
                  value={[params.topK]}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, topK: v }))
                  }
                  min={1}
                  max={100}
                  step={1}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="flex items-center gap-2">
                    Top P
                    <HelpTooltip content={PARAMETER_HELP.topP} />
                  </Label>
                  <span className="text-sm text-muted-foreground w-12 text-right">
                    {params.topP.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[params.topP]}
                  onValueChange={([v]) =>
                    setParams((p) => ({ ...p, topP: v }))
                  }
                  min={0.1}
                  max={1.0}
                  step={0.05}
                />
              </div>

              {useVoiceClone && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="flex items-center gap-2">
                      Style Strength
                      <HelpTooltip content={PARAMETER_HELP.styleStrength} />
                    </Label>
                    <span className="text-sm text-muted-foreground w-12 text-right">
                      {params.styleStrength.toFixed(2)}
                    </span>
                  </div>
                  <Slider
                    value={[params.styleStrength]}
                    onValueChange={([v]) =>
                      setParams((p) => ({ ...p, styleStrength: v }))
                    }
                    min={0.0}
                    max={1.0}
                    step={0.05}
                  />
                </div>
              )}
            </CardContent>
          </Card>

          <Button
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
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
                Generate with Vevo
              </>
            )}
          </Button>

          {isGenerating && (
            <div className="space-y-2">
              <Progress value={progress} />
              <p className="text-sm text-center text-muted-foreground">
                {progress < 30 && "Preparing..."}
                {progress >= 30 && progress < 70 && "Generating audio..."}
                {progress >= 70 && progress < 100 && "Finalizing..."}
                {progress === 100 && "Complete!"}
              </p>
            </div>
          )}

          {generatedAudio && (
            <Card>
              <CardHeader>
                <CardTitle>Generated Audio</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <AudioPlayer src={generatedAudio} />
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
