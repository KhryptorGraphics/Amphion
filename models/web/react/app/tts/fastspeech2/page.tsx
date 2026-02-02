"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Play, Download, Wand2, Loader2, Mic, ArrowLeft } from "lucide-react";
import Link from "next/link";

interface FastSpeech2Params {
  pitchControl: number;
  energyControl: number;
  durationControl: number;
  voiceId?: string;
}

const PARAMETER_HELP = {
  pitchControl: "Pitch variation multiplier (1.0 = normal, >1.0 = higher pitch, <1.0 = lower pitch).",
  energyControl: "Energy/loudness variation multiplier.",
  durationControl: "Speaking rate control (<1.0 = faster, >1.0 = slower).",
};

const VOICE_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "female-1", label: "Female 1" },
  { value: "female-2", label: "Female 2" },
  { value: "male-1", label: "Male 1" },
  { value: "male-2", label: "Male 2" },
];

export default function FastSpeech2Page() {
  const { toast } = useToast();
  const [text, setText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [params, setParams] = useState<FastSpeech2Params>({
    pitchControl: 1.0,
    energyControl: 1.0,
    durationControl: 1.0,
    voiceId: "default",
  });

  const handleGenerate = async () => {
    if (!text.trim()) {
      toast({ title: "Error", description: "Please enter text", variant: "destructive" });
      return;
    }
    setIsGenerating(true);
    setProgress(0);
    try {
      const formData = new FormData();
      formData.append("text", text);
      formData.append("pitch_control", params.pitchControl.toString());
      formData.append("energy_control", params.energyControl.toString());
      formData.append("duration_control", params.durationControl.toString());
      if (params.voiceId && params.voiceId !== "default") {
        formData.append("voice_id", params.voiceId);
      }
      const progressInterval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? prev : prev + Math.random() * 15));
      }, 500);
      const response = await fetch("/api/tts/fastspeech2", { method: "POST", body: formData });
      clearInterval(progressInterval);
      if (!response.ok) throw new Error("Generation failed");
      const blob = await response.blob();
      setGeneratedAudio(URL.createObjectURL(blob));
      setProgress(100);
      toast({ title: "Success", description: "Audio generated with FastSpeech2" });
    } catch (error) {
      toast({ title: "Error", description: String(error), variant: "destructive" });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link href="/tts" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />Back to TTS
        </Link>
        <h1 className="text-3xl font-bold">FastSpeech2</h1>
        <p className="text-muted-foreground">Fast and high-quality end-to-end neural TTS with prosody control</p>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Mic className="h-5 w-5" />Text Input</CardTitle></CardHeader>
            <CardContent>
              <Textarea value={text} onChange={(e) => setText(e.target.value)} rows={8} placeholder="Enter text..." />
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle>Voice</CardTitle></CardHeader>
            <CardContent>
              <Select value={params.voiceId} onValueChange={(v) => setParams((p) => ({ ...p, voiceId: v }))}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>{VOICE_OPTIONS.map((v) => <SelectItem key={v.value} value={v.value}>{v.label}</SelectItem>)}</SelectContent>
              </Select>
            </CardContent>
          </Card>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Wand2 className="h-5 w-5" />Prosody Control</CardTitle></CardHeader>
            <CardContent className="space-y-6">
              {[
                { key: "pitchControl", label: "Pitch Control", min: 0.5, max: 2.0 },
                { key: "energyControl", label: "Energy Control", min: 0.5, max: 2.0 },
                { key: "durationControl", label: "Duration Control (Speed)", min: 0.5, max: 2.0 },
              ].map(({ key, label, min, max }) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="flex items-center gap-2">{label} <HelpTooltip content={PARAMETER_HELP[key as keyof typeof PARAMETER_HELP]} /></Label>
                    <span className="text-sm text-muted-foreground">{params[key as keyof FastSpeech2Params]?.toFixed(2)}x</span>
                  </div>
                  <Slider
                    value={[params[key as keyof FastSpeech2Params] as number]}
                    onValueChange={([v]) => setParams((p) => ({ ...p, [key]: v }))}
                    min={min} max={max} step={0.05}
                  />
                </div>
              ))}
            </CardContent>
          </Card>
          <Button onClick={handleGenerate} disabled={isGenerating || !text.trim()} className="w-full h-12">
            {isGenerating ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" />Generating...</> : <><Play className="mr-2 h-5 w-5" />Generate</>}
          </Button>
          {isGenerating && <Progress value={progress} />}
          {generatedAudio && (
            <Card>
              <CardHeader><CardTitle>Generated Audio</CardTitle></CardHeader>
              <CardContent>
                <AudioPlayer src={generatedAudio} />
                <Button variant="outline" className="w-full mt-4" onClick={() => {
                  const a = document.createElement("a");
                  a.href = generatedAudio;
                  a.download = `fastspeech2-${Date.now()}.wav`;
                  a.click();
                }}>
                  <Download className="mr-2 h-4 w-4" />Download
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
