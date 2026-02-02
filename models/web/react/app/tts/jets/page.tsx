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
import { Play, Download, Wand2, Loader2, Mic, FileAudio, ArrowLeft } from "lucide-react";
import Link from "next/link";

interface JetsParams {
  alpha: number;
  beta: number;
  noiseScale: number;
  lengthScale: number;
  voiceId?: string;
}

const PARAMETER_HELP = {
  alpha: "Flow matching guidance scale for ODE solver.",
  beta: "Variance of the stochastic sampler.",
  noiseScale: "Stochastic duration predictor noise scale.",
  lengthScale: "Speed control - < 1.0 is faster, > 1.0 is slower.",
};

const VOICE_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "female-1", label: "Female 1" },
  { value: "female-2", label: "Female 2" },
  { value: "male-1", label: "Male 1" },
  { value: "male-2", label: "Male 2" },
];

export default function JetsPage() {
  const { toast } = useToast();
  const [text, setText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [useVoiceClone, setUseVoiceClone] = useState(false);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);

  const [params, setParams] = useState<JetsParams>({
    alpha: 1.0,
    beta: 0.0,
    noiseScale: 0.667,
    lengthScale: 1.0,
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
      formData.append("alpha", params.alpha.toString());
      formData.append("beta", params.beta.toString());
      formData.append("noise_scale", params.noiseScale.toString());
      formData.append("length_scale", params.lengthScale.toString());
      if (useVoiceClone && referenceAudio) {
        formData.append("reference_audio", referenceAudio);
      } else if (params.voiceId && params.voiceId !== "default") {
        formData.append("voice_id", params.voiceId);
      }
      const progressInterval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? prev : prev + 5));
      }, 1000);
      const response = await fetch("/api/tts/jets", { method: "POST", body: formData });
      clearInterval(progressInterval);
      if (!response.ok) throw new Error("Generation failed");
      const blob = await response.blob();
      setGeneratedAudio(URL.createObjectURL(blob));
      setProgress(100);
      toast({ title: "Success", description: "Audio generated with Jets" });
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
        <h1 className="text-3xl font-bold">Jets</h1>
        <p className="text-muted-foreground">Joint training of FastSpeech2 and HiFi-GAN for fast, high-quality TTS</p>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Mic className="h-5 w-5" />Text Input</CardTitle></CardHeader>
            <CardContent>
              <Textarea value={text} onChange={(e) => setText(e.target.value)} rows={8} placeholder="Enter text to synthesize..." />
              <p className="text-sm text-muted-foreground mt-2">{text.length} characters</p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><FileAudio className="h-5 w-5" />Voice</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Use Voice Clone</Label>
                <Switch checked={useVoiceClone} onCheckedChange={setUseVoiceClone} />
              </div>
              {useVoiceClone ? (
                <FileUpload accept="audio/*" onFileSelect={setReferenceAudio} selectedFile={referenceAudio} />
              ) : (
                <Select value={params.voiceId} onValueChange={(v) => setParams((p) => ({ ...p, voiceId: v }))}>
                  <SelectTrigger><SelectValue placeholder="Select voice" /></SelectTrigger>
                  <SelectContent>{VOICE_OPTIONS.map((v) => <SelectItem key={v.value} value={v.value}>{v.label}</SelectItem>)}</SelectContent>
                </Select>
              )}
            </CardContent>
          </Card>
        </div>
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Wand2 className="h-5 w-5" />Parameters</CardTitle></CardHeader>
            <CardContent className="space-y-6">
              {[
                { key: "alpha", label: "Alpha (Flow Scale)", min: 0.5, max: 2.0, step: 0.05 },
                { key: "beta", label: "Beta (Stochastic Variance)", min: 0, max: 1.0, step: 0.05 },
                { key: "noiseScale", label: "Noise Scale", min: 0, max: 1, step: 0.01 },
                { key: "lengthScale", label: "Length Scale (Speed)", min: 0.5, max: 2, step: 0.05 },
              ].map(({ key, label, min, max, step }) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="flex items-center gap-2">{label} <HelpTooltip content={PARAMETER_HELP[key as keyof typeof PARAMETER_HELP]} /></Label>
                    <span className="text-sm text-muted-foreground">{(params[key as keyof JetsParams] as number).toFixed(2)}</span>
                  </div>
                  <Slider
                    value={[params[key as keyof JetsParams] as number]}
                    onValueChange={([v]) => setParams((p) => ({ ...p, [key]: v }))}
                    min={min} max={max} step={step}
                  />
                </div>
              ))}
            </CardContent>
          </Card>
          <Button onClick={handleGenerate} disabled={isGenerating || !text.trim()} className="w-full h-12">
            {isGenerating ? <><Loader2 className="mr-2 h-5 w-5 animate-spin" />Generating...</> : <><Play className="mr-2 h-5 w-5" />Generate with Jets</>}
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
                  a.download = `jets-${Date.now()}.wav`;
                  a.click();
                }}>
                  <Download className="mr-2 h-4 w-4" />Download WAV
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
