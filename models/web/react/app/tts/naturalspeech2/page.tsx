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

interface NS2Params {
  temperature: number;
  topK: number;
  topP: number;
  diffusionSteps: number;
  voiceId?: string;
}

const PARAMETER_HELP = {
  temperature: "Controls randomness.",
  topK: "Limits token selection.",
  topP: "Nucleus sampling threshold.",
  diffusionSteps: "Number of diffusion steps (more = higher quality but slower).",
};

const VOICE_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "female-1", label: "Female 1" },
  { value: "male-1", label: "Male 1" },
];

export default function NaturalSpeech2Page() {
  const { toast } = useToast();
  const [text, setText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [useVoiceClone, setUseVoiceClone] = useState(false);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);

  const [params, setParams] = useState<NS2Params>({
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
    diffusionSteps: 100,
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
      formData.append("temperature", params.temperature.toString());
      formData.append("top_k", params.topK.toString());
      formData.append("top_p", params.topP.toString());
      formData.append("diffusion_steps", params.diffusionSteps.toString());
      if (useVoiceClone && referenceAudio) {
        formData.append("reference_audio", referenceAudio);
      } else if (params.voiceId && params.voiceId !== "default") {
        formData.append("voice_id", params.voiceId);
      }
      const progressInterval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? prev : prev + 5));
      }, 1000);
      const response = await fetch("/api/tts/naturalspeech2", { method: "POST", body: formData });
      clearInterval(progressInterval);
      if (!response.ok) throw new Error("Generation failed");
      const blob = await response.blob();
      setGeneratedAudio(URL.createObjectURL(blob));
      setProgress(100);
      toast({ title: "Success", description: "Audio generated with NaturalSpeech2" });
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
        <h1 className="text-3xl font-bold">NaturalSpeech2</h1>
        <p className="text-muted-foreground">Zero-shot speech synthesis with natural-sounding voices</p>
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <Card>
            <CardHeader><CardTitle className="flex items-center gap-2"><Mic className="h-5 w-5" />Text Input</CardTitle></CardHeader>
            <CardContent><Textarea value={text} onChange={(e) => setText(e.target.value)} rows={6} placeholder="Enter text..." /></CardContent>
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
                  <SelectTrigger><SelectValue /></SelectTrigger>
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
                { key: "temperature", label: "Temperature", min: 0.1, max: 1.5, step: 0.1 },
                { key: "topK", label: "Top K", min: 1, max: 100, step: 1 },
                { key: "topP", label: "Top P", min: 0.1, max: 1, step: 0.05 },
                { key: "diffusionSteps", label: "Diffusion Steps", min: 10, max: 1000, step: 10 },
              ].map(({ key, label, min, max, step }) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="flex items-center gap-2">{label} <HelpTooltip content={PARAMETER_HELP[key as keyof typeof PARAMETER_HELP]} /></Label>
                    <span>{params[key as keyof NS2Params]?.toString()}</span>
                  </div>
                  <Slider value={[params[key as keyof NS2Params] as number]} onValueChange={([v]) => setParams((p) => ({ ...p, [key]: v }))} min={min} max={max} step={step} />
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
                <Button variant="outline" className="w-full mt-4" onClick={() => { const a = document.createElement("a"); a.href = generatedAudio; a.download = `ns2-${Date.now()}.wav`; a.click(); }}>
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
