"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
import { Play, Download, Wand2, Loader2, Mic, FileAudio, ArrowLeft, Volume2, Scissors } from "lucide-react";
import Link from "next/link";

interface MetisTTSParams {
  task: "tts";
  temperature: number;
  topK: number;
  topP: number;
  maxLength: number;
  voiceId?: string;
}

interface MetisVCParams {
  task: "vc";
  temperature: number;
  topK: number;
  topP: number;
}

interface MetisTSEParams {
  task: "tse";
  temperature: number;
  topK: number;
  targetSpeaker: string;
}

interface MetisSEParams {
  task: "se";
  temperature: number;
  denoiseStrength: number;
}

type MetisParams = MetisTTSParams | MetisVCParams | MetisTSEParams | MetisSEParams;

const PARAMETER_HELP = {
  temperature: "Controls randomness in generation.",
  topK: "Limits token selection to top K most likely tokens.",
  topP: "Nucleus sampling threshold.",
  maxLength: "Maximum length of generated audio.",
  targetSpeaker: "Target speaker for target speaker extraction.",
  denoiseStrength: "Strength of speech enhancement (0.0-1.0).",
};

const VOICE_OPTIONS = [
  { value: "default", label: "Default" },
  { value: "female-1", label: "Female 1" },
  { value: "female-2", label: "Female 2" },
  { value: "male-1", label: "Male 1" },
  { value: "male-2", label: "Male 2" },
];

export default function MetisPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("tts");
  const [text, setText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [sourceAudio, setSourceAudio] = useState<File | null>(null);
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null);

  const [ttsParams, setTtsParams] = useState<MetisTTSParams>({
    task: "tts",
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
    maxLength: 2048,
    voiceId: "default",
  });

  const [vcParams, setVcParams] = useState<MetisVCParams>({
    task: "vc",
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
  });

  const [tseParams, setTseParams] = useState<MetisTSEParams>({
    task: "tse",
    temperature: 0.7,
    topK: 50,
    targetSpeaker: "speaker1",
  });

  const [seParams, setSeParams] = useState<MetisSEParams>({
    task: "se",
    temperature: 0.7,
    denoiseStrength: 0.5,
  });

  const handleGenerate = async () => {
    if (activeTab === "tts" && !text.trim()) {
      toast({ title: "Error", description: "Please enter text", variant: "destructive" });
      return;
    }
    if ((activeTab === "vc" || activeTab === "tse" || activeTab === "se") && !sourceAudio) {
      toast({ title: "Error", description: "Please upload source audio", variant: "destructive" });
      return;
    }
    if (activeTab === "vc" && !referenceAudio) {
      toast({ title: "Error", description: "Please upload reference audio", variant: "destructive" });
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    try {
      const formData = new FormData();
      formData.append("task", activeTab);

      if (activeTab === "tts") {
        formData.append("text", text);
        formData.append("temperature", ttsParams.temperature.toString());
        formData.append("top_k", ttsParams.topK.toString());
        formData.append("top_p", ttsParams.topP.toString());
        formData.append("max_length", ttsParams.maxLength.toString());
        if (ttsParams.voiceId && ttsParams.voiceId !== "default") {
          formData.append("voice_id", ttsParams.voiceId);
        }
      } else if (activeTab === "vc") {
        formData.append("source_audio", sourceAudio!);
        formData.append("reference_audio", referenceAudio!);
        formData.append("temperature", vcParams.temperature.toString());
        formData.append("top_k", vcParams.topK.toString());
        formData.append("top_p", vcParams.topP.toString());
      } else if (activeTab === "tse") {
        formData.append("source_audio", sourceAudio!);
        formData.append("temperature", tseParams.temperature.toString());
        formData.append("top_k", tseParams.topK.toString());
        formData.append("target_speaker", tseParams.targetSpeaker);
      } else if (activeTab === "se") {
        formData.append("source_audio", sourceAudio!);
        formData.append("temperature", seParams.temperature.toString());
        formData.append("denoise_strength", seParams.denoiseStrength.toString());
      }

      const progressInterval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? prev : prev + Math.random() * 10));
      }, 500);

      const response = await fetch("/api/tts/metis", { method: "POST", body: formData });
      clearInterval(progressInterval);

      if (!response.ok) throw new Error("Generation failed");
      const blob = await response.blob();
      setGeneratedAudio(URL.createObjectURL(blob));
      setProgress(100);
      toast({ title: "Success", description: `Audio generated with Metis (${activeTab.toUpperCase()})` });
    } catch (error) {
      toast({ title: "Error", description: String(error), variant: "destructive" });
    } finally {
      setIsGenerating(false);
    }
  };

  const renderTTSParams = () => (
    <div className="space-y-6">
      {[
        { key: "temperature", label: "Temperature", min: 0.1, max: 1.5, step: 0.1 },
        { key: "topK", label: "Top K", min: 1, max: 100, step: 1 },
        { key: "topP", label: "Top P", min: 0.1, max: 1, step: 0.05 },
        { key: "maxLength", label: "Max Length", min: 256, max: 4096, step: 256 },
      ].map(({ key, label, min, max, step }) => (
        <div key={key} className="space-y-2">
          <div className="flex justify-between">
            <Label className="flex items-center gap-2">{label} <HelpTooltip content={PARAMETER_HELP[key as keyof typeof PARAMETER_HELP]} /></Label>
            <span>{ttsParams[key as keyof MetisTTSParams]?.toString()}</span>
          </div>
          <Slider value={[ttsParams[key as keyof MetisTTSParams] as number]} onValueChange={([v]) => setTtsParams((p) => ({ ...p, [key]: v }))} min={min} max={max} step={step} />
        </div>
      ))}
      <div className="space-y-2">
        <Label>Voice</Label>
        <Select value={ttsParams.voiceId} onValueChange={(v) => setTtsParams((p) => ({ ...p, voiceId: v }))}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>{VOICE_OPTIONS.map((v) => <SelectItem key={v.value} value={v.value}>{v.label}</SelectItem>)}</SelectContent>
        </Select>
      </div>
    </div>
  );

  const renderVCParams = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <Label className="flex items-center gap-2"><FileAudio className="h-4 w-4" />Source Audio</Label>
        <FileUpload accept="audio/*" onFileSelect={setSourceAudio} selectedFile={sourceAudio} />
      </div>
      <div className="space-y-4">
        <Label className="flex items-center gap-2"><Volume2 className="h-4 w-4" />Reference Audio (Target Voice)</Label>
        <FileUpload accept="audio/*" onFileSelect={setReferenceAudio} selectedFile={referenceAudio} />
      </div>
      {[
        { key: "temperature", label: "Temperature", min: 0.1, max: 1.5, step: 0.1 },
        { key: "topK", label: "Top K", min: 1, max: 100, step: 1 },
        { key: "topP", label: "Top P", min: 0.1, max: 1, step: 0.05 },
      ].map(({ key, label, min, max, step }) => (
        <div key={key} className="space-y-2">
          <div className="flex justify-between">
            <Label className="flex items-center gap-2">{label} <HelpTooltip content={PARAMETER_HELP[key as keyof typeof PARAMETER_HELP]} /></Label>
            <span>{vcParams[key as keyof MetisVCParams]?.toString()}</span>
          </div>
          <Slider value={[vcParams[key as keyof MetisVCParams] as number]} onValueChange={([v]) => setVcParams((p) => ({ ...p, [key]: v }))} min={min} max={max} step={step} />
        </div>
      ))}
    </div>
  );

  const renderTSEParams = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <Label className="flex items-center gap-2"><Scissors className="h-4 w-4" />Mixed Audio (with multiple speakers)</Label>
        <FileUpload accept="audio/*" onFileSelect={setSourceAudio} selectedFile={sourceAudio} />
      </div>
      {[
        { key: "temperature", label: "Temperature", min: 0.1, max: 1.5, step: 0.1 },
        { key: "topK", label: "Top K", min: 1, max: 100, step: 1 },
      ].map(({ key, label, min, max, step }) => (
        <div key={key} className="space-y-2">
          <div className="flex justify-between">
            <Label className="flex items-center gap-2">{label} <HelpTooltip content={PARAMETER_HELP[key as keyof typeof PARAMETER_HELP]} /></Label>
            <span>{tseParams[key as keyof MetisTSEParams]?.toString()}</span>
          </div>
          <Slider value={[tseParams[key as keyof MetisTSEParams] as number]} onValueChange={([v]) => setTseParams((p) => ({ ...p, [key]: v }))} min={min} max={max} step={step} />
        </div>
      ))}
      <div className="space-y-2">
        <Label className="flex items-center gap-2">Target Speaker <HelpTooltip content={PARAMETER_HELP.targetSpeaker} /></Label>
        <Select value={tseParams.targetSpeaker} onValueChange={(v) => setTseParams((p) => ({ ...p, targetSpeaker: v }))}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="speaker1">Speaker 1</SelectItem>
            <SelectItem value="speaker2">Speaker 2</SelectItem>
            <SelectItem value="speaker3">Speaker 3</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );

  const renderSEParams = () => (
    <div className="space-y-6">
      <div className="space-y-4">
        <Label className="flex items-center gap-2"><Volume2 className="h-4 w-4" />Noisy Audio</Label>
        <FileUpload accept="audio/*" onFileSelect={setSourceAudio} selectedFile={sourceAudio} />
      </div>
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label className="flex items-center gap-2">Temperature <HelpTooltip content={PARAMETER_HELP.temperature} /></Label>
          <span>{seParams.temperature}</span>
        </div>
        <Slider value={[seParams.temperature]} onValueChange={([v]) => setSeParams((p) => ({ ...p, temperature: v }))} min={0.1} max={1.5} step={0.1} />
      </div>
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label className="flex items-center gap-2">Denoise Strength <HelpTooltip content={PARAMETER_HELP.denoiseStrength} /></Label>
          <span>{seParams.denoiseStrength}</span>
        </div>
        <Slider value={[seParams.denoiseStrength]} onValueChange={([v]) => setSeParams((p) => ({ ...p, denoiseStrength: v }))} min={0} max={1} step={0.05} />
      </div>
    </div>
  );

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link href="/tts" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />Back to TTS
        </Link>
        <h1 className="text-3xl font-bold">Metis</h1>
        <p className="text-muted-foreground">Foundation model for unified speech generation (TTS, VC, TSE, SE)</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="tts"><Mic className="mr-2 h-4 w-4" />TTS</TabsTrigger>
          <TabsTrigger value="vc"><Volume2 className="mr-2 h-4 w-4" />VC</TabsTrigger>
          <TabsTrigger value="tse"><Scissors className="mr-2 h-4 w-4" />TSE</TabsTrigger>
          <TabsTrigger value="se"><Wand2 className="mr-2 h-4 w-4" />SE</TabsTrigger>
        </TabsList>

        <div className="grid gap-6 lg:grid-cols-2">
          <div className="space-y-4">
            {activeTab === "tts" && (
              <Card>
                <CardHeader><CardTitle className="flex items-center gap-2"><Mic className="h-5 w-5" />Text Input</CardTitle></CardHeader>
                <CardContent>
                  <Textarea value={text} onChange={(e) => setText(e.target.value)} rows={8} placeholder="Enter text to synthesize..." />
                  <p className="text-sm text-muted-foreground mt-2">{text.length} characters</p>
                </CardContent>
              </Card>
            )}
          </div>

          <div className="space-y-4">
            <Card>
              <CardHeader><CardTitle className="flex items-center gap-2"><Wand2 className="h-5 w-5" />Parameters</CardTitle></CardHeader>
              <CardContent>
                <TabsContent value="tts" className="mt-0">{renderTTSParams()}</TabsContent>
                <TabsContent value="vc" className="mt-0">{renderVCParams()}</TabsContent>
                <TabsContent value="tse" className="mt-0">{renderTSEParams()}</TabsContent>
                <TabsContent value="se" className="mt-0">{renderSEParams()}</TabsContent>
              </CardContent>
            </Card>

            <Button
              onClick={handleGenerate}
              disabled={isGenerating || (activeTab === "tts" ? !text.trim() : !sourceAudio) || (activeTab === "vc" && !referenceAudio)}
              className="w-full h-12"
            >
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
                    a.download = `metis-${activeTab}-${Date.now()}.wav`;
                    a.click();
                  }}>
                    <Download className="mr-2 h-4 w-4" />Download WAV
                  </Button>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </Tabs>
    </div>
  );
}
