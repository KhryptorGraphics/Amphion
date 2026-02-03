"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { FileUpload } from "@/components/ui/file-upload";
import { HelpTooltip } from "@/components/ui/help-tooltip";
import { AudioPlayer } from "@/components/ui/audio-player";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { Download, ArrowLeft, Loader2, Upload, Disc3, ArrowRightLeft, Copy, Check } from "lucide-react";
import Link from "next/link";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function DualCodecPage() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("encode");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [tokens, setTokens] = useState<string>("");
  const [resultAudio, setResultAudio] = useState<string | null>(null);
  const [encodedTokens, setEncodedTokens] = useState<string>("");
  const [copied, setCopied] = useState(false);

  const handleEncode = async () => {
    if (!audioFile) {
      toast({ title: "Error", description: "Please upload an audio file", variant: "destructive" });
      return;
    }

    setIsProcessing(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("audio", audioFile);

      const progressInterval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? prev : prev + 10));
      }, 300);

      const response = await fetch("/api/codec/dualcodec/encode", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Encoding failed");
      }

      const data = await response.json();
      setEncodedTokens(JSON.stringify(data.tokens, null, 2));
      setProgress(100);

      toast({ title: "Success", description: "Audio encoded successfully" });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to encode audio",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDecode = async () => {
    if (!tokens.trim()) {
      toast({ title: "Error", description: "Please enter tokens", variant: "destructive" });
      return;
    }

    setIsProcessing(true);
    setProgress(0);

    try {
      const formData = new FormData();
      formData.append("tokens", tokens);

      const progressInterval = setInterval(() => {
        setProgress((prev) => (prev >= 90 ? prev : prev + 10));
      }, 300);

      const response = await fetch("/api/codec/dualcodec/decode", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Decoding failed");
      }

      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      setResultAudio(audioUrl);
      setProgress(100);

      toast({ title: "Success", description: "Audio decoded successfully" });
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to decode audio",
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
    a.download = `dualcodec-decoded-${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const copyTokens = () => {
    navigator.clipboard.writeText(encodedTokens);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link href="/codec" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Codecs
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">DualCodec</h1>
        <p className="text-muted-foreground">Neural audio codec with dual-stream encoding for high-quality compression</p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="encode" className="flex items-center gap-2">
            <Upload className="h-4 w-4" />
            Encode
          </TabsTrigger>
          <TabsTrigger value="decode" className="flex items-center gap-2">
            <Disc3 className="h-4 w-4" />
            Decode
          </TabsTrigger>
        </TabsList>

        <TabsContent value="encode" className="mt-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5" />
                  Upload Audio
                  <HelpTooltip content="Upload audio file to encode into tokens" />
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <FileUpload
                  accept="audio/*"
                  onFileSelect={setAudioFile}
                  selectedFile={audioFile}
                  label="Upload audio to encode"
                  description="WAV or MP3 format recommended"
                />
                <Button onClick={handleEncode} disabled={isProcessing || !audioFile} className="w-full">
                  {isProcessing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ArrowRightLeft className="mr-2 h-4 w-4" />}
                  Encode Audio
                </Button>
                {isProcessing && <Progress value={progress} />}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Disc3 className="h-5 w-5" />
                  Encoded Tokens
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={encodedTokens}
                  readOnly
                  placeholder="Encoded tokens will appear here..."
                  className="min-h-[200px] font-mono text-xs"
                />
                {encodedTokens && (
                  <Button variant="outline" className="w-full" onClick={copyTokens}>
                    {copied ? <Check className="mr-2 h-4 w-4" /> : <Copy className="mr-2 h-4 w-4" />}
                    {copied ? "Copied!" : "Copy Tokens"}
                  </Button>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="decode" className="mt-6">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Disc3 className="h-5 w-5" />
                  Input Tokens
                  <HelpTooltip content="Enter tokens (JSON array format) to decode into audio" />
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  value={tokens}
                  onChange={(e) => setTokens(e.target.value)}
                  placeholder='[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]'
                  className="min-h-[200px] font-mono text-xs"
                />
                <Button onClick={handleDecode} disabled={isProcessing || !tokens.trim()} className="w-full">
                  {isProcessing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <ArrowRightLeft className="mr-2 h-4 w-4" />}
                  Decode to Audio
                </Button>
                {isProcessing && <Progress value={progress} />}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Decoded Audio</CardTitle>
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
                  <p className="text-muted-foreground text-center py-8">Decoded audio will appear here</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
