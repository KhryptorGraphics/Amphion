"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileUpload } from "@/components/ui/file-upload";
import { AudioComparisonPlayer } from "@/components/ui/audio-comparison";
import { ArrowLeft, BarChart3, Download, RefreshCw } from "lucide-react";
import Link from "next/link";

export default function AudioComparePage() {
  const [fileA, setFileA] = useState<File | null>(null);
  const [fileB, setFileB] = useState<File | null>(null);
  const [urlA, setUrlA] = useState<string | null>(null);
  const [urlB, setUrlB] = useState<string | null>(null);

  const handleFileASelect = (file: File | null) => {
    if (urlA) URL.revokeObjectURL(urlA);
    setFileA(file);
    setUrlA(file ? URL.createObjectURL(file) : null);
  };

  const handleFileBSelect = (file: File | null) => {
    if (urlB) URL.revokeObjectURL(urlB);
    setFileB(file);
    setUrlB(file ? URL.createObjectURL(file) : null);
  };

  const handleClear = () => {
    if (urlA) URL.revokeObjectURL(urlA);
    if (urlB) URL.revokeObjectURL(urlB);
    setFileA(null);
    setFileB(null);
    setUrlA(null);
    setUrlB(null);
  };

  const handleDownload = (url: string, filename: string) => {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
  };

  const bothLoaded = urlA !== null && urlB !== null;

  return (
    <div className="container mx-auto py-6">
      <div className="mb-6">
        <Link
          href="/"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Home
        </Link>
        <div className="flex items-center gap-3">
          <h1 className="text-3xl font-bold">Audio Comparison</h1>
          <BarChart3 className="h-8 w-8 text-primary" />
        </div>
        <p className="text-muted-foreground">
          Compare two audio files side-by-side with synchronized playback
        </p>
      </div>

      {!bothLoaded ? (
        <Card>
          <CardHeader>
            <CardTitle>Load Audio Files</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-2">
                <p className="text-sm font-medium">Original Audio</p>
                <FileUpload
                  accept="audio/*"
                  onFileSelect={handleFileASelect}
                  selectedFile={fileA}
                  label="Drop original audio here or click to browse"
                  description="The reference or original track"
                />
              </div>
              <div className="space-y-2">
                <p className="text-sm font-medium">Generated Audio</p>
                <FileUpload
                  accept="audio/*"
                  onFileSelect={handleFileBSelect}
                  selectedFile={fileB}
                  label="Drop generated audio here or click to browse"
                  description="The synthesized or processed track"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Synchronized Playback</CardTitle>
            </CardHeader>
            <CardContent>
              <AudioComparisonPlayer
                srcA={urlA}
                srcB={urlB}
                labelA={fileA?.name ?? "Original Audio"}
                labelB={fileB?.name ?? "Generated Audio"}
              />
            </CardContent>
          </Card>

          <div className="flex flex-wrap gap-3">
            <Button
              variant="outline"
              onClick={() => handleDownload(urlA, fileA?.name ?? "original-audio")}
            >
              <Download className="mr-2 h-4 w-4" />
              Download Original
            </Button>
            <Button
              variant="outline"
              onClick={() => handleDownload(urlB, fileB?.name ?? "generated-audio")}
            >
              <Download className="mr-2 h-4 w-4" />
              Download Generated
            </Button>
            <Button variant="outline" onClick={handleClear}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Load Different Files
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
