"use client";

import { useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { FileUpload } from "@/components/ui/file-upload";
import { useToast } from "@/hooks/use-toast";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Play,
  Download,
  FileText,
  Upload,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  Trash2,
} from "lucide-react";

interface TTSModel {
  id: string;
  name: string;
  description: string;
}

interface BatchItem {
  id: string;
  text: string;
  status: "pending" | "processing" | "completed" | "error";
  audioUrl?: string;
  error?: string;
}

interface BatchProcessorProps {
  model: TTSModel;
}

export function BatchProcessor({ model }: BatchProcessorProps) {
  const { toast } = useToast();
  const [items, setItems] = useState<BatchItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [progress, setProgress] = useState(0);
  const [jsonInput, setJsonInput] = useState("");

  const parseCSV = (content: string): string[] => {
    return content
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
  };

  const parseJSON = (content: string): string[] => {
    try {
      const parsed = JSON.parse(content);
      if (Array.isArray(parsed)) {
        return parsed.map((item) =>
          typeof item === "string" ? item : item.text || ""
        );
      }
      return [];
    } catch {
      return [];
    }
  };

  const handleFileUpload = (file: File | null) => {
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      let texts: string[] = [];

      if (file.name.endsWith(".json")) {
        texts = parseJSON(content);
      } else {
        texts = parseCSV(content);
      }

      const newItems: BatchItem[] = texts.map((text, index) => ({
        id: `item-${Date.now()}-${index}`,
        text,
        status: "pending",
      }));

      setItems((prev) => [...prev, ...newItems]);

      toast({
        title: "File loaded",
        description: `${newItems.length} items added to batch`,
      });
    };
    reader.readAsText(file);
  };

  const handleJsonAdd = () => {
    const texts = parseJSON(jsonInput);
    if (texts.length === 0) {
      toast({
        title: "Error",
        description: "Invalid JSON format. Expected array of strings or objects with 'text' property.",
        variant: "destructive",
      });
      return;
    }

    const newItems: BatchItem[] = texts.map((text, index) => ({
      id: `item-${Date.now()}-${index}`,
      text,
      status: "pending",
    }));

    setItems((prev) => [...prev, ...newItems]);
    setJsonInput("");

    toast({
      title: "Items added",
      description: `${newItems.length} items added to batch`,
    });
  };

  const processBatch = async () => {
    if (items.length === 0) return;

    setIsProcessing(true);
    setCurrentIndex(0);
    setProgress(0);

    for (let i = 0; i < items.length; i++) {
      setCurrentIndex(i);
      setProgress((i / items.length) * 100);

      setItems((prev) =>
        prev.map((item, idx) =>
          idx === i ? { ...item, status: "processing" } : item
        )
      );

      try {
        const formData = new FormData();
        formData.append("text", items[i].text);
        formData.append("model", model.id);

        const response = await fetch("/api/tts/generate", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("Generation failed");
        }

        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);

        setItems((prev) =>
          prev.map((item, idx) =>
            idx === i ? { ...item, status: "completed", audioUrl } : item
          )
        );
      } catch (error) {
        setItems((prev) =>
          prev.map((item, idx) =>
            idx === i
              ? { ...item, status: "error", error: "Failed to generate" }
              : item
          )
        );
      }
    }

    setProgress(100);
    setIsProcessing(false);

    toast({
      title: "Batch complete",
      description: "All items processed",
    });
  };

  const downloadAll = () => {
    items.forEach((item, index) => {
      if (item.audioUrl) {
        const a = document.createElement("a");
        a.href = item.audioUrl;
        a.download = `tts-${model.id}-batch-${index + 1}.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }
    });
  };

  const removeItem = (id: string) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  };

  const clearAll = () => {
    setItems([]);
    setProgress(0);
  };

  const completedCount = items.filter((i) => i.status === "completed").length;
  const errorCount = items.filter((i) => i.status === "error").length;

  return (
    <div className="space-y-4">
      <Tabs defaultValue="upload" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="upload">File Upload</TabsTrigger>
          <TabsTrigger value="json">JSON Input</TabsTrigger>
        </TabsList>

        <TabsContent value="upload" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload CSV or JSON
              </CardTitle>
            </CardHeader>
            <CardContent>
              <FileUpload
                accept=".csv,.json,.txt"
                onFileSelect={handleFileUpload}
                label="Drop CSV, JSON, or text file here"
                description="Each line will be treated as a separate text to synthesize"
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="json" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                JSON Input
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder={`[\n  "Text to synthesize 1",\n  "Text to synthesize 2",\n  { "text": "Text to synthesize 3" }\n]`}
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                rows={8}
                className="font-mono text-sm"
              />
              <Button onClick={handleJsonAdd} disabled={!jsonInput.trim()}>
                Add to Batch
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {items.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                Batch Items
                <Badge variant="secondary">{items.length}</Badge>
              </CardTitle>
              <div className="flex gap-2">
                {completedCount > 0 && (
                  <Button variant="outline" size="sm" onClick={downloadAll}>
                    <Download className="mr-2 h-4 w-4" />
                    Download All
                  </Button>
                )}
                <Button
                  variant="outline"
                  size="sm"
                  onClick={clearAll}
                  disabled={isProcessing}
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Clear
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {isProcessing && (
              <div className="space-y-2">
                <Progress value={progress} />
                <p className="text-sm text-center text-muted-foreground">
                  Processing item {currentIndex + 1} of {items.length}
                </p>
              </div>
            )}

            <div className="border rounded-lg">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-12">#</TableHead>
                    <TableHead>Text</TableHead>
                    <TableHead className="w-24">Status</TableHead>
                    <TableHead className="w-12"></TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {items.map((item, index) => (
                    <TableRow key={item.id}>
                      <TableCell className="font-mono text-xs">
                        {index + 1}
                      </TableCell>
                      <TableCell className="max-w-md truncate">
                        {item.text}
                      </TableCell>
                      <TableCell>
                        {item.status === "pending" && (
                          <Badge variant="outline">
                            <Clock className="mr-1 h-3 w-3" />
                            Pending
                          </Badge>
                        )}
                        {item.status === "processing" && (
                          <Badge variant="secondary">
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                            Processing
                          </Badge>
                        )}
                        {item.status === "completed" && (
                          <Badge variant="default" className="bg-green-600">
                            <CheckCircle className="mr-1 h-3 w-3" />
                            Done
                          </Badge>
                        )}
                        {item.status === "error" && (
                          <Badge variant="destructive">
                            <XCircle className="mr-1 h-3 w-3" />
                            Error
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeItem(item.id)}
                          disabled={isProcessing}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            <Button
              onClick={processBatch}
              disabled={isProcessing || items.length === 0}
              className="w-full"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Process Batch ({items.length} items)
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
