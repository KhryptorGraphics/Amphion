"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { AudioPlayer } from "@/components/ui/audio-player";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Search,
  Download,
  Play,
  Trash2,
  Clock,
  FileAudio,
  Filter,
  X,
} from "lucide-react";

interface TTSHistoryItem {
  id: string;
  text: string;
  model: string;
  voiceId: string;
  audioUrl: string;
  timestamp: number;
  duration?: number;
}

const MODEL_LABELS: Record<string, string> = {
  maskgct: "MaskGCT",
  "dualcodec-valle": "DualCodec-VALLE",
  "vevo-tts": "Vevo TTS",
  metis: "Metis",
};

export function TTSHistory() {
  const [history, setHistory] = useState<TTSHistoryItem[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [modelFilter, setModelFilter] = useState<string>("all");
  const [selectedItem, setSelectedItem] = useState<TTSHistoryItem | null>(null);

  useEffect(() => {
    // Load history from localStorage
    const stored = localStorage.getItem("tts-history");
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setHistory(parsed);
      } catch {
        console.error("Failed to parse TTS history");
      }
    }
  }, []);

  const saveHistory = (items: TTSHistoryItem[]) => {
    localStorage.setItem("tts-history", JSON.stringify(items));
    setHistory(items);
  };

  const deleteItem = (id: string) => {
    const newHistory = history.filter((item) => item.id !== id);
    saveHistory(newHistory);
    if (selectedItem?.id === id) {
      setSelectedItem(null);
    }
  };

  const clearAll = () => {
    if (confirm("Are you sure you want to clear all history?")) {
      saveHistory([]);
      setSelectedItem(null);
    }
  };

  const downloadItem = (item: TTSHistoryItem) => {
    const a = document.createElement("a");
    a.href = item.audioUrl;
    a.download = `tts-${item.model}-${new Date(item.timestamp).getTime()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.text
      .toLowerCase()
      .includes(searchQuery.toLowerCase());
    const matchesModel = modelFilter === "all" || item.model === modelFilter;
    return matchesSearch && matchesModel;
  });

  const uniqueModels = Array.from(new Set(history.map((item) => item.model)));

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatDuration = (seconds?: number) => {
    if (!seconds) return "--:--";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Filters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by text content..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
              {searchQuery && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="absolute right-1 top-1 h-8 w-8 p-0"
                  onClick={() => setSearchQuery("")}
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>
            <Select value={modelFilter} onValueChange={setModelFilter}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Filter by model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Models</SelectItem>
                {uniqueModels.map((model) => (
                  <SelectItem key={model} value={model}>
                    {MODEL_LABELS[model] || model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">
              {filteredHistory.length} items
              {searchQuery || modelFilter !== "all" ? " (filtered)" : ""}
            </span>
            {history.length > 0 && (
              <Button variant="ghost" size="sm" onClick={clearAll}>
                <Trash2 className="mr-2 h-4 w-4" />
                Clear All
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* History List */}
      {filteredHistory.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <FileAudio className="mx-auto h-12 w-12 text-muted-foreground/50" />
            <p className="mt-4 text-lg font-medium">No history yet</p>
            <p className="text-muted-foreground">
              Generated audio will appear here
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4">
          {selectedItem && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>Selected Audio</CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedItem(null)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <AudioPlayer src={selectedItem.audioUrl} />
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => downloadItem(selectedItem)}
                  >
                    <Download className="mr-2 h-4 w-4" />
                    Download
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={() => deleteItem(selectedItem.id)}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    Delete
                  </Button>
                </div>
                <div className="text-sm text-muted-foreground space-y-1">
                  <p>
                    <strong>Model:</strong>{" "}
                    {MODEL_LABELS[selectedItem.model] || selectedItem.model}
                  </p>
                  <p>
                    <strong>Voice:</strong> {selectedItem.voiceId}
                  </p>
                  <p>
                    <strong>Created:</strong>{" "}
                    {formatDate(selectedItem.timestamp)}
                  </p>
                  {selectedItem.duration && (
                    <p>
                      <strong>Duration:</strong>{" "}
                      {formatDuration(selectedItem.duration)}
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid gap-2">
            {filteredHistory.map((item) => (
              <Card
                key={item.id}
                className="cursor-pointer hover:bg-accent/50 transition-colors"
                onClick={() => setSelectedItem(item)}
              >
                <CardContent className="p-4">
                  <div className="flex items-center gap-4">
                    <div className="h-10 w-10 rounded-full bg-primary/10 flex items-center justify-center">
                      <Play className="h-5 w-5 text-primary" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{item.text}</p>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Badge variant="outline" className="text-xs">
                          {MODEL_LABELS[item.model] || item.model}
                        </Badge>
                        <span>•</span>
                        <Clock className="h-3 w-3" />
                        <span>{formatDate(item.timestamp)}</span>
                      </div>
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        downloadItem(item);
                      }}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteItem(item.id);
                      }}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
