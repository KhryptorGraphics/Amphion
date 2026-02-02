"use client";

import { useCallback } from "react";
import { cn } from "@/lib/utils";
import { Upload, File, X } from "lucide-react";

interface FileUploadProps {
  accept?: string;
  onFileSelect: (file: File | null) => void;
  selectedFile?: File | null;
  label?: string;
  description?: string;
}

export function FileUpload({
  accept,
  onFileSelect,
  selectedFile,
  label = "Drop file here or click to browse",
  description,
}: FileUploadProps) {
  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) {
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const handleRemove = () => {
    onFileSelect(null);
  };

  if (selectedFile) {
    return (
      <div className="flex items-center gap-3 p-3 border rounded-lg bg-muted/50">
        <File className="h-8 w-8 text-primary" />
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm truncate">{selectedFile.name}</p>
          <p className="text-xs text-muted-foreground">
            {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
        <button
          onClick={handleRemove}
          className="p-1 hover:bg-destructive/10 rounded transition-colors"
        >
          <X className="h-4 w-4 text-destructive" />
        </button>
      </div>
    );
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      className={cn(
        "border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer",
        "hover:border-primary hover:bg-primary/5"
      )}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleChange}
        className="hidden"
        id="file-upload"
      />
      <label htmlFor="file-upload" className="cursor-pointer block">
        <Upload className="mx-auto h-8 w-8 text-muted-foreground mb-2" />
        <p className="font-medium">{label}</p>
        {description && (
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
        )}
        {accept && (
          <p className="text-xs text-muted-foreground mt-2">
            Accepted: {accept}
          </p>
        )}
      </label>
    </div>
  );
}
