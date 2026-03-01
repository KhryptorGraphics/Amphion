"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { usePresets } from "@/hooks/use-presets";
import { BookmarkPlus, Trash2 } from "lucide-react";

interface PresetManagerProps {
  modelId: string;
  modelType: string;
  currentParams: Record<string, unknown>;
  onLoadPreset: (params: Record<string, unknown>) => void;
}

export function PresetManager({
  modelId,
  modelType,
  currentParams,
  onLoadPreset,
}: PresetManagerProps) {
  const { presets, loading, error, savePreset, deletePreset } = usePresets(
    modelId,
    modelType
  );

  const [selectedPresetId, setSelectedPresetId] = useState<string>("");
  const [showSaveForm, setShowSaveForm] = useState(false);
  const [saveName, setSaveName] = useState("");
  const [saveDescription, setSaveDescription] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const selectedPreset = presets.find((p) => p.id === selectedPresetId);

  const handleLoad = () => {
    if (!selectedPreset) return;
    onLoadPreset(selectedPreset.parameters);
  };

  const handleSaveToggle = () => {
    setShowSaveForm((prev) => !prev);
    setSaveName("");
    setSaveDescription("");
  };

  const handleSaveSubmit = async () => {
    if (!saveName.trim()) return;
    setIsSaving(true);
    const created = await savePreset(
      saveName.trim(),
      saveDescription.trim(),
      currentParams
    );
    setIsSaving(false);
    if (created) {
      setShowSaveForm(false);
      setSaveName("");
      setSaveDescription("");
      setSelectedPresetId(created.id);
    }
  };

  const handleDeleteClick = () => {
    if (!selectedPreset) return;
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    handleDeleteConfirm();
  };

  const handleDeleteConfirm = async () => {
    if (!selectedPreset) return;
    setIsDeleting(true);
    const success = await deletePreset(selectedPreset.id);
    setIsDeleting(false);
    setConfirmDelete(false);
    if (success) {
      setSelectedPresetId("");
    }
  };

  const handleDeleteCancel = () => {
    setConfirmDelete(false);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BookmarkPlus className="h-5 w-5" />
          Presets
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}

        {loading ? (
          <p className="text-sm text-muted-foreground">Loading presets...</p>
        ) : presets.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No presets saved yet. Save your current settings to create one.
          </p>
        ) : (
          <div className="space-y-2">
            <Label>Select Preset</Label>
            <Select
              value={selectedPresetId}
              onValueChange={(value) => {
                setSelectedPresetId(value);
                setConfirmDelete(false);
              }}
            >
              <SelectTrigger>
                <SelectValue placeholder="Choose a preset..." />
              </SelectTrigger>
              <SelectContent>
                {presets.map((preset) => (
                  <SelectItem key={preset.id} value={preset.id}>
                    {preset.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {selectedPreset?.description && (
              <p className="text-xs text-muted-foreground">
                {selectedPreset.description}
              </p>
            )}
          </div>
        )}

        <div className="flex flex-wrap gap-2">
          {presets.length > 0 && (
            <Button
              variant="secondary"
              size="sm"
              disabled={!selectedPresetId}
              onClick={handleLoad}
            >
              Load
            </Button>
          )}

          <Button
            variant="outline"
            size="sm"
            onClick={handleSaveToggle}
          >
            <BookmarkPlus className="mr-1.5 h-4 w-4" />
            {showSaveForm ? "Cancel" : "Save Current"}
          </Button>

          {selectedPresetId && !confirmDelete && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleDeleteClick}
              disabled={isDeleting}
            >
              <Trash2 className="mr-1.5 h-4 w-4" />
              Delete
            </Button>
          )}

          {confirmDelete && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Are you sure?</span>
              <Button
                variant="destructive"
                size="sm"
                onClick={handleDeleteConfirm}
                disabled={isDeleting}
              >
                <Trash2 className="mr-1.5 h-4 w-4" />
                {isDeleting ? "Deleting..." : "Confirm"}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDeleteCancel}
                disabled={isDeleting}
              >
                Cancel
              </Button>
            </div>
          )}
        </div>

        {showSaveForm && (
          <div className="space-y-3 border rounded-md p-3">
            <div className="space-y-1.5">
              <Label htmlFor="preset-name">Name</Label>
              <Input
                id="preset-name"
                placeholder="My preset name"
                value={saveName}
                onChange={(e) => setSaveName(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="preset-description">Description (optional)</Label>
              <Input
                id="preset-description"
                placeholder="Describe these settings..."
                value={saveDescription}
                onChange={(e) => setSaveDescription(e.target.value)}
              />
            </div>
            <Button
              size="sm"
              disabled={!saveName.trim() || isSaving}
              onClick={handleSaveSubmit}
            >
              {isSaving ? "Saving..." : "Save Preset"}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
