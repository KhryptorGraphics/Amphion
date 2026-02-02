"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Check, Cpu } from "lucide-react";

interface TTSModel {
  id: string;
  name: string;
  description: string;
}

interface ModelSelectorProps {
  models: TTSModel[];
  selected: TTSModel;
  onSelect: (model: TTSModel) => void;
}

export function ModelSelector({ models, selected, onSelect }: ModelSelectorProps) {
  return (
    <Card className="h-fit">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Cpu className="h-5 w-5" />
          Select Model
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {models.map((model) => (
            <button
              key={model.id}
              onClick={() => onSelect(model)}
              className={cn(
                "w-full text-left p-3 rounded-lg transition-all",
                "hover:bg-accent hover:shadow-sm",
                "border-2",
                selected.id === model.id
                  ? "border-primary bg-primary/5"
                  : "border-transparent bg-muted/50"
              )}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5">
                  {selected.id === model.id ? (
                    <div className="h-5 w-5 rounded-full bg-primary flex items-center justify-center">
                      <Check className="h-3 w-3 text-primary-foreground" />
                    </div>
                  ) : (
                    <div className="h-5 w-5 rounded-full border-2 border-muted-foreground/30" />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-sm">{model.name}</div>
                  <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                    {model.description}
                  </p>
                </div>
              </div>
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
