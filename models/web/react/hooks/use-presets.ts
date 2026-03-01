"use client"

import { useState, useEffect, useCallback } from "react"

export interface Preset {
  id: string
  name: string
  description: string
  model_type: string
  model_id: string
  parameters: Record<string, unknown>
  created_at: string
  updated_at: string
}

interface UsePresetsReturn {
  presets: Preset[]
  loading: boolean
  error: string | null
  savePreset: (
    name: string,
    description: string,
    params: Record<string, unknown>
  ) => Promise<Preset | null>
  deletePreset: (id: string) => Promise<boolean>
  refreshPresets: () => Promise<void>
}

export function usePresets(
  modelId: string,
  modelType: string
): UsePresetsReturn {
  const [presets, setPresets] = useState<Preset[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshPresets = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const params = new URLSearchParams({ model_id: modelId, model_type: modelType })
      const response = await fetch(`/api/presets?${params}`)
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || "Failed to fetch presets")
      }
      const data: Preset[] = await response.json()
      setPresets(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch presets")
    } finally {
      setLoading(false)
    }
  }, [modelId, modelType])

  useEffect(() => {
    refreshPresets()
  }, [refreshPresets])

  const savePreset = useCallback(
    async (
      name: string,
      description: string,
      params: Record<string, unknown>
    ): Promise<Preset | null> => {
      setError(null)
      try {
        const response = await fetch("/api/presets", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name,
            description,
            model_type: modelType,
            model_id: modelId,
            parameters: params,
          }),
        })
        if (!response.ok) {
          const data = await response.json()
          throw new Error(data.detail || "Failed to save preset")
        }
        const created: Preset = await response.json()
        setPresets((prev) => [...prev, created])
        return created
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to save preset")
        return null
      }
    },
    [modelId, modelType]
  )

  const deletePreset = useCallback(async (id: string): Promise<boolean> => {
    setError(null)
    try {
      const response = await fetch(`/api/presets/${id}`, { method: "DELETE" })
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || "Failed to delete preset")
      }
      setPresets((prev) => prev.filter((p) => p.id !== id))
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete preset")
      return false
    }
  }, [])

  return { presets, loading, error, savePreset, deletePreset, refreshPresets }
}
