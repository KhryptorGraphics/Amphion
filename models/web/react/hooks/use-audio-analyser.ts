"use client"

import * as React from "react"

interface AudioAnalyserState {
  isReady: boolean
  analyser: AnalyserNode | null
  dataArray: Uint8Array | null
  bufferLength: number
}

interface UseAudioAnalyserOptions {
  fftSize?: number
  smoothingTimeConstant?: number
  minDecibels?: number
  maxDecibels?: number
}

const DEFAULT_OPTIONS: Required<UseAudioAnalyserOptions> = {
  fftSize: 2048,
  smoothingTimeConstant: 0.8,
  minDecibels: -90,
  maxDecibels: -10,
}

let audioContext: AudioContext | null = null

function getAudioContext(): AudioContext {
  if (!audioContext) {
    audioContext = new (window.AudioContext || (window as typeof window & { webkitAudioContext: typeof AudioContext }).webkitAudioContext)()
  }
  return audioContext
}

function createAnalyser(
  context: AudioContext,
  options: Required<UseAudioAnalyserOptions>
): AnalyserNode {
  const analyser = context.createAnalyser()
  analyser.fftSize = options.fftSize
  analyser.smoothingTimeConstant = options.smoothingTimeConstant
  analyser.minDecibels = options.minDecibels
  analyser.maxDecibels = options.maxDecibels
  return analyser
}

export function useAudioAnalyser(options: UseAudioAnalyserOptions = {}) {
  const mergedOptions = { ...DEFAULT_OPTIONS, ...options }
  const [state, setState] = React.useState<AudioAnalyserState>({
    isReady: false,
    analyser: null,
    dataArray: null,
    bufferLength: 0,
  })

  const analyserRef = React.useRef<AnalyserNode | null>(null)
  const sourceRef = React.useRef<MediaElementAudioSourceNode | null>(null)

  const connectToElement = React.useCallback(
    (element: HTMLAudioElement | null) => {
      if (!element) {
        if (sourceRef.current) {
          sourceRef.current.disconnect()
          sourceRef.current = null
        }
        return
      }

      try {
        const context = getAudioContext()

        if (!analyserRef.current) {
          analyserRef.current = createAnalyser(context, mergedOptions)
        }

        if (!sourceRef.current) {
          sourceRef.current = context.createMediaElementSource(element)
          sourceRef.current.connect(analyserRef.current)
          analyserRef.current.connect(context.destination)

          const bufferLength = analyserRef.current.frequencyBinCount
          const dataArray = new Uint8Array(bufferLength)

          setState({
            isReady: true,
            analyser: analyserRef.current,
            dataArray,
            bufferLength,
          })
        }
      } catch (error) {
        // Handle case where audio element is already connected to another source
        // or other Web Audio API errors
        setState((prev) => ({
          ...prev,
          isReady: false,
        }))
      }
    },
    [mergedOptions]
  )

  const getByteFrequencyData = React.useCallback(
    (array: Uint8Array): void => {
      if (!analyserRef.current) {
        return
      }
      analyserRef.current.getByteFrequencyData(array as Uint8Array<ArrayBuffer>)
    },
    []
  )

  const getByteTimeDomainData = React.useCallback(
    (array: Uint8Array): void => {
      if (!analyserRef.current) {
        return
      }
      analyserRef.current.getByteTimeDomainData(array as Uint8Array<ArrayBuffer>)
    },
    []
  )

  const reset = React.useCallback(() => {
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }
    if (analyserRef.current) {
      analyserRef.current.disconnect()
      analyserRef.current = null
    }
    setState({
      isReady: false,
      analyser: null,
      dataArray: null,
      bufferLength: 0,
    })
  }, [])

  React.useEffect(() => {
    return () => {
      if (sourceRef.current) {
        sourceRef.current.disconnect()
        sourceRef.current = null
      }
      if (analyserRef.current) {
        analyserRef.current.disconnect()
        analyserRef.current = null
      }
    }
  }, [])

  return {
    ...state,
    connectToElement,
    getByteFrequencyData,
    getByteTimeDomainData,
    reset,
  }
}

export type { AudioAnalyserState, UseAudioAnalyserOptions }
