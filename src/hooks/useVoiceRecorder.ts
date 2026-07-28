import { useState, useRef, useCallback, useEffect } from 'react';
import type { RecordingState } from '../types';

export interface VoiceRecorderStreamOptions {
  batchIntervalSeconds: number;
  onBatch: (blob: Blob, mimeType: string, batchIndex: number) => void;
}

interface ActiveRecordingContext {
  generation: number;
  recorder: MediaRecorder;
  stream: MediaStream;
  chunks: Blob[];
  trimChunks: number;
  sentChunkCount: number;
  batchIndex: number;
}

export function useVoiceRecorder(streamOptions?: VoiceRecorderStreamOptions) {
  const [state, setState] = useState<RecordingState>({
    isRecording: false,
    isPaused: false,
    duration: 0,
    audioBlob: null,
  });
  const [isStopping, setIsStopping] = useState(false);

  const activeRecordingRef = useRef<ActiveRecordingContext | null>(null);
  const recordingGenerationRef = useRef(0);
  const startInFlightRef = useRef(false);
  const stopInFlightRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const batchTimerRef = useRef<number | null>(null);
  const streamOptionsRef = useRef(streamOptions);

  useEffect(() => {
    streamOptionsRef.current = streamOptions;
  }, [streamOptions]);

  const pickSupportedMimeType = useCallback((): string => {
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];

    for (const type of candidates) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }

    return '';
  }, []);

  const updateDuration = useCallback(() => {
    setState((prev) => ({ ...prev, duration: prev.duration + 1 }));
  }, []);

  const startRecording = useCallback(async () => {
    if (
      startInFlightRef.current
      || stopInFlightRef.current
      || activeRecordingRef.current !== null
    ) {
      return false;
    }
    startInFlightRef.current = true;
    const generation = ++recordingGenerationRef.current;
    let acquiredStream: MediaStream | null = null;
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Микрофон недоступен в этом контексте. Используйте HTTPS или localhost.');
      }

      acquiredStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      if (generation !== recordingGenerationRef.current) {
        acquiredStream.getTracks().forEach((track) => track.stop());
        return false;
      }

      const mimeType = pickSupportedMimeType();
      const recorderOptions: MediaRecorderOptions = {};
      if (mimeType) recorderOptions.mimeType = mimeType;
      // Повышаем битрейт для лучшего качества распознавания речи
      recorderOptions.audioBitsPerSecond = 128000;
      const mediaRecorder = new MediaRecorder(acquiredStream, recorderOptions);
      const context: ActiveRecordingContext = {
        generation,
        recorder: mediaRecorder,
        stream: acquiredStream,
        chunks: [],
        trimChunks: 0,
        sentChunkCount: 0,
        batchIndex: 0,
      };
      activeRecordingRef.current = context;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          context.chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const recorderMimeType = mediaRecorder.mimeType || mimeType || 'audio/webm';
        // Trim last N chunks if requested (to cut off stop phrase from audio)
        const trimCount = context.trimChunks;
        const chunks = trimCount > 0 && context.chunks.length > trimCount
          ? context.chunks.slice(0, -trimCount)
          : context.chunks;

        // Streaming: flush оставшиеся необработанные чанки как финальный батч
        const activeStreamOptions = streamOptionsRef.current;
        if (activeStreamOptions?.onBatch) {
          const sent = context.sentChunkCount;
          if (sent < chunks.length) {
            const remaining = chunks.slice(sent);
            const finalChunks = sent === 0 ? remaining : [chunks[0], ...remaining];
            const finalBlob = new Blob(finalChunks, { type: recorderMimeType });
            try {
              activeStreamOptions.onBatch(finalBlob, recorderMimeType, context.batchIndex++);
            } catch (error) {
              console.error('Error flushing final audio batch:', error);
            }
          }
        }

        context.stream.getTracks().forEach((track) => track.stop());
        const isCurrent =
          generation === recordingGenerationRef.current
          && activeRecordingRef.current === context;
        if (isCurrent) {
          if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
          }
          if (batchTimerRef.current) {
            clearInterval(batchTimerRef.current);
            batchTimerRef.current = null;
          }
          activeRecordingRef.current = null;
          stopInFlightRef.current = false;
          setIsStopping(false);
          const audioBlob = new Blob(chunks, { type: recorderMimeType });
          setState((prev) => ({
            ...prev,
            isRecording: false,
            isPaused: false,
            audioBlob,
          }));
        }
      };

      mediaRecorder.start(1000);

      timerRef.current = window.setInterval(updateDuration, 1000);

      // Streaming: отправляем батчи чанков во время записи
      const activeStreamOptions = streamOptionsRef.current;
      if (activeStreamOptions) {
        const intervalMs = activeStreamOptions.batchIntervalSeconds * 1000;
        batchTimerRef.current = window.setInterval(() => {
          const currentStreamOptions = streamOptionsRef.current;
          if (
            !currentStreamOptions
            || generation !== recordingGenerationRef.current
            || activeRecordingRef.current !== context
          ) return;
          const curr = context.chunks.length;
          const sent = context.sentChunkCount;
          // Ждём минимум 5 чанков (5 сек), чтобы webm-контейнер был валидным
          if (curr - sent < 5) return;
          const newChunks = curr === sent ? [] : context.chunks.slice(sent);
          if (newChunks.length === 0) return;
          context.sentChunkCount = curr;
          // Батч 0 содержит header, остальные батчи префиксируем chunk[0] (webm header)
          const batchChunks = sent === 0 ? newChunks : [context.chunks[0], ...newChunks];
          const blob = new Blob(batchChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
          try {
            currentStreamOptions.onBatch(
              blob,
              mediaRecorder.mimeType || 'audio/webm',
              context.batchIndex++,
            );
          } catch (error) {
            console.error('Error sending audio batch:', error);
          }
        }, intervalMs);
      }

      setIsStopping(false);
      setState({
        isRecording: true,
        isPaused: false,
        duration: 0,
        audioBlob: null,
      });
      return true;
    } catch (error) {
      if (acquiredStream) {
        acquiredStream.getTracks().forEach((track) => track.stop());
      }
      if (activeRecordingRef.current?.generation === generation) {
        activeRecordingRef.current = null;
      }
      console.error('Error accessing microphone:', error);
      throw new Error('Не удалось получить доступ к микрофону');
    } finally {
      startInFlightRef.current = false;
    }
  }, [pickSupportedMimeType, updateDuration]);

  const pauseRecording = useCallback(() => {
    const context = activeRecordingRef.current;
    if (context && state.isRecording && !state.isPaused) {
      context.recorder.pause();
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setState((prev) => ({ ...prev, isPaused: true }));
    }
  }, [state.isRecording, state.isPaused]);

  const resumeRecording = useCallback(() => {
    const context = activeRecordingRef.current;
    if (context && state.isRecording && state.isPaused) {
      context.recorder.resume();
      timerRef.current = window.setInterval(updateDuration, 1000);
      setState((prev) => ({ ...prev, isPaused: false }));
    }
  }, [state.isRecording, state.isPaused, updateDuration]);

  const stopRecording = useCallback((trimSeconds = 0) => {
    const context = activeRecordingRef.current;
    if (
      context
      && !stopInFlightRef.current
      && context.recorder.state !== 'inactive'
    ) {
      stopInFlightRef.current = true;
      setIsStopping(true);
      // Each chunk is ~1 second (start(1000)), so trimSeconds ≈ chunks to drop
      context.trimChunks = trimSeconds;

      if (batchTimerRef.current) {
        clearInterval(batchTimerRef.current);
        batchTimerRef.current = null;
      }

      context.recorder.stop();

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      context.stream.getTracks().forEach((track) => track.stop());

      setState((prev) => ({ ...prev, isRecording: false, isPaused: false }));
    }
  }, []);

  const resetRecording = useCallback(() => {
    recordingGenerationRef.current += 1;
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (batchTimerRef.current) {
      clearInterval(batchTimerRef.current);
      batchTimerRef.current = null;
    }

    const context = activeRecordingRef.current;
    activeRecordingRef.current = null;
    if (context) {
      context.recorder.ondataavailable = null;
      context.recorder.onstop = null;
      if (context.recorder.state !== 'inactive') {
        try {
          context.recorder.stop();
        } catch {
          // Already stopping.
        }
      }
      context.stream.getTracks().forEach((track) => track.stop());
    }

    stopInFlightRef.current = false;
    setIsStopping(false);

    setState({
      isRecording: false,
      isPaused: false,
      duration: 0,
      audioBlob: null,
    });
  }, []);

  useEffect(() => {
    return () => {
      recordingGenerationRef.current += 1;
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (batchTimerRef.current) {
        clearInterval(batchTimerRef.current);
      }
      const context = activeRecordingRef.current;
      activeRecordingRef.current = null;
      if (context) {
        context.recorder.ondataavailable = null;
        context.recorder.onstop = null;
        if (context.recorder.state !== 'inactive') {
          try {
            context.recorder.stop();
          } catch {
            // Already stopping.
          }
        }
        context.stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const formatDuration = useCallback((seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }, []);

  return {
    ...state,
    isStopping,
    formattedDuration: formatDuration(state.duration),
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording,
    resetRecording,
  };
}
