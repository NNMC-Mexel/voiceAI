import { useState, useRef, useCallback, useEffect } from 'react';
import type { RecordingState } from '../types';

export interface VoiceRecorderStreamOptions {
  batchIntervalSeconds: number;
  onBatch: (blob: Blob, mimeType: string, batchIndex: number) => void;
}

export function useVoiceRecorder(streamOptions?: VoiceRecorderStreamOptions) {
  const [state, setState] = useState<RecordingState>({
    isRecording: false,
    isPaused: false,
    duration: 0,
    audioBlob: null,
  });

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const trimChunksRef = useRef(0);
  const batchTimerRef = useRef<number | null>(null);
  const sentChunkCountRef = useRef(0);
  const batchIndexRef = useRef(0);

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
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Микрофон недоступен в этом контексте. Используйте HTTPS или localhost.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 48000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      chunksRef.current = [];

      const mimeType = pickSupportedMimeType();
      const recorderOptions: MediaRecorderOptions = {};
      if (mimeType) recorderOptions.mimeType = mimeType;
      // Повышаем битрейт для лучшего качества распознавания речи
      recorderOptions.audioBitsPerSecond = 128000;
      const mediaRecorder = new MediaRecorder(stream, recorderOptions);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const recorderMimeType = mediaRecorder.mimeType || mimeType || 'audio/webm';
        // Trim last N chunks if requested (to cut off stop phrase from audio)
        const trimCount = trimChunksRef.current;
        trimChunksRef.current = 0;
        const chunks = trimCount > 0 && chunksRef.current.length > trimCount
          ? chunksRef.current.slice(0, -trimCount)
          : chunksRef.current;

        // Streaming: flush оставшиеся необработанные чанки как финальный батч
        if (streamOptions?.onBatch) {
          const sent = sentChunkCountRef.current;
          if (sent < chunks.length) {
            const remaining = chunks.slice(sent);
            const finalChunks = sent === 0 ? remaining : [chunks[0], ...remaining];
            const finalBlob = new Blob(finalChunks, { type: recorderMimeType });
            streamOptions.onBatch(finalBlob, recorderMimeType, batchIndexRef.current++);
          }
        }

        const audioBlob = new Blob(chunks, { type: recorderMimeType });
        setState((prev) => ({ ...prev, audioBlob }));
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(1000);

      timerRef.current = window.setInterval(updateDuration, 1000);

      // Streaming: отправляем батчи чанков во время записи
      if (streamOptions) {
        sentChunkCountRef.current = 0;
        batchIndexRef.current = 0;
        const intervalMs = streamOptions.batchIntervalSeconds * 1000;
        batchTimerRef.current = window.setInterval(() => {
          const curr = chunksRef.current.length;
          const sent = sentChunkCountRef.current;
          // Ждём минимум 5 чанков (5 сек), чтобы webm-контейнер был валидным
          if (curr - sent < 5) return;
          const newChunks = curr === sent ? [] : chunksRef.current.slice(sent);
          if (newChunks.length === 0) return;
          sentChunkCountRef.current = curr;
          // Батч 0 содержит header, остальные батчи префиксируем chunk[0] (webm header)
          const batchChunks = sent === 0 ? newChunks : [chunksRef.current[0], ...newChunks];
          const blob = new Blob(batchChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
          streamOptions.onBatch(blob, mediaRecorder.mimeType || 'audio/webm', batchIndexRef.current++);
        }, intervalMs);
      }

      setState({
        isRecording: true,
        isPaused: false,
        duration: 0,
        audioBlob: null,
      });
    } catch (error) {
      console.error('Error accessing microphone:', error);
      throw new Error('Не удалось получить доступ к микрофону');
    }
  }, [pickSupportedMimeType, updateDuration]);

  const pauseRecording = useCallback(() => {
    if (mediaRecorderRef.current && state.isRecording && !state.isPaused) {
      mediaRecorderRef.current.pause();
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      setState((prev) => ({ ...prev, isPaused: true }));
    }
  }, [state.isRecording, state.isPaused]);

  const resumeRecording = useCallback(() => {
    if (mediaRecorderRef.current && state.isRecording && state.isPaused) {
      mediaRecorderRef.current.resume();
      timerRef.current = window.setInterval(updateDuration, 1000);
      setState((prev) => ({ ...prev, isPaused: false }));
    }
  }, [state.isRecording, state.isPaused, updateDuration]);

  const stopRecording = useCallback((trimSeconds = 0) => {
    if (mediaRecorderRef.current && state.isRecording) {
      // Each chunk is ~1 second (start(1000)), so trimSeconds ≈ chunks to drop
      trimChunksRef.current = trimSeconds;

      if (batchTimerRef.current) {
        clearInterval(batchTimerRef.current);
        batchTimerRef.current = null;
      }

      mediaRecorderRef.current.stop();

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      setState((prev) => ({ ...prev, isRecording: false, isPaused: false }));
    }
  }, [state.isRecording]);

  const resetRecording = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (batchTimerRef.current) {
      clearInterval(batchTimerRef.current);
      batchTimerRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    mediaRecorderRef.current = null;
    chunksRef.current = [];
    sentChunkCountRef.current = 0;
    batchIndexRef.current = 0;

    setState({
      isRecording: false,
      isPaused: false,
      duration: 0,
      audioBlob: null,
    });
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (batchTimerRef.current) {
        clearInterval(batchTimerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
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
    formattedDuration: formatDuration(state.duration),
    startRecording,
    pauseRecording,
    resumeRecording,
    stopRecording,
    resetRecording,
  };
}
