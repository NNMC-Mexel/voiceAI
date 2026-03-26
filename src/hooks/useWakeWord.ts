import { useEffect, useRef, useCallback, useState } from 'react';

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

interface SpeechRecognitionInstance extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
}

declare global {
  interface Window {
    SpeechRecognition: new () => SpeechRecognitionInstance;
    webkitSpeechRecognition: new () => SpeechRecognitionInstance;
  }
}

const WAKE_PHRASES = ['нави', 'navi'];
const STOP_PHRASES = ['стоп нави', 'stop navi', 'стоп-нави', 'stop-navi'];

// Cooldown after action to prevent re-triggering from buffered results
const ACTION_COOLDOWN_MS = 3000;

interface UseWakeWordOptions {
  enabled: boolean;
  isRecording: boolean;
  onWakeWord: () => void;
  onStopWord: () => void;
}

export function useWakeWord({ enabled, isRecording, onWakeWord, onStopWord }: UseWakeWordOptions) {
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);
  const restartTimeoutRef = useRef<number | null>(null);
  const enabledRef = useRef(enabled);
  const lastActionTimeRef = useRef(0);
  // Flag: stop already fired this recording session, don't fire again
  const stopFiredRef = useRef(false);
  const [isListening, setIsListening] = useState(false);
  const [isSupported] = useState(() => {
    return typeof window !== 'undefined' &&
      !!(window.SpeechRecognition || window.webkitSpeechRecognition);
  });

  const onWakeWordRef = useRef(onWakeWord);
  const onStopWordRef = useRef(onStopWord);
  const isRecordingRef = useRef(isRecording);

  useEffect(() => { onWakeWordRef.current = onWakeWord; }, [onWakeWord]);
  useEffect(() => { onStopWordRef.current = onStopWord; }, [onStopWord]);
  useEffect(() => {
    isRecordingRef.current = isRecording;
    // Reset stopFired when recording starts
    if (isRecording) {
      stopFiredRef.current = false;
    }
  }, [isRecording]);
  useEffect(() => { enabledRef.current = enabled; }, [enabled]);

  const containsStopPhrase = (text: string): boolean => {
    for (const phrase of STOP_PHRASES) {
      if (text.includes(phrase)) return true;
    }
    return false;
  };

  const containsWakePhrase = (text: string): boolean => {
    for (const phrase of WAKE_PHRASES) {
      if (text.includes(phrase)) return true;
    }
    return false;
  };

  const createAndStart = useCallback(() => {
    if (!isSupported) return;

    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.abort();
      recognitionRef.current = null;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'ru-RU';
    recognition.maxAlternatives = 3;

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      // Cooldown — ignore results too soon after last action
      const now = Date.now();
      if (now - lastActionTimeRef.current < ACTION_COOLDOWN_MS) {
        return;
      }

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const isFinal = result.isFinal;

        for (let j = 0; j < result.length; j++) {
          const transcript = result[j].transcript.toLowerCase().trim();
          if (!transcript) continue;

          if (isRecordingRef.current) {
            // RECORDING state: detect stop phrase
            // Use INTERIM results for fast response — but fire only ONCE per session
            if (stopFiredRef.current) continue;

            if (containsStopPhrase(transcript)) {
              stopFiredRef.current = true;
              lastActionTimeRef.current = Date.now();
              console.log('[WakeWord] Stop phrase detected:', transcript, isFinal ? '(final)' : '(interim)');
              onStopWordRef.current();
              return;
            }
          } else {
            // IDLE state: detect wake phrase
            // Skip if transcript contains a stop phrase (to avoid "стоп нави" → "нави" match)
            if (containsStopPhrase(transcript)) continue;

            // Only trigger on FINAL results for accuracy
            if (!isFinal) continue;

            if (containsWakePhrase(transcript)) {
              lastActionTimeRef.current = Date.now();
              console.log('[WakeWord] Wake phrase detected (final):', transcript);
              onWakeWordRef.current();
              return;
            }
          }
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === 'no-speech' || event.error === 'aborted') return;
      console.warn('[WakeWord] Recognition error:', event.error);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (enabledRef.current) {
        restartTimeoutRef.current = window.setTimeout(() => {
          if (enabledRef.current) {
            createAndStart();
          }
        }, 300);
      }
    };

    recognitionRef.current = recognition;

    try {
      recognition.start();
    } catch {
      // Already started — ignore
    }
  }, [isSupported]);

  const stop = useCallback(() => {
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.abort();
      recognitionRef.current = null;
    }
    setIsListening(false);
  }, []);

  useEffect(() => {
    if (enabled && isSupported) {
      createAndStart();
    } else {
      stop();
    }
    return () => stop();
  }, [enabled, isSupported, createAndStart, stop]);

  return { isListening, isSupported };
}
