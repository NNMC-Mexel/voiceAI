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
  // Flag: microphone permission denied — stop all restart attempts
  const permissionDeniedRef = useRef(false);
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
    return /(?<![а-яёa-z])стоп[\s\-]*нави(?![а-яёa-z])/i.test(text) ||
      /(?<![а-яёa-z])stop[\s\-]*navi(?![а-яёa-z])/i.test(text);
  };

  const containsWakePhrase = (text: string): boolean => {
    return /(?<![а-яёa-z])нави(?![а-яёa-z])/i.test(text) ||
      /(?<![а-яёa-z])navi(?![a-z])/i.test(text);
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
      if (event.error === 'not-allowed') {
        if (isRecordingRef.current) {
          // Mic busy with MediaRecorder — expected, will retry after recording ends
          return;
        }
        // Actual permission denied — stop all attempts
        console.warn('[WakeWord] Microphone permission denied — wake word disabled until page reload');
        permissionDeniedRef.current = true;
        return;
      }
      console.warn('[WakeWord] Recognition error:', event.error);
    };

    recognition.onend = () => {
      setIsListening(false);
      if (enabledRef.current && !permissionDeniedRef.current) {
        restartTimeoutRef.current = window.setTimeout(() => {
          if (enabledRef.current && !permissionDeniedRef.current) {
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

  // When recording ends, force-restart speech recognition after a delay
  // (MediaRecorder may have taken exclusive mic access, causing recognition to die)
  const prevRecordingRef = useRef(isRecording);
  useEffect(() => {
    const wasRecording = prevRecordingRef.current;
    prevRecordingRef.current = isRecording;
    if (wasRecording && !isRecording && enabledRef.current) {
      console.log('[WakeWord] Recording ended, scheduling recognition restart...');
      setTimeout(() => {
        if (enabledRef.current && !isRecordingRef.current) {
          console.log('[WakeWord] Restarting recognition after recording ended');
          createAndStart();
        }
      }, 1000);
    }
  }, [isRecording, createAndStart]);

  useEffect(() => {
    if (enabled && isSupported) {
      // Reset permission denied flag when user re-enables wake word
      permissionDeniedRef.current = false;
      createAndStart();
    } else {
      stop();
    }
    return () => stop();
  }, [enabled, isSupported, createAndStart, stop]);

  // Watchdog: if enabled but not listening, force restart every 5 seconds
  const isListeningRef = useRef(isListening);
  useEffect(() => { isListeningRef.current = isListening; }, [isListening]);
  useEffect(() => {
    if (!enabled || !isSupported) return;
    const watchdog = setInterval(() => {
      // Skip if permission denied or mic busy with recording
      if (permissionDeniedRef.current || isRecordingRef.current) return;
      if (enabledRef.current && !isListeningRef.current) {
        console.log('[WakeWord] Watchdog: not listening, restarting...');
        createAndStart();
      }
    }, 5000);
    return () => clearInterval(watchdog);
  }, [enabled, isSupported, createAndStart]);

  return { isListening, isSupported };
}
