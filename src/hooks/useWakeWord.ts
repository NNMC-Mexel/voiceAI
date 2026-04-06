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
  // Flag: real microphone permission denied by the browser — stop all restart attempts
  const permissionDeniedRef = useRef(false);
  // Deduplicate interim wake word triggers — store last fired transcript
  const lastWakeTranscriptRef = useRef('');
  // Consecutive not-allowed count — drives exponential backoff (resets on successful onstart)
  const notAllowedCountRef = useRef(0);
  // Restart delay for next onend (ms) — set by onerror, consumed by onend
  const pendingRestartDelayRef = useRef(300);

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
    if (isRecording) {
      stopFiredRef.current = false;
      lastWakeTranscriptRef.current = '';
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
      console.log('[WakeWord] Recognition started');
      notAllowedCountRef.current = 0;
      pendingRestartDelayRef.current = 300;
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
            // RECORDING state: detect stop phrase on interim OR final — fire only once
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

            if (!containsWakePhrase(transcript)) continue;

            // Accept INTERIM results for faster response, but deduplicate:
            // don't fire again if this transcript is same/subset of what already fired
            const alreadyFired = lastWakeTranscriptRef.current &&
              transcript.includes(lastWakeTranscriptRef.current);
            if (alreadyFired) continue;

            // On final result — always fire (clears interim dedup)
            // On interim — fire immediately but record transcript to avoid double-fire
            lastWakeTranscriptRef.current = isFinal ? '' : transcript;
            lastActionTimeRef.current = Date.now();
            console.log('[WakeWord] Wake phrase detected:', transcript, isFinal ? '(final)' : '(interim)');
            onWakeWordRef.current();
            return;
          }
        }
      }
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === 'no-speech' || event.error === 'aborted') return;

      if (event.error === 'not-allowed') {
        notAllowedCountRef.current++;
        // Exponential backoff: 300ms → 600 → 1200 → 2400 → max 10s
        // Never permanently disabled — mic may be transitioning from MediaRecorder
        const delay = Math.min(300 * Math.pow(2, notAllowedCountRef.current - 1), 10000);
        pendingRestartDelayRef.current = delay;
        if (notAllowedCountRef.current <= 3) {
          console.log(`[WakeWord] not-allowed #${notAllowedCountRef.current}, retrying in ${delay}ms (mic transitioning)`);
        } else {
          console.warn(`[WakeWord] not-allowed #${notAllowedCountRef.current}, retrying in ${delay}ms — check mic permissions`);
        }
        return;
      }

      console.warn('[WakeWord] Recognition error:', event.error);
    };

    recognition.onend = () => {
      setIsListening(false);
      const willRestart = enabledRef.current && !permissionDeniedRef.current;
      if (willRestart) {
        const delay = pendingRestartDelayRef.current;
        pendingRestartDelayRef.current = 300; // reset for next session
        restartTimeoutRef.current = window.setTimeout(() => {
          if (enabledRef.current && !permissionDeniedRef.current) {
            createAndStart();
          }
        }, delay);
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

  // When recording ends, force-restart speech recognition after a delay.
  // MediaRecorder may have taken exclusive mic access, causing recognition to die silently.
  const prevRecordingRef = useRef(isRecording);
  useEffect(() => {
    const wasRecording = prevRecordingRef.current;
    prevRecordingRef.current = isRecording;
    if (wasRecording && !isRecording && enabledRef.current) {
      console.log('[WakeWord] Recording ended, scheduling recognition restart...');
      setTimeout(() => {
        if (enabledRef.current && !isRecordingRef.current && !permissionDeniedRef.current) {
          console.log('[WakeWord] Restarting recognition after recording ended');
          createAndStart();
        }
      }, 1500);
    }
  }, [isRecording, createAndStart]);

  // Enable/disable effect — resets state on re-enable
  useEffect(() => {
    if (enabled && isSupported) {
      permissionDeniedRef.current = false;
      lastWakeTranscriptRef.current = '';
      notAllowedCountRef.current = 0;
      pendingRestartDelayRef.current = 300;
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
