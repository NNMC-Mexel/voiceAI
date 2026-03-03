import { useEffect, useRef, useState } from 'react';

interface WaveformVisualizerProps {
  isActive: boolean;
  isPaused: boolean;
}

export function WaveformVisualizer({ isActive, isPaused }: WaveformVisualizerProps) {
  const [bars] = useState(() => Array.from({ length: 32 }, () => Math.random()));
  const animationRef = useRef<number[]>(
    bars.map(() => Math.random() * 0.5 + 0.5)
  );

  useEffect(() => {
    if (isActive && !isPaused) {
      const interval = setInterval(() => {
        animationRef.current = animationRef.current.map(
          () => Math.random() * 0.7 + 0.3
        );
      }, 100);
      return () => clearInterval(interval);
    }
  }, [isActive, isPaused]);

  return (
    <div className="flex items-center justify-center gap-1 h-16 px-4">
      {bars.map((_, index) => {
        const delay = index * 0.05;
        const height = isActive && !isPaused 
          ? `${animationRef.current[index] * 100}%`
          : '30%';

        return (
          <div
            key={index}
            className={`w-1.5 rounded-full transition-all duration-150 ${
              isActive
                ? isPaused
                  ? 'bg-amber-400'
                  : 'bg-gradient-to-t from-medical-500 to-medical-300'
                : 'bg-slate-300'
            }`}
            style={{
              height,
              animationDelay: `${delay}s`,
              opacity: isActive ? 1 : 0.5,
            }}
          />
        );
      })}
    </div>
  );
}
