import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [beatUrl, setBeatUrl] = useState(null);
  const beatRef = useRef(null);
  const waveCanvasRef = useRef(null);
  const waveformImageRef = useRef(null);
  const rafRef = useRef(null);
  const visualizerContextRef = useRef(null);
  const playbackContextRef = useRef(null);
  const playbackSourceRef = useRef(null);
  // removed manual play/pause control; beat auto-plays when selected or recording starts
  // predefined beats list
  const BEATS = [
    { id: 'boom-bap', name: 'Boom Bap (90 BPM)', url: process.env.PUBLIC_URL + '/beats/beat1.wav', profile: { mood: 'Classic, punchy', energy: 'Medium', swing: 'Tight' } },
    { id: 'trap-140', name: 'Trap (140 BPM)', url: process.env.PUBLIC_URL + '/beats/trap-140.wav', profile: { mood: 'Dark, modern', energy: 'High', swing: 'Triplet hats' } },
    { id: 'lofi-80', name: 'Lo-Fi (80 BPM)', url: process.env.PUBLIC_URL + '/beats/lofi-80.wav', profile: { mood: 'Chill, warm', energy: 'Low', swing: 'Loose' } },
    { id: 'house-124', name: 'House (124 BPM)', url: process.env.PUBLIC_URL + '/beats/house-124.wav', profile: { mood: 'Uplifting', energy: 'High', swing: 'Four-on-the-floor' } },
  ];
  // waveform sizing parameters
  const PX_PER_SECOND = 200; // horizontal pixels per second of audio
  const MIN_CANVAS_WIDTH = 600; // px
  const MAX_CANVAS_WIDTH = 6000; // px
  const MAX_SECONDS = Math.floor(MAX_CANVAS_WIDTH / PX_PER_SECOND);
  const [tones] = useState(['aggressive', 'witty', 'sarcastic', 'neutral']);
  const [selectedTones, setSelectedTones] = useState({
    aggressive: false,
    witty: false,
    sarcastic: false,
    neutral: true,
  });
  const [proportions, setProportions] = useState({
    aggressive: 0.0,
    witty: 0.0,
    sarcastic: 0.0,
    neutral: 1.0,
  });
  // Voice mix: other voices take from your voice (remainder)
  const [voices, setVoices] = useState({
    rapperA: 0.0,
    rapperB: 0.0,
    rapperC: 0.0,
    yourVoice: 1.0,
  });
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const turnIntervalRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const [currentTurn, setCurrentTurn] = useState(null);
  const turnDuration = 5; // Duration of each turn in seconds
  const [threadId, setThreadId] = useState(null);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      // Clear intervals
      if (turnIntervalRef.current) {
        clearInterval(turnIntervalRef.current);
      }
      
      // Close audio contexts
      if (visualizerContextRef.current) {
        visualizerContextRef.current.close();
      }
      if (playbackContextRef.current) {
        playbackContextRef.current.close();
      }
    };
  }, []);




  // URL by fetching the audio
  const drawWaveformFromUrl = async (url) => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    try {
      const res = await fetch(url);
      const arrayBuffer = await res.arrayBuffer();
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!visualizerContextRef.current) {
        visualizerContextRef.current = new AudioCtx();
      }
      const ac = visualizerContextRef.current;
      const audioBuffer = await ac.decodeAudioData(arrayBuffer.slice(0));
      const channelData = audioBuffer.numberOfChannels > 0 ? audioBuffer.getChannelData(0) : null;
      if (!channelData) return;
      const duration = audioBuffer.duration || (channelData.length / (audioBuffer.sampleRate || 44100));
      if (duration > MAX_SECONDS) {
        alert(`Selected beat is too long. Maximum allowed length is ${MAX_SECONDS} seconds.`);
        try { setBeatUrl(null); } catch (_) {}
        clearWaveform();
        try { stopRaf(); } catch (_) {}
        return;
      }
      const desiredWidth = Math.max(MIN_CANVAS_WIDTH, Math.min(MAX_CANVAS_WIDTH, Math.ceil(duration * PX_PER_SECOND)));
      const dpr = window.devicePixelRatio || 1;
      const height = canvas.clientHeight || 100;
      canvas.width = Math.floor(desiredWidth * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${desiredWidth}px`;
      canvas.style.height = `${height}px`;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const step = Math.ceil(channelData.length / canvas.width);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 1 * dpr;
      ctx.strokeStyle = '#000000';
      ctx.beginPath();
      const midY = canvas.height / 2;
      for (let i = 0; i < canvas.width; i++) {
        const start = i * step;
        let min = 1.0;
        let max = -1.0;
        for (let j = 0; j < step && start + j < channelData.length; j++) {
          const v = channelData[start + j];
          if (v < min) min = v;
          if (v > max) max = v;
        }
        const x = i;
        const y1 = midY + (min * midY);
        const y2 = midY + (max * midY);
        ctx.moveTo(x, y1);
        ctx.lineTo(x, y2);
      }
      ctx.stroke();
      try {
        const dataUrl = canvas.toDataURL();
        const img = new Image();
        img.src = dataUrl;
        waveformImageRef.current = img;
      } catch (err) {
        console.warn('Could not create waveform image', err);
      }
    } catch (err) {
      console.warn('Waveform draw from URL failed', err);
    }
  };

  const drawPlayhead = React.useCallback(() => {
    const canvas = waveCanvasRef.current;
    const ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
    const img = waveformImageRef.current;
    if (!canvas || !ctx || !img) return;
    // draw base waveform image
    try {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    } catch (err) {
      // image may not be ready yet
      return;
    }
    // compute playhead x
    const audio = beatRef.current;
    if (!audio || !audio.duration || isNaN(audio.duration)) return;
    const t = Math.max(0, Math.min(audio.currentTime || 0, audio.duration));
    const x = Math.floor((t / audio.duration) * canvas.width);
    // draw playhead
    ctx.save();
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, canvas.height);
    ctx.stroke();
    ctx.restore();
  }, []);

  const startRaf = React.useCallback(() => {
    if (rafRef.current) return;
    const loop = () => {
      drawPlayhead();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, [drawPlayhead]);

  const stopRaf = React.useCallback(() => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    drawPlayhead();
  }, [drawPlayhead]);

  const clearWaveform = () => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    waveformImageRef.current = null;
  };

  // update a non-neutral tone proportion in 10% steps; neutrality fills the remainder
  const updateToneProportion = (tone, pct) => {
    if (tone === 'neutral') return; // neutrality is computed, not directly set
    const snappedPct = Math.max(0, Math.min(100, Math.round(pct / 10) * 10));
    const val = snappedPct / 100;
    // current values
    const a = proportions.aggressive || 0;
    const w = proportions.witty || 0;
    const s = proportions.sarcastic || 0;
    const current = { aggressive: a, witty: w, sarcastic: s };
    // sum others excluding the tone being changed
    const sumOthers = Object.entries(current)
      .filter(([k]) => k !== tone)
      .reduce((sum, [, v]) => sum + v, 0);
    // cap this tone so total never exceeds 1.0
    const maxForTone = Math.max(0, 1 - sumOthers);
    const capped = Math.min(val, maxForTone);
    const nextNonNeutralSum = sumOthers + capped;
    const nextNeutral = Math.max(0, 1 - nextNonNeutralSum);
    setProportions((p) => ({
      ...p,
      aggressive: tone === 'aggressive' ? capped : (p.aggressive || 0),
      witty: tone === 'witty' ? capped : (p.witty || 0),
      sarcastic: tone === 'sarcastic' ? capped : (p.sarcastic || 0),
      neutral: nextNeutral,
    }));
  };

  // update a non-yourVoice proportion; yourVoice is computed as remainder
  const updateVoiceProportion = (key, pct) => {
    if (key === 'yourVoice') return;
    const snappedPct = Math.max(0, Math.min(100, Math.round(pct / 10) * 10));
    const val = snappedPct / 100;
    const current = {
      rapperA: voices.rapperA || 0,
      rapperB: voices.rapperB || 0,
      rapperC: voices.rapperC || 0,
    };
    const sumOthers = Object.entries(current)
      .filter(([k]) => k !== key)
      .reduce((sum, [, v]) => sum + v, 0);
    const maxForKey = Math.max(0, 1 - sumOthers);
    const capped = Math.min(val, maxForKey);
    const nextNonYourSum = sumOthers + capped;
    const nextYour = Math.max(0, 1 - nextNonYourSum);
    setVoices((v) => ({
      ...v,
      rapperA: key === 'rapperA' ? capped : (v.rapperA || 0),
      rapperB: key === 'rapperB' ? capped : (v.rapperB || 0),
      rapperC: key === 'rapperC' ? capped : (v.rapperC || 0),
      yourVoice: nextYour,
    }));
  };

  // start animation frame loop whenever a new beat URL is set and can play
  React.useEffect(() => {
    if (!beatUrl) return;
    const attemptPlay = () => {
      if (!beatRef.current) return;
      try {
        beatRef.current.play().then(() => {
          startRaf();
        }).catch(() => {});
      } catch (_) {}
    };
    const t = setTimeout(attemptPlay, 120);
    return () => clearTimeout(t);
  }, [beatUrl, startRaf]);

  // Resize observer to make vertical sliders fill their boxes
  React.useEffect(() => {
    if (typeof ResizeObserver === 'undefined') return;
    const updateHeightVar = (el) => {
      try {
        const h = el.getBoundingClientRect().height;
        if (h > 0) {
          el.style.setProperty('--bar-height', `${Math.floor(h)}px`);
        }
      } catch (_) {}
    };
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        updateHeightVar(entry.target);
      }
    });
    // Wait for DOM to be ready
    setTimeout(() => {
      const nodes = document.querySelectorAll('.vslider-wrap');
      nodes.forEach((n) => {
        updateHeightVar(n);
        ro.observe(n);
      });
    }, 100);
    return () => {
      try { ro.disconnect(); } catch (_) {}
    };
  }, []);

  const startRecording = async () => {
    if (recording) return;

    // Start countdown from 3
    setCountdown(3);
    setCurrentTurn('Countdown');

    const countdownTimer = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownTimer);
          return null;
        }
        return prev - 1;
      });
    }, 1000);

    // Wait for countdown to finish
    setTimeout(async () => {
      try {
        // Set up beat playback with its own AudioContext
        if (beatRef.current) {
          beatRef.current.currentTime = 0;

          try {
            // Create new playback context if needed
            if (!playbackContextRef.current) {
              const AudioCtx = window.AudioContext || window.webkitAudioContext;
              playbackContextRef.current = new AudioCtx();
            }

            // Resume the context if it's suspended
            if (playbackContextRef.current.state === 'suspended') {
              await playbackContextRef.current.resume();
            }

            // Create and connect audio source only once
            if (!playbackSourceRef.current) {
              const source = playbackContextRef.current.createMediaElementSource(beatRef.current);
              source.connect(playbackContextRef.current.destination);
              playbackSourceRef.current = source;
            }

            // Set audio routing hints
            beatRef.current.mozAudioChannelType = 'content'; // Firefox
            beatRef.current.preservesPitch = false; // Hint for mobile
            
            // Play the beat with full routing setup
            await beatRef.current.play();
          } catch (err) {
            console.warn('Initial beat play failed:', err);
          }
        }

        // Initialize recording with completely separate context and constraints
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
            latency: 0,
            sampleRate: 48000,
            channelCount: 1,
            // Ensure we're using a different audio device routing
            googAudioMirroring: false,
            googAutoGainControl: false,
            googAutoGainControl2: false,
            googDucking: false,
            googHighpassFilter: false,
            googNoiseSuppression: false,
            googTypingNoiseDetection: false
          }
        });

        // Set stream audio routing priority
        if (stream.getAudioTracks().length > 0) {
          const track = stream.getAudioTracks()[0];
          // Request concurrent audio capture/playback
          if (track.getSettings) {
            const settings = track.getSettings();
            if (settings.deviceId) {
              try {
                await track.applyConstraints({
                  advanced: [{ deviceId: settings.deviceId }]
                });
              } catch (err) {
                console.warn('Could not apply advanced constraints:', err);
              }
            }
          }
        }
        
        const mr = new MediaRecorder(stream, {
          mimeType: 'audio/webm;codecs=opus',
          audioBitsPerSecond: 128000
        });
        mediaRecorderRef.current = mr;
        chunksRef.current = [];
        
        mr.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) chunksRef.current.push(e.data);
        };
        
        mr.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
          const url = URL.createObjectURL(blob);
          setRecordedUrl(url);
          // stop microphone tracks
          stream.getTracks().forEach((t) => t.stop());
        };
        mr.start();
        setRecording(true);
        
        // Clear any existing interval
        if (turnIntervalRef.current) {
          clearInterval(turnIntervalRef.current);
        }

        // Start with user's turn
        setCurrentTurn("Your turn!");

        // Create an interval to alternate turns
        turnIntervalRef.current = setInterval(() => {
          setCurrentTurn(current => current === "Your turn!" ? "Shimon's turn!" : "Your turn!");
        }, turnDuration * 1000);

        // Make sure beat is still playing
        if (beatRef.current && beatRef.current.paused) {
          const playBeatWithRetry = async () => {
            try {
              await beatRef.current.play();
            } catch (err) {
              if (err.name === 'NotAllowedError') {
                setTimeout(playBeatWithRetry, 100);
              } else {
                console.warn('Beat playback failed:', err);
              }
            }
          };
          await playBeatWithRetry();
        }

      } catch (err) {
        console.error(err);
        alert('Microphone access denied or unavailable');
        setCountdown(null);
        setCurrentTurn(null);
      }
    }, 3000); // Wait for 3 second countdown
  };

  const stopRecording = () => {
    if (!recording) return;
    setRecording(false);
    
    // Stop the media recorder
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    // Clear the turn interval
    if (turnIntervalRef.current) {
      clearInterval(turnIntervalRef.current);
      turnIntervalRef.current = null;
    }
    
    setCurrentTurn(null); // Clear the turn indicator when stopping
  };

  return (
    <div className="App rap-container">
      <div className="debug-banner">App mounted</div>
      <h1 className="rap-title">Shimon Raps</h1>

      <div className="top-panels">
        <div className="panel left-panel">
          <div className="beats-picker">
            <div className="beats-list">
              {BEATS.map((b) => (
                <div key={b.id} className={"beat-card" + (beatUrl === b.url ? ' selected' : '')}>
                  <div className="beat-header">
                    <input
                      type="radio"
                      name="selectedBeat"
                      checked={beatUrl === b.url}
                      onChange={async () => {
                        if (beatRef.current) {
                          try { beatRef.current.pause(); } catch (_) {}
                        }
                        setBeatUrl(b.url);
                        // draw waveform shortly after setting URL
                        setTimeout(() => {
                          drawWaveformFromUrl(b.url);
                        }, 50);
                      }}
                    />
                    <div className="beat-name">{b.name}</div>
                  </div>
                  <div className="beat-profile">
                    <span>{b.profile.mood}</span>
                    <span> • </span>
                    <span>Energy: {b.profile.energy}</span>
                    <span> • </span>
                    <span>{b.profile.swing}</span>
                  </div>
                  <audio src={b.url} controls preload="none" className="beat-preview" />
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="panel right-panel">
          <div className="mixers-row">
            <div className="mixer-group">
              <div className="mixer-title">Emotion mix</div>
              <div className="sliders-column sliders-row">
              {/* Aggressive */}
              <div className="slider-box">
                <div className="slider-inner">
                  <div className="slider-row vertical-row">
                    <div className="vslider-wrap">
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={10}
                        value={Math.round((proportions.aggressive || 0) * 100)}
                        onChange={(e) => updateToneProportion('aggressive', Number(e.target.value))}
                        className="vertical-range"
                      />
                    </div>
                    <div className="slider-pct">{Math.round((proportions.aggressive || 0) * 100)}%</div>
                  </div>
                  <div className="slider-caption">Aggressive</div>
                </div>
              </div>

              {/* Witty */}
              <div className="slider-box">
                <div className="slider-inner">
                  <div className="slider-row vertical-row">
                    <div className="vslider-wrap">
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={10}
                        value={Math.round((proportions.witty || 0) * 100)}
                        onChange={(e) => updateToneProportion('witty', Number(e.target.value))}
                        className="vertical-range"
                      />
                    </div>
                    <div className="slider-pct">{Math.round((proportions.witty || 0) * 100)}%</div>
                  </div>
                  <div className="slider-caption">Witty</div>
                </div>
              </div>

              {/* Sarcastic */}
              <div className="slider-box">
                <div className="slider-inner">
                  <div className="slider-row vertical-row">
                    <div className="vslider-wrap">
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={10}
                        value={Math.round((proportions.sarcastic || 0) * 100)}
                        onChange={(e) => updateToneProportion('sarcastic', Number(e.target.value))}
                        className="vertical-range"
                      />
                    </div>
                    <div className="slider-pct">{Math.round((proportions.sarcastic || 0) * 100)}%</div>
                  </div>
                  <div className="slider-caption">Sarcastic</div>
                </div>
              </div>

              {/* Neutral (computed) */}
              <div className="slider-box">
                <div className="slider-inner">
                  <div className="slider-row vertical-row">
                    <div className="vslider-wrap">
                      <input
                        type="range"
                        min={0}
                        max={100}
                        step={10}
                        value={Math.round((proportions.neutral || 0) * 100)}
                        disabled
                        className="vertical-range"
                      />
                    </div>
                    <div className="slider-pct">{Math.round((proportions.neutral || 0) * 100)}%</div>
                  </div>
                  <div className="slider-caption">Neutral</div>
                </div>
              </div>
            </div>
            </div>
            <div className="mixer-group">
              <div className="mixer-title">Voice mix</div>
              <div className="sliders-column sliders-row">
                {/* Rapper A */}
                <div className="slider-box">
                  <div className="slider-inner">
                    <div className="slider-row vertical-row">
                      <div className="vslider-wrap">
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={10}
                          value={Math.round((voices.rapperA || 0) * 100)}
                          onChange={(e) => updateVoiceProportion('rapperA', Number(e.target.value))}
                          className="vertical-range"
                        />
                      </div>
                      <div className="slider-pct">{Math.round((voices.rapperA || 0) * 100)}%</div>
                    </div>
                    <div className="slider-caption">Rapper A</div>
                  </div>
                </div>

                {/* Rapper B */}
                <div className="slider-box">
                  <div className="slider-inner">
                    <div className="slider-row vertical-row">
                      <div className="vslider-wrap">
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={10}
                          value={Math.round((voices.rapperB || 0) * 100)}
                          onChange={(e) => updateVoiceProportion('rapperB', Number(e.target.value))}
                          className="vertical-range"
                        />
                      </div>
                      <div className="slider-pct">{Math.round((voices.rapperB || 0) * 100)}%</div>
                    </div>
                    <div className="slider-caption">Rapper B</div>
                  </div>
                </div>

                {/* Rapper C */}
                <div className="slider-box">
                  <div className="slider-inner">
                    <div className="slider-row vertical-row">
                      <div className="vslider-wrap">
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={10}
                          value={Math.round((voices.rapperC || 0) * 100)}
                          onChange={(e) => updateVoiceProportion('rapperC', Number(e.target.value))}
                          className="vertical-range"
                        />
                      </div>
                      <div className="slider-pct">{Math.round((voices.rapperC || 0) * 100)}%</div>
                    </div>
                    <div className="slider-caption">Rapper C</div>
                  </div>
                </div>

                {/* Your Voice (computed) */}
                <div className="slider-box">
                  <div className="slider-inner">
                    <div className="slider-row vertical-row">
                      <div className="vslider-wrap">
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={10}
                          value={Math.round((voices.yourVoice || 0) * 100)}
                          disabled
                          className="vertical-range"
                        />
                      </div>
                      <div className="slider-pct">{Math.round((voices.yourVoice || 0) * 100)}%</div>
                    </div>
                    <div className="slider-caption">Your Voice</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {beatUrl && (
        <div className="beat-row">
          <audio ref={beatRef} src={beatUrl} loop className="hidden-audio" />
          <div className="player-row">
            <div className="waveform-canvas">
              <canvas ref={waveCanvasRef} className="wave-canvas" />
            </div>
          </div>
        </div>
      )}

      

      <div className="controls-row">
        {countdown ? (
          <div className="countdown">{countdown}</div>
        ) : recording ? (
          <button onClick={stopRecording} className="recording-btn">Stop Recording</button>
        ) : (
          <button onClick={startRecording} className="recording-btn">Start Recording</button>
        )}
        {currentTurn && <div className="turn-indicator">{currentTurn}</div>}
      </div>

      {recordedUrl && (
        <div className="recorded-section">
          <h3>Your recording</h3>
          <audio src={recordedUrl} controls />
          <div>
            <a href={recordedUrl} download="rap_recording.webm">Download</a>
          </div>
        </div>
      )}

      <div className="tone-mix tone-mix-centered">
        <div>Tone mix:</div>
        <div className="tone-mix-list">
          {['aggressive','witty','sarcastic','neutral'].map((t) => (
            <div key={t} className="tone-mix-item">
              {t}: {Math.round((proportions[t] || 0) * 100)}%
            </div>
          ))}
        </div>
      </div>

      <div className="tone-mix tone-mix-centered">
        <div>Voice mix:</div>
        <div className="tone-mix-list">
          {['rapperA','rapperB','rapperC','yourVoice'].map((v) => (
            <div key={v} className="tone-mix-item">
              {v}: {Math.round((voices[v] || 0) * 100)}%
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
