import React, { useState, useRef, useCallback } from 'react';
import './App.css';

function App() {
  const [beatUrl, setBeatUrl] = useState(null);
  const beatRef = useRef(null);
  const waveCanvasRef = useRef(null);
  const waveformImageRef = useRef(null);
  const rafRef = useRef(null);
  const visualizerContextRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioInputs, setAudioInputs] = useState([]);
  const [audioOutputs, setAudioOutputs] = useState([]);
  const [selectedInputId, setSelectedInputId] = useState(null);
  const [selectedOutputId, setSelectedOutputId] = useState(null);
  const [sinkSupported] = useState(typeof HTMLMediaElement !== 'undefined' && 'setSinkId' in HTMLMediaElement.prototype);
  
  const BEATS = [
    { id: 'beat1', name: 'Beat 1 (100 BPM)', url: process.env.PUBLIC_URL + '/beats/beat1.wav', beats: [0, 2.43809524, 4.82975057, 7.2214059, 9.63628118, 12.02793651, 14.41959184, 16.83446712] },
    { id: 'beat2', name: 'Beat 2 (100 BPM)', url: process.env.PUBLIC_URL + '/beats/beat2.wav',  beats: [0, 2.43809524, 4.82975057, 7.2214059, 9.63628118, 12.02793651, 14.41959184, 16.83446712]  },
  ];

  const PX_PER_SECOND = 200;
  const MIN_CANVAS_WIDTH = 600;
  const MAX_CANVAS_WIDTH = 6000;
  const MAX_SECONDS = Math.floor(MAX_CANVAS_WIDTH / PX_PER_SECOND);

  const [proportions, setProportions] = useState({
    angry: 0.0,
    sad: 0.0,
    witty: 0.0,
    neutral: 1.0,
  });
  // Refs to always access the most recent slider values when sending requests
  const proportionsRef = useRef(proportions);
  React.useEffect(() => { proportionsRef.current = proportions; }, [proportions]);
  const [voices, setVoices] = useState({
    eminem: 0.0,
    nicki: 0.0,
    default: 1.0,
  });
  const voicesRef = useRef(voices);
  React.useEffect(() => { voicesRef.current = voices; }, [voices]);

  const mediaRecorderRef = useRef(null);
  const shimonAudioRef = useRef(null);
  const chunksRef = useRef([]);
  const turnIntervalRef = useRef(null); // You might not need this anymore if not auto-alternating
  const [recording, setRecording] = useState(false);
  const [recordedUrl, setRecordedUrl] = useState(null);
  const [countdown, setCountdown] = useState(null);
  const [currentTurn, setCurrentTurn] = useState(null); // Can be "Your turn!"
  // Removed downbeat scheduling; recording restarts immediately after Shimon finishes

  // --- NEW STATE VARIABLES ---
  const [threadId, setThreadId] = useState(null);
  const [round, setRound] = useState(0); // To count turns
  const [isGenerating, setIsGenerating] = useState(false); // Loading state
  const [shimonResponseUrl, setShimonResponseUrl] = useState(null); // For the AI's rap
  const [shimonPlaybackStartTime, setShimonPlaybackStartTime] = useState(null); // Beat timestamp when Shimon should start
  const shimonPlaybackTimeoutRef = useRef(null); // Timeout for scheduled playback
  const humanStartTimeoutRef = useRef(null); // Timeout for scheduling human start at next downbeat
  const [shimonTranscription, setShimonTranscription] = useState(null); // Last transcription from backend header

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      if (turnIntervalRef.current) clearInterval(turnIntervalRef.current);
      if (visualizerContextRef.current) {
        try { visualizerContextRef.current.close(); } catch (_) {}
      }
      if (shimonPlaybackTimeoutRef.current) {
        try { clearTimeout(shimonPlaybackTimeoutRef.current); } catch (_) {}
      }
      if (humanStartTimeoutRef.current) {
        try { clearTimeout(humanStartTimeoutRef.current); } catch (_) {}
      }
      // No auto-stop timeout to clear
    };
  }, []);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      setAudioInputs(devices.filter(d => d.kind === 'audioinput'));
      setAudioOutputs(devices.filter(d => d.kind === 'audiooutput'));
    } catch (e) {
      console.warn('enumerateDevices failed', e);
    }
  }, []);

  React.useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  React.useEffect(() => {
    if (!sinkSupported) return;
    const el = beatRef.current;
    if (!el || !selectedOutputId) return;
    el.setSinkId(selectedOutputId).catch(err => console.warn('setSinkId failed', err));
  }, [sinkSupported, selectedOutputId]);

  // --- NEW ---
  // This function handles the API call to your FastAPI backend
  const uploadAndGenerateRap = async (audioBlob) => {
    setIsGenerating(true); // Show loading indicator
    setShimonResponseUrl(null); // Clear previous response
    setRecordedUrl(null); // Clear user's recording display (optional)

    let currentThreadId = threadId;
    if (!currentThreadId) {
      // Create a new thread ID for this session
      const newThreadId = `web_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
      setThreadId(newThreadId);
      currentThreadId = newThreadId;
    }

    // Create a FormData object to send the file and data
    const formData = new FormData();
    formData.append("recording", audioBlob, "recording.webm");
    
    // Append the *most up-to-date* slider values (use refs to avoid stale closures)
    const latestTones = proportionsRef.current || proportions;
    const latestVoices = voicesRef.current || voices;
    formData.append("tones", JSON.stringify(latestTones));
    formData.append("voices", JSON.stringify(latestVoices));
    
    // Append session and turn data
    formData.append("thread_id", currentThreadId);
    formData.append("turn", round);

    // Log FormData contents for debugging
    console.log("=== FormData contents ===");
    for (let [key, value] of formData.entries()) {
      if (value instanceof Blob) {
        console.log(key, ":", `[Blob: ${value.type}, ${value.size} bytes]`);
      } else {
        console.log(key, ":", value);
      }
    }
    console.log("========================");

    try {
      // Replace with your server's IP and port
      const response = await fetch("/generate", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errText}`);
      }


      // Record current beat timestamp when response arrives
      const currentBeatTime = (beatRef.current && !isNaN(beatRef.current.currentTime)) ? beatRef.current.currentTime : 0;
      console.log('Current beat time on response:', currentBeatTime);

      // Find next downbeat from beat list
      const beatMeta = BEATS.find(b => b.url === beatUrl);
      const downbeats = beatMeta && Array.isArray(beatMeta.beats) ? beatMeta.beats : [];
      let nextDownbeat = downbeats.find(t => t > currentBeatTime);
      
      // Handle wrap-around if no future downbeat in current loop
      if (nextDownbeat === undefined && downbeats.length > 0) {
        nextDownbeat = downbeats[0];
      }
      
      console.log('Next downbeat for Shimon playback:', nextDownbeat);

      // Get the generated .wav file as a blob
      const responseAudioBlob = await response.blob();
      const responseAudioUrl = URL.createObjectURL(responseAudioBlob);

      const raw = response.headers.get('X-Rap-Transcript')

      let transcript = null;
      if (raw) {
        const bytes = Uint8Array.from(atob(raw), c => c.charCodeAt(0));
        transcript = new TextDecoder().decode(bytes);
      }

      console.log("Generated rap transcript:", transcript);
      setShimonTranscription(transcript);

      // Set the URL and next downbeat time
      setShimonResponseUrl(responseAudioUrl);
      setShimonPlaybackStartTime(nextDownbeat || 0);
      setRound(prevRound => prevRound + 1); // Increment the turn counter

    } catch (error) {
      console.error("Error generating rap:", error);
      alert(`Error generating rap: ${error.message}`);
    } finally {
      setIsGenerating(false); // Hide loading indicator
    }
  };

  // Schedule Shimon's playback to start at the specified beat timestamp
  React.useEffect(() => {
    if (!shimonResponseUrl || shimonPlaybackStartTime === null || !beatRef.current) return;
    
    const el = shimonAudioRef.current;
    if (!el) return;

    // Clear any existing scheduled playback
    if (shimonPlaybackTimeoutRef.current) {
      clearTimeout(shimonPlaybackTimeoutRef.current);
      shimonPlaybackTimeoutRef.current = null;
    }

    const schedulePlayback = () => {
      const currentBeatTime = beatRef.current.currentTime;
      const targetDownbeat = shimonPlaybackStartTime;
      const beatDuration = beatRef.current.duration || 0;
      
      // Calculate delay to next downbeat
      let delay = (targetDownbeat - currentBeatTime) * 1000;
      
      // Handle wrap-around: if target is earlier than current, it's in the next loop
      if (delay < 0 && beatDuration > 0) {
        delay = ((beatDuration - currentBeatTime) + targetDownbeat) * 1000;
      }
      
      // If very close (within 100ms) or past, play immediately
      if (delay < 100) {
        console.log('Playing Shimon audio immediately (downbeat reached)');
        el.play().catch(err => console.warn('Shimon play failed', err));
        return;
      }
      
      console.log(`Scheduling Shimon playback in ${(delay/1000).toFixed(2)}s (next downbeat: ${targetDownbeat.toFixed(3)})`);
      
      shimonPlaybackTimeoutRef.current = setTimeout(() => {
        console.log('Starting Shimon playback at beat time:', beatRef.current?.currentTime);
        el.play().catch(err => console.warn('Shimon play failed', err));
        shimonPlaybackTimeoutRef.current = null;
      }, delay);
    };

    schedulePlayback();

    return () => {
      if (shimonPlaybackTimeoutRef.current) {
        clearTimeout(shimonPlaybackTimeoutRef.current);
        shimonPlaybackTimeoutRef.current = null;
      }
    };
  }, [shimonResponseUrl, shimonPlaybackStartTime]);

  // When Shimon's response finishes playing, immediately start the next human recording (no downbeat wait)
  React.useEffect(() => {
    const el = shimonAudioRef.current;
    if (!el) return;
    const onEnded = () => {
      if (recording || !beatUrl || !beatRef.current) return;

      const currentBeatTime = !isNaN(beatRef.current.currentTime) ? (beatRef.current.currentTime || 0) : 0;
      const beatMeta = BEATS.find(b => b.url === beatUrl);
      const downbeats = beatMeta && Array.isArray(beatMeta.beats) ? beatMeta.beats : [];

      // Find the next downbeat strictly greater than current time; wrap to first if none
      let nextDownbeat = downbeats.find(t => t > currentBeatTime);
      if (nextDownbeat === undefined && downbeats.length > 0) {
        nextDownbeat = downbeats[0];
      }

      // If we don't have downbeats, start immediately
      if (nextDownbeat === undefined) {
        startRecording({ skipCountdown: true });
        return;
      }

      const beatDuration = beatRef.current.duration || 0;
      let delay = (nextDownbeat - currentBeatTime) * 1000;
      if (delay < 0 && beatDuration > 0) {
        // Wrap-around to the next loop
        delay = ((beatDuration - currentBeatTime) + nextDownbeat) * 1000;
      }

      // If very close (<100ms), start immediately
      if (delay < 100) {
        startRecording({ skipCountdown: true });
        return;
      }

      // Indicate waiting during the break until human turn starts
      setCurrentTurn('Waiting...');

      // Clear any prior scheduled human start
      if (humanStartTimeoutRef.current) {
        clearTimeout(humanStartTimeoutRef.current);
        humanStartTimeoutRef.current = null;
      }

      humanStartTimeoutRef.current = setTimeout(() => {
        startRecording({ skipCountdown: true });
        humanStartTimeoutRef.current = null;
      }, delay);
    };
    el.addEventListener('ended', onEnded);
    return () => {
      try { el.removeEventListener('ended', onEnded); } catch (_) {}
    };
  }, [shimonResponseUrl, recording, beatUrl]);


  // --- UNCHANGED HELPER FUNCTIONS ---
  // (drawWaveformFromUrl, drawPlayhead, startRaf, stopRaf, clearWaveform)
  // (updateToneProportion, updateVoiceProportion)
  // (useEffect for beat playback, useEffect for slider resizing)
  // ... All those functions stay exactly the same ...
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

  const drawPlayhead = useCallback(() => {
    const canvas = waveCanvasRef.current;
    const ctx = canvas && canvas.getContext ? canvas.getContext('2d') : null;
    const img = waveformImageRef.current;
    if (!canvas || !ctx || !img) return;
    try {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    } catch (err) {
      return;
    }
    const audio = beatRef.current;
    if (!audio || !audio.duration || isNaN(audio.duration)) return;
    const t = Math.max(0, Math.min(audio.currentTime || 0, audio.duration));
    const x = Math.floor((t / audio.duration) * canvas.width);
    ctx.save();
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x + 0.5, 0);
    ctx.lineTo(x + 0.5, canvas.height);
    ctx.stroke();
    ctx.restore();
  }, []);

  const startRaf = useCallback(() => {
    if (rafRef.current) return;
    const loop = () => {
      drawPlayhead();
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, [drawPlayhead]);

  const stopRaf = useCallback(() => {
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

  const updateToneProportion = (tone, pct) => {
    if (tone === 'neutral') return;
    const snappedPct = Math.max(0, Math.min(100, Math.round(pct / 10) * 10));
    const val = snappedPct / 100;
    const a = proportions.angry || 0;
    const s = proportions.sad || 0;
    const w = proportions.witty || 0;
    const current = { angry: a, sad: s, witty: w };
    const sumOthers = Object.entries(current)
      .filter(([k]) => k !== tone)
      .reduce((sum, [, v]) => sum + v, 0);
    const maxForTone = Math.max(0, 1 - sumOthers);
    const capped = Math.min(val, maxForTone);
    const nextNonNeutralSum = sumOthers + capped;
    const nextNeutral = Math.max(0, 1 - nextNonNeutralSum);
    setProportions((p) => ({
      ...p,
      angry: tone === 'angry' ? capped : (p.angry || 0),
      sad: tone === 'sad' ? capped : (p.sad || 0),
      witty: tone === 'witty' ? capped : (p.witty || 0),
      neutral: nextNeutral,
    }));
  };

  const updateVoiceProportion = (key, pct) => {
    if (key === 'default') return;
    const snappedPct = Math.max(0, Math.min(100, Math.round(pct / 10) * 10));
    const val = snappedPct / 100;
    const current = {
      eminem: voices.eminem || 0,
      nicki: voices.nicki || 0,
    };
    const sumOthers = Object.entries(current)
      .filter(([k]) => k !== key)
      .reduce((sum, [, v]) => sum + v, 0);
    const maxForKey = Math.max(0, 1 - sumOthers);
    const capped = Math.min(val, maxForKey);
    const nextNonDefaultSum = sumOthers + capped;
    const nextDefault = Math.max(0, 1 - nextNonDefaultSum);
    setVoices((v) => ({
      ...v,
      eminem: key === 'eminem' ? capped : (v.eminem || 0),
      nicki: key === 'nicki' ? capped : (v.nicki || 0),
      default: nextDefault,
    }));
  };

  React.useEffect(() => {
      const audio = beatRef.current;
      if (!audio) return;
      const onPlay = () => { setIsPlaying(true); startRaf(); };
      const onPause = () => { setIsPlaying(false); stopRaf(); };
      audio.addEventListener('play', onPlay);
      audio.addEventListener('pause', onPause);
      return () => {
        try {
          audio.removeEventListener('play', onPlay);
          audio.removeEventListener('pause', onPause);
        } catch (_) {}
      };
    }, [beatUrl, startRaf, stopRaf]);

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
  
  // --- MODIFIED ---
  // This function now just handles the countdown and *starting* the recording.
  async function startRecording(opts = {}) {
    const { skipCountdown = false } = opts;
    if (recording) return;

    // Clear any previous recordings
    setRecordedUrl(null);
    setShimonResponseUrl(null);
    // No downbeat scheduling to cancel

    let countdownDelayMs = 0;
    if (!skipCountdown) {
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
      countdownDelayMs = 3000; // original countdown length
    } else {
      setCountdown(null);
      setCurrentTurn('Preparing');
    }

    setTimeout(async () => {
      try {
        // Request mic access first; some browsers pause audio during prompt
        const audioConstraints = {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        };
        if (selectedInputId) {
          audioConstraints.deviceId = { exact: selectedInputId };
        }
        const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });

        // Refresh devices post-permission to get labels
        refreshDevices();

        // Firefox/Safari can suspend AudioContexts after getUserMedia
        if (visualizerContextRef.current && visualizerContextRef.current.state === 'suspended') {
          try { await visualizerContextRef.current.resume(); } catch (_) {}
        }

        // Ensure beat is playing and (if not already aligned by scheduler) align to next downbeat before starting recorder
        if (beatRef.current) {
          try {
            // Play if paused
            if (beatRef.current.paused) {
              if (isNaN(beatRef.current.currentTime)) beatRef.current.currentTime = 0;
              beatRef.current.muted = false;
              beatRef.current.volume = 0.2;
              await beatRef.current.play();
            }
            // No alignment to downbeat; start immediately
          } catch (err) {
            console.warn('Beat alignment failed', err);
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
        
        // --- MODIFIED onstop HANDLER ---
        // This is where the upload is triggered
        mr.onstop = () => {
          const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
          const url = URL.createObjectURL(blob);
          setRecordedUrl(url); // Show user their recording
          
          // --- THIS IS GOAL 2 & 3 ---
          // Call the FastAPI app with the audio and slider data
          uploadAndGenerateRap(blob);
          
          stream.getTracks().forEach((t) => t.stop()); // Stop mic
        };
        
        mr.start();
        setRecording(true);
        setCurrentTurn("Your turn!"); // Show "Your turn!"

        // Remove the auto-alternating turn interval
        if (turnIntervalRef.current) {
          clearInterval(turnIntervalRef.current);
        }
        // We don't need the interval anymore, user stops manually
        // turnIntervalRef.current = setInterval(() => { ... });

      } catch (err) {
        console.error(err);
        alert('Microphone access denied or unavailable');
        setCountdown(null);
        setCurrentTurn(null);
      }
    }, countdownDelayMs); // Wait for countdown if not skipped
  }

  // (downbeat scheduler removed)

  // --- MODIFIED ---
  // This function is called when the user clicks "Stop Recording"
  const stopRecording = () => {
    if (!recording) return;
    setRecording(false);
    // No timeout clearing necessary
    
    // Stop the media recorder
    // This will automatically trigger the 'mr.onstop' handler we defined
    // which then triggers the uploadAndGenerateRap function.
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    if (turnIntervalRef.current) {
      clearInterval(turnIntervalRef.current);
      turnIntervalRef.current = null;
    }
    
    setCurrentTurn("Shimon's turn!");
  };

  return (
    <div className="App rap-container">
      <div className="debug-banner">App mounted</div>
      <h1 className="rap-title">Shimon Raps</h1>

      <div className="top-panels">
        {/* ... Left Panel (Beats) ... */}
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
                         setTimeout(() => {
                           drawWaveformFromUrl(b.url);
                         }, 50);
                       }}
                     />
                     <div className="beat-name">{b.name}</div>
                   </div>
                   <audio src={b.url} controls preload="none" className="beat-preview" />
                 </div>
               ))}
             </div>
           </div>
         </div>
        {/* ... Right Panel (Mixers) ... */}
        <div className="panel right-panel">
           <div className="mixers-row">
             <div className="mixer-group">
               <div className="mixer-title">Emotion mix</div>
               <div className="sliders-column sliders-row">
               {/* Angry */}
               <div className="slider-box">
                 <div className="slider-inner">
                   <div className="slider-row vertical-row">
                     <div className="vslider-wrap">
                       <input
                         type="range"
                         min={0}
                         max={100}
                         step={10}
                         value={Math.round((proportions.angry || 0) * 100)}
                         onChange={(e) => updateToneProportion('angry', Number(e.target.value))}
                         className="vertical-range"
                       />
                     </div>
                     <div className="slider-pct">{Math.round((proportions.angry || 0) * 100)}%</div>
                   </div>
                   <div className="slider-caption">Angry</div>
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
               {/* Sad */}
               <div className="slider-box">
                 <div className="slider-inner">
                   <div className="slider-row vertical-row">
                     <div className="vslider-wrap">
                       <input
                         type="range"
                         min={0}
                         max={100}
                         step={10}
                         value={Math.round((proportions.sad || 0) * 100)}
                         onChange={(e) => updateToneProportion('sad', Number(e.target.value))}
                         className="vertical-range"
                       />
                     </div>
                     <div className="slider-pct">{Math.round((proportions.sad || 0) * 100)}%</div>
                   </div>
                   <div className="slider-caption">Sad</div>
                 </div>
               </div>
               {/* Neutral */}
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
                 {/* Eminem */}
                 <div className="slider-box">
                   <div className="slider-inner">
                     <div className="slider-row vertical-row">
                       <div className="vslider-wrap">
                         <input
                           type="range"
                           min={0}
                           max={100}
                           step={10}
                           value={Math.round((voices.eminem || 0) * 100)}
                           onChange={(e) => updateVoiceProportion('eminem', Number(e.target.value))}
                           className="vertical-range"
                         />
                       </div>
                       <div className="slider-pct">{Math.round((voices.eminem || 0) * 100)}%</div>
                     </div>
                      <div className="slider-caption">Eminem</div>
                   </div>
                 </div>
                 {/* nicki */}
                 <div className="slider-box">
                   <div className="slider-inner">
                     <div className="slider-row vertical-row">
                       <div className="vslider-wrap">
                         <input
                           type="range"
                           min={0}
                           max={100}
                           step={10}
                           value={Math.round((voices.nicki || 0) * 100)}
                           onChange={(e) => updateVoiceProportion('nicki', Number(e.target.value))}
                           className="vertical-range"
                         />
                       </div>
                       <div className="slider-pct">{Math.round((voices.nicki || 0) * 100)}%</div>
                     </div>
                     <div className="slider-caption">Nicki</div>
                   </div>
                 </div>
                 {/* Default (computed) */}
                 <div className="slider-box">
                   <div className="slider-inner">
                     <div className="slider-row vertical-row">
                       <div className="vslider-wrap">
                         <input
                           type="range"
                           min={0}
                           max={100}
                           step={10}
                           value={Math.round((voices.default || 0) * 100)}
                           disabled
                           className="vertical-range"
                         />
                       </div>
                       <div className="slider-pct">{Math.round((voices.default || 0) * 100)}%</div>
                     </div>
                     <div className="slider-caption">Kendrick</div>
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
                <button
                  className="play-btn"
                  onClick={async () => {
                    if (!beatRef.current) return;
                    try {
                      if (beatRef.current.paused) {
                        await beatRef.current.play();
                      } else {
                        beatRef.current.pause();
                      }
                    } catch (err) {
                      console.warn('Play failed', err);
                    }
                  }}
                >
                  {isPlaying ? '⏸ Pause' : '▶ Play'}
                </button>
              </div>
        </div>
      )}
      
      {/* --- MODIFIED --- Controls Row */}
      <div className="device-row">
        <label>
          Mic:
          <select value={selectedInputId || ''} onChange={(e) => setSelectedInputId(e.target.value || null)}>
            <option value="">System default</option>
            {audioInputs.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label || d.deviceId}</option>
            ))}
          </select>
        </label>
        <label>
          Output:
          <select value={selectedOutputId || ''} onChange={(e) => setSelectedOutputId(e.target.value || null)} disabled={!sinkSupported}>
            <option value="">System default</option>
            {audioOutputs.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label || d.deviceId}</option>
            ))}
          </select>
          {!sinkSupported && <span className="note"> (output selection not supported)</span>}
        </label>
      </div>
      <div className="controls-row">
        {countdown ? (
          <div className="countdown">{countdown}</div>
        ) : recording ? (
          <button onClick={stopRecording} className="recording-btn">Stop Recording</button>
        ) : isGenerating ? (
          <div className="loading-indicator">Shimon is thinking...</div>
        ) : (
          // Disable button if no beat is selected
          <button onClick={startRecording} className="recording-btn" disabled={!beatUrl}>
            {beatUrl ? "Start Recording" : "Select a beat first"}
          </button>
        )}
        {currentTurn && <div className="turn-indicator">{currentTurn}</div>}
      </div>
      {/* Turn length input removed; manual stop controls duration */}

      {/* --- MODIFIED --- Show your recording */}
      {recordedUrl && !isGenerating && (
        <div className="recorded-section">
          <h3>Your recording</h3>
          <audio src={recordedUrl} controls />
          <div>
            <a href={recordedUrl} download="rap_recording.webm">Download</a>
          </div>
        </div>
      )}
      
      {/* --- NEW --- Show Shimon's response */}
      {shimonResponseUrl && (
        <div className="recorded-section shimon-response">
          <h3>Shimon's Response</h3>
          {shimonTranscription && (
            <div className="transcription-box">
              <strong>Transcription:</strong>
              <div className="transcription-text">{shimonTranscription}</div>
            </div>
          )}
          <audio ref={shimonAudioRef} src={shimonResponseUrl} controls />
        </div>
      )}

      {/* ... Tone/Voice Mix display ... */}
      <div className="tone-mix tone-mix-centered">
        <div>Tone mix:</div>
        <div className="tone-mix-list">
          {['angry','sad','witty','neutral'].map((t) => (
            <div key={t} className="tone-mix-item">
              {t}: {Math.round((proportions[t] || 0) * 100)}%
            </div>
          ))}
        </div>
      </div>
      <div className="tone-mix tone-mix-centered">
        <div>Voice mix:</div>
        <div className="tone-mix-list">
          {['eminem','nicki','default'].map((v) => (
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