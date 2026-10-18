import React, { useState, useEffect, useRef } from 'react';
import { Download, Share2, AlertCircle, Loader2, Copy, Check, Wand2, Type, Languages, FileText, Scissors, Crosshair, TrendingUp, ImagePlus, X } from 'lucide-react';
import { getApiUrl } from '../config';
import { apiFetch } from '../lib/api';
import SubtitleModal from './SubtitleModal';
import HookModal from './HookModal';
import TranslateModal from './TranslateModal';
import Modal from './ui/Modal';
import WatermarkModal, { watermarkNoticeDismissed } from './WatermarkModal';
import { useAuth } from '../contexts/AuthContext';
import { renderInBrowser } from '../lib/renderInBrowser';

const QUIET_BTN = 'group flex flex-col items-center justify-center gap-1 py-2.5 sm:py-2 px-1 rounded-input border border-rule hover:bg-paper3 text-[11px] lowercase text-ink2 whitespace-nowrap transition-colors disabled:opacity-45 disabled:cursor-not-allowed';

function clipDurationSeconds(clip) {
    // A recut clip's start/end are the covering source range (segments may be
    // non-contiguous or reordered); its real duration is the segment sum.
    const segments = clip.recipe?.segments;
    if (segments?.length) {
        return segments.reduce((acc, s) => acc + (s.end - s.start), 0);
    }
    return clip.end && clip.start ? clip.end - clip.start : NaN;
}

function formatDuration(clip) {
    const secs = Math.floor(clipDurationSeconds(clip));
    if (!Number.isFinite(secs) || secs < 0) return null;
    return `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`;
}

export default function ResultCard({ clip, index, jobId, durable, geminiApiKey, elevenLabsKey, isManaged, postizConfigured, postizPostType, clipAiThumbnailsEnabled, onPlay, onPause, onBulkSubtitle, clipCount = 1, bulkProgress, initialState = null, onStateChange, onEditClip = null, onReframeClip = null, onUpgrade = null }) {
    // The "why" line is clamped to two lines so cards in a row stay level;
    // when it overflows, a hover (desktop) or tap (touch) shows the whole
    // sentence in a popover that floats over the card instead of growing it.
    const whyRef = useRef(null);
    const [whyOpen, setWhyOpen] = useState(false);
    const [whyClamped, setWhyClamped] = useState(false);
    useEffect(() => {
        const el = whyRef.current;
        if (!el) return;
        const check = () => setWhyClamped(el.scrollHeight > el.clientHeight + 1);
        check();
        window.addEventListener('resize', check);
        return () => window.removeEventListener('resize', check);
    }, [clip.why]);
    const [showDescModal, setShowDescModal] = useState(false);
    const [showSubtitleModal, setShowSubtitleModal] = useState(false);
    const [showWatermarkModal, setShowWatermarkModal] = useState(false);
    const { plan } = useAuth();
    const videoRef = React.useRef(null);
    // Pristine base clip (no burned subtitles/hook), stable regardless of how
    // clip.video_url mutates after server edits. Used as the compositing base
    // for the Remotion preview so it never stacks subtitles over an already-
    // subtitled file (double-subtitle bug).
    const stripBurns = (filename) => {
        let f = filename || '', prev;
        do { prev = f; f = f.replace(/^subtitled_\d+_/, '').replace(/^hooked_\d+_/, '').replace(/^hook_/, ''); } while (f !== prev);
        return f;
    };
    const originalVideoUrl = getApiUrl((clip.video_url || '').replace(/[^/]+$/, stripBurns((clip.video_url || '').split('/').pop())));
    const [currentVideoUrl, setCurrentVideoUrl] = useState(getApiUrl(clip.video_url));
    // Where the <video> element pulls its bytes from. The clips are archived to
    // R2 anyway, and R2 egress is free and edge-served, while /videos is served
    // by the same single-worker API process that is running the renders. So play
    // from R2 when possible.
    //
    // ONLY when R2 holds exactly the file the server considers current for this
    // clip (durable.filename === serverVideoFile). Every server-side edit sends
    // input_filename: serverVideoFile and rewrites clip.video_url, but the R2
    // re-archive behind it is fire-and-forget, so for a few seconds the durable
    // copy is the PRE-edit clip. Preferring it blindly would silently show the
    // clip without the subtitles/hook the user just burned.
    //
    // This is display-only: currentVideoUrl remains the source of truth for the
    // download button and for every server operation, so no edit can be routed
    // to the wrong file by this.
    const [durableSrc, setDurableSrc] = useState(null);
    const [durableFailed, setDurableFailed] = useState(false);
    // Switching src reloads the element and restarts playback, so the durable copy
    // is only ever adopted before the user has touched this player. After an edit
    // the chase in App.jsx can land while they are watching the result, and losing
    // their position to save a few seconds of buffering is a bad trade.
    const [hasPlayed, setHasPlayed] = useState(false);

    // A delivered clip is tens of MB, and on a slow link the old silent
    // fetch-then-save took minutes with nothing on screen, which reads as a dead
    // button. Stream it instead and report progress.
    const [downloadPct, setDownloadPct] = useState(null);

    const downloadClip = async () => {
        try {
            setDownloadPct(0);
            const response = await fetch(currentVideoUrl);
            if (!response.ok) throw new Error('Download failed');
            const total = Number(response.headers.get('content-length')) || 0;
            let blob;
            // No body reader (old browser) or no length to measure against: fall
            // back to the plain path rather than lose the download.
            if (!response.body || !total) {
                blob = await response.blob();
            } else {
                const reader = response.body.getReader();
                const chunks = [];
                let received = 0;
                for (;;) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    received += value.length;
                    setDownloadPct(Math.min(99, Math.round((received / total) * 100)));
                }
                blob = new Blob(chunks, { type: 'video/mp4' });
            }
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `clip-${index + 1}.mp4`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            console.error('Download error:', err);
            window.open(currentVideoUrl, '_blank');
        } finally {
            setDownloadPct(null);
        }
    };
    // Latest file that exists ON THE SERVER (blob: previews don't count).
    // All server-side operations must chain from this, so burned-in edits
    // (subtitles, hooks, effects) never get silently dropped.
    // A reopened project seeds it from the persisted project state.
    const [serverVideoFile, setServerVideoFile] = useState(initialState?.server_file || (clip.video_url || '').split('/').pop());
    const [videoErrored, setVideoErrored] = useState(false);
    const [resolution, setResolution] = useState(null);

    // Adopt the durable copy only while it matches the current server file, and
    // pin the first signed URL seen for that file: /api/history mints a fresh
    // signature on every call, and swapping src mid-playback restarts the video.
    useEffect(() => {
        if (durable?.url && durable.filename && durable.filename === serverVideoFile && !hasPlayed) {
            setDurableSrc((prev) => prev || durable.url);
        } else {
            setDurableSrc(null);
        }
    }, [durable?.url, durable?.filename, serverVideoFile, hasPlayed]);

    // If the local video failed and a durable R2 URL is (now) available, use it.
    // Handles the race where the video errors before the durable URL has loaded.
    // Deliberately NOT version-gated: reaching here means the local file is gone
    // (retention sweep after a reload), so an older durable copy still beats a
    // broken player.
    useEffect(() => {
        if (videoErrored && durable?.url && currentVideoUrl !== durable.url) {
            setCurrentVideoUrl(durable.url);
            setVideoErrored(false);
        }
    }, [videoErrored, durable, currentVideoUrl]);

    // When an external refresh changes this clip's server file (e.g. bulk
    // subtitles applied from another card), adopt it so the card shows the
    // freshly subtitled video instead of a stale one.
    useEffect(() => {
        const serverUrl = getApiUrl(clip.video_url);
        const serverName = (clip.video_url || '').split('/').pop();
        if (serverName && serverName !== serverVideoFile) {
            setServerVideoFile(serverName);
            setCurrentVideoUrl(serverUrl);
            setDurableFailed(false);
            setHasPlayed(false);
            if (videoRef.current) videoRef.current.load();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [clip.video_url]);

    const [postizSending, setPostizSending] = useState(false);
    const [postizResult, setPostizResult] = useState(null);
    const postizThumbnailInputRef = useRef(null);
    const [postizThumbnail, setPostizThumbnail] = useState(null);
    const [showAiThumbnailModal, setShowAiThumbnailModal] = useState(false);
    const [aiThumbnailInstructions, setAiThumbnailInstructions] = useState('');
    const [aiThumbnailCount, setAiThumbnailCount] = useState(1);
    const [aiThumbnailState, setAiThumbnailState] = useState(null);
    const [aiThumbnailLoading, setAiThumbnailLoading] = useState(false);
    const [aiThumbnailError, setAiThumbnailError] = useState(null);
    const [copied, setCopied] = useState(null);

    useEffect(() => {
        setPostizResult(null);
        setPostizThumbnail(null);
        setAiThumbnailState(null);
        setAiThumbnailError(null);
        if (postizThumbnailInputRef.current) postizThumbnailInputRef.current.value = '';
    }, [clip.video_url]);

    const loadAiThumbnailState = async () => {
        try {
            const res = await apiFetch(`/api/clip-thumbnail/${jobId}/${index}`);
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || 'Could not load AI thumbnail details.');
            setAiThumbnailState(data);
            setAiThumbnailError(null);
            return data;
        } catch (e) {
            setAiThumbnailError(e.message || 'Could not load AI thumbnail details.');
            return null;
        }
    };

    useEffect(() => {
        if (!clipAiThumbnailsEnabled || isManaged) return undefined;
        void loadAiThumbnailState();
        // The current rendered filename is part of the server-side thumbnail
        // version key. Reload state after a subtitle/edit/reframe replacement.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [jobId, index, clip.video_url, clipAiThumbnailsEnabled, isManaged]);

    const handleCopy = async (field, text) => {
        try {
            await navigator.clipboard.writeText(text || '');
            setCopied(field);
            setTimeout(() => setCopied(null), 2000);
        } catch {
            // clipboard unavailable — silent
        }
    };

    const [isEditing, setIsEditing] = useState(false);
    const [isSubtitling, setIsSubtitling] = useState(false);
    const [isHooking, setIsHooking] = useState(false);
    const [isTranslating, setIsTranslating] = useState(false);
    const [showHookModal, setShowHookModal] = useState(false);
    const [showTranslateModal, setShowTranslateModal] = useState(false);
    const [editError, setEditError] = useState(null);

    const [clipDuration, setClipDuration] = useState(() => {
        const secs = clipDurationSeconds(clip);
        return Number.isFinite(secs) ? secs : 30;
    });

    // Accumulate Remotion layers across operations. A reopened project restores
    // the layers persisted in its project state, so the next edit composes over
    // them instead of silently dropping previous browser-side work.
    const [activeLayers, setActiveLayers] = useState(initialState?.active_layers || { subtitles: null, hook: null, effects: null });

    // Report edit state upward (debounced sync to the project record). Skip the
    // mount run: only user-driven changes are worth persisting.
    const stateReported = React.useRef(false);
    useEffect(() => {
        if (!stateReported.current) { stateReported.current = true; return; }
        onStateChange?.(index, { activeLayers, serverVideoFile });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [activeLayers, serverVideoFile]);

    // True when the current server file already carries burned-in content.
    // Browser (Remotion) renders compose over the ORIGINAL clip, so using them
    // here would silently drop those burns — chain via server FFmpeg instead.
    const hasServerBurns = /(^|_)(subtitled|hook|hooked)_/.test(serverVideoFile || '');

    // The hook currently burned into the server file (auto-hook or a manual
    // one). /api/hook REPLACES it; tracked locally so the modal stays honest
    // after edits without refetching the job.
    const [burnedHook, setBurnedHook] = useState(clip.auto_hook?.text || null);

    // Fetch clip duration from transcript endpoint
    useEffect(() => {
        if (!jobId || index === undefined) return;
        apiFetch(`/api/clip/${jobId}/${index}/transcript`)
            .then(res => res.ok ? res.json() : null)
            .then(data => {
                if (data && data.durationSec) setClipDuration(data.durationSec);
            })
            .catch(() => {});
    }, [jobId, index]);

    const handleAutoEdit = async () => {
        setIsEditing(true);
        setEditError(null);
        try {
            const apiKey = geminiApiKey || localStorage.getItem('gemini_key');

            // Managed (paid) users get the Gemini key resolved server-side;
            // only BYOK/self-host needs a local key.
            if (!apiKey && !isManaged) {
                throw new Error("Gemini API Key is missing. Please set it in Settings.");
            }
            const geminiHeaders = apiKey ? { 'X-Gemini-Key': apiKey } : {};

            // Try Remotion effects endpoint first
            const effectsRes = hasServerBurns ? null : await apiFetch('/api/effects/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...geminiHeaders
                },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: index,
                    input_filename: serverVideoFile
                })
            });

            if (effectsRes && effectsRes.ok) {
                const data = await effectsRes.json();
                if (data.effects && data.effects.segments) {
                    const newLayers = { ...activeLayers, effects: data.effects };
                    setActiveLayers(newLayers);
                    const blobUrl = await renderInBrowser({
                        videoUrl: originalVideoUrl,
                        durationInSeconds: clipDuration,
                        subtitles: newLayers.subtitles,
                        hook: newLayers.hook,
                        effects: newLayers.effects,
                    });
                    setCurrentVideoUrl(blobUrl);
                    if (videoRef.current) videoRef.current.load();
                    return;
                }
            }

            // Fallback: legacy FFmpeg edit endpoint
            const res = await apiFetch('/api/edit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...geminiHeaders
                },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: index,
                    input_filename: serverVideoFile
                })
            });

            if (!res.ok) {
                const errText = await res.text();
                try {
                    const jsonErr = JSON.parse(errText);
                    throw new Error(jsonErr.detail || errText);
                } catch (e) {
                    throw new Error(errText);
                }
            }

            const data = await res.json();
            if (data.new_video_url) {
                setCurrentVideoUrl(getApiUrl(data.new_video_url));
                setServerVideoFile(data.new_video_url.split('/').pop());
                if (videoRef.current) {
                    videoRef.current.load();
                }
            }

        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsEditing(false);
        }
    };

    // Clips are captioned by default, so "no captions" has to be reachable.
    // Nothing is re-encoded: the server still holds the clean file next to the
    // captioned one and just points this clip back at it.
    const handleRemoveSubtitles = async () => {
        setIsSubtitling(true);
        setEditError(null);
        try {
            const res = await apiFetch('/api/subtitle/remove', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId, clip_index: index, input_filename: serverVideoFile,
                }),
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            if (data.new_video_url) {
                const serverUrl = getApiUrl(data.new_video_url);
                setServerVideoFile(data.new_video_url.split('/').pop());
                const remaining = { ...activeLayers, subtitles: null };
                setActiveLayers(remaining);
                if (remaining.hook || remaining.effects) {
                    setCurrentVideoUrl(await renderInBrowser({
                        videoUrl: serverUrl,
                        durationInSeconds: clipDuration,
                        subtitles: null,
                        hook: remaining.hook,
                        effects: remaining.effects,
                    }));
                } else {
                    setCurrentVideoUrl(serverUrl);
                }
                if (videoRef.current) videoRef.current.load();
                setShowSubtitleModal(false);
            }
        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsSubtitling(false);
        }
    };

    const handleSubtitle = async (options) => {
        setIsSubtitling(true);
        setEditError(null);
        try {
            // Karaoke styles are burned server-side (ASS word-highlight render);
            // the in-browser Remotion path only handles classic styles, and only
            // when the server file has no burned-in content to preserve.
            if (options.remotion && options.style !== 'karaoke' && !hasServerBurns) {
                // Accumulate layer and render all layers together
                const newLayers = { ...activeLayers, subtitles: options.remotion };
                setActiveLayers(newLayers);
                const blobUrl = await renderInBrowser({
                    videoUrl: originalVideoUrl,
                    durationInSeconds: clipDuration,
                    subtitles: newLayers.subtitles,
                    hook: newLayers.hook,
                    effects: newLayers.effects,
                });
                setCurrentVideoUrl(blobUrl);
                if (videoRef.current) videoRef.current.load();
                setShowSubtitleModal(false);
                return;
            }

            // Fallback: legacy FFmpeg
            const res = await apiFetch('/api/subtitle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: index,
                    position: options.position,
                    font_size: options.fontSize,
                    font_name: options.fontName,
                    font_color: options.fontColor,
                    border_color: options.borderColor,
                    border_width: options.borderWidth,
                    bg_color: options.bgColor,
                    bg_opacity: options.bgOpacity,
                    style: options.style || 'classic',
                    highlight_color: options.highlightColor || '#FFD700',
                    effect: options.effect || 'none',
                    base_opacity: options.baseOpacity ?? 1.0,
                    uppercase: options.uppercase || false,
                    reveal: options.reveal || false,
                    shadow: options.shadow || 0,
                    max_chars: options.maxChars ?? null,
                    max_duration: options.maxDuration ?? null,
                    input_filename: serverVideoFile,
                    // Edited caption text (clip-relative ms); null = server
                    // regenerates from the transcript as before.
                    words: options.captions || null
                })
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            if (data.new_video_url) {
                const serverUrl = getApiUrl(data.new_video_url);
                setServerVideoFile(data.new_video_url.split('/').pop());
                // Subtitles are burned into the server file now — drop the
                // browser subtitle layer and re-compose any remaining browser
                // layers (hook/effects) over the new file so they aren't lost.
                const remaining = { ...activeLayers, subtitles: null };
                setActiveLayers(remaining);
                if (remaining.hook || remaining.effects) {
                    const blobUrl = await renderInBrowser({
                        videoUrl: serverUrl,
                        durationInSeconds: clipDuration,
                        subtitles: null,
                        hook: remaining.hook,
                        effects: remaining.effects,
                    });
                    setCurrentVideoUrl(blobUrl);
                } else {
                    setCurrentVideoUrl(serverUrl);
                }
                if (videoRef.current) videoRef.current.load();
                setShowSubtitleModal(false);
            }
        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsSubtitling(false);
        }
    };

    const handleHook = async (hookData) => {
        setIsHooking(true);
        setEditError(null);
        try {
            if (hookData.remotion && !hasServerBurns) {
                // Accumulate layer and render all layers together
                const newLayers = { ...activeLayers, hook: hookData.remotion };
                setActiveLayers(newLayers);
                const blobUrl = await renderInBrowser({
                    videoUrl: originalVideoUrl,
                    durationInSeconds: clipDuration,
                    subtitles: newLayers.subtitles,
                    hook: newLayers.hook,
                    effects: newLayers.effects,
                });
                setCurrentVideoUrl(blobUrl);
                if (videoRef.current) videoRef.current.load();
                setShowHookModal(false);
                return;
            }

            // Fallback: legacy FFmpeg
            const payload = typeof hookData === 'string'
                ? { text: hookData, position: 'top', size: 'M' }
                : hookData;

            const res = await apiFetch('/api/hook', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: index,
                    text: payload.text,
                    position: payload.position,
                    size: payload.size,
                    style: payload.style || 'pill',
                    font: payload.font || null,
                    duration_seconds: payload.remotion?.displayDurationSec ?? null,
                    input_filename: serverVideoFile
                })
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            if (data.new_video_url) {
                setCurrentVideoUrl(getApiUrl(data.new_video_url));
                setServerVideoFile(data.new_video_url.split('/').pop());
                setBurnedHook(data.burned_hook?.text ?? payload.text ?? null);
                if (videoRef.current) videoRef.current.load();
                setShowHookModal(false);
            }
        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsHooking(false);
        }
    };

    // Strip the burned hook (auto-hook or manual) off the server file.
    const handleRemoveHook = async () => {
        setIsHooking(true);
        setEditError(null);
        try {
            const res = await apiFetch('/api/hook', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: index,
                    remove: true,
                    input_filename: serverVideoFile,
                }),
            });
            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            if (data.new_video_url) {
                setCurrentVideoUrl(getApiUrl(data.new_video_url));
                setServerVideoFile(data.new_video_url.split('/').pop());
                setBurnedHook(null);
                if (videoRef.current) videoRef.current.load();
                setShowHookModal(false);
            }
        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsHooking(false);
        }
    };

    const handleTranslate = async (options) => {
        console.log('[Translate] Starting translation with options:', options);
        setIsTranslating(true);
        setEditError(null);
        try {
            const apiKey = elevenLabsKey;
            console.log('[Translate] API Key available:', !!apiKey);

            if (!apiKey) {
                throw new Error("ElevenLabs API Key is missing. Please set it in Settings.");
            }

            const requestBody = {
                job_id: jobId,
                clip_index: index,
                target_language: options.targetLanguage,
                input_filename: serverVideoFile
            };
            console.log('[Translate] Request body:', requestBody);
            console.log('[Translate] Sending request to /api/translate');

            const res = await apiFetch('/api/translate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-ElevenLabs-Key': apiKey
                },
                body: JSON.stringify(requestBody)
            });

            console.log('[Translate] Response status:', res.status);

            if (!res.ok) {
                const errText = await res.text();
                console.error('[Translate] Error response:', errText);
                try {
                    const jsonErr = JSON.parse(errText);
                    throw new Error(jsonErr.detail || errText);
                } catch (e) {
                    if (e.message !== errText) throw e;
                    throw new Error(errText);
                }
            }

            const data = await res.json();
            console.log('[Translate] Success response:', data);
            if (data.new_video_url) {
                setCurrentVideoUrl(getApiUrl(data.new_video_url));
                setServerVideoFile(data.new_video_url.split('/').pop());
                if (videoRef.current) {
                    videoRef.current.load();
                }
                setShowTranslateModal(false);
            }

        } catch (e) {
            console.error('[Translate] Exception:', e);
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsTranslating(false);
        }
    };

    const handleSendToPostiz = async () => {
        if (!postizConfigured) {
            setPostizResult({
                success: false,
                msg: "Configure Postiz in OpenShorts .env, then restart the backend."
            });
            return;
        }
        setPostizSending(true);
        setPostizResult(null);
        try {
            let path = '/api/postiz/publish';
            let options = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_id: jobId, clip_index: index })
            };
            if (postizThumbnail) {
                const form = new FormData();
                form.append('job_id', jobId);
                form.append('clip_index', String(index));
                form.append('thumbnail', postizThumbnail);
                path = '/api/postiz/publish-thumbnail';
                options = { method: 'POST', body: form };
            }
            const res = await apiFetch(path, options);
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || "Could not send this clip to Postiz.");
            setPostizResult({
                success: true,
                reused: data.reused,
                msg: data.message || "Sent to Postiz."
            });
        } catch (e) {
            setPostizResult({ success: false, msg: e.message || "Could not send this clip to Postiz." });
        } finally {
            setPostizSending(false);
        }
    };

    const handlePostizThumbnailChange = (event) => {
        const selected = event.target.files?.[0] || null;
        if (!selected) return;
        if (selected.type && !['image/jpeg', 'image/png'].includes(selected.type)) {
            setPostizResult({ success: false, msg: 'Thumbnail must be a JPEG or PNG image.' });
            event.target.value = '';
            return;
        }
        if (selected.size > 2 * 1024 * 1024) {
            setPostizResult({ success: false, msg: 'Thumbnail image exceeds the 2 MB limit.' });
            event.target.value = '';
            return;
        }
        setPostizResult(null);
        setPostizThumbnail(selected);
        // A browser file wins for this send, but clear the persisted AI
        // selection too so a later send cannot unexpectedly reuse it.
        void selectAiThumbnail(null);
    };

    const selectAiThumbnail = async (candidateId) => {
        try {
            const res = await apiFetch(`/api/clip-thumbnail/${jobId}/${index}/select`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ candidate_id: candidateId }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || 'Could not select this AI thumbnail.');
            setAiThumbnailState(data);
            setAiThumbnailError(null);
            if (candidateId) {
                setPostizThumbnail(null);
                if (postizThumbnailInputRef.current) postizThumbnailInputRef.current.value = '';
            }
        } catch (e) {
            setAiThumbnailError(e.message || 'Could not select this AI thumbnail.');
        }
    };

    const generateAiThumbnail = async () => {
        setAiThumbnailLoading(true);
        setAiThumbnailError(null);
        try {
            const apiKey = geminiApiKey || localStorage.getItem('gemini_key');
            const headers = { 'Content-Type': 'application/json' };
            if (apiKey) headers['X-Gemini-Key'] = apiKey;
            const res = await apiFetch('/api/clip-thumbnail/generate', {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: index,
                    instructions: aiThumbnailInstructions,
                    count: aiThumbnailCount,
                    request_id: crypto.randomUUID(),
                }),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.detail || 'Could not generate an AI thumbnail.');
            setAiThumbnailState(data);
            setShowAiThumbnailModal(false);
            setAiThumbnailInstructions('');
        } catch (e) {
            setAiThumbnailError(e.message || 'Could not generate an AI thumbnail.');
        } finally {
            setAiThumbnailLoading(false);
        }
    };

    const isVideoChanging = isEditing || isSubtitling || isHooking || isTranslating
        || (bulkProgress?.running ?? false);
    const aiThumbnailBusy = aiThumbnailLoading || aiThumbnailState?.hasRunningBatch === true;
    const postizAlreadySent = !!postizResult?.success || aiThumbnailState?.postizLocked === true;

    // Browser-rendered previews (Remotion) live in a blob: URL that exists only
    // in this tab, so they always win over the durable copy.
    const playbackUrl = (durableSrc && !durableFailed && !String(currentVideoUrl || '').startsWith('blob:'))
        ? durableSrc
        : currentVideoUrl;

    const durationReadout = formatDuration(clip);

    return (
        <div className="card overflow-hidden flex flex-col md:flex-row group hover:border-rule2 transition-colors animate-fade md:min-h-[420px]" style={{ animationDelay: `${index * 0.1}s` }}>
            {/* Left: Video Preview — 9:16 column matching the fixed card height */}
            {/* A full-width 9:16 preview on a phone is ~640px tall on its own,
                which pushed the title, captions and every action off-screen.
                Capping the height and centring keeps the whole card scannable
                without letterboxing the clip. */}
            <div className="w-full max-w-[calc(64vh*0.5625)] md:max-w-none mx-auto md:mx-0 md:w-[236px] bg-black relative shrink-0 aspect-[9/16] md:aspect-auto group/video">
                <video
                    ref={videoRef}
                    src={playbackUrl}
                    controls
                    className="w-full h-full object-contain"
                    playsInline
                    onLoadedMetadata={(e) => {
                        if (e.target.videoWidth) setResolution(`${e.target.videoWidth}×${e.target.videoHeight}`);
                    }}
                    onError={() => {
                        // The durable copy is unreachable (signature expired after an
                        // hour on an idle tab, object purged) → serve from the API for
                        // the rest of this card's life.
                        if (playbackUrl === durableSrc) {
                            setDurableFailed(true);
                            return;
                        }
                        // Local /videos/ file gone (e.g. cleaned up after a reload) →
                        // fall back to the durable R2 copy for managed users. If the
                        // durable URL hasn't loaded yet, the effect above retries.
                        if (durable?.url && currentVideoUrl !== durable.url) setCurrentVideoUrl(durable.url);
                        else setVideoErrored(true);
                    }}
                    onPlay={() => {
                        setHasPlayed(true);
                        const currentTime = videoRef.current ? videoRef.current.currentTime : 0;
                        onPlay && onPlay(clip.start + currentTime);
                    }}
                    onPause={() => onPause && onPause()}
                    onEnded={() => {
                        if (videoRef.current) {
                            videoRef.current.currentTime = 0;
                            videoRef.current.play();
                        }
                    }}
                />
                <div className="absolute top-3 left-3 flex gap-2">
                    {/* Stays the clip's own number, not its rank: the cards are
                        ordered by score, but this is what the downloaded file
                        is called (clip-N.mp4) and what every api call indexes. */}
                    <span className="bg-black/70 text-ink font-mono text-micro uppercase px-2 py-1 rounded-full">
                        Clip {index + 1}
                    </span>
                    {/* A bare number on a thumbnail reads as a duration, a
                        position, anything — it has to name itself and carry
                        its scale, or it is decoration. */}
                    {Number.isFinite(clip.predicted_score) && (
                        <span
                            className="bg-black/70 font-mono text-micro uppercase px-2 py-1 rounded-full flex items-center gap-1"
                            title="openshorts' prediction of how well this clip will perform, from 0 to 100"
                        >
                            <TrendingUp size={11} className="shrink-0 text-muted" />
                            <span className="text-muted">viral</span>
                            <b className={
                                clip.predicted_score >= 80 ? 'text-ok'
                                    : clip.predicted_score >= 65 ? 'text-brass'
                                        : 'text-ink2'
                            }>
                                {clip.predicted_score}
                            </b>
                            <span className="text-muted">/100</span>
                        </span>
                    )}
                </div>

                {/* Auto Edit Overlay if Processing */}
                {isEditing && (
                    <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center z-10 p-4 text-center">
                        <Loader2 size={28} className="text-brass animate-spin mb-3" />
                        <span className="text-xs text-ink lowercase">ai magic in progress…</span>
                        <span className="readout mt-1.5">APPLYING VIRAL EDITS · ZOOMS</span>
                    </div>
                )}
            </div>

            {/* Right: Content & Details */}
            <div className="flex-1 p-4 md:p-5 flex flex-col overflow-hidden min-w-0">
                <div className="mb-4">
                    <h3 className="text-base font-medium text-ink leading-tight line-clamp-2 mb-2 break-words" title={clip.video_title_for_youtube_short}>
                        {clip.video_title_for_youtube_short || "Viral Clip Generated"}
                    </h3>
                    {/* The score alone says how much, not why. One line from
                        the selection pass on what this moment has going for it. */}
                    {clip.why && (
                        <div
                            className="relative mb-2"
                            onMouseEnter={() => setWhyOpen(true)}
                            onMouseLeave={() => setWhyOpen(false)}
                        >
                            <p
                                ref={whyRef}
                                className={`text-xs text-muted leading-snug line-clamp-2 break-words ${whyClamped ? 'cursor-pointer' : ''}`}
                                onClick={() => whyClamped && setWhyOpen(v => !v)}
                                aria-label="why openshorts picked this moment"
                            >
                                {clip.why}
                            </p>
                            {whyClamped && whyOpen && (
                                <div
                                    role="tooltip"
                                    className="absolute left-0 right-0 top-full mt-1 z-20 bg-paper border border-rule rounded-input px-3 py-2 text-xs text-ink leading-snug shadow-lg"
                                >
                                    {clip.why}
                                </div>
                            )}
                        </div>
                    )}
                    <div className="flex flex-wrap gap-1.5">
                        {durationReadout && <span className="readout bg-paper3 px-2 py-0.5 rounded-full shrink-0">{durationReadout}</span>}
                        {resolution && <span className="readout bg-paper3 px-2 py-0.5 rounded-full shrink-0">{resolution}</span>}
                        <span className="readout bg-paper3 px-2 py-0.5 rounded-full shrink-0">#shorts</span>
                        <span className="readout bg-paper3 px-2 py-0.5 rounded-full shrink-0">#viral</span>
                    </div>
                </div>

                {/* Descriptions (compact) — full text lives in the modal */}
                <div className="flex-1 min-h-0 space-y-2 mb-4">
                    <div className="bg-paper rounded-input px-3 py-2 border border-rule flex items-center gap-2 min-w-0 overflow-hidden">
                        <span className="eyebrow shrink-0">YOUTUBE</span>
                        <p className="text-xs text-ink2 truncate flex-1 min-w-0">
                            {clip.video_title_for_youtube_short || "Viral Short Video"}
                        </p>
                        <button
                            onClick={() => handleCopy('youtube', clip.video_title_for_youtube_short || "Viral Short Video")}
                            aria-label="copy youtube title"
                            className="p-1 rounded-full text-muted hover:text-brass transition-colors shrink-0"
                        >
                            {copied === 'youtube' ? <Check size={14} className="text-ok" /> : <Copy size={14} />}
                        </button>
                    </div>

                    <div className="bg-paper rounded-input px-3 py-2 border border-rule flex items-center gap-2 min-w-0 overflow-hidden">
                        <span className="eyebrow shrink-0">TIKTOK · IG</span>
                        <p className="text-xs text-ink2 truncate flex-1 min-w-0">
                            {clip.video_description_for_tiktok || clip.video_description_for_instagram}
                        </p>
                        <button
                            onClick={() => handleCopy('caption', clip.video_description_for_tiktok || clip.video_description_for_instagram)}
                            aria-label="copy caption"
                            className="p-1 rounded-full text-muted hover:text-brass transition-colors shrink-0"
                        >
                            {copied === 'caption' ? <Check size={14} className="text-ok" /> : <Copy size={14} />}
                        </button>
                    </div>

                    <button
                        onClick={() => setShowDescModal(true)}
                        className="w-full flex items-center justify-center gap-2 py-2 rounded-input border border-dashed border-rule text-xs lowercase text-muted hover:text-brass hover:border-rule2 transition-colors"
                    >
                        <FileText size={14} /> view descriptions
                    </button>

                    {aiThumbnailState?.youtube && (
                        <details className="bg-paper rounded-input px-3 py-2 border border-rule text-xs text-ink2">
                            <summary className="cursor-pointer text-muted lowercase">youtube details</summary>
                            <div className="mt-2 space-y-2">
                                <p className="break-words"><span className="eyebrow mr-2">TITLE</span>{aiThumbnailState.youtube.title}</p>
                                <p className="whitespace-pre-wrap break-words">{aiThumbnailState.youtube.description || 'No separate YouTube description was generated.'}</p>
                                {aiThumbnailState.youtube.tags?.length > 0 && (
                                    <p className="break-words text-muted">tags: {aiThumbnailState.youtube.tags.join(', ')}</p>
                                )}
                                {aiThumbnailState.youtube.truncated && (
                                    <p className="text-warn">Description was shortened to YouTube’s size limit.</p>
                                )}
                                {aiThumbnailState.youtube.profileWarning && (
                                    <p className="text-warn">{aiThumbnailState.youtube.profileWarning}</p>
                                )}
                            </div>
                        </details>
                    )}

                    {aiThumbnailState?.message && (
                        <p className={`text-xs rounded-input p-2 ${
                            aiThumbnailState.batchState === 'unknown' || aiThumbnailState.batchState === 'failed'
                                ? 'text-warn bg-[color-mix(in_oklab,var(--color-warn)_10%,transparent)]'
                                : 'text-muted bg-paper'
                        }`}>
                            {aiThumbnailState.message}
                        </p>
                    )}

                    {aiThumbnailState?.candidates?.length > 0 && (
                        <div className="grid grid-cols-2 gap-2">
                            {aiThumbnailState.candidates.map((candidate) => (
                                <button
                                    key={candidate.id}
                                    onClick={() => void selectAiThumbnail(candidate.id)}
                                    disabled={aiThumbnailBusy || postizAlreadySent}
                                    title={candidate.selected ? 'Selected AI thumbnail' : 'Use this AI thumbnail for Send to Postiz'}
                                    className={`rounded-input overflow-hidden border text-left transition-colors ${candidate.selected ? 'border-brass ring-1 ring-brass' : 'border-rule hover:border-rule2'} disabled:opacity-50`}
                                >
                                    <img
                                        src={getApiUrl(candidate.url)}
                                        alt="Generated AI thumbnail candidate"
                                        className="w-full aspect-video object-cover bg-paper3"
                                    />
                                    <span className="block px-2 py-1 text-[10px] lowercase text-muted">
                                        {candidate.selected ? 'AI thumbnail selected' : 'use this thumbnail'}
                                    </span>
                                </button>
                            ))}
                        </div>
                    )}
                </div>

                {/* Error Message */}
                {editError && (
                    <div className="mb-3 px-3 py-2 rounded-input text-xs text-danger bg-[color-mix(in_oklab,var(--color-danger)_10%,transparent)] flex items-center gap-2">
                        <AlertCircle size={14} className="shrink-0" />
                        {editError}
                    </div>
                )}
                {postizResult && (
                    <div className={`mb-3 px-3 py-2 rounded-input text-xs flex items-center gap-2 ${postizResult.success ? 'text-ok bg-[color-mix(in_oklab,var(--color-ok)_10%,transparent)]' : 'text-danger bg-[color-mix(in_oklab,var(--color-danger)_10%,transparent)]'}`}>
                        {postizResult.success ? <Check size={14} className="shrink-0" /> : <AlertCircle size={14} className="shrink-0" />}
                        {postizResult.msg}
                    </div>
                )}

                {/* Actions Footer */}
                <div className="grid grid-cols-[repeat(auto-fill,minmax(104px,1fr))] gap-2 mt-auto pt-4 border-t border-rule">
                    {onEditClip && (
                        <button
                            onClick={() => onEditClip(index)}
                            className={QUIET_BTN}
                        >
                            <Scissors size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />
                            edit clip
                        </button>
                    )}

                    {onReframeClip && (
                        <button
                            onClick={() => onReframeClip(index)}
                            className={QUIET_BTN}
                        >
                            <Crosshair size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />
                            reframing
                        </button>
                    )}

                    <button
                        onClick={handleAutoEdit}
                        disabled={isEditing}
                        className={QUIET_BTN}
                    >
                        {isEditing ? <Loader2 size={16} className="animate-spin text-brass shrink-0" /> : <Wand2 size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />}
                        {isEditing ? 'editing…' : 'auto edit'}
                    </button>

                    <button
                        onClick={() => setShowSubtitleModal(true)}
                        disabled={isSubtitling}
                        className={QUIET_BTN}
                    >
                        {isSubtitling ? <Loader2 size={16} className="animate-spin text-brass shrink-0" /> : <Type size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />}
                        {isSubtitling ? 'adding…' : 'subtitles'}
                    </button>

                    <button
                        onClick={() => setShowHookModal(true)}
                        disabled={isHooking}
                        className={QUIET_BTN}
                    >
                        {isHooking ? <Loader2 size={16} className="animate-spin text-brass shrink-0" /> : <Wand2 size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />}
                        {isHooking ? 'saving…' : 'edit hook'}
                    </button>

                    <button
                        onClick={() => setShowTranslateModal(true)}
                        disabled={isTranslating}
                        className={QUIET_BTN}
                    >
                        {isTranslating ? <Loader2 size={16} className="animate-spin text-brass shrink-0" /> : <Languages size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />}
                        {isTranslating ? 'translating…' : 'dub voice'}
                    </button>

                    <input
                        ref={postizThumbnailInputRef}
                        type="file"
                        accept="image/jpeg,image/png"
                        className="hidden"
                        onChange={handlePostizThumbnailChange}
                    />
                    <button
                        onClick={() => postizThumbnailInputRef.current?.click()}
                        disabled={postizSending || isVideoChanging || aiThumbnailBusy || postizAlreadySent}
                        title={postizThumbnail
                            ? `Selected thumbnail: ${postizThumbnail.name}. Click to replace it.`
                            : 'Optional: select a JPEG or PNG YouTube thumbnail (maximum 2 MB)'}
                        className={QUIET_BTN}
                    >
                        <ImagePlus size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />
                        {postizThumbnail ? 'thumb ready' : 'add thumbnail'}
                    </button>
                    {!isManaged && clipAiThumbnailsEnabled && (
                        <button
                            onClick={() => setShowAiThumbnailModal(true)}
                            disabled={postizSending || isVideoChanging || aiThumbnailBusy || postizAlreadySent}
                            title={postizAlreadySent
                                ? 'This clip was already sent to Postiz. Update its draft in Postiz.'
                                : 'Create a Gemini AI thumbnail only for this clip'}
                            className={QUIET_BTN}
                        >
                            {aiThumbnailBusy
                                ? <Loader2 size={16} className="animate-spin text-brass shrink-0" />
                                : <Wand2 size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />}
                            {aiThumbnailBusy ? 'generating…' : 'generate AI thumb'}
                        </button>
                    )}
                    {postizThumbnail && (
                        <button
                            onClick={() => {
                                setPostizThumbnail(null);
                                if (postizThumbnailInputRef.current) postizThumbnailInputRef.current.value = '';
                            }}
                            disabled={postizSending || postizAlreadySent}
                            title="Remove the selected thumbnail before sending"
                            className={QUIET_BTN}
                        >
                            <X size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />
                            clear thumb
                        </button>
                    )}
                    {aiThumbnailState?.selectedCandidateId && (
                        <button
                            onClick={() => void selectAiThumbnail(null)}
                            disabled={postizSending || isVideoChanging || aiThumbnailBusy || postizAlreadySent}
                            title="Do not send an AI thumbnail with this clip"
                            className={QUIET_BTN}
                        >
                            <X size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />
                            clear AI thumb
                        </button>
                    )}

                    <button
                        onClick={handleSendToPostiz}
                        disabled={postizSending || !postizConfigured || isVideoChanging || aiThumbnailBusy || postizAlreadySent}
                        title={isVideoChanging
                            ? 'Wait for the current video edit to finish before sending it to Postiz'
                            : postizAlreadySent
                            ? 'This clip was already sent to Postiz. Update its draft in Postiz.'
                            : postizConfigured
                            ? (postizPostType === 'now'
                                ? 'Send this clip to Postiz for YouTube publishing'
                                : 'Create a Postiz YouTube draft')
                            : 'Configure Postiz in OpenShorts .env, then restart the backend'}
                        className="btn-primary flex-col gap-1 py-2.5 sm:py-2 px-1 text-[11px] leading-none rounded-input whitespace-nowrap disabled:opacity-45 disabled:cursor-not-allowed"
                    >
                        {postizSending
                            ? <Loader2 size={16} className="animate-spin shrink-0" />
                            : <Share2 size={16} className="shrink-0" />}
                        {postizSending
                            ? 'sending…'
                            : postizConfigured
                                ? 'send to postiz'
                                : 'configure postiz'}
                    </button>
                    <button
                        onClick={(e) => {
                            e.preventDefault();
                            // Free clips are watermarked — surface the upsell once
                            // before the first download, then get out of the way.
                            if (plan === 'free' && !watermarkNoticeDismissed(jobId)) {
                                setShowWatermarkModal(true);
                                return;
                            }
                            downloadClip();
                        }}
                        className={`${QUIET_BTN}${onEditClip ? ' col-span-2' : ''}`}
                    >
                        <Download size={16} className="text-muted group-hover:text-brass transition-colors shrink-0" />
                        {downloadPct === null ? 'download' : `downloading ${downloadPct}%`}
                    </button>
                </div>
            </div>

            <Modal
                isOpen={showAiThumbnailModal}
                onClose={() => !aiThumbnailLoading && setShowAiThumbnailModal(false)}
                eyebrow="ON-DEMAND AI"
                title="generate clip thumbnail"
                size="md"
            >
                <div className="space-y-4">
                    <p className="text-sm text-muted leading-relaxed">
                        This sends one locally extracted frame from this final clip, its
                        clip-specific metadata, and your direction to Gemini. It does not
                        upload the video file. Each candidate is a separate image request.
                    </p>
                    {aiThumbnailState?.thumbnailBrief && (
                        <div className="bg-paper rounded-input p-3 border border-rule text-xs text-ink2 space-y-1">
                            <p><span className="eyebrow mr-2">MOMENT</span>{aiThumbnailState.thumbnailBrief.moment || 'Use the selected clip context.'}</p>
                            <p><span className="eyebrow mr-2">TEXT</span>{aiThumbnailState.thumbnailBrief.thumbnail_text || 'No fixed headline requested.'}</p>
                        </div>
                    )}
                    {aiThumbnailState?.referenceUrl && (
                        <img
                            src={getApiUrl(aiThumbnailState.referenceUrl)}
                            alt="Locally selected reference frame"
                            className="w-full rounded-input border border-rule aspect-video object-cover bg-paper3"
                        />
                    )}
                    <div>
                        <label className="eyebrow block mb-1.5" htmlFor={`thumb-instructions-${jobId}-${index}`}>CREATIVE DIRECTION (OPTIONAL)</label>
                        <textarea
                            id={`thumb-instructions-${jobId}-${index}`}
                            value={aiThumbnailInstructions}
                            onChange={(event) => setAiThumbnailInstructions(event.target.value.slice(0, 600))}
                            maxLength={600}
                            rows={3}
                            placeholder="Example: emphasize the boss reaction, cinematic lighting, no extra text."
                            className="w-full bg-paper rounded-input border border-rule px-3 py-2 text-sm text-ink2 outline-none focus:border-brass resize-y"
                        />
                    </div>
                    <div>
                        <label className="eyebrow block mb-1.5">CANDIDATES</label>
                        <select
                            value={aiThumbnailCount}
                            onChange={(event) => setAiThumbnailCount(Number(event.target.value))}
                            className="bg-paper rounded-input border border-rule px-3 py-2 text-sm text-ink2 outline-none focus:border-brass"
                        >
                            <option value={1}>1 candidate</option>
                            <option value={2}>2 candidates</option>
                            <option value={3}>3 candidates</option>
                        </select>
                    </div>
                    {aiThumbnailError && (
                        <p className="text-xs text-danger bg-[color-mix(in_oklab,var(--color-danger)_10%,transparent)] rounded-input p-3">
                            {aiThumbnailError}
                        </p>
                    )}
                    <div className="flex justify-end gap-2">
                        <button
                            onClick={() => setShowAiThumbnailModal(false)}
                            disabled={aiThumbnailLoading}
                            className="btn-ghost px-3 py-2 text-xs"
                        >
                            cancel
                        </button>
                        <button
                            onClick={() => void generateAiThumbnail()}
                            disabled={aiThumbnailLoading}
                            className="btn-primary px-3 py-2 text-xs"
                        >
                            {aiThumbnailLoading ? 'generating…' : 'generate'}
                        </button>
                    </div>
                </div>
            </Modal>

            {/* Descriptions Modal */}
            <Modal
                isOpen={showDescModal}
                onClose={() => setShowDescModal(false)}
                eyebrow="GENERATED COPY"
                title="descriptions"
                size="md"
            >
                <div className="space-y-4">
                    <div>
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                            <label className="eyebrow">YOUTUBE TITLE</label>
                            <button
                                onClick={() => handleCopy('youtube', clip.video_title_for_youtube_short || "Viral Short Video")}
                                aria-label="copy youtube title"
                                className="p-1 rounded-full text-muted hover:text-brass transition-colors shrink-0"
                            >
                                {copied === 'youtube' ? <Check size={14} className="text-ok" /> : <Copy size={14} />}
                            </button>
                        </div>
                        <p className="text-sm text-ink2 select-all break-words bg-paper rounded-input p-3 border border-rule">
                            {clip.video_title_for_youtube_short || "Viral Short Video"}
                        </p>
                    </div>

                    <div>
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                            <label className="eyebrow">TIKTOK · IG CAPTION</label>
                            <button
                                onClick={() => handleCopy('caption', clip.video_description_for_tiktok || clip.video_description_for_instagram)}
                                aria-label="copy caption"
                                className="p-1 rounded-full text-muted hover:text-brass transition-colors shrink-0"
                            >
                                {copied === 'caption' ? <Check size={14} className="text-ok" /> : <Copy size={14} />}
                            </button>
                        </div>
                        <p className="text-sm text-ink2 select-all break-words bg-paper rounded-input p-3 border border-rule whitespace-pre-wrap">
                            {clip.video_description_for_tiktok || clip.video_description_for_instagram}
                        </p>
                    </div>
                </div>
            </Modal>

            <SubtitleModal
                isOpen={showSubtitleModal}
                onClose={() => setShowSubtitleModal(false)}
                onGenerate={handleSubtitle}
                onApplyAll={onBulkSubtitle ? async (options) => {
                    await onBulkSubtitle(options);
                    setShowSubtitleModal(false);
                } : undefined}
                onRemove={handleRemoveSubtitles}
                bulkCount={clipCount}
                bulkProgress={bulkProgress}
                isProcessing={isSubtitling || (bulkProgress?.running ?? false)}
                videoUrl={originalVideoUrl}
                jobId={jobId}
                clipIndex={index}
                existingHook={activeLayers.hook}
            />

            <HookModal
                isOpen={showHookModal}
                onClose={() => setShowHookModal(false)}
                onGenerate={handleHook}
                isProcessing={isHooking}
                videoUrl={originalVideoUrl}
                initialText={clip.viral_hook_text}
                durationInSeconds={clip.end && clip.start ? clip.end - clip.start : 30}
                existingSubtitles={activeLayers.subtitles}
                hasCaptions={!!activeLayers.subtitles || /(^|_)subtitled_/.test(serverVideoFile || '')}
                serverRender={hasServerBurns}
                burnedHook={burnedHook}
                onRemove={burnedHook ? handleRemoveHook : null}
            />

            <TranslateModal
                isOpen={showTranslateModal}
                onClose={() => setShowTranslateModal(false)}
                onTranslate={handleTranslate}
                isProcessing={isTranslating}
                videoUrl={currentVideoUrl}
                hasApiKey={!!elevenLabsKey}
            />

            {showWatermarkModal && (
                <WatermarkModal
                    source="download"
                    jobId={jobId}
                    onUpgrade={onUpgrade}
                    onClose={() => setShowWatermarkModal(false)}
                    onContinue={downloadClip}
                />
            )}

        </div>
    );
}
