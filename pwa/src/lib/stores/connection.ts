/**
 * WebSocket client with reconnection (exponential backoff) and offline command queue.
 *
 * Uses native Svelte 5 stores ($state-compatible writable stores).
 * Queue messages when readyState !== OPEN, flush on reconnect.
 */

import { writable, get } from 'svelte/store';

// ── Types ────────────────────────────────────────────────────────────

export interface JarvisMessage {
	type: string;
	[key: string]: unknown;
}

export interface ChatMessage {
	role: 'user' | 'assistant' | 'system';
	text: string;
	timestamp: number;
}

// ── Stores ───────────────────────────────────────────────────────────

/** Connection status: 'connecting' | 'connected' | 'disconnected' */
export const connectionStatus = writable<'connecting' | 'connected' | 'disconnected'>('disconnected');

/** Orchestrator status (Listening, Thinking, Speaking, etc.) */
export const orchestratorStatus = writable<string>('Idle');

/** Conversation history */
export const chatHistory = writable<ChatMessage[]>([]);

/** Latest detections from vision scan */
export const detections = writable<{ detections: unknown[]; description: string }>({
	detections: [],
	description: ''
});

/** Whether a wake word was just detected */
export const wakeDetected = writable<boolean>(false);

/** Latest proactive message */
export const proactiveMessage = writable<string>('');

/** Latest error */
export const lastError = writable<string>('');

/** System stats pushed via WebSocket (type: system_status) */
export const wsSystemStats = writable<string | null>(null);

/** Server-side interim transcript (STT partial from Jetson) */
export const serverInterimTranscript = writable<string>('');

/** Orchestration thinking steps – live feed of what Jarvis is doing */
export interface ThinkingStep {
	step: string;   // e.g. 'heard', 'vision', 'context', 'reasoning', 'tool', 'speaking', 'done'
	detail: string; // human-readable description
	timestamp: number;
}
export const thinkingSteps = writable<ThinkingStep[]>([]);

/** Streaming reply – used for typewriter effect. Null when not streaming. */
export const streamingReply = writable<string | null>(null);

/** Tracked object from enriched vision pipeline */
export interface TrackedObjectData {
	track_id: number;
	xyxy: number[];
	cls: number;
	class_name: string;
	conf: number;
	velocity: number[];
	depth: number | null;
	frames_seen: number;
	age: number;
}

/** Tracked objects store — updated every scan_result (5s) + hologram (15s) */
export const trackedObjects = writable<TrackedObjectData[]>([]);

/** Hologram data (point cloud + tracked objects) */
export interface HologramData {
	point_cloud: Array<{ x: number; y: number; z: number; r: number; g: number; b: number }>;
	tracked_objects: TrackedObjectData[];
	description: string;
}
export const hologramData = writable<HologramData | null>(null);

/** Vitals data */
export interface VitalsData {
	fatigue: string;
	posture: string;
	heart_rate: number | null;
	hr_confidence: number;
	alerts: string[];
}
export const vitalsData = writable<VitalsData | null>(null);

/** Threat assessment data (pushed by enriched vision pipeline) */
export interface ThreatData {
	level: string;       // 'clear' | 'low' | 'moderate' | 'high' | 'critical'
	score: number;       // 0-1
	summary: string;
}
export const threatData = writable<ThreatData | null>(null);

// ── Typewriter effect ────────────────────────────────────────────────

let _typewriterTimer: ReturnType<typeof setTimeout> | null = null;
let _typewriterFullText: string | null = null;

function _commitPendingTypewriter() {
	// If a typewriter is in progress, commit its full text to history immediately
	if (_typewriterTimer) {
		clearTimeout(_typewriterTimer);
		_typewriterTimer = null;
	}
	if (_typewriterFullText !== null) {
		const text = _typewriterFullText;
		_typewriterFullText = null;
		streamingReply.set(null);
		chatHistory.update((h) => [
			...h,
			{ role: 'assistant', text, timestamp: Date.now() },
		]);
	}
}

function _typewriterReply(fullText: string) {
	// Commit any in-progress typewriter first (prevents lost messages)
	_commitPendingTypewriter();

	// Clear thinking steps when reply starts
	thinkingSteps.set([]);

	_typewriterFullText = fullText;
	const chars = [...fullText];
	let index = 0;
	// Chars per tick: scale speed with message length for consistent feel
	const charsPerTick = Math.max(1, Math.ceil(chars.length / 80));
	const tickMs = 18;

	streamingReply.set('');

	function tick() {
		if (index >= chars.length) {
			// Done streaming: move to permanent history
			_typewriterFullText = null;
			streamingReply.set(null);
			chatHistory.update((h) => [
				...h,
				{ role: 'assistant', text: fullText, timestamp: Date.now() },
			]);
			_typewriterTimer = null;
			return;
		}
		const end = Math.min(index + charsPerTick, chars.length);
		const partial = chars.slice(0, end).join('');
		streamingReply.set(partial);
		index = end;
		_typewriterTimer = setTimeout(tick, tickMs);
	}
	tick();
}

// ── WebSocket Manager ────────────────────────────────────────────────

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempt = 0;
const MAX_RECONNECT_DELAY = 30_000;
const BASE_RECONNECT_DELAY = 1_000;

/** Offline command queue – flushed on reconnect */
const offlineQueue: string[] = [];

/** Track texts sent from the PWA so we can deduplicate server echoes */
const _recentSentTexts: Set<string> = new Set();

/** Current server host (configurable, persisted in localStorage) */
function _loadServerHost(): string {
	if (typeof window === 'undefined') return 'localhost:8000';
	try {
		const saved = localStorage.getItem('jarvis_server_host');
		if (saved) return saved;
	} catch { /* ignore */ }
	return window.location.hostname + ':8000';
}
export const serverHost = writable<string>(_loadServerHost());
// Persist host changes
if (typeof window !== 'undefined') {
	serverHost.subscribe((host) => {
		try { localStorage.setItem('jarvis_server_host', host); } catch { /* ignore */ }
	});
}

/** Return the correct protocol (http/https) based on the page origin. */
function getHttpProtocol(): string {
	return typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'https' : 'http';
}

/** Build a full URL for an API/stream path, respecting http vs https. */
export function getApiUrl(path: string): string {
	const host = get(serverHost);
	return `${getHttpProtocol()}://${host}${path.startsWith('/') ? path : '/' + path}`;
}

function getWsUrl(): string {
	const host = get(serverHost);
	const protocol = typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss' : 'ws';
	return `${protocol}://${host}/ws`;
}

function handleMessage(event: MessageEvent) {
	try {
		const msg: JarvisMessage = JSON.parse(event.data);

		switch (msg.type) {
			case 'status':
				orchestratorStatus.set(msg.status as string);
				break;

			case 'wake':
				wakeDetected.set(true);
				setTimeout(() => wakeDetected.set(false), 3000);
				break;

		case 'transcript_interim':
				serverInterimTranscript.set((msg.text as string) || '');
				break;

			case 'transcript_final': {
				serverInterimTranscript.set('');
				const tfText = (msg.text as string) || '';
				// Skip if this text was already added optimistically by sendText()
				if (_recentSentTexts.has(tfText)) {
					_recentSentTexts.delete(tfText);
				} else {
					chatHistory.update((h) => [
						...h,
						{ role: 'user', text: tfText, timestamp: Date.now() }
					]);
				}
				break;
			}

			case 'thinking_step': {
				const step = (msg.step as string) || '';
				const detail = (msg.detail as string) || '';
				if (step === 'done') {
					// Clear thinking steps when orchestration completes
					thinkingSteps.set([]);
				} else {
					thinkingSteps.update((s) => [
						...s,
						{ step, detail, timestamp: Date.now() },
					]);
				}
				break;
			}

			case 'reply':
				// Trigger typewriter effect: put text in streamingReply,
				// then after animation completes, move to chatHistory
				_typewriterReply(msg.text as string);
				break;

			case 'detections':
			case 'scan_result':
				detections.set({
					detections: (msg.detections as unknown[]) || [],
					description: (msg.description as string) || ''
				});
				// Update tracked objects for HUD overlay (sent every 5s with scan_result)
				if (msg.tracked && Array.isArray(msg.tracked)) {
					trackedObjects.set(msg.tracked as TrackedObjectData[]);
				}
				// If threat data is bundled with scan result, update it
				if (msg.threat) {
					threatData.set(msg.threat as ThreatData);
				}
				break;

			case 'proactive':
				proactiveMessage.set(msg.text as string);
				chatHistory.update((h) => [
					...h,
					{ role: 'system', text: msg.text as string, timestamp: Date.now() }
				]);
				break;

			case 'system_status':
				wsSystemStats.set((msg.status as string) || null);
				break;

			case 'hologram': {
				const hData = (msg.data as HologramData) || null;
				hologramData.set(hData);
				if (hData?.tracked_objects) {
					trackedObjects.set(hData.tracked_objects);
				}
				break;
			}

			case 'vitals':
				vitalsData.set((msg.data as VitalsData) || null);
				break;

			case 'threat':
				threatData.set((msg.data as ThreatData) || null);
				break;

			case 'error':
				lastError.set(msg.message as string);
				break;
		}
	} catch {
		// Ignore malformed messages
	}
}

function flushQueue() {
	while (offlineQueue.length > 0 && ws?.readyState === WebSocket.OPEN) {
		ws.send(offlineQueue.shift()!);
	}
}

function scheduleReconnect() {
	if (reconnectTimer) return;
	const delay = Math.min(BASE_RECONNECT_DELAY * 2 ** reconnectAttempt, MAX_RECONNECT_DELAY);
	reconnectAttempt++;
	reconnectTimer = setTimeout(() => {
		reconnectTimer = null;
		connect();
	}, delay);
}

export function connect() {
	if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
		return;
	}

	const url = getWsUrl();
	connectionStatus.set('connecting');

	try {
		ws = new WebSocket(url);
	} catch {
		connectionStatus.set('disconnected');
		scheduleReconnect();
		return;
	}

	ws.onopen = () => {
		connectionStatus.set('connected');
		reconnectAttempt = 0;
		flushQueue();
	};

	ws.onmessage = handleMessage;

	ws.onclose = () => {
		connectionStatus.set('disconnected');
		ws = null;
		scheduleReconnect();
	};

	ws.onerror = () => {
		// onclose will fire after this
	};
}

export function disconnect() {
	if (reconnectTimer) {
		clearTimeout(reconnectTimer);
		reconnectTimer = null;
	}
	reconnectAttempt = 0;
	if (ws) {
		ws.onclose = null;
		ws.close();
		ws = null;
	}
	connectionStatus.set('disconnected');
}

/** Send a JSON message; queues if offline. */
export function sendMessage(msg: JarvisMessage) {
	const payload = JSON.stringify(msg);
	if (ws?.readyState === WebSocket.OPEN) {
		ws.send(payload);
	} else {
		offlineQueue.push(payload);
	}
}

/** Convenience: send user text */
export function sendText(text: string) {
	sendMessage({ type: 'text', text });
	// Track so we can deduplicate the server's transcript_final echo
	_recentSentTexts.add(text);
	// Clean up after 10s in case the echo never arrives
	setTimeout(() => _recentSentTexts.delete(text), 10_000);
	// Optimistically add to history
	chatHistory.update((h) => [...h, { role: 'user', text, timestamp: Date.now() }]);
}

/** Convenience: request a scan */
export function sendScan() {
	sendMessage({ type: 'scan' });
}

/** Convenience: request system status */
export function sendGetStatus() {
	sendMessage({ type: 'get_status' });
}

/** Convenience: toggle sarcasm */
export function sendSarcasmToggle(enabled: boolean) {
	sendMessage({ type: 'sarcasm_toggle', enabled });
}

/** Convenience: request hologram render */
export function sendHologramRequest() {
	sendMessage({ type: 'hologram_request' });
}

/** Convenience: request vitals update */
export function sendVitalsRequest() {
	sendMessage({ type: 'vitals_request' });
}

// ── Chat history localStorage persistence ────────────────────────────

const CHAT_STORAGE_KEY = 'jarvis_chat_history';
const CHAT_MAX_PERSISTED = 50;

/** Hydrate chat history from localStorage on load. */
export function hydrateChatHistory() {
	if (typeof window === 'undefined') return;
	try {
		const raw = localStorage.getItem(CHAT_STORAGE_KEY);
		if (raw) {
			const parsed = JSON.parse(raw) as ChatMessage[];
			if (Array.isArray(parsed) && parsed.length > 0) {
				chatHistory.set(parsed.slice(-CHAT_MAX_PERSISTED));
			}
		}
	} catch { /* ignore corrupt data */ }
}

/** Subscribe to chatHistory changes and debounce-write to localStorage. */
let _chatDebounce: ReturnType<typeof setTimeout> | null = null;
if (typeof window !== 'undefined') {
	chatHistory.subscribe((h) => {
		if (_chatDebounce) clearTimeout(_chatDebounce);
		_chatDebounce = setTimeout(() => {
			try {
				localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(h.slice(-CHAT_MAX_PERSISTED)));
			} catch { /* storage full or unavailable */ }
		}, 500);
	});
}
