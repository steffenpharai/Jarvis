/**
 * WebSocket client with reconnection (exponential backoff) and offline command queue.
 *
 * Enhanced with:
 *   - Message sequence numbers for ordering (prevents WS race conditions)
 *   - Request-response acknowledgement for button actions
 *   - Heartbeat ping/pong for connection health monitoring
 *   - Debounced action dispatching
 *
 * Uses native Svelte 5 stores ($state-compatible writable stores).
 * Queue messages when readyState !== OPEN, flush on reconnect.
 */

import { writable, get } from 'svelte/store';

// ── Types ────────────────────────────────────────────────────────────

export interface JarvisMessage {
	type: string;
	request_id?: string;
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

/** Last processed server sequence number for ordering */
let _lastSeq = 0;

/** Pending ack callbacks: request_id → { resolve, timer } */
const _pendingAcks: Map<string, { resolve: () => void; timer: ReturnType<typeof setTimeout> }> = new Map();

/** Debounce timers for button actions */
const _actionDebounce: Map<string, ReturnType<typeof setTimeout>> = new Map();

/** Heartbeat interval and health tracking */
let _heartbeatTimer: ReturnType<typeof setInterval> | null = null;
const HEARTBEAT_INTERVAL = 30_000;
let _lastPongTime = 0;
let _missedPongs = 0;
const PONG_TIMEOUT = 5_000;
const MAX_MISSED_PONGS = 3;

/** Connection health: 'good' | 'degraded' | 'lost' */
export const connectionHealth = writable<'good' | 'degraded' | 'lost'>('good');

/** Generate a unique request ID */
function _genRequestId(): string {
	return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/** Debounce an action by key (prevents double-tap) */
function _debouncedAction(key: string, fn: () => void, delayMs = 500) {
	const existing = _actionDebounce.get(key);
	if (existing) {
		clearTimeout(existing);
	}
	_actionDebounce.set(key, setTimeout(() => {
		_actionDebounce.delete(key);
		fn();
	}, delayMs));
}

/** Send message with ack tracking (returns promise that resolves on ack) */
function _sendWithAck(msg: JarvisMessage, timeoutMs = 10_000): Promise<void> {
	return new Promise((resolve) => {
		const reqId = _genRequestId();
		msg.request_id = reqId;
		const timer = setTimeout(() => {
			_pendingAcks.delete(reqId);
			resolve(); // resolve anyway after timeout
		}, timeoutMs);
		_pendingAcks.set(reqId, { resolve, timer });
		sendMessage(msg);
	});
}

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

function _processMessage(msg: JarvisMessage) {
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
}

function handleMessage(event: MessageEvent) {
	try {
		const msg: JarvisMessage = JSON.parse(event.data);

		// Handle ack messages
		if (msg.type === 'ack' && msg.request_id) {
			const pending = _pendingAcks.get(msg.request_id as string);
			if (pending) {
				clearTimeout(pending.timer);
				pending.resolve();
				_pendingAcks.delete(msg.request_id as string);
			}
			return;
		}

		// Handle pong (heartbeat response) — track health
		if (msg.type === 'pong') {
			_lastPongTime = Date.now();
			_missedPongs = 0;
			connectionHealth.set('good');
			return;
		}

		// Track sequence numbers for ordering
		const seq = msg._seq as number | undefined;
		if (seq !== undefined) {
			// If we detect a gap > 1, hold the message briefly to allow
			// the missing message to arrive (prevents "reply before transcript" race)
			if (seq > _lastSeq + 1 && _lastSeq > 0) {
				setTimeout(() => {
					_processMessage(msg);
				}, 50);
				_lastSeq = Math.max(_lastSeq, seq);
				return;
			}
			_lastSeq = Math.max(_lastSeq, seq);
		}

		_processMessage(msg);
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
		connectionHealth.set('good');
		reconnectAttempt = 0;
		_lastSeq = 0;
		_missedPongs = 0;
		_lastPongTime = Date.now();
		flushQueue();
		// Resync state after reconnect
		sendMessage({ type: 'get_status' });
		// Start heartbeat with health tracking
		if (_heartbeatTimer) clearInterval(_heartbeatTimer);
		_heartbeatTimer = setInterval(() => {
			if (ws?.readyState === WebSocket.OPEN) {
				// Check if previous pong was received
				if (Date.now() - _lastPongTime > PONG_TIMEOUT) {
					_missedPongs++;
					if (_missedPongs >= MAX_MISSED_PONGS) {
						connectionHealth.set('lost');
						// Force reconnect
						ws?.close();
						return;
					} else {
						connectionHealth.set('degraded');
					}
				}
				ws.send(JSON.stringify({ type: 'ping' }));
			}
		}, HEARTBEAT_INTERVAL);
	};

	ws.onmessage = handleMessage;

	ws.onclose = () => {
		connectionStatus.set('disconnected');
		connectionHealth.set('lost');
		ws = null;
		if (_heartbeatTimer) {
			clearInterval(_heartbeatTimer);
			_heartbeatTimer = null;
		}
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

/** Convenience: request a scan (debounced) */
export function sendScan() {
	_debouncedAction('scan', () => _sendWithAck({ type: 'scan' }), 300);
}

/** Async: request scan, resolves when server acks */
export function sendScanAsync(): Promise<void> {
	return _sendWithAck({ type: 'scan' });
}

/** Convenience: request system status (debounced) */
export function sendGetStatus() {
	_debouncedAction('get_status', () => _sendWithAck({ type: 'get_status' }), 300);
}

/** Async: request status, resolves when server acks */
export function sendGetStatusAsync(): Promise<void> {
	return _sendWithAck({ type: 'get_status' });
}

/** Convenience: toggle sarcasm (debounced) */
export function sendSarcasmToggle(enabled: boolean) {
	_debouncedAction('sarcasm', () => _sendWithAck({ type: 'sarcasm_toggle', enabled }), 300);
}

/** Convenience: request hologram render (debounced) */
export function sendHologramRequest() {
	_debouncedAction('hologram', () => _sendWithAck({ type: 'hologram_request' }), 500);
}

/** Async: request hologram, resolves when server acks */
export function sendHologramRequestAsync(): Promise<void> {
	return _sendWithAck({ type: 'hologram_request' });
}

/** Convenience: request vitals update (debounced) */
export function sendVitalsRequest() {
	_debouncedAction('vitals', () => _sendWithAck({ type: 'vitals_request' }), 500);
}

/** Async: request vitals, resolves when server acks */
export function sendVitalsRequestAsync(): Promise<void> {
	return _sendWithAck({ type: 'vitals_request' });
}

/** Convenience: send interrupt (immediate, no debounce) */
export function sendInterrupt() {
	sendMessage({ type: 'interrupt' });
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
