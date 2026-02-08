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

			case 'reply':
				chatHistory.update((h) => [
					...h,
					{ role: 'assistant', text: msg.text as string, timestamp: Date.now() }
				]);
				break;

			case 'detections':
			case 'scan_result':
				detections.set({
					detections: (msg.detections as unknown[]) || [],
					description: (msg.description as string) || ''
				});
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
