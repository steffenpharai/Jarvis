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

// ── WebSocket Manager ────────────────────────────────────────────────

let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectAttempt = 0;
const MAX_RECONNECT_DELAY = 30_000;
const BASE_RECONNECT_DELAY = 1_000;

/** Offline command queue – flushed on reconnect */
const offlineQueue: string[] = [];

/** Current server host (configurable) */
export const serverHost = writable<string>(
	typeof window !== 'undefined' ? window.location.hostname + ':8000' : 'localhost:8000'
);

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
				// Could show interim transcript in UI; for now skip
				break;

			case 'transcript_final':
				chatHistory.update((h) => [
					...h,
					{ role: 'user', text: msg.text as string, timestamp: Date.now() }
				]);
				break;

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
				// Handled by dashboard
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
