/**
 * Web Speech API wrapper – push-to-talk (default) and always-on modes.
 *
 * Uses SpeechRecognition with continuous: true, interimResults: true, lang: "en-GB".
 * On result, sends transcript to backend via WebSocket.
 * Fallback: could capture MediaStream and send audio to Jetson for Faster-Whisper.
 */

import { writable, get } from 'svelte/store';
import { sendText } from './connection';

// ── Types ────────────────────────────────────────────────────────────

type VoiceMode = 'push-to-talk' | 'always-on';

// Web Speech API types (not always in default TS lib)
/* eslint-disable @typescript-eslint/no-explicit-any */
type SpeechRecognitionAny = any;
/* eslint-enable @typescript-eslint/no-explicit-any */

// ── Stores ───────────────────────────────────────────────────────────

/** Whether the mic is currently active/recording */
export const isListening = writable<boolean>(false);

/** Current interim transcript (live while speaking) */
export const interimTranscript = writable<string>('');

/** Voice input mode */
export const voiceMode = writable<VoiceMode>('push-to-talk');

/** Whether Web Speech API is supported */
export const speechSupported = writable<boolean>(false);

// ── SpeechRecognition ────────────────────────────────────────────────

let recognition: SpeechRecognitionAny | null = null;
let isRunning = false;

function getSpeechRecognition(): SpeechRecognitionAny | null {
	if (typeof window === 'undefined') return null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const w = window as any;
	const SR = w.SpeechRecognition || w.webkitSpeechRecognition;
	if (!SR) return null;
	return new SR();
}

export function initSpeechRecognition() {
	recognition = getSpeechRecognition();
	if (!recognition) {
		speechSupported.set(false);
		return;
	}
	speechSupported.set(true);

	recognition.continuous = true;
	recognition.interimResults = true;
	recognition.lang = 'en-GB';

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	recognition.onresult = (event: any) => {
		let interim = '';
		let final = '';

		for (let i = event.resultIndex; i < event.results.length; i++) {
			const transcript = event.results[i][0].transcript;
			if (event.results[i].isFinal) {
				final += transcript;
			} else {
				interim += transcript;
			}
		}

		interimTranscript.set(interim);

		if (final.trim()) {
			interimTranscript.set('');
			sendText(final.trim());

			// In push-to-talk mode, stop after getting a final result
			if (get(voiceMode) === 'push-to-talk') {
				stopListening();
			}
		}
	};

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	recognition.onerror = (event: any) => {
		if (event.error === 'no-speech' || event.error === 'aborted') {
			// Normal in push-to-talk
			return;
		}
		if (event.error === 'network') {
			// Network error (common on Jetson LAN) — retry after 1s
			console.warn('Speech recognition network error — retrying in 1s');
			isRunning = false;
			setTimeout(() => {
				if (get(isListening)) {
					try {
						recognition?.start();
						isRunning = true;
					} catch {
						isListening.set(false);
					}
				}
			}, 1000);
			return;
		}
		if (event.error === 'not-allowed') {
			// Mic permission denied — disable mic button
			console.warn('Microphone permission denied');
			speechSupported.set(false);
			isListening.set(false);
			isRunning = false;
			return;
		}
		console.warn('Speech recognition error:', event.error);
		isListening.set(false);
		isRunning = false;
	};

	recognition.onend = () => {
		isRunning = false;
		// In always-on mode, restart automatically
		if (get(voiceMode) === 'always-on' && get(isListening)) {
			try {
				recognition?.start();
				isRunning = true;
			} catch {
				isListening.set(false);
			}
		} else {
			isListening.set(false);
		}
	};
}

export function startListening() {
	if (!recognition || isRunning) return;

	// Haptic feedback
	if (navigator.vibrate) {
		navigator.vibrate(50);
	}

	try {
		recognition.start();
		isRunning = true;
		isListening.set(true);
		interimTranscript.set('');
	} catch {
		isListening.set(false);
	}
}

export function stopListening() {
	if (!recognition || !isRunning) return;

	try {
		recognition.stop();
	} catch {
		// Ignore
	}
	isRunning = false;
	isListening.set(false);
	interimTranscript.set('');
}

export function toggleListening() {
	if (get(isListening)) {
		stopListening();
	} else {
		startListening();
	}
}

export function setVoiceMode(mode: VoiceMode) {
	voiceMode.set(mode);
	if (mode === 'push-to-talk' && get(isListening)) {
		stopListening();
	}
}
