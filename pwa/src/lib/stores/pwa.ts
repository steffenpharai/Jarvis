/**
 * PWA install prompt handler.
 *
 * Captures `beforeinstallprompt` and exposes it so the UI can show
 * an "Add to home screen" button after second visit.
 */

import { writable } from 'svelte/store';

export const canInstall = writable<boolean>(false);

let deferredPrompt: BeforeInstallPromptEvent | null = null;

interface BeforeInstallPromptEvent extends Event {
	prompt(): Promise<void>;
	userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

export function initPwaPrompt() {
	if (typeof window === 'undefined') return;

	window.addEventListener('beforeinstallprompt', (e) => {
		e.preventDefault();
		deferredPrompt = e as BeforeInstallPromptEvent;
		canInstall.set(true);
	});
}

export async function installPwa() {
	if (!deferredPrompt) return;
	await deferredPrompt.prompt();
	const { outcome } = await deferredPrompt.userChoice;
	if (outcome === 'accepted') {
		canInstall.set(false);
	}
	deferredPrompt = null;
}
