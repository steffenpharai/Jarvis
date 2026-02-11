<!--
  Voice controls: mic button (push-to-talk), text input, quick actions (Scan, Status, Sarcasm).
  Keyboard: Tab to mic, Enter to send.
-->
<script lang="ts">
	import { sendText, sendScan, sendGetStatus, sendSarcasmToggle, sendHologramRequest, sendVitalsRequest, connectionStatus } from '$lib/stores/connection';
	import {
		isListening,
		speechSupported,
		voiceMode,
		startListening,
		stopListening,
		toggleListening,
		setVoiceMode
	} from '$lib/stores/voice';

	let textInput = $state('');
	let listening = $derived($isListening);
	let supported = $derived($speechSupported);
	let mode = $derived($voiceMode);
	let sarcasmOn = $state(false);
	let connected = $derived($connectionStatus === 'connected');

	// Loading states for buttons (visual feedback while waiting for ack)
	let scanLoading = $state(false);
	let statusLoading = $state(false);
	let holoLoading = $state(false);
	let vitalsLoading = $state(false);

	function handleSend() {
		const text = textInput.trim();
		if (!text) return;
		sendText(text);
		textInput = '';
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	}

	function handleMicDown() {
		if (mode === 'push-to-talk') {
			startListening();
		}
	}

	function handleMicUp() {
		if (mode === 'push-to-talk') {
			stopListening();
		}
	}

	function toggleSarcasm() {
		sarcasmOn = !sarcasmOn;
		sendSarcasmToggle(sarcasmOn);
	}

	function handleScan() {
		if (scanLoading) return;
		scanLoading = true;
		sendScan();
		setTimeout(() => { scanLoading = false; }, 3000);
	}

	function handleStatus() {
		if (statusLoading) return;
		statusLoading = true;
		sendGetStatus();
		setTimeout(() => { statusLoading = false; }, 3000);
	}

	function handleHolo() {
		if (holoLoading) return;
		holoLoading = true;
		sendHologramRequest();
		setTimeout(() => { holoLoading = false; }, 5000);
	}

	function handleVitals() {
		if (vitalsLoading) return;
		vitalsLoading = true;
		sendVitalsRequest();
		setTimeout(() => { vitalsLoading = false; }, 3000);
	}
</script>

<div class="glass border-t border-[var(--color-jarvis-border)] p-3 space-y-3">
	<!-- Quick actions row -->
	<div class="flex items-center justify-center gap-2 flex-wrap">
		<button
			onclick={handleScan}
			disabled={!connected || scanLoading}
			class="px-3 py-1.5 rounded-lg text-xs font-medium uppercase tracking-wider
				bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)]
				text-[var(--color-jarvis-cyan)] hover:border-[var(--color-jarvis-cyan)]/40
				transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
			aria-label="Scan with camera"
		>
			{scanLoading ? 'Scanning...' : 'Scan'}
		</button>
		<button
			onclick={handleStatus}
			disabled={!connected || statusLoading}
			class="px-3 py-1.5 rounded-lg text-xs font-medium uppercase tracking-wider
				bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)]
				text-[var(--color-jarvis-text)] hover:border-[var(--color-jarvis-cyan)]/40
				transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
			aria-label="Get system status"
		>
			{statusLoading ? 'Loading...' : 'Status'}
		</button>
		<button
			onclick={handleHolo}
			disabled={!connected || holoLoading}
			class="px-3 py-1.5 rounded-lg text-xs font-medium uppercase tracking-wider
				bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)]
				text-[var(--color-jarvis-magenta)] hover:border-[var(--color-jarvis-magenta)]/40
				transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
			aria-label="Request 3D hologram"
		>
			{holoLoading ? 'Rendering...' : 'Holo'}
		</button>
		<button
			onclick={handleVitals}
			disabled={!connected || vitalsLoading}
			class="px-3 py-1.5 rounded-lg text-xs font-medium uppercase tracking-wider
				bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)]
				text-[var(--color-jarvis-green)] hover:border-[var(--color-jarvis-green)]/40
				transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
			aria-label="Request vitals update"
		>
			{vitalsLoading ? 'Loading...' : 'Vitals'}
		</button>
		<button
			onclick={toggleSarcasm}
			class="px-3 py-1.5 rounded-lg text-xs font-medium uppercase tracking-wider
				border transition-colors
				{sarcasmOn
					? 'bg-[var(--color-jarvis-magenta)]/20 border-[var(--color-jarvis-magenta)]/40 text-[var(--color-jarvis-magenta)]'
					: 'bg-[var(--color-jarvis-card)] border-[var(--color-jarvis-border)] text-[var(--color-jarvis-muted)]'}"
			aria-label="Toggle sarcasm mode"
			aria-pressed={sarcasmOn}
		>
			Sarcasm
		</button>
		<button
			onclick={() => setVoiceMode(mode === 'push-to-talk' ? 'always-on' : 'push-to-talk')}
			class="px-3 py-1.5 rounded-lg text-xs font-medium uppercase tracking-wider
				bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)]
				text-[var(--color-jarvis-muted)] hover:border-[var(--color-jarvis-cyan)]/40
				transition-colors"
			aria-label="Toggle voice mode"
		>
			{mode === 'push-to-talk' ? 'PTT' : 'Always On'}
		</button>
	</div>

	<!-- Input row: text + mic -->
	<div class="flex items-center gap-2">
		<input
			type="text"
			bind:value={textInput}
			onkeydown={handleKeydown}
			placeholder="Type a message…"
			class="flex-1 bg-[var(--color-jarvis-card)] border border-[var(--color-jarvis-border)] rounded-xl
				px-4 py-2.5 text-sm text-[var(--color-jarvis-text)] placeholder-[var(--color-jarvis-muted)]
				focus:outline-none focus:border-[var(--color-jarvis-cyan)]/50 transition-colors"
			aria-label="Type a message"
		/>

		<button
			onclick={handleSend}
			class="p-2.5 rounded-xl bg-[var(--color-jarvis-cyan)]/10 border border-[var(--color-jarvis-cyan)]/30
				text-[var(--color-jarvis-cyan)] hover:bg-[var(--color-jarvis-cyan)]/20 transition-colors"
			aria-label="Send message"
		>
			<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<line x1="22" y1="2" x2="11" y2="13"></line>
				<polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
			</svg>
		</button>

		{#if supported}
			<button
				onpointerdown={handleMicDown}
				onpointerup={handleMicUp}
				onpointerleave={handleMicUp}
				onclick={() => { if (mode === 'always-on') toggleListening(); }}
				class="p-3 rounded-full border-2 transition-all duration-300
					{listening
						? 'bg-[var(--color-jarvis-cyan)]/20 border-[var(--color-jarvis-cyan)] glow-pulse'
						: 'bg-[var(--color-jarvis-card)] border-[var(--color-jarvis-border)] hover:border-[var(--color-jarvis-cyan)]/40'}"
				aria-label={listening ? 'Stop listening' : 'Start listening'}
				aria-pressed={listening}
			>
				<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none"
					stroke={listening ? 'var(--color-jarvis-cyan)' : 'var(--color-jarvis-muted)'}
					stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
					<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
					<path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
					<line x1="12" y1="19" x2="12" y2="23"></line>
					<line x1="8" y1="23" x2="16" y2="23"></line>
				</svg>
			</button>
		{/if}
	</div>
</div>
