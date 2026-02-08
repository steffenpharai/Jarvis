<!--
  Main page: fold-aware layout for Pixel 10 Pro Fold.

  Cover / folded (~6.4", < 600px): single column – status, orb, chat, controls.
  Inner / unfolded (8", >= 900px): grid – left 40% chat, right 60% camera + dashboard + reminders.
  Transition zone (600–900px): two-column with stacked right pane.

  Uses CSS Container Queries on the main container and media queries for breakpoints.
  Optional Device Posture API (navigator.devicePosture) as progressive enhancement.
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { connect, disconnect, hydrateChatHistory } from '$lib/stores/connection';
	import { initSpeechRecognition } from '$lib/stores/voice';

	import StatusBar from '$lib/components/StatusBar.svelte';
	import SettingsPanel from '$lib/components/SettingsPanel.svelte';
	import ListeningOrb from '$lib/components/ListeningOrb.svelte';
	import ChatPanel from '$lib/components/ChatPanel.svelte';
	import VoiceControls from '$lib/components/VoiceControls.svelte';
	import CameraStream from '$lib/components/CameraStream.svelte';
	import Dashboard from '$lib/components/Dashboard.svelte';
	import Reminders from '$lib/components/Reminders.svelte';
	import Toast from '$lib/components/Toast.svelte';

	let isFolded = $state(true);
	let isMidSize = $state(false);

	onMount(() => {
		// Hydrate persisted chat before connecting
		hydrateChatHistory();

		// Connect to Jarvis backend
		connect();
		initSpeechRecognition();

		// Fold detection via width (primary) and Device Posture API (progressive enhancement)
		const mqLarge = window.matchMedia('(min-width: 900px)');
		const mqMid = window.matchMedia('(min-width: 600px) and (max-width: 899px)');
		const updateLayout = () => {
			isFolded = !mqLarge.matches && !mqMid.matches;
			isMidSize = mqMid.matches;
		};
		updateLayout();
		mqLarge.addEventListener('change', updateLayout);
		mqMid.addEventListener('change', updateLayout);

		// Device Posture API (experimental, Chrome)
		if ('devicePosture' in navigator) {
			const posture = (navigator as unknown as { devicePosture: { type: string; addEventListener: (e: string, cb: () => void) => void } }).devicePosture;
			const updatePosture = () => {
				if (posture.type === 'folded') {
					isFolded = true;
				}
			};
			updatePosture();
			posture.addEventListener('change', updatePosture);
		}

		return () => {
			disconnect();
			mqLarge.removeEventListener('change', updateLayout);
			mqMid.removeEventListener('change', updateLayout);
		};
	});
</script>

<svelte:head>
	<title>Jarvis</title>
</svelte:head>

<Toast />

<div class="h-full flex flex-col" style="container-type: inline-size;">
	<!-- Top bar -->
	<div class="flex items-center justify-between">
		<div class="flex-1">
			<StatusBar />
		</div>
		<div class="absolute right-2 top-1.5 z-40">
			<SettingsPanel />
		</div>
	</div>

	<!-- Main content area -->
	{#if isFolded}
		<!-- ═══ FOLDED / COVER LAYOUT (~6.4") ═══ -->
		<div class="flex-1 flex flex-col min-h-0">
			<!-- Orb (centered, prominent) -->
			<div class="flex items-center justify-center py-6">
				<ListeningOrb />
			</div>

			<!-- Chat (scrollable, takes remaining space) -->
			<div class="flex-1 min-h-0">
				<ChatPanel />
			</div>

			<!-- Controls (fixed at bottom) -->
			<VoiceControls />
		</div>
	{:else if isMidSize}
		<!-- ═══ MID-SIZE LAYOUT (600–899px) ═══ -->
		<div class="flex-1 grid grid-cols-[1fr_1fr] gap-0 min-h-0">
			<!-- Left pane: orb + chat + controls -->
			<div class="flex flex-col min-h-0 border-r border-[var(--color-jarvis-border)]">
				<div class="flex items-center justify-center py-3 border-b border-[var(--color-jarvis-border)]">
					<ListeningOrb />
				</div>
				<div class="flex-1 min-h-0">
					<ChatPanel />
				</div>
				<VoiceControls />
			</div>

			<!-- Right pane: camera only (compact) -->
			<div class="flex flex-col min-h-0 overflow-y-auto p-3 gap-3">
				<CameraStream />
				<Dashboard />
			</div>
		</div>
	{:else}
		<!-- ═══ UNFOLDED / INNER LAYOUT (8"+) ═══ -->
		<div class="flex-1 grid grid-cols-[2fr_3fr] gap-0 min-h-0">
			<!-- Left pane: conversation (~40%) -->
			<div class="flex flex-col min-h-0 border-r border-[var(--color-jarvis-border)]">
				<!-- Small orb in unfolded mode -->
				<div class="flex items-center justify-center py-3 border-b border-[var(--color-jarvis-border)]">
					<ListeningOrb />
				</div>
				<div class="flex-1 min-h-0">
					<ChatPanel />
				</div>
				<VoiceControls />
			</div>

			<!-- Right pane: camera + dashboard + reminders (~60%) -->
			<div class="flex flex-col min-h-0 overflow-y-auto p-4 gap-4">
				<CameraStream />
				<div class="grid grid-cols-2 gap-4">
					<Dashboard />
					<Reminders />
				</div>
			</div>
		</div>
	{/if}
</div>

<style>
	/* Device Posture API media query (progressive enhancement) */
	@media (device-posture: folded) {
		/* Force single-column when physically folded */
	}

	/* High-contrast mode (toggled via settings) */
	:global(.high-contrast) {
		--color-jarvis-text: #ffffff;
		--color-jarvis-bg: #000000;
		--color-jarvis-cyan: #00ffff;
		--color-jarvis-card: #000000;
		--color-jarvis-border: #ffffff;
	}
</style>
