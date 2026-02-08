<!--
  Central listening orb – animated based on orchestrator status.
  Visible primarily in folded (cover) mode as the main visual focus.
-->
<script lang="ts">
	import { orchestratorStatus, wakeDetected } from '$lib/stores/connection';
	import { isListening } from '$lib/stores/voice';

	let status = $derived($orchestratorStatus);
	let listening = $derived($isListening);
	let wake = $derived($wakeDetected);

	let orbClass = $derived.by(() => {
		if (wake) return 'border-[var(--color-jarvis-red)] shadow-[0_0_40px_rgba(255,0,170,0.4)]';
		if (listening) return 'border-[var(--color-jarvis-cyan)] glow-pulse';
		if (status.includes('Thinking')) return 'border-[var(--color-jarvis-cyan)] animate-[spin-slow_3s_linear_infinite]';
		if (status.includes('Speaking')) return 'border-[var(--color-jarvis-green)] glow-cyan';
		return 'border-[var(--color-jarvis-border)]';
	});

	let label = $derived.by(() => {
		if (wake) return 'Wake Detected';
		if (listening) return 'Listening…';
		if (status.includes('Thinking')) return 'Thinking…';
		if (status.includes('Speaking')) return 'Speaking…';
		return 'Ready';
	});
</script>

<div class="flex flex-col items-center gap-4" role="status" aria-live="polite" aria-label={label}>
	<div
		class="w-32 h-32 rounded-full border-2 flex items-center justify-center transition-all duration-500 {orbClass}"
	>
		<div class="w-20 h-20 rounded-full bg-[var(--color-jarvis-surface)] flex items-center justify-center">
			<span class="font-[var(--font-heading)] text-[var(--color-jarvis-cyan)] text-2xl font-bold">J</span>
		</div>
	</div>
	<span class="text-sm text-[var(--color-jarvis-muted)] uppercase tracking-wider">{label}</span>
</div>
