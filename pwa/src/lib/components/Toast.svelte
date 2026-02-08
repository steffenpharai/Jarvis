<!--
  Auto-dismiss error toast. Shows lastError, dismisses after 5s or on click.
-->
<script lang="ts">
	import { lastError } from '$lib/stores/connection';

	let visible = $state(false);
	let message = $state('');
	let timer: ReturnType<typeof setTimeout> | null = null;

	$effect(() => {
		const err = $lastError;
		if (err) {
			message = err;
			visible = true;
			if (timer) clearTimeout(timer);
			timer = setTimeout(() => {
				visible = false;
				lastError.set('');
			}, 5000);
		}
	});

	function dismiss() {
		visible = false;
		lastError.set('');
		if (timer) clearTimeout(timer);
	}
</script>

{#if visible}
	<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
	<div
		onclick={dismiss}
		onkeydown={(e) => { if (e.key === 'Escape' || e.key === 'Enter') dismiss(); }}
		class="fixed top-14 right-3 z-50 max-w-sm px-4 py-3 rounded-xl text-sm
			bg-[var(--color-jarvis-red)]/15 border border-[var(--color-jarvis-red)]/40
			text-[var(--color-jarvis-red)] backdrop-blur-md shadow-lg
			cursor-pointer hover:bg-[var(--color-jarvis-red)]/25 transition-all"
		role="alert"
		aria-live="assertive"
		tabindex="0"
	>
		<div class="flex items-start gap-2">
			<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="flex-shrink-0 mt-0.5">
				<circle cx="12" cy="12" r="10"></circle>
				<line x1="12" y1="8" x2="12" y2="12"></line>
				<line x1="12" y1="16" x2="12.01" y2="16"></line>
			</svg>
			<span>{message}</span>
		</div>
	</div>
{/if}
