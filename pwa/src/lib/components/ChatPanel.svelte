<!--
  Conversation history panel. Shows user/assistant/system messages.
  ARIA live region for new messages.
-->
<script lang="ts">
	import { chatHistory } from '$lib/stores/connection';
	import { interimTranscript } from '$lib/stores/voice';
	import { tick } from 'svelte';

	let scrollContainer: HTMLDivElement;
	let history = $derived($chatHistory);
	let interim = $derived($interimTranscript);

	$effect(() => {
		// Auto-scroll to bottom on new messages
		if (history.length) {
			tick().then(() => {
				if (scrollContainer) {
					scrollContainer.scrollTop = scrollContainer.scrollHeight;
				}
			});
		}
	});
</script>

<div
	class="flex flex-col h-full overflow-hidden"
	role="log"
	aria-label="Conversation"
	aria-live="polite"
>
	<div
		bind:this={scrollContainer}
		class="flex-1 overflow-y-auto px-3 py-4 space-y-3"
	>
		{#if history.length === 0}
			<p class="text-center text-[var(--color-jarvis-muted)] text-sm mt-8">
				Say something or type a message…
			</p>
		{/if}

		{#each history as msg (msg.timestamp)}
			{@const isUser = msg.role === 'user'}
			{@const isSystem = msg.role === 'system'}
			<div
				class="flex {isUser ? 'justify-end' : 'justify-start'}"
			>
				<div
					class="max-w-[85%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed
						{isUser
							? 'bg-[var(--color-jarvis-cyan)]/10 text-[var(--color-jarvis-cyan)] border border-[var(--color-jarvis-cyan)]/20'
							: isSystem
								? 'bg-[var(--color-jarvis-magenta)]/10 text-[var(--color-jarvis-magenta)] border border-[var(--color-jarvis-magenta)]/20'
								: 'glass glow-cyan text-[var(--color-jarvis-text)]'}"
				>
					{msg.text}
				</div>
			</div>
		{/each}

		{#if interim}
			<div class="flex justify-end">
				<div class="max-w-[85%] px-4 py-2.5 rounded-2xl text-sm opacity-60 italic
					bg-[var(--color-jarvis-cyan)]/5 text-[var(--color-jarvis-cyan)] border border-[var(--color-jarvis-cyan)]/10">
					{interim}…
				</div>
			</div>
		{/if}
	</div>
</div>
