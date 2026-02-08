<!--
  Conversation history panel. Shows user/assistant/system messages,
  server-side interim transcripts, and a typing indicator while LLM thinks.
-->
<script lang="ts">
	import { chatHistory, orchestratorStatus, serverInterimTranscript } from '$lib/stores/connection';
	import { interimTranscript } from '$lib/stores/voice';
	import { tick } from 'svelte';

	let scrollContainer: HTMLDivElement;
	let history = $derived($chatHistory);
	let interim = $derived($interimTranscript);
	let serverInterim = $derived($serverInterimTranscript);
	let orchStatus = $derived($orchestratorStatus);
	let isThinking = $derived(orchStatus.includes('Thinking'));

	$effect(() => {
		// Auto-scroll to bottom on new messages, interim, or thinking
		if (history.length || interim || serverInterim || isThinking) {
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
		{#if history.length === 0 && !isThinking}
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
					{#if isSystem}
						<span class="text-[0.65rem] font-bold uppercase tracking-wider opacity-70 mr-1.5">Proactive</span>
					{/if}
					{msg.text}
				</div>
			</div>
		{/each}

		<!-- Server-side interim transcript (STT partial from Jetson) -->
		{#if serverInterim}
			<div class="flex justify-end">
				<div class="max-w-[85%] px-4 py-2.5 rounded-2xl text-sm opacity-50 italic
					bg-[var(--color-jarvis-cyan)]/5 text-[var(--color-jarvis-cyan)] border border-[var(--color-jarvis-cyan)]/10">
					{serverInterim}…
				</div>
			</div>
		{/if}

		<!-- Client-side Web Speech interim transcript -->
		{#if interim}
			<div class="flex justify-end">
				<div class="max-w-[85%] px-4 py-2.5 rounded-2xl text-sm opacity-60 italic
					bg-[var(--color-jarvis-cyan)]/5 text-[var(--color-jarvis-cyan)] border border-[var(--color-jarvis-cyan)]/10">
					{interim}…
				</div>
			</div>
		{/if}

		<!-- Typing indicator while LLM is thinking -->
		{#if isThinking}
			<div class="flex justify-start">
				<div class="px-4 py-3 rounded-2xl glass glow-cyan flex items-center gap-1.5">
					<span class="typing-dot"></span>
					<span class="typing-dot" style="animation-delay: 0.15s"></span>
					<span class="typing-dot" style="animation-delay: 0.3s"></span>
				</div>
			</div>
		{/if}
	</div>
</div>

<style>
	.typing-dot {
		display: inline-block;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--color-jarvis-cyan);
		opacity: 0.6;
		animation: typing-bounce 1s ease-in-out infinite;
	}

	@keyframes typing-bounce {
		0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
		30% { transform: translateY(-6px); opacity: 1; }
	}
</style>
