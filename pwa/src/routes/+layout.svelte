<script lang="ts">
	let { children } = $props();
	import '../app.css';
	import { onMount } from 'svelte';
	import { initPwaPrompt } from '$lib/stores/pwa';

	onMount(() => {
		initPwaPrompt();
	});
</script>

{@render children()}
