import { $ } from "bun";

console.log("[prunella] Building...");

await $`rm -rf dist`;
await Bun.build({
	entrypoints: ["src/index.ts"],
	outdir: "dist",
	target: "node",
	format: "esm",
	external: ["ai", "dedent"],
});

await $`bunx tsc -p tsconfig.build.json --emitDeclarationOnly`;

console.log("[prunella] Build complete.");
