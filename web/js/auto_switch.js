import { app } from "../../../scripts/app.js";

// LGraphEventMode.BYPASS = 4 (confirmed from ComfyUI frontend bundle)
const BYPASS_MODE = 4;

app.registerExtension({
    name: "yaple.AutoSwitch",

    setup() {
        const origGraphToPrompt = app.graphToPrompt.bind(app);

        app.graphToPrompt = async function (...args) {
            const result = await origGraphToPrompt(...args);

            for (const node of app.graph._nodes) {
                if (node.comfyClass !== "AutoSwitch") continue;

                const nodeId = String(node.id);
                if (!result.output[nodeId]) continue;

                const inputs = result.output[nodeId].inputs;

                for (const inputName of ["a", "b"]) {
                    // Find the LiteGraph input slot by name
                    const slot = node.inputs?.find((s) => s.name === inputName);
                    if (!slot || slot.link == null) continue;

                    // Check the direct source node (before any bypass chain following)
                    const link = app.graph.links[slot.link];
                    if (!link) continue;

                    const sourceNode = app.graph.getNodeById(link.origin_id);
                    if (!sourceNode) continue;

                    // If the direct source is bypassed, remove this input so
                    // Python receives None and auto-detection kicks in
                    if (sourceNode.mode === BYPASS_MODE) {
                        delete inputs[inputName];
                    }
                }
            }

            return result;
        };
    },
});
