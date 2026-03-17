import { app } from "../../scripts/app.js";

// Mask ollama_api_key widget as a password field on the WebSearch node
app.registerExtension({
    name: "OllamaDescriber.ApiKeyMask",
    async nodeCreated(node) {
        if (node.comfyClass !== "OllamaTool_WebSearch") return;

        // Find the widget by name and change its DOM input to type=password
        const applyMask = () => {
            for (const widget of node.widgets ?? []) {
                if (widget.name === "ollama_api_key" && widget.inputEl) {
                    widget.inputEl.type = "password";
                    widget.inputEl.autocomplete = "off";
                    break;
                }
            }
        };

        // Apply immediately and also after a short delay (ComfyUI may render widgets async)
        applyMask();
        setTimeout(applyMask, 300);
    }
});
