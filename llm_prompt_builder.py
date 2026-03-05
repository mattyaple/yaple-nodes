import os
import folder_paths


class LLMPromptBuilder:
    """
    Builds LLM prompts with optional system prompt from file and creative enhancement flag.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get ComfyUI base path and construct LLM_input directory path
        comfy_path = os.path.dirname(folder_paths.models_dir)
        llm_input_dir = os.path.join(comfy_path, "LLM_input")

        # Scan for .md and .txt files
        prompt_files = []
        if os.path.exists(llm_input_dir):
            for file in sorted(os.listdir(llm_input_dir)):
                if file.endswith(('.md', '.txt')):
                    prompt_files.append(file)

        # If no files found, provide a default placeholder
        if not prompt_files:
            prompt_files = ["(no files found)"]

        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Your main prompt or instructions"
                }),
                "system_prompt_file": (prompt_files, {
                    "tooltip": "Select a .md or .txt file from ComfyUI/LLM_input directory"
                }),
                "creatively_enhance": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If checked, prepends '**CREATIVELY ENHANCE PROMPT**' to the prompt output"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "system_prompt")
    FUNCTION = "build_prompt"
    CATEGORY = "llm/prompting"

    def build_prompt(self, prompt, system_prompt_file, creatively_enhance):
        """
        Build LLM prompt and system prompt.

        Args:
            prompt: User's main prompt text
            system_prompt_file: Selected file name from LLM_input directory
            creatively_enhance: Whether to add creative enhancement header

        Returns:
            Tuple of (prompt, system_prompt)
        """
        # Build the main prompt
        output_prompt = prompt
        if creatively_enhance:
            output_prompt = "**CREATIVELY ENHANCE PROMPT**\n" + output_prompt
            print("[LLMPromptBuilder] Creative enhancement enabled")

        # Read system prompt from file
        system_prompt = ""
        if system_prompt_file and system_prompt_file != "(no files found)":
            comfy_path = os.path.dirname(folder_paths.models_dir)
            llm_input_dir = os.path.join(comfy_path, "LLM_input")
            file_path = os.path.join(llm_input_dir, system_prompt_file)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    system_prompt = f.read()
                print(f"[LLMPromptBuilder] Loaded system prompt from: {system_prompt_file}")
                print(f"[LLMPromptBuilder] System prompt length: {len(system_prompt)} characters")
            except Exception as e:
                print(f"[LLMPromptBuilder] ERROR reading file {file_path}: {e}")
                system_prompt = f"Error reading file: {e}"
        else:
            print("[LLMPromptBuilder] No system prompt file available")

        print(f"[LLMPromptBuilder] Output prompt length: {len(output_prompt)} characters")

        return (output_prompt, system_prompt)


NODE_CLASS_MAPPINGS = {
    "LLMPromptBuilder": LLMPromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptBuilder": "LLM Prompt Builder",
}
