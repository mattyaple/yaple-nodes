import os
import re


class FilenameFormatter:
    """
    Formats a filename string for use as an output name.

    Processing order:
      1. Strip file extension
      2. Strip trailing numbered suffix (e.g. _00001, -042)
      3. Find / replace
      4. Prepend prefix and/or append suffix
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"forceInput": True}),
                "strip_numbered_suffix": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove a trailing separator+digits before the extension (e.g. _00001, -042)",
                }),
                "find": ("STRING", {
                    "default": "",
                    "tooltip": "Text to find in the stem. Leave blank to skip.",
                }),
                "replace": ("STRING", {
                    "default": "",
                    "tooltip": "Replacement text. Only used when 'find' is non-empty.",
                }),
                "prefix": ("STRING", {
                    "default": "",
                    "tooltip": "Prepend this to the result.",
                }),
                "suffix": ("STRING", {
                    "default": "",
                    "tooltip": "Append this to the result.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "format_filename"
    CATEGORY = "yaple/utils"

    def format_filename(self, filename, strip_numbered_suffix, find, replace, prefix, suffix):
        # 1. Strip extension
        stem = os.path.splitext(filename)[0]

        # 2. Strip trailing numbered suffix: separator + digits at end of stem
        #    Matches patterns like _00001, -42, _1 but not bare digits inside a name
        if strip_numbered_suffix:
            stem = re.sub(r'[_\-\s]\d+$', '', stem)

        # 3. Find / replace
        if find:
            stem = stem.replace(find, replace)

        # 4. Prefix / suffix
        result = f"{prefix}{stem}{suffix}"

        return (result,)


NODE_CLASS_MAPPINGS = {
    "FilenameFormatter": FilenameFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FilenameFormatter": "Filename Formatter",
}
