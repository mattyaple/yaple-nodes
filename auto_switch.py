class AnyType(str):
    """Wildcard type that satisfies ComfyUI's type-checking for any connection."""
    def __ne__(self, other):
        return False


any_type = AnyType("*")


class AutoSwitch:
    """
    Smart switch that auto-selects the active input.

    Priority:
      1. If only one of A/B is connected (or the other's source node is muted),
         automatically use the active input — no toggle needed.
      2. If both are active, use the 'select_b' boolean to choose.
      3. If neither is active, returns None.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select_b": ("BOOLEAN", {
                    "default": False,
                    "label_on": "B",
                    "label_off": "A",
                    "tooltip": "Fallback toggle used only when both A and B are active",
                }),
            },
            "optional": {
                "a": (any_type,),
                "b": (any_type,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "yaple/utils"

    def _node_label(self, unique_id, extra_pnginfo):
        """Return the user-assigned node title, falling back to 'AutoSwitch'."""
        try:
            nodes = (extra_pnginfo or {}).get("workflow", {}).get("nodes", [])
            for node in nodes:
                if str(node.get("id")) == str(unique_id):
                    return node.get("title") or "AutoSwitch"
        except Exception:
            pass
        return "AutoSwitch"

    def switch(self, select_b, unique_id=None, extra_pnginfo=None, a=None, b=None):
        label = self._node_label(unique_id, extra_pnginfo)
        a_active = a is not None
        b_active = b is not None

        if a_active and not b_active:
            print(f"[{label}] Auto-selected A (B is inactive)")
            return (a,)
        if b_active and not a_active:
            print(f"[{label}] Auto-selected B (A is inactive)")
            return (b,)

        if not a_active and not b_active:
            print(f"[{label}] WARNING: Both inputs inactive — returning None")
            return (None,)

        # Both active — use the boolean toggle
        chosen = "B" if select_b else "A"
        print(f"[{label}] Both inputs active — toggle selected {chosen}")
        return (b,) if select_b else (a,)


NODE_CLASS_MAPPINGS = {
    "AutoSwitch": AutoSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoSwitch": "Auto Switch",
}
