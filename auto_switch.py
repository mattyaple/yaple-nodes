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
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"
    CATEGORY = "yaple/utils"

    def switch(self, select_b, a=None, b=None):
        a_active = a is not None
        b_active = b is not None

        if a_active and not b_active:
            return (a,)
        if b_active and not a_active:
            return (b,)

        # Both active or neither active — fall back to the boolean toggle
        return (b,) if select_b else (a,)


NODE_CLASS_MAPPINGS = {
    "AutoSwitch": AutoSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoSwitch": "Auto Switch",
}
