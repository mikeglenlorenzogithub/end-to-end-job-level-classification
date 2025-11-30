
# LLM Gemini Dependencies
from typing import List, Tuple, Optional, Any
from google.genai import types, Client

# Convert Field Defs to Gemini Schema
def _build_schema_from_defs(defs: Any) -> types.Schema:
    if isinstance(defs, str):
        # Primitive type
        if defs == "array_of_strings" or defs == "array":
            # Array of simple strings
            return types.Schema(type="array", items=types.Schema(type="string"))
        return types.Schema(type=defs)

    if isinstance(defs, tuple):
        if len(defs) == 2:
            name, content = defs
            # "v" tuple just unwraps
            if name == "v":
                return _build_schema_from_defs(content)
            return types.Schema(type="object", properties=_build_schema_from_defs(content))
        elif len(defs) == 3:
            # (name, children, "array") -> array of objects
            name, children, marker = defs
            if marker == "array":
                return types.Schema(type="array", items=_build_schema_from_defs(children))

    if isinstance(defs, list):
        props = {}
        required = []
        for part in defs:
            if len(part) == 3 and part[2] == "array":
                # Array of objects
                name, sub, _ = part
                props[name] = types.Schema(type="array", items=_build_schema_from_defs(sub))
            else:
                name, sub = part
                if sub == "array_of_strings" or sub == "array":
                    props[name] = types.Schema(type="array", items=types.Schema(type="string"))
                else:
                    props[name] = _build_schema_from_defs(sub)
            required.append(name)
        return types.Schema(type="object", properties=props, required=required)

    raise ValueError(f"Invalid schema definition: {defs}")


# Build Structured Output Format (Mime)
def _build_types_schema(field_defs: List[Tuple[str, Any]]) -> types.Schema:
    """
    Build the top-level schema as an ARRAY of OBJECTS.

    Each top-level tuple (name, sub) becomes a property on the item object.
      - If sub is a list -> property = array(items = object defined by sub)
      - If sub is a string/tuple -> property = schema returned by _build_schema_from_defs
    """
    item_props = {}
    item_required = []

    for name, sub in field_defs:
        if isinstance(sub, list):
            item_schema = _build_schema_from_defs(sub)
            item_props[name] = types.Schema(type="array", items=item_schema)
        else:
            item_props[name] = _build_schema_from_defs(sub)

        item_required.append(name)

    return types.Schema(
        type="array",
        items=types.Schema(
            type="object",
            properties=item_props,
            required=item_required
        )
    )


# Process Gemini Response
def gemini_process_response(
    model_version: str,
    api_key: str,
    prompt_template: str,
    input: str,
    column_uid: str,
    response_key_list: Optional[List[Tuple[str, str]]] = None
):
    """
    Request Gemini model and parse structured list of dicts
    """
    # Ensure uid is always included
    if response_key_list is None:
        response_key_list = []
    response_key_list = [(column_uid, "string")] + response_key_list

    # Build schema from response_key_list
    response_schema = _build_types_schema(response_key_list)

    client = Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_version,
        contents=(
            prompt_template +
            " Output must include uid exactly as provided in the input, without trimming, chopping, or normalization."
            " list input: " + input
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0,
            top_p=1,
            top_k=1,
        ),
    )

    if response.candidates[0].finish_reason.name != 'STOP':
        raise ValueError("response hit token output limit:", response.usage_metadata.candidates_token_count)

    return response