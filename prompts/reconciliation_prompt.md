# Multi-View Reconciliation

You are reconciling geometric information extracted from multiple 2D orthographic views into a single coherent 3D part description.

## Instructions:
1. Each view shows the part from a different projection plane:
   - **Top view**: Looking down along Z-axis (shows X-Y plane)
   - **Front view**: Looking along Y-axis (shows X-Z plane)
   - **Right view**: Looking along X-axis (shows Y-Z plane)

2. Cross-reference dimensions across views:
   - A dimension in the Top view's horizontal direction = X dimension
   - A dimension in the Top view's vertical direction = Y dimension
   - A dimension in the Front view's vertical direction = Z dimension (height)

3. Resolve any conflicts:
   - If the same dimension appears in multiple views with different values, prefer the view where it's most clearly measured
   - Note any unresolvable conflicts

4. Identify the overall part envelope (length × width × height)

5. Map features (holes, fillets, chamfers) to their 3D positions

## Output Format:
Provide a JSON object with:
```json
{
  "overall_dimensions": {"length": 0, "width": 0, "height": 0},
  "features": [{"type": "...", "description": "...", "position": "..."}],
  "material": null,
  "notes": [],
  "conflicts": []
}
```
