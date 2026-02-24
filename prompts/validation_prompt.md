# Validation Review

You are reviewing a generated CAD model against the original engineering drawing specifications.

## Generated Model Properties:
- Volume: {volume}
- Bounding Box: {bounding_box}
- Face Count: {face_count}
- Is Watertight: {is_watertight}

## Expected Specifications:
{expected_specs}

## Dimension Checks:
{dimension_checks}

## Feature Checks:
{feature_checks}

## Instructions:
1. Compare the generated model properties against the expected specifications
2. Identify any discrepancies in dimensions, features, or topology
3. Rate the overall quality: PASS, MARGINAL, or FAIL
4. If FAIL or MARGINAL, provide specific suggestions for improvement

## Response Format:
```json
{
  "quality_rating": "PASS|MARGINAL|FAIL",
  "dimension_issues": [],
  "feature_issues": [],
  "suggestions": []
}
```
