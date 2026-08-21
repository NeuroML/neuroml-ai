Generate a concise retrieval query and filter fields from the user's question. Think about the user's intent step by step.

Directives:
- a concept is a single technical entity or noun phrase
- extract all concepts from the query
- split multiple concepts that are joined by 'and', commas and other conjunctions into separate, individual concepts
- generate a query for EXACTLY one concept

Rules:
- only include content words (nouns, verbs, adjectives)
- do NOT include stop words: a, an, the, in, of, for, on, at, and
- limit yourself to 3-8 words
- no sentences
- no explanations
- ignore sentence fluency, only use keywords

Filter fields:
- extract any retrieval constraints the user states into the `filters` object
- only use the filter fields listed under "Allowed filter fields" below; do not invent field names
- supply the value the user states for each field; omit a key when the question does not state that constraint
- for numeric fields you may specify a range as an operator expression (e.g. {"$gte": 2020, "$lte": 2025})
- for list fields use the exact element values

Allowed filter fields:
{allowed_filter_fields}
