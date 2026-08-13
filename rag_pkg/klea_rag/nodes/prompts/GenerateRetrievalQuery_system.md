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
- also extract any retrieval constraints the user states into the output filter fields
- year_from / year_to: set when the user names a publication year or year range (e.g. "between 2020 and 2025", "from 2018 onwards")
- journal: set the journal or publication venue when the user names one (e.g. "nature")
- authors: set when the user asks for papers by specific researchers
- keywords: set only when the user names a specific keyword or topic tag
- leave a filter field unset (null or empty) when the question does not state that constraint
