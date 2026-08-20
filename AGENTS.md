# AGENTS.md

Guidance for coding agents that work in this repository.

`CLAUDE.md` holds the full guidance. Read it. This file repeats the language
rule, because that rule applies to every reply and every file you write.

## Language: ASD-STE100

Write all text in ASD-STE100 Simplified Technical English (STE).

This rule applies to:

- Replies to the user
- Code comments and docstrings
- Comments in config files
- Documentation and commit messages

### Writing rules

| Rule | Requirement |
|------|-------------|
| Words | Use only approved words. Technical names and technical verbs are permitted. |
| One meaning | Give each word one meaning and one part of speech. |
| Sentence length | Instructions: 20 words maximum. Descriptions: 25 words maximum. |
| Paragraphs | Write one topic in one paragraph. Use 6 sentences maximum. |
| Voice | Use the active voice. |
| Tense | Use the simple tenses. Do not use the perfect or the progressive tenses. |
| Verbs that end in -ing | Do not use them, unless they are a technical name. Write a clause. |
| Instructions | Write one instruction in one sentence. |
| Articles | Write `the` or `a` where you can. |
| Noun clusters | Use 3 words maximum. |
| Lists | Put complex data in a vertical list. |
| Warnings | Write the warning before the step, not after it. |

### Words to replace

| Do not write | Write |
|--------------|-------|
| utilize | use |
| perform | do |
| prior to | before |
| in order to | to |
| due to | because of |
| ensure | make sure |
| via | with, by |
| obtain | get |
| attempt | try |
| terminate | stop |
| initiate | start |
| additional | more |
| approximately | about |
| however | but |
| therefore | thus |

### Examples

| Do not write | Write |
|--------------|-------|
| Removing the persona changed the judgments. | The judgments changed when we removed the persona. |
| The cache is being shared by concurrent jobs. | Concurrent jobs share the cache. |
| It's uninterpretable without an anchor. | You cannot read this value without an anchor. |
| This has been fixed. | We fixed this. |

## Documentation

Write project information in `vlm-narratives-docs/`.

**Warning:** The Obsidian wiki at `vault/context/` is no longer maintained. Do
not write to it. Read it only for history.
