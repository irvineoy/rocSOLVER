**Role & Goal**

You are a code-aware data curator. Build a **Supervised Fine-Tuning (SFT)** dataset for HIP GPU kernels from the rocSOLVER repository. You need to go through all the digestion YAMLs under kernelgen folder. For each Yaml file, please generate adequate data entries for each level of Level1/Level2/Level3. The dataset must be **self-contained**: every item includes the **exact code excerpts** needed to reason about the question **and**—for coding tasks—the **reference implementation from the rocsolver repo** inside the answer.

  

**Primary Sources**

- All digestion YAML files under kernelgen/ (e.g., kernelgen/roclapack_*.yaml).
- Source files referenced by these YAMLs **within this repo**.
- **Do not** hallucinate profiling outputs. 
  

**Hard Requirements**

1. **Embed code, not just paths.** Each data item must contain exact code excerpts (code_blocks) for context. Path strings are allowed for provenance only.
2. **Answer must include reference code for coding tasks.** For items tagged as coding tasks, include a vetted code excerpt from the rocsolver repo in answer_code_blocks (see schema).
3. **Repo-independent at inference.** Every item must be understandable without checking out the repo. Include minimal but sufficient code excerpts.
4. **No hallucinations.** Only assert facts grounded in YAML + code.
    


## **Question Design**

**Levels**
- Level 1 (Single Function/Kernel): Parameters, memory access patterns, LDS usage, synchronization, launch config, etc.
    
- Level 2 (Subsystem scope): Questions about how a small set of related kernels (2–5) work together to solve a specific subproblem. These require understanding data flow, ordering, or dependencies between kernels. Areas for optimization may include: 
	- Reducing redundant memory transfers between kernels 
	- Improving synchronization strategies or stream usage 
	- Load balancing across thread blocks when multiple kernels share data 
	- Kernel fusion opportunities (combining simple kernels to reduce launch overhead); Example: "How do the update kernels cooperate to apply both left and right orthogonal transformations in the reduction?" 

- Level 3 (Interface scope): Questions about the entire YAML-defined interface—including all host-side functions and GPU kernels that collectively implement one rocSOLVER API routine. These focus on high-level orchestration, design intent, and how the YAML entries expose functionality to end users. 
Areas for optimization may include: 
	- Choosing between algorithmic variants (blocked vs. unblocked, batched vs. non-batched) 
	- Managing workspace allocation and reuse across multiple kernels 
	- Streamlining host–device coordination across the entire routine 
	- Designing interfaces to maximize reusability and composability in rocSOLVER; Example: "How does the gebrd interface coordinate host routines and GPU kernels to transform a dense matrix into bidiagonal form?"


**Coding-heavy bias**
- Target **≥50%** of items as **coding tasks** (tag coding).
- Coding task types (examples):
    - “Implement/complete a device function or kernel given constraints.”
    - “Refactor for coalesced access / shared memory tiling / warp-shuffle.”
    - “Translate a host-side routine into a minimal GPU kernel skeleton.”
    - “Write a correctness-preserving change (e.g., boundary checks, leading dimension handling, batched loops).”
    - “Extract and explain an idiom (e.g., Householder update scaffolding) and provide the reference implementation.”
        


## **JSONL Schema (one JSON object per line)**

**Required fields**
- id (string): str(Timestamp)
- level (string): "L1" | "L2" | "L3".
- interface (string): e.g., "gebrd".
- instruction (string): Clear, testable prompt.
- context_text (string): Brief context from YAML (related source files/functions, roles).
- code_blocks (array): Context snippets you’re asking the model to reason about.
    - object fields:
        - path (string): repo-relative path for provenance
        - language (string): "cpp" | "hip" | "cuda-like-cpp" | "c"
        - content (string): **Exact** excerpt; minimal but sufficient
- answer (string): Best, concise solution/analysis.
- rationale (string): Why the answer is correct, grounded in the excerpts.
- tags (array): e.g., ["coding","lds","memory-coalescing","synchronization","panel-reduction", etc].


This is the overall prompt, I want you to do it one by one, please process the kernelgen/roclapack_gerqf.yaml, generate a jsonl file with the same name under kernelgen/dataset folder. Please read all related source files and generate more than 10 entries for each yaml. Feel free to use script to help you for the schema format


roclapack_gebrd.yaml - /root/rocSOLVER/kernelgen/roclapack_gebrd.yaml
roclapack_geqrf.yaml - /root/rocSOLVER/kernelgen/roclapack_geqrf.yaml
roclapack_potf2.yaml - /root/rocSOLVER/kernelgen/roclapack_potf2.yaml
roclapack_geqlf.yaml - /root/rocSOLVER/kernelgen/roclapack_geqlf.yaml
roclapack_gerqf.yaml - /root/rocSOLVER/kernelgen/roclapack_gerqf.yaml