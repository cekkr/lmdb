# **A Database-Native Statistical Language Model (DB-SLM): An Architectural Blueprint for Tensor-Free Generation in MariaDB**

## **Executive Summary**

This report presents a novel architectural blueprint for a generative language model operating entirely within the MariaDB ecosystem, adhering to the specified constraint of zero tensor-based operations. The proposed system is a hybrid-model, hybrid-engine architecture designed to meet the three-level requirements specified:

* **Level 1 (Statistical Generation):** Implemented as a large-scale, read-optimized N-gram probability model. This layer is stored in relational tables utilizing the **Aria** storage engine, which is optimized for the read-heavy, analytical queries required for statistical lookup.  
* **Level 2 (Stateful Memory):** Implemented as a series of relational tables for conversation logging, contextual retrieval, and federated learning from corrections. To handle the distinct workloads, chat history logging will use the write-optimized **MyRocks** engine, while metadata and correction-loop tables will use the ACID-compliant **InnoDB** engine.  
* **Level 3 (Conceptual Generation):** Implemented as a hierarchical generation system that addresses the performance limitations of a pure token-level model. A high-level statistical model predicts abstract "concepts," which are stored in an **InnoDB** table. A "verbalization" module then decodes these concepts into human-readable text using relational templates.

The central thesis of this analysis is that while a *purely* token-level statistical model in SQL (Level 1\) is operationally non-viable for real-time applications due to I/O latency, the proposed **Level 3 conceptual generation is the critical innovation** that makes the entire system feasible. By amortizing the high cost of database queries over a sequence of "verbalized" tokens rather than a single token, the model shifts from a latency-bound token-generator to a more efficient concept-generator.

## **Part 1: Feasibility Analysis and Core Architectural Principles**

This section establishes the theoretical foundation for a "no-tensor" language model, distinguishing between the impractical approach of re-implementing Transformer models in SQL and the proposed, database-native statistical model.

### **1.1 Deconstructing the "No-Tensor" Constraint: Transformer vs. Statistical Models**

The "no-tensor" constraint is the defining design parameter. This requires a fundamental departure from modern Large Language Model (LLM) architecture, which is inextricably linked to tensor operations (i.e., matrix multiplications).1

* **The Transformer-in-SQL Fallacy:** Recent research, such as TranSQL, demonstrates the *theoretical* possibility of translating Transformer operations (like matrix multiplication and attention) into a series of SQL queries and relational tables.3 TranSQL represents model weights as relational tables and neural operations as SQL queries, leveraging database features like caching and data management.3 However, this is a *transpilation* of a tensor-based architecture onto a relational platform. It is extraordinarily inefficient, complex, and does not align with the query's request for a model based on "statistical linkage."  
* **The Database-Native Statistical Model (DB-SLM):** The query's "Level 1" requirement for "basic training" and "statistical linkage" points directly to classical statistical language models, specifically N-gram models 4 or Markov Chains.6 These models are *not* tensor-based. An N-gram model, which calculates the probability of the next word based on the *n-1* preceding words, is, at its core, a massive conditional probability lookup table.4 Such a model *is* a database.

Therefore, this design will not implement a Transformer. It will implement a true Database-Native Statistical Language Model (DB-SLM), for which a Relational Database Management System (RDBMS) is the natural, if not traditionally performant, execution environment. The "no-tensor" constraint is met by *model selection*, not by transpilation.

### **1.2 The Core Computational Trade-Off: GPU Memory Bandwidth vs. Database I/O Latency**

The choice of a DB-SLM over a tensor-based LLM results in a direct and significant trade-off in the system's computational bottleneck.

* **Transformer Bottlenecks:** Modern LLMs are bound by a dichotomy of computation and memory. The prefill stage (processing the prompt) is compute-intensive, while the decoding stage (generating tokens) is memory-bound, limited by the speed at which weights can be read from GPU memory (memory bandwidth).8 The entire architecture is designed for massive, parallel in-memory operations on tensors.2  
* **DB-SLM Bottlenecks:** The proposed DB-SLM will be fundamentally **I/O-bound** and **CPU-bound**.  
  * **I/O Bound:** Every generated token will require, at minimum, one SELECT query against a potentially petabyte-scale probability table. The system's performance will be limited by disk speed, index efficiency, and the database's ability to cache the "hot" parts of the N-gram table.12  
  * **CPU Bound:** Complex retrievals for Level 2 memory (a form of Retrieval-Augmented Generation, or RAG) and the Level 3 concept-stitching will involve complex JOINs, GROUP BY clauses, and other relational operations, which are CPU-intensive.14

This trade-off reveals the central challenge: a standard LLM performs *one* forward pass (a memory-bound operation) to derive the probability distribution for *all* tokens in its vocabulary. In contrast, the DB-SLM must perform *one* database query (an I/O-bound operation) to retrieve the *same* information. The true bottleneck is this **per-token query latency**. A 200-token response would necessitate 200 sequential, high-latency database queries, making real-time interaction impossible.  
This seemingly fatal flaw is the primary justification for the proposed **Level 3 conceptual prediction model**. A system that predicts *one concept* and then "verbalizes" it into *50 tokens* amortizes this I/O cost by a factor of 50\. This innovation, which has parallels in Multi-Token Prediction (MTP) research 16, makes Level 3 the *core* of the architecture, not a mere add-on.

## **Part 2: The Hybrid-Engine MariaDB Strategy**

The query requests the "fastest engine for this task." However, the system's three functional levels (statistical lookup, memory logging, and relational control) have radically different workloads. No single engine is "fastest" for all of them. The solution is a hybrid-engine architecture, leveraging MariaDB's pluggable storage engine feature.19

### **2.1 Workload Analysis and Engine Selection**

#### **Workload 1 (Level 1): Statistical Lookups**

* **Profile:** This workload is defined by the Level 1 N-gram probability tables, which could contain trillions of rows. This table will be almost exclusively read-only, with appends occurring only during large, offline "retraining" events. The primary generation query will involve a WHERE clause on the context, followed by GROUP BY or ORDER BY... LIMIT 1 to find the token with the maximum probability.  
* **Engine:** **Aria**. The Aria storage engine is MariaDB's modern, crash-safe replacement for MyISAM.20 Benchmarks demonstrate that Aria is "much faster at the GROUP BY level than either InnoDB or MyISAM".22 In one performance test, Aria was shown to be **four times faster than InnoDB** for these types of queries.22 Its optimization for read-heavy workloads and superior caching makes it the clear choice for the primary statistical lookup tables.22

#### **Workload 2 (Level 2): Memory Logging**

* **Profile:** This workload is defined by the messages table, which logs every user and assistant turn. This is a write-intensive, append-only workload. The table will grow indefinitely and could become the largest in the system by disk footprint.  
* **Engine:** **MyRocks**. MyRocks is a write-optimized storage engine based on a Log-Structured Merge-tree (LSM-tree) architecture.23 It is designed for high write performance and high compression. It features "10x less write amplification" and "2x better compression than InnoDB".19 This is critical for managing the storage footprint and improving the endurance of flash storage for a table that will receive billions of writes.25 We must, however, accept the known trade-off: MyRocks is "behind InnoDB in quick 'index lookup' queries".26 This is an acceptable compromise for a log-structured table where write performance and storage efficiency are paramount.

#### **Workload 3 (Level 2/3): Relational Metadata and Control**

* **Profile:** This workload covers the "control plane" of the model. This includes tables for conversations, users, the correction\_log, and the Level 3 concept\_repository and verbalization\_templates. These tables are general-purpose, require high concurrency, and, most importantly, demand **ACID compliance, transactions, and foreign key constraints** to maintain relational integrity.  
* **Engine:** **InnoDB**. As the default, general-purpose, ACID-compliant storage engine for MariaDB, InnoDB is the only appropriate choice for these tables.21

### **2.2 Architectural Insight: A Multi-Engine Database**

The "model" will not be a single database; it will be a single MariaDB instance running tables with three different storage engines simultaneously. This hybrid approach is the only way to satisfy the "fastest engine" requirement for all components of the system.  
**Table 1: Hybrid-Engine Table Allocation**

| Table Name | Level | Workload | Selected Engine | Justification |
| :---- | :---- | :---- | :---- | :---- |
| tbl\_l1\_ngram\_probs | 1 | Read-Heavy, GROUP BY | **Aria** | 4x faster GROUP BY performance than InnoDB.22 |
| tbl\_l3\_concept\_probs | 3 | Read-Heavy, GROUP BY | **Aria** | Same as Level 1; optimized for statistical lookups. |
| tbl\_l2\_messages | 2 | Write-Intensive, Append-Only | **MyRocks** | 10x less write amplification, 2x better compression.23 |
| tbl\_l2\_conversations | 2 | General Purpose, ACID | **InnoDB** | Requires transactions and foreign key constraints.27 |
| tbl\_l2\_correction\_log | 2 | General Purpose, ACID | **InnoDB** | "Learning" loop requires transactional integrity. |
| tbl\_l3\_concept\_repo | 3 | General Purpose, ACID | **InnoDB** | Control plane table for concept definitions. |
| tbl\_l3\_verbal\_templates | 3 | General Purpose, ACID | **InnoDB** | Control plane table for verbalization templates. |
| tbl\_l1\_vocabulary | 1 | General Purpose, ACID | **InnoDB** | Token-to-ID mapping; requires reliable lookups. |

## **Part 3: Level 1 Implementation (The Statistical Prediction Layer)**

This layer is the statistical "motor" of the model, implementing a classical N-gram model as a large-scale relational table.

### **3.1 Schema Design for N-Gram Probabilities**

Storing N-grams presents a "curse of dimensionality" challenge. As *N* increases, the number of potential N-grams grows exponentially. Storing a complete 5-gram or 6-gram table from a trillion-token corpus is unfeasible.28 Therefore, the schema must be built using efficiency techniques identified in linguistic research.28  
We will use three primary optimization techniques:

1. **Pruning:** The N-gram tables will be pruned. Only N-grams with a count greater than a specified threshold (*k*) will be stored. This uses entropy to discard less-important (low-probability) N-grams.28  
2. **Hashing:** Storing the N-1 context (e.g., "Mary had a little") as a VARCHAR string is impossibly slow and storage-intensive. Instead, the context string will be represented in memory as a 64-bit hash.28  
3. **Quantization:** Probabilities do not need 8-byte FLOAT precision. Research indicates they can be quantized to 4-8 bits.28 We will store probabilities as a TINYINT (8 bits, 0-255), representing a quantized probability scale. This reduces the row size from (e.g., BIGINT+INT+FLOAT \= 20 bytes) to (BIGINT+INT+TINYINT \= 13 bytes), a \~35% storage saving that vastly improves I/O and cache efficiency for the Aria engine.

#### **Database Schema (Level 1\)**

**tbl\_l1\_vocabulary** (ENGINE=InnoDB)

* token\_id (INT, PRIMARY KEY): A unique ID for each word/token.  
* token\_text (VARCHAR(255), UNIQUE): The actual text of the token.

**tbl\_l1\_ngram\_probs** (ENGINE=Aria)

* context\_hash (BIGINT, non-unique): The 64-bit hash of the *N-1* preceding token\_ids.  
* next\_token\_id (INT, non-unique, FK to tbl\_l1\_vocabulary): The predicted next token.  
* quantized\_prob (TINYINT): The quantized probability (0-255) of this next\_token\_id given the context\_hash.  
* **Indexes:** A composite index on (context\_hash, quantized\_prob) is critical. This allows the database to instantly find all potential next tokens for a given context and then efficiently sort them to find the most probable one.

### **3.2 The "Generative" SQL Query (Inference)**

To generate the next token (the "inference pass"), the application layer performs these steps:

1. Takes the current context (e.g., "Mary had a little").  
2. Looks up the token\_ids for the *N-1* context (e.g., "had a little").  
3. Computes the 64-bit hash of this token\_id sequence (e.g., hash(52, 10, 811)).  
4. Executes the following SQL query:

SQL

SELECT  
    v.token\_text,  
    p.quantized\_prob  
FROM  
    tbl\_l1\_ngram\_probs AS p  
JOIN  
    tbl\_l1\_vocabulary AS v ON p.next\_token\_id \= v.token\_id  
WHERE  
    p.context\_hash \= \[computed\_hash\_value\]  
ORDER BY  
    p.quantized\_prob DESC  
LIMIT 1;

The performance of this single query, executed once *per token*, is the primary latency bottleneck of the entire system. Its speed is entirely dependent on the Aria engine's optimization for this WHERE...ORDER BY...LIMIT 1 pattern.22

## **Part 4: Level 2 Implementation (The Stateful Memory and Learning System)**

This layer provides the model with stateful memory and the ability to "learn" from corrections. This is not "retraining" the Level 1 model; rather, it is a **Correctional Retrieval-Augmented Generation (RAG)** loop that intercepts and overrides the statistical model. This component draws inspiration from schemas designed for agent memory.29

### **4.1 Schema for Episodic and Semantic Memory**

We will adapt the "Agent Memory" schema design 29 to log all interactions, using the hybrid-engine strategy for optimization.  
**tbl\_l2\_conversations** (ENGINE=InnoDB)

* *Purpose:* Tracks overall conversation sessions.29  
* *Schema:*  
  SQL  
  CREATE TABLE tbl\_l2\_conversations (  
    id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
    user\_id UUID,  
    agent\_name TEXT,  
    created\_at TIMESTAMP DEFAULT now()  
  );

**tbl\_l2\_messages** (ENGINE=MyRocks)

* *Purpose:* Logs every single turn from the user and assistant for episodic memory.29  
* *Schema:*  
  SQL  
  CREATE TABLE tbl\_l2\_messages (  
    id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
    conversation\_id UUID REFERENCES tbl\_l2\_conversations(id),  
    sender TEXT CHECK (sender IN ('user', 'assistant')),  
    content TEXT,  
    created\_at TIMESTAMP DEFAULT now()  
  );

* *Engine Choice:* This table is explicitly set to MyRocks to handle the massive, continuous write-volume of chat logging. The high compression and low write amplification of MyRocks are essential for the long-term scalability and cost-effectiveness of this episodic memory.23

### **4.2 The "Learning from Errors" Mechanism**

This is the core of the Level 2 requirement. "Learning" is achieved by storing explicit user corrections in a dedicated table, which is then used as a high-priority RAG source. This concept is inspired by systems designed to automatically correct and log schema errors.30  
**tbl\_l2\_correction\_log** (ENGINE=InnoDB)

* *Purpose:* Stores explicit user corrections, creating a "long-term memory" of facts and preferences.32  
* *Schema:*  
  SQL  
  CREATE TABLE tbl\_l2\_correction\_log (  
    correction\_id UUID PRIMARY KEY DEFAULT gen\_random\_uuid(),  
    conversation\_id UUID REFERENCES tbl\_l2\_conversations(id),  
    error\_message\_id UUID REFERENCES tbl\_l2\_messages(id),  
    correction\_message\_id UUID REFERENCES tbl\_l2\_messages(id),  
    error\_context TEXT,  
    corrected\_fact\_json JSON,  
    created\_at TIMESTAMP DEFAULT now()  
  );

#### **The Correctional RAG Learning Process:**

1. **Logging Error:** The model generates an error (e.g., "The capital of France is Lyon"). The user corrects it (e.g., "No, the capital is Paris"). An external process (or a SQL trigger/user-feedback mechanism) identifies this error/correction pair 30 and creates a new entry in tbl\_l2\_correction\_log. The corrected\_fact\_json might store {"entity": "France", "fact": "capital", "value": "Paris"}.  
2. **Retrieval (The "Learning"):** Before the Level 1 query is *ever* generated, the system *first* queries this table using the current context:  
   SQL  
   SELECT corrected\_fact\_json  
   FROM tbl\_l2\_correction\_log  
   WHERE \[current\_query\_context\] LIKE CONCAT('%', error\_context, '%')  
   ORDER BY created\_at DESC  
   LIMIT 5;

3. **Augmentation:** If a matching correction is found, the system *overrides* the Level 1 statistical model. It will use the corrected\_fact\_json to populate a response, perhaps using a Level 3 template (e.g., "A\_correction\_template: You previously told me that the capital of France is Paris."). This is a database-native RAG system 33 that provides stateful, long-term memory and the ability to learn from errors without retraining.34

## **Part 5: Level 3 Implementation (The Conceptual Prediction Engine)**

This layer implements the most advanced proposal: predicting a high-level "concept" and then "verbalizing" it. This hierarchical model 35 is the key to solving the Level 1 latency bottleneck.

### **5.1 Theoretical Framework: Narrativized Embeddings**

The proposed "concept-level prediction" maps directly to the research concept of **"narrativized embeddings"** 37 and "semantic tokenizers".38  
The core idea is to "convert structured data into narratives, using templates".37

* A "concept" is a piece of structured data (e.g., Concept: GetWeather, Payload: {city: "London"}).  
* The "verbalization" is a template-driven process (e.g., "The weather in {city} is...").

This creates a two-level hierarchical generation process 35:

1. **Top Level (Concept Model):** Predicts the next *semantic concept* based on context.  
2. **Bottom Level (Token Model):** Uses the Level 1 N-gram model to generate the *individual words* that fill out and "stitch" these concepts together.

### **5.2 Schema for Conceptual Generation**

To implement this, we need a parallel N-gram model for concepts and tables to store the concept definitions and their verbalization templates.  
**tbl\_l3\_concept\_repo** (ENGINE=InnoDB)

* *Purpose:* A dictionary of all possible "concepts" or "semantic tokens" the model can predict.  
* concept\_id (INT, PRIMARY KEY)  
* concept\_name (VARCHAR(255), UNIQUE): (e.g., 'ReportWeather', 'DefineTerm', 'CorrectError').  
* metadata\_schema (JSON): A schema defining the data required to verbalize this concept (e.g., {'city': 'string', 'unit': 'string'}).

**tbl\_l3\_verbal\_templates** (ENGINE=InnoDB)

* *Purpose:* Stores the templates used to "verbalize" a concept into text.  
* template\_id (INT, PRIMARY KEY)  
* concept\_id (INT, FK to tbl\_l3\_concept\_repo)  
* template\_string (TEXT): (e.g., "The current weather in {city} is {condition} and {temp}° {unit}.").  
* language\_code (CHAR(5))

**tbl\_l3\_concept\_probs** (ENGINE=Aria)

* *Purpose:* A parallel N-gram model, but for concepts instead of words.  
* context\_hash (BIGINT): Hash of the preceding *N* tokens/concepts.  
* next\_concept\_id (INT, FK to tbl\_l3\_concept\_repo)  
* quantized\_prob (TINYINT): The probability of this concept being the next logical step.

### **5.3 The Hierarchical Generation Process**

This is the synthesized query loop that combines all three levels:

1. **Level 2 (Memory):** The application first queries tbl\_l2\_correction\_log and tbl\_l2\_messages (RAG) to fetch recent context and any overriding corrected facts.  
2. **Level 3 (Concept Prediction):** The application hashes the current context. It queries the **Aria**\-powered tbl\_l3\_concept\_probs to find the most probable next\_concept\_id.  
   * SELECT next\_concept\_id FROM tbl\_l3\_concept\_probs WHERE context\_hash \=? ORDER BY quantized\_prob DESC LIMIT 1;  
3. Level 3 (Verbalization): If a high-probability concept is predicted (e.g., concept\_id \= 101, "ReportWeather"):  
   a. The application queries tbl\_l3\_verbal\_templates for its template\_string.  
   b. It uses Level 2 memory (or external API calls, if permitted) to find the data needed for the template's variables (e.g., {city: "London"}).  
   c. The resulting string ("The current weather in London is sunny.") is appended to the response buffer.  
4. **Level 1 (Token Stitching):** The *end* of that verbalized string ("...is sunny.") becomes the *new* context. The application now queries the Level 1 tbl\_l1\_ngram\_probs (from Part 3.2) to generate "stitching" tokens (e.g., "Furthermore,", "Is there..."). This continues until the Level 3 model predicts a new, high-probability concept.

This hierarchical process solves the "dynamic embedding" problem without vectors. The query about "dynamic embeddings" 41 implies a need for vector search.43 This architecture *bypasses* vector math entirely. The "embedding" is not a high-dimensional vector; it is a *structured, relational concept* (concept\_id) linked to *procedural, template-based logic* (template\_string).

## **Part 6: Systemic Bottlenecks and Strategic Recommendations**

### **6.1 A Critical Analysis of the True Bottlenecks**

While this architecture is theoretically sound and meets all constraints, it is critical to understand its performance limitations.

* **Primary Bottleneck: I/O Latency.** As established, the system's speed is governed by the *number of sequential queries* it must make. The hierarchical Level 3 model is a *mitigation*, not a *solution*. A response requiring 5 concepts and 20 stitch-tokens still requires 25 sequential database queries. This will be *orders of magnitude slower* than a tensor-based model's single, parallelized forward pass.10  
* **Secondary Bottleneck: Schema and Query Complexity.** The RAG queries for Level 2 "learning" 33 and the multi-table JOINs for Level 3 verbalization are complex. Research into Text-to-SQL highlights that as schema complexity grows, query generation and optimization become significant, non-trivial challenges.47 This system *is*, by definition, a highly complex schema.  
* **Tertiary Bottleneck: The "Training" ETL.** The "training" of this model is the population of the Aria N-gram tables. For a trillion-token corpus 28, this Extract, Transform, Load (ETL) process is a monumental batch operation, far removed from the more dynamic training and fine-tuning paradigms of neural networks.

### **6.2 Strategic Recommendations and Conclusion**

1. **Viability Assessment:** The proposed DB-SLM is a *theoretically sound* architecture that directly answers every component of the user's query. It is a "no-tensor" model, it uses MariaDB's fastest engines in a hybrid design, it implements N-gram statistics (Level 1), it has a stateful, learning memory (Level 2), and it features a conceptual-verbalization layer (Level 3).  
2. **Operational Warning:** This system *cannot* compete with modern tensor-based LLMs on latency, throughput, or real-time performance. Its design is best suited for asynchronous generation, academic research into non-tensor models, or niche applications where absolute data-sovereignty and audibility (every "thought" is a SQL query log) are the primary business drivers.  
3. **Recommended Path Forward (The Hybrid-RAG Model):** A more practical and performant architecture would be a **Database-Augmented** model, not a Database-Native one. This "best-of-both-worlds" approach would be:  
   * **Keep:** The entire **Level 2 (Memory) and Level 3 (Concept)** schemas on MariaDB (using InnoDB and MyRocks). This system is a state-of-the-art, SQL-native RAG and memory system.32  
   * **Replace:** The **Level 1 (N-Gram) layer** (the Aria tables) with a standard, tensor-based LLM (e.g., Llama, Mistral).1  
   * **New Workflow:** In this Hybrid-RAG model, the application first queries the MariaDB Level 2/3 system to retrieve all relevant facts, corrections, and concepts. These are compiled into a rich, database-augmented prompt. This prompt is *then* fed to the tensor-based LLM for fast, fluent, and coherent generation.

This final, hybrid-RAG model leverages the database for what it excels at (structured data storage, relational integrity, and RAG) 53 and the LLM for what *it* excels at (parallelized, context-aware next-token prediction).11 This approach achieves the *spirit* of the original query—a deeply database-integrated model that learns and remembers—while remaining operationally viable in a production environment.

#### **Bibliografia**

1. LLMs: What's a large language model? | Machine Learning \- Google for Developers, accesso eseguito il giorno novembre 9, 2025, [https://developers.google.com/machine-learning/crash-course/llm/transformers](https://developers.google.com/machine-learning/crash-course/llm/transformers)  
2. LLM Transformer Model Visually Explained \- Polo Club of Data Science, accesso eseguito il giorno novembre 9, 2025, [https://poloclub.github.io/transformer-explainer/](https://poloclub.github.io/transformer-explainer/)  
3. Database is All You Need: Serving LLMs with Relational Queries, accesso eseguito il giorno novembre 9, 2025, [https://openproceedings.org/2025/conf/edbt/paper-326.pdf](https://openproceedings.org/2025/conf/edbt/paper-326.pdf)  
4. Word n-gram language model \- Wikipedia, accesso eseguito il giorno novembre 9, 2025, [https://en.wikipedia.org/wiki/Word\_n-gram\_language\_model](https://en.wikipedia.org/wiki/Word_n-gram_language_model)  
5. Cracking the Language Code: A Comprehensive Guide to N-Gram Models and Text Generation | by Sundharesan Kumaresan | Medium, accesso eseguito il giorno novembre 9, 2025, [https://medium.com/@sundharesansk11/cracking-the-language-code-a-comprehensive-guide-to-n-gram-models-and-text-generation-b670335ce6e8](https://medium.com/@sundharesansk11/cracking-the-language-code-a-comprehensive-guide-to-n-gram-models-and-text-generation-b670335ce6e8)  
6. Markov Chain: SQL Database and Java Representation \- Stack Overflow, accesso eseguito il giorno novembre 9, 2025, [https://stackoverflow.com/questions/6610058/markov-chain-sql-database-and-java-representation](https://stackoverflow.com/questions/6610058/markov-chain-sql-database-and-java-representation)  
7. A Beginner's Guide to Markov Chains, Conditional Probability, and Independence | by Hezekiah J. Branch | Towards AI, accesso eseguito il giorno novembre 9, 2025, [https://pub.towardsai.net/a-beginners-guide-to-markov-chains-conditional-probability-and-independence-b35887a9032](https://pub.towardsai.net/a-beginners-guide-to-markov-chains-conditional-probability-and-independence-b35887a9032)  
8. The New LLM Bottleneck: A Systems Perspective on Latent Attention and Mixture-of-Experts, Yun et al. 2025 \- Reddit, accesso eseguito il giorno novembre 9, 2025, [https://www.reddit.com/r/mlscaling/comments/1n2axga/the\_new\_llm\_bottleneck\_a\_systems\_perspective\_on/](https://www.reddit.com/r/mlscaling/comments/1n2axga/the_new_llm_bottleneck_a_systems_perspective_on/)  
9. The New LLM Bottleneck: A Systems Perspective on Latent Attention and Mixture-of-Experts, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/html/2507.15465v1](https://arxiv.org/html/2507.15465v1)  
10. Scaling LLM Inference: Innovations in Tensor Parallelism, Context Parallelism, and Expert Parallelism \- Engineering at Meta, accesso eseguito il giorno novembre 9, 2025, [https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)  
11. Are advances in LLMs due to transformers per se, or more of the scale of data and compute?, accesso eseguito il giorno novembre 9, 2025, [https://www.reddit.com/r/learnmachinelearning/comments/1hjse9z/are\_advances\_in\_llms\_due\_to\_transformers\_per\_se/](https://www.reddit.com/r/learnmachinelearning/comments/1hjse9z/are_advances_in_llms_due_to_transformers_per_se/)  
12. Handling Large Data Volumes with MySQL and MariaDB | Severalnines, accesso eseguito il giorno novembre 9, 2025, [https://severalnines.com/blog/handling-large-data-volumes-mysql-and-mariadb/](https://severalnines.com/blog/handling-large-data-volumes-mysql-and-mariadb/)  
13. A Look at MyRocks Performance \- Hacker News, accesso eseguito il giorno novembre 9, 2025, [https://news.ycombinator.com/item?id=16963484](https://news.ycombinator.com/item?id=16963484)  
14. MariaDB Performance Tuning: Best Practices and Techniques \- Cloudways, accesso eseguito il giorno novembre 9, 2025, [https://www.cloudways.com/blog/mariadb-performance-tuning/](https://www.cloudways.com/blog/mariadb-performance-tuning/)  
15. Improve MariaDB Performance using Query Profiling \- IDERA, accesso eseguito il giorno novembre 9, 2025, [https://www.idera.com/blogs/improve-mariadb-performance-using-query-profiling/](https://www.idera.com/blogs/improve-mariadb-performance-using-query-profiling/)  
16. \[2502.09419\] On multi-token prediction for efficient LLM inference \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/abs/2502.09419](https://arxiv.org/abs/2502.09419)  
17. Better & Faster Large Language Models via Multi-token ... \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/abs/2404.19737](https://arxiv.org/abs/2404.19737)  
18. \[2505.17505\] L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/abs/2505.17505](https://arxiv.org/abs/2505.17505)  
19. Exploring Storage Engine Options for MariaDB | Severalnines, accesso eseguito il giorno novembre 9, 2025, [https://severalnines.com/blog/exploring-storage-engine-options-mariadb/](https://severalnines.com/blog/exploring-storage-engine-options-mariadb/)  
20. Choosing the Right Storage Engine | Server | MariaDB Documentation, accesso eseguito il giorno novembre 9, 2025, [https://mariadb.com/docs/server/server-usage/storage-engines/choosing-the-right-storage-engine](https://mariadb.com/docs/server/server-usage/storage-engines/choosing-the-right-storage-engine)  
21. An Introduction to MariaDB Storage Engines, accesso eseguito il giorno novembre 9, 2025, [https://www.mariadbtutorial.com/mariadb-basics/mariadb-storage-engines/](https://www.mariadbtutorial.com/mariadb-basics/mariadb-storage-engines/)  
22. Aria Storage Engine: Benefits & Comparisons | MariaDB, accesso eseguito il giorno novembre 9, 2025, [https://mariadb.com/resources/blog/storage-engine-choice-aria/](https://mariadb.com/resources/blog/storage-engine-choice-aria/)  
23. MyRocks Storage Engine | MariaDB Documentation, accesso eseguito il giorno novembre 9, 2025, [https://mariadb.com/docs/platform/mariadb-faqs/storage-engines/myrocks-storage-engine](https://mariadb.com/docs/platform/mariadb-faqs/storage-engines/myrocks-storage-engine)  
24. Exploring MariaDB's Storage Engine Options \- Datavail, accesso eseguito il giorno novembre 9, 2025, [https://www.datavail.com/blog/exploring-mariadbs-storage-engine-options/](https://www.datavail.com/blog/exploring-mariadbs-storage-engine-options/)  
25. Increase write throughput on Amazon RDS for MariaDB using the MyRocks storage engine, accesso eseguito il giorno novembre 9, 2025, [https://aws.amazon.com/blogs/database/increase-write-throughput-on-amazon-rds-for-mariadb-using-the-myrocks-storage-engine/](https://aws.amazon.com/blogs/database/increase-write-throughput-on-amazon-rds-for-mariadb-using-the-myrocks-storage-engine/)  
26. MyRocks Use Case: Big Dataset \- Percona, accesso eseguito il giorno novembre 9, 2025, [https://www.percona.com/blog/myrocks-use-case-big-dataset/](https://www.percona.com/blog/myrocks-use-case-big-dataset/)  
27. Choosing the Right Storage Engine for MySQL Tables \- Navicat, accesso eseguito il giorno novembre 9, 2025, [https://www.navicat.com/en/company/aboutus/blog/2385-choosing-the-right-storage-engine-for-mysql-tables](https://www.navicat.com/en/company/aboutus/blog/2385-choosing-the-right-storage-engine-for-mysql-tables)  
28. N-gram Language Models \- Stanford University, accesso eseguito il giorno novembre 9, 2025, [https://web.stanford.edu/\~jurafsky/slp3/3.pdf](https://web.stanford.edu/~jurafsky/slp3/3.pdf)  
29. Schema Design for Agent Memory and LLM History | by Pranav ..., accesso eseguito il giorno novembre 9, 2025, [https://medium.com/@pranavprakash4777/schema-design-for-agent-memory-and-llm-history-38f5cbc126fb](https://medium.com/@pranavprakash4777/schema-design-for-agent-memory-and-llm-history-38f5cbc126fb)  
30. An automatic correction tool for relational database schemas \- ResearchGate, accesso eseguito il giorno novembre 9, 2025, [https://www.researchgate.net/publication/4204651\_An\_automatic\_correction\_tool\_for\_relational\_database\_schemas](https://www.researchgate.net/publication/4204651_An_automatic_correction_tool_for_relational_database_schemas)  
31. An Automatic Correction Tool For Relational Database Schemas \- CORE, accesso eseguito il giorno novembre 9, 2025, [https://core.ac.uk/download/pdf/132550206.pdf](https://core.ac.uk/download/pdf/132550206.pdf)  
32. LLM4LLM: Longer-Lasting Memory for LLMs | UC Berkeley School of Information, accesso eseguito il giorno novembre 9, 2025, [https://www.ischool.berkeley.edu/projects/2024/llm4llm-longer-lasting-memory-llms](https://www.ischool.berkeley.edu/projects/2024/llm4llm-longer-lasting-memory-llms)  
33. Chat history and semantic search \- API \- OpenAI Developer Community, accesso eseguito il giorno novembre 9, 2025, [https://community.openai.com/t/chat-history-and-semantic-search/460984](https://community.openai.com/t/chat-history-and-semantic-search/460984)  
34. Persistent Memory for Chatbots using PostgreSQL and LangChain \- HexaCluster, accesso eseguito il giorno novembre 9, 2025, [https://hexacluster.ai/blog/postgresql/postgres-for-chat-history-langchain-postgres-postgreschatmessagehistory/](https://hexacluster.ai/blog/postgresql/postgres-for-chat-history-langchain-postgres-postgreschatmessagehistory/)  
35. A Hierarchical Model for Data-to-Text Generation \- PMC \- NIH, accesso eseguito il giorno novembre 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC7148215/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7148215/)  
36. Retrieval-style In-context Learning for Few-shot Hierarchical Text Classification | Transactions of the Association for Computational Linguistics \- MIT Press Direct, accesso eseguito il giorno novembre 9, 2025, [https://direct.mit.edu/tacl/article/doi/10.1162/tacl\_a\_00697/124630/Retrieval-style-In-context-Learning-for-Few-shot](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00697/124630/Retrieval-style-In-context-Learning-for-Few-shot)  
37. Narrativized Embeddings: Bridging Structured Data and Semantic ..., accesso eseguito il giorno novembre 9, 2025, [https://xponentl.ai/news/narrativized-embeddings-bridging-structured-data-and-semantic-understanding](https://xponentl.ai/news/narrativized-embeddings-bridging-structured-data-and-semantic-understanding)  
38. Semantic Tokenizer for Enhanced Natural Language Processing \- ResearchGate, accesso eseguito il giorno novembre 9, 2025, [https://www.researchgate.net/publication/370262546\_Semantic\_Tokenizer\_for\_Enhanced\_Natural\_Language\_Processing](https://www.researchgate.net/publication/370262546_Semantic_Tokenizer_for_Enhanced_Natural_Language_Processing)  
39. From Principles to Applications: A Comprehensive Survey of Discrete Tokenizers in Generation, Comprehension, Recommendation, and Information Retrieval \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/html/2502.12448v1](https://arxiv.org/html/2502.12448v1)  
40. \[1912.10011\] A Hierarchical Model for Data-to-Text Generation \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/abs/1912.10011](https://arxiv.org/abs/1912.10011)  
41. Data Embeddings: Types and Storage Solutions \- DEV Community, accesso eseguito il giorno novembre 9, 2025, [https://dev.to/ankush\_mahore/understanding-data-embeddings-types-and-storage-solutions-12jp](https://dev.to/ankush_mahore/understanding-data-embeddings-types-and-storage-solutions-12jp)  
42. Embeddings and Vector Databases With ChromaDB \- Real Python, accesso eseguito il giorno novembre 9, 2025, [https://realpython.com/chromadb-vector-database/](https://realpython.com/chromadb-vector-database/)  
43. Embeddings, Vector Databases, and Semantic Search: A Comprehensive Guide, accesso eseguito il giorno novembre 9, 2025, [https://dev.to/imsushant12/embeddings-vector-databases-and-semantic-search-a-comprehensive-guide-2j01](https://dev.to/imsushant12/embeddings-vector-databases-and-semantic-search-a-comprehensive-guide-2j01)  
44. Embeddings and Vector Databases. This is an excerpt from Chapter 5… | by Vlad Rișcuția, accesso eseguito il giorno novembre 9, 2025, [https://medium.com/@vladris/embeddings-and-vector-databases-732f9927b377](https://medium.com/@vladris/embeddings-and-vector-databases-732f9927b377)  
45. Speed Always Wins: A Survey on Efficient Architectures for Large Language Models \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/html/2508.09834v1](https://arxiv.org/html/2508.09834v1)  
46. Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/html/2406.08426v1](https://arxiv.org/html/2406.08426v1)  
47. Why LLMs Need Help for Accurate SQL Generation at Scale | GigaSpaces AI, accesso eseguito il giorno novembre 9, 2025, [https://www.gigaspaces.com/blog/llms-accurate-sql-generation-at-scale](https://www.gigaspaces.com/blog/llms-accurate-sql-generation-at-scale)  
48. From Natural Language to SQL: Approaches and Challenges in Text2SQL \- Ashish Agarwal, accesso eseguito il giorno novembre 9, 2025, [https://toashishagarwal.medium.com/from-natural-language-to-sql-approaches-and-challenges-in-text2sql-d1252ff86321](https://toashishagarwal.medium.com/from-natural-language-to-sql-approaches-and-challenges-in-text2sql-d1252ff86321)  
49. Challenges of Building Production-Ready LLM Text-to-SQL — and How to Overcome Them, accesso eseguito il giorno novembre 9, 2025, [https://blog.aidetic.in/challenges-of-building-production-ready-llm-text-to-sql-and-how-to-overcome-them-04a6d32cc5e8](https://blog.aidetic.in/challenges-of-building-production-ready-llm-text-to-sql-and-how-to-overcome-them-04a6d32cc5e8)  
50. Enterprise-grade natural language to SQL generation using LLMs: Balancing accuracy, latency, and scale | Artificial Intelligence \- Amazon AWS, accesso eseguito il giorno novembre 9, 2025, [https://aws.amazon.com/blogs/machine-learning/enterprise-grade-natural-language-to-sql-generation-using-llms-balancing-accuracy-latency-and-scale/](https://aws.amazon.com/blogs/machine-learning/enterprise-grade-natural-language-to-sql-generation-using-llms-balancing-accuracy-latency-and-scale/)  
51. Large language model \- Wikipedia, accesso eseguito il giorno novembre 9, 2025, [https://en.wikipedia.org/wiki/Large\_language\_model](https://en.wikipedia.org/wiki/Large_language_model)  
52. Large Language Models (LLMs) vs Transformers \- GeeksforGeeks, accesso eseguito il giorno novembre 9, 2025, [https://www.geeksforgeeks.org/nlp/large-language-models-llms-vs-transformers/](https://www.geeksforgeeks.org/nlp/large-language-models-llms-vs-transformers/)  
53. How to Build Large Language Model Application Using Vector Database? \- Analytics Vidhya, accesso eseguito il giorno novembre 9, 2025, [https://www.analyticsvidhya.com/blog/2023/10/how-to-build-llm-apps-using-vector-database/](https://www.analyticsvidhya.com/blog/2023/10/how-to-build-llm-apps-using-vector-database/)  
54. LLM and Vector Databases \- skymod.tech \- Medium, accesso eseguito il giorno novembre 9, 2025, [https://medium.com/skymod-tech/llm-and-vector-databases-b17e5b667da7](https://medium.com/skymod-tech/llm-and-vector-databases-b17e5b667da7)  
55. Relational Database Augmented Large Language Model \- arXiv, accesso eseguito il giorno novembre 9, 2025, [https://arxiv.org/html/2407.15071v1](https://arxiv.org/html/2407.15071v1)  
56. Introducing the embedding() function: Semantic search made easy with SQL\! \- MotherDuck, accesso eseguito il giorno novembre 9, 2025, [https://motherduck.com/blog/sql-embeddings-for-semantic-meaning-in-text-and-rag/](https://motherduck.com/blog/sql-embeddings-for-semantic-meaning-in-text-and-rag/)