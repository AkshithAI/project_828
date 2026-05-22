"""
Curated evaluation prompts for the Project 828 eval suite.

Two prompt banks aligned with the Phase 2 datamix:
  1. CODE_PROMPTS  — 36 code completion prompts (6 per language x 6 languages)
  2. CS_QA_QUESTIONS — 40 multiple-choice CS knowledge questions (8 per category)

Each prompt maps to a specific Phase 2 data source so scores directly measure
whether the model is learning from its training data.
"""

# ═══════════════════════════════════════════════════════════════════
#  1. Multilingual Code Completion Prompts  →  Code Replay (35%)
# ═══════════════════════════════════════════════════════════════════
#
#  Scoring: structural validity + keyword presence + bracket balance
#  + no repetition + sufficient length
#
#  Fields:
#    lang     — programming language
#    prompt   — code prefix for the model to complete
#    kw       — expected keywords in the completion
#    desc     — short description for reporting

CODE_PROMPTS = [
    # ── Python (14%) ──────────────────────────────────────────
    {"lang": "python", "desc": "context manager",
     "prompt": "class FileLogger:\n    def __init__(self, path):\n        self.path = path\n\n    def __enter__(self):\n        self.file = open(self.path, 'a')\n        return self\n\n    def __exit__(self, exc_type, exc_val, exc_tb):",
     "kw": ["self.file", "close", "return"]},
    {"lang": "python", "desc": "decorator with args",
     "prompt": "import functools\nimport time\n\ndef rate_limit(max_calls, period=60):\n    def decorator(func):\n        calls = []\n        @functools.wraps(func)\n        def wrapper(*args, **kwargs):",
     "kw": ["calls", "time", "return"]},
    {"lang": "python", "desc": "generator pipeline",
     "prompt": "def read_chunks(file_path, chunk_size=1024):\n    with open(file_path, 'rb') as f:\n        while True:\n            chunk = f.read(chunk_size)\n            if not chunk:",
     "kw": ["yield", "break"]},
    {"lang": "python", "desc": "dataclass methods",
     "prompt": "from dataclasses import dataclass\nfrom typing import List\n\n@dataclass\nclass Student:\n    name: str\n    grades: List[float]\n\n    @property\n    def gpa(self):",
     "kw": ["return", "self.grades", "len"]},
    {"lang": "python", "desc": "async gather",
     "prompt": "import asyncio\nfrom typing import List\n\nasync def fetch_all(urls: List[str]) -> List[dict]:\n    async def fetch_one(session, url):\n        async with session.get(url) as resp:\n            return await resp.json()\n\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_one(session, u) for u in urls]\n        return await asyncio.",
     "kw": ["gather", "tasks"]},
    {"lang": "python", "desc": "binary search",
     "prompt": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:",
     "kw": ["left", "mid", "right", "return"]},

    # ── JavaScript (7%) ───────────────────────────────────────
    {"lang": "javascript", "desc": "event emitter",
     "prompt": "class EventEmitter {\n  constructor() {\n    this._events = new Map();\n  }\n\n  on(event, listener) {\n    if (!this._events.has(event)) {\n      this._events.set(event, []);\n    }\n    this._events.get(event).push(listener);\n    return this;\n  }\n\n  emit(event, ...args) {",
     "kw": ["this._events", "forEach", "listener"]},
    {"lang": "javascript", "desc": "promise retry",
     "prompt": "async function fetchWithRetry(url, retries = 3, delay = 1000) {\n  for (let i = 0; i < retries; i++) {\n    try {\n      const response = await fetch(url);\n      if (!response.ok) throw new Error(`HTTP ${response.status}`);\n      return await response.json();\n    } catch (err) {",
     "kw": ["await", "delay", "throw"]},
    {"lang": "javascript", "desc": "debounce",
     "prompt": "function debounce(func, wait) {\n  let timeout;\n  return function(...args) {",
     "kw": ["clearTimeout", "setTimeout", "timeout"]},
    {"lang": "javascript", "desc": "deep merge",
     "prompt": "function deepMerge(target, source) {\n  for (const key in source) {\n    if (source.hasOwnProperty(key)) {\n      if (typeof source[key] === 'object' && source[key] !== null) {",
     "kw": ["target", "deepMerge", "Object"]},
    {"lang": "javascript", "desc": "linked list",
     "prompt": "class LinkedList {\n  constructor() {\n    this.head = null;\n    this.size = 0;\n  }\n\n  push(value) {\n    const node = { value, next: null };\n    if (!this.head) {\n      this.head = node;\n    } else {",
     "kw": ["node", "next", "this.size"]},
    {"lang": "javascript", "desc": "array groupBy",
     "prompt": "function groupBy(array, keyFn) {\n  return array.reduce((result, item) => {\n    const key = keyFn(item);\n    if (!result[key]) {",
     "kw": ["result", "push", "return"]},

    # ── TypeScript (4%) ───────────────────────────────────────
    {"lang": "typescript", "desc": "generic repository",
     "prompt": "interface Entity { id: string; }\n\nclass InMemoryRepo<T extends Entity> {\n  private store = new Map<string, T>();\n\n  async findById(id: string): Promise<T | null> {",
     "kw": ["this.store", "get", "return"]},
    {"lang": "typescript", "desc": "type guard",
     "prompt": "interface Circle { kind: 'circle'; radius: number; }\ninterface Square { kind: 'square'; side: number; }\ntype Shape = Circle | Square;\n\nfunction area(shape: Shape): number {\n  switch (shape.kind) {",
     "kw": ["circle", "square", "radius", "return"]},
    {"lang": "typescript", "desc": "async fetcher",
     "prompt": "async function fetchData<T>(url: string, schema: z.ZodSchema<T>): Promise<T> {\n  const response = await fetch(url);\n  if (!response.ok) {",
     "kw": ["throw", "json", "parse", "return"]},
    {"lang": "typescript", "desc": "builder pattern",
     "prompt": "class QueryBuilder {\n  private conditions: string[] = [];\n  private params: unknown[] = [];\n\n  where(field: string, value: unknown): this {\n    this.conditions.push(`${field} = ?`);\n    this.params.push(value);\n    return this;\n  }\n\n  build(): { sql: string; params: unknown[] } {",
     "kw": ["this.conditions", "join", "return"]},
    {"lang": "typescript", "desc": "discriminated union",
     "prompt": "type Result<T, E = Error> =\n  | { ok: true; value: T }\n  | { ok: false; error: E };\n\nfunction map<T, U, E>(result: Result<T, E>, fn: (v: T) => U): Result<U, E> {\n  if (result.ok) {",
     "kw": ["return", "value", "fn"]},
    {"lang": "typescript", "desc": "enum usage",
     "prompt": "enum LogLevel {\n  DEBUG = 0,\n  INFO = 1,\n  WARN = 2,\n  ERROR = 3,\n}\n\nclass Logger {\n  constructor(private level: LogLevel = LogLevel.INFO) {}\n\n  log(level: LogLevel, message: string): void {\n    if (level >= this.level) {",
     "kw": ["console", "LogLevel", "message"]},

    # ── C++ (5%) ──────────────────────────────────────────────
    {"lang": "cpp", "desc": "template stack",
     "prompt": "#include <vector>\n#include <stdexcept>\n\ntemplate<typename T>\nclass Stack {\n    std::vector<T> data;\npublic:\n    void push(const T& val) { data.push_back(val); }\n\n    T pop() {\n        if (data.empty())\n            throw std::runtime_error(\"stack underflow\");\n        T val = data.back();",
     "kw": ["data", "pop_back", "return"]},
    {"lang": "cpp", "desc": "smart pointer RAII",
     "prompt": "#include <memory>\n#include <iostream>\n\nclass Connection {\n    std::string host;\npublic:\n    Connection(const std::string& h) : host(h) {\n        std::cout << \"Connected to \" << host << std::endl;\n    }\n    ~Connection() {",
     "kw": ["cout", "host", "Disconnected"]},
    {"lang": "cpp", "desc": "STL algorithm",
     "prompt": "#include <vector>\n#include <algorithm>\n#include <numeric>\n\nstd::vector<int> topK(std::vector<int>& nums, int k) {\n    std::partial_sort(nums.begin(), nums.begin() + k, nums.end(),\n        [](int a, int b) {",
     "kw": ["return", "vector", "begin"]},
    {"lang": "cpp", "desc": "move semantics",
     "prompt": "#include <string>\n#include <utility>\n\nclass Buffer {\n    char* data;\n    size_t size;\npublic:\n    Buffer(size_t sz) : data(new char[sz]), size(sz) {}\n    ~Buffer() { delete[] data; }\n\n    Buffer(Buffer&& other) noexcept\n        : data(other.data), size(other.size) {",
     "kw": ["other.data", "nullptr", "other.size"]},
    {"lang": "cpp", "desc": "thread safe queue",
     "prompt": "#include <queue>\n#include <mutex>\n#include <condition_variable>\n\ntemplate<typename T>\nclass SafeQueue {\n    std::queue<T> queue;\n    std::mutex mtx;\n    std::condition_variable cv;\npublic:\n    void push(T val) {\n        std::lock_guard<std::mutex> lock(mtx);",
     "kw": ["queue", "push", "notify", "cv"]},
    {"lang": "cpp", "desc": "iterator implementation",
     "prompt": "#include <iterator>\n\ntemplate<typename T>\nclass Range {\n    T start_, end_, step_;\npublic:\n    Range(T start, T end, T step = 1) : start_(start), end_(end), step_(step) {}\n\n    class iterator {\n        T current;\n        T step;\n    public:\n        iterator(T val, T s) : current(val), step(s) {}\n        T operator*() const { return current; }\n        iterator& operator++() {",
     "kw": ["current", "step", "return"]},

    # ── Go (5%) ───────────────────────────────────────────────
    {"lang": "go", "desc": "worker pool",
     "prompt": "package main\n\nimport (\n\t\"sync\"\n)\n\nfunc workerPool(jobs <-chan int, results chan<- int, numWorkers int) {\n\tvar wg sync.WaitGroup\n\tfor i := 0; i < numWorkers; i++ {\n\t\twg.Add(1)\n\t\tgo func() {\n\t\t\tdefer wg.Done()\n\t\t\tfor job := range jobs {",
     "kw": ["results", "job", "wg"]},
    {"lang": "go", "desc": "interface struct",
     "prompt": "package main\n\nimport \"fmt\"\n\ntype Shape interface {\n\tArea() float64\n\tPerimeter() float64\n}\n\ntype Rectangle struct {\n\tWidth, Height float64\n}\n\nfunc (r Rectangle) Area() float64 {",
     "kw": ["return", "Width", "Height"]},
    {"lang": "go", "desc": "error wrapping",
     "prompt": "package main\n\nimport (\n\t\"fmt\"\n\t\"os\"\n)\n\nfunc readConfig(path string) ([]byte, error) {\n\tdata, err := os.ReadFile(path)\n\tif err != nil {",
     "kw": ["return", "fmt.Errorf", "err"]},
    {"lang": "go", "desc": "HTTP middleware",
     "prompt": "package main\n\nimport (\n\t\"log\"\n\t\"net/http\"\n\t\"time\"\n)\n\nfunc loggingMiddleware(next http.Handler) http.Handler {\n\treturn http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {\n\t\tstart := time.Now()",
     "kw": ["log", "next.ServeHTTP", "time.Since"]},
    {"lang": "go", "desc": "concurrent map",
     "prompt": "package main\n\nimport \"sync\"\n\ntype SafeMap struct {\n\tmu   sync.RWMutex\n\tdata map[string]interface{}\n}\n\nfunc (m *SafeMap) Get(key string) (interface{}, bool) {\n\tm.mu.RLock()\n\tdefer m.mu.RUnlock()",
     "kw": ["return", "m.data", "key"]},
    {"lang": "go", "desc": "context timeout",
     "prompt": "package main\n\nimport (\n\t\"context\"\n\t\"fmt\"\n\t\"time\"\n)\n\nfunc fetchWithTimeout(ctx context.Context, url string) (string, error) {\n\tctx, cancel := context.WithTimeout(ctx, 5*time.Second)\n\tdefer cancel()\n\n\tselect {",
     "kw": ["case", "ctx.Done", "return"]},

    # ── Rust (5%) ─────────────────────────────────────────────
    {"lang": "rust", "desc": "trait impl",
     "prompt": "trait Summary {\n    fn summarize(&self) -> String;\n    fn preview(&self) -> String {\n        format!(\"{}...\", &self.summarize()[..20])\n    }\n}\n\nstruct Article {\n    title: String,\n    content: String,\n}\n\nimpl Summary for Article {\n    fn summarize(&self) -> String {",
     "kw": ["self.title", "self.content", "format"]},
    {"lang": "rust", "desc": "Result chain",
     "prompt": "use std::fs;\nuse std::io;\n\nfn read_username_from_file(path: &str) -> Result<String, io::Error> {\n    let contents = fs::read_to_string(path)?;\n    let first_line = contents.lines().next()",
     "kw": ["ok_or", "Ok", "trim"]},
    {"lang": "rust", "desc": "enum methods",
     "prompt": "enum Coin {\n    Penny,\n    Nickel,\n    Dime,\n    Quarter,\n}\n\nimpl Coin {\n    fn value(&self) -> u32 {\n        match self {",
     "kw": ["Penny", "Nickel", "Dime", "Quarter"]},
    {"lang": "rust", "desc": "iterator chain",
     "prompt": "fn top_words(text: &str, n: usize) -> Vec<(String, usize)> {\n    let mut counts = std::collections::HashMap::new();\n    for word in text.split_whitespace() {\n        *counts.entry(word.to_lowercase()).or_insert(0) += 1;\n    }\n    let mut pairs: Vec<_> = counts.into_iter()",
     "kw": ["collect", "sort", "truncate"]},
    {"lang": "rust", "desc": "struct lifetime",
     "prompt": "struct Parser<'a> {\n    input: &'a str,\n    pos: usize,\n}\n\nimpl<'a> Parser<'a> {\n    fn new(input: &'a str) -> Self {\n        Parser { input, pos: 0 }\n    }\n\n    fn peek(&self) -> Option<char> {",
     "kw": ["self.input", "self.pos", "chars", "nth"]},
    {"lang": "rust", "desc": "builder pattern",
     "prompt": "struct Config {\n    host: String,\n    port: u16,\n    workers: usize,\n}\n\nstruct ConfigBuilder {\n    host: String,\n    port: u16,\n    workers: usize,\n}\n\nimpl ConfigBuilder {\n    fn new() -> Self {\n        ConfigBuilder {\n            host: \"localhost\".to_string(),\n            port: 8080,\n            workers: 4,\n        }\n    }\n\n    fn host(mut self, h: &str) -> Self {",
     "kw": ["self.host", "self"]},
]


# ═══════════════════════════════════════════════════════════════════
#  2. CS Knowledge QA (MCQ)  →  CS Knowledge (18%)
# ═══════════════════════════════════════════════════════════════════
#
#  Scoring: log-likelihood of each choice → pick highest → compare
#  Fields:
#    q   — question text
#    c   — list of 4 choices
#    a   — correct answer index (0-based)
#    cat — category for reporting

CS_QA_QUESTIONS = [
    # ── Data Structures & Algorithms (8) ──────────────────────
    {"q": "What is the average-case time complexity of searching for a key in a hash table?",
     "c": ["O(n)", "O(log n)", "O(1)", "O(n log n)"], "a": 2, "cat": "algorithms"},
    {"q": "Which data structure does BFS (Breadth-First Search) use?",
     "c": ["Stack", "Queue", "Priority queue", "Deque"], "a": 1, "cat": "algorithms"},
    {"q": "Which sorting algorithm is stable?",
     "c": ["Quick sort", "Heap sort", "Merge sort", "Selection sort"], "a": 2, "cat": "algorithms"},
    {"q": "What is the time complexity of inserting into a balanced BST?",
     "c": ["O(1)", "O(n)", "O(log n)", "O(n log n)"], "a": 2, "cat": "algorithms"},
    {"q": "What is the key property required for dynamic programming?",
     "c": ["Greedy choice", "Optimal substructure", "No cycles", "Sorted input"], "a": 1, "cat": "algorithms"},
    {"q": "What is the time complexity of extracting the minimum from a min-heap?",
     "c": ["O(1)", "O(log n)", "O(n)", "O(n log n)"], "a": 1, "cat": "algorithms"},
    {"q": "Which data structure would you use to check if parentheses are balanced?",
     "c": ["Queue", "Hash map", "Stack", "Linked list"], "a": 2, "cat": "algorithms"},
    {"q": "What is the space complexity of merge sort?",
     "c": ["O(1)", "O(log n)", "O(n)", "O(n^2)"], "a": 2, "cat": "algorithms"},

    # ── Networking (8) ────────────────────────────────────────
    {"q": "Which protocol provides reliable, ordered delivery of data?",
     "c": ["UDP", "TCP", "ICMP", "ARP"], "a": 1, "cat": "networking"},
    {"q": "What is the difference between HTTP PUT and PATCH?",
     "c": ["PUT is faster", "PUT replaces the entire resource, PATCH updates partially",
           "PATCH replaces the entire resource", "There is no difference"], "a": 1, "cat": "networking"},
    {"q": "What does DNS primarily do?",
     "c": ["Encrypt traffic", "Route packets", "Translate domain names to IP addresses",
           "Filter malicious traffic"], "a": 2, "cat": "networking"},
    {"q": "Which HTTP status code indicates a resource was not found?",
     "c": ["400", "401", "403", "404"], "a": 3, "cat": "networking"},
    {"q": "What are the three steps of the TCP three-way handshake?",
     "c": ["SYN, SYN-ACK, ACK", "SYN, ACK, FIN", "ACK, SYN, FIN", "SYN, FIN, ACK"], "a": 0, "cat": "networking"},
    {"q": "Which protocol is preferred for real-time video streaming?",
     "c": ["TCP", "FTP", "UDP", "SMTP"], "a": 2, "cat": "networking"},
    {"q": "What does HTTPS add on top of HTTP?",
     "c": ["Compression", "Caching", "TLS encryption", "Load balancing"], "a": 2, "cat": "networking"},
    {"q": "What is a key constraint of RESTful APIs?",
     "c": ["Must use XML", "Must be stateless", "Must use WebSockets", "Must use POST only"], "a": 1, "cat": "networking"},

    # ── Systems (8) ───────────────────────────────────────────
    {"q": "What is the key difference between a process and a thread?",
     "c": ["Threads are faster", "Threads share the same memory space",
           "Processes share memory", "There is no difference"], "a": 1, "cat": "systems"},
    {"q": "What does virtual memory allow?",
     "c": ["Faster CPU clock speed", "Programs to use more memory than physically available",
           "Direct hardware access", "Network communication"], "a": 1, "cat": "systems"},
    {"q": "In LRU cache eviction, which item is removed first?",
     "c": ["Most recently used", "Least recently used", "Largest item", "Oldest item by creation"], "a": 1, "cat": "systems"},
    {"q": "How many conditions are needed for a deadlock to occur?",
     "c": ["2", "3", "4", "5"], "a": 2, "cat": "systems"},
    {"q": "Where are local variables stored in memory?",
     "c": ["Heap", "Stack", "Global segment", "Code segment"], "a": 1, "cat": "systems"},
    {"q": "What is the primary difference between a mutex and a semaphore?",
     "c": ["Mutex is faster", "Mutex allows only one thread, semaphore allows N",
           "Semaphore is binary only", "They are identical"], "a": 1, "cat": "systems"},
    {"q": "What scheduling algorithm assigns each process a fixed time slice?",
     "c": ["FIFO", "Priority scheduling", "Round-robin", "Shortest job first"], "a": 2, "cat": "systems"},
    {"q": "What is a race condition?",
     "c": ["A CPU overheating issue", "When two threads access shared data concurrently without synchronization",
           "A network timeout", "A disk I/O error"], "a": 1, "cat": "systems"},

    # ── Software Engineering (8) ──────────────────────────────
    {"q": "What does the Single Responsibility Principle state?",
     "c": ["A class should inherit from only one parent", "A class should have only one reason to change",
           "Use only one design pattern per class", "Functions should have one parameter"], "a": 1, "cat": "software_eng"},
    {"q": "Which design pattern ensures a class has only one instance?",
     "c": ["Factory", "Observer", "Singleton", "Strategy"], "a": 2, "cat": "software_eng"},
    {"q": "What is the main difference between unit tests and integration tests?",
     "c": ["Unit tests are slower", "Unit tests test individual components in isolation",
           "Integration tests don't use real data", "Unit tests test the whole system"], "a": 1, "cat": "software_eng"},
    {"q": "What does git rebase do compared to git merge?",
     "c": ["Creates a merge commit", "Replays commits on top of another branch for linear history",
           "Deletes the source branch", "Reverts all changes"], "a": 1, "cat": "software_eng"},
    {"q": "In CI/CD, what does CI stand for?",
     "c": ["Code Inspection", "Continuous Integration", "Central Intelligence", "Code Integration"], "a": 1, "cat": "software_eng"},
    {"q": "What is a code smell?",
     "c": ["A syntax error", "A hint that there might be a deeper problem in the code",
           "A security vulnerability", "A runtime exception"], "a": 1, "cat": "software_eng"},
    {"q": "What is a key advantage of microservices over monoliths?",
     "c": ["Simpler debugging", "Independent deployment and scaling",
           "Less network overhead", "Shared database"], "a": 1, "cat": "software_eng"},
    {"q": "Which HTTP method should be used to create a new resource?",
     "c": ["GET", "PUT", "DELETE", "POST"], "a": 3, "cat": "software_eng"},

    # ── Databases (8) ─────────────────────────────────────────
    {"q": "Why are B-tree indexes preferred for range queries over hash indexes?",
     "c": ["B-trees are smaller", "B-trees maintain sorted order enabling range scans",
           "Hash indexes are slower", "B-trees use less memory"], "a": 1, "cat": "databases"},
    {"q": "What does the 'A' in ACID stand for?",
     "c": ["Availability", "Atomicity", "Authentication", "Aggregation"], "a": 1, "cat": "databases"},
    {"q": "What does Third Normal Form (3NF) eliminate?",
     "c": ["Duplicate tables", "Transitive dependencies", "All nulls", "Foreign keys"], "a": 1, "cat": "databases"},
    {"q": "When would you choose a NoSQL database over SQL?",
     "c": ["When you need ACID transactions", "When schema is flexible and data is unstructured",
           "When you need complex JOINs", "When data integrity is critical"], "a": 1, "cat": "databases"},
    {"q": "What does an INNER JOIN return?",
     "c": ["All rows from both tables", "Only rows that have matching values in both tables",
           "All rows from the left table", "All rows from the right table"], "a": 1, "cat": "databases"},
    {"q": "What is the purpose of a database index?",
     "c": ["Backup data", "Speed up query lookups", "Enforce constraints", "Compress data"], "a": 1, "cat": "databases"},
    {"q": "Which isolation level provides the strongest guarantees?",
     "c": ["Read uncommitted", "Read committed", "Repeatable read", "Serializable"], "a": 3, "cat": "databases"},
    {"q": "In the CAP theorem, what does 'P' stand for?",
     "c": ["Performance", "Partition tolerance", "Persistence", "Parallelism"], "a": 1, "cat": "databases"},
]
