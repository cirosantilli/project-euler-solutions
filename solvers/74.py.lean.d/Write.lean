import solvers.«74.py.lean.d».Walk
import lang.py.FormalSW.ReverseCalls

/-! Cache write batches retain the live path object used by Python's iterators. -/
namespace solvers.«74».write
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number := factorials.number

abbrev tailBody : List Stmt :=
  [.augAssign (.name "remaining") .add (.constant (.int 1)),
   .assign [.subscript (.name "lengths") (.name "value")] (.name "remaining")]

abbrev cycleBody : List Stmt :=
  [.assign [.subscript (.name "lengths") (.name "value")] (.name "cycle_length")]

abbrev cycleIterable : Expr :=
  .subscript (.name "path") (.slice (some (.name "cycle_start")) none none)

abbrev prefixIterable : Expr :=
  .subscript (.name "path") (.slice none (some (.name "cycle_start")) none)

abbrev cachedCode : List Stmt :=
  [.assign [.name "remaining"] (.subscript (.name "lengths") (.name "current")),
   .forReversed "value" (.name "path") tailBody []]

abbrev cycleCode : List Stmt :=
  [.assign [.name "cycle_start"] (.subscript (.name "positions") (.name "current")),
   .assign [.name "cycle_length"]
     (.binOp (.builtinCall .len [.name "path"]) .sub (.name "cycle_start")),
   .forIn (.name "value") cycleIterable cycleBody [],
   .assign [.name "remaining"] (.name "cycle_length"),
   .forReversed "value" prefixIterable tailBody []]

abbrev code : List Stmt :=
  [.ifThen (.membership (.name "current") (.name "lengths") false) cachedCode cycleCode]

theorem code_source :
    ((executableBody ((findFunctionIn program.body "chain_length").getD
      ([], [])).2).drop 4).take 1 = code := rfl

def tailFill (entries : store.Entries) (remaining : Nat) : List Nat → store.Entries
  | [] => entries
  | value :: rest => tailFill (store.insert entries value (remaining + 1)) (remaining + 1) rest

def tailLocals (locals : HeapCalls.Env) (remaining : Nat) : List Nat → HeapCalls.Env
  | [] => locals
  | value :: rest => tailLocals
      (setLocal (setLocal locals "value" (number value)) "remaining" (number (remaining + 1)))
      (remaining + 1) rest

theorem tailLocals_frame (locals : HeapCalls.Env) (remaining : Nat) (values : List Nat)
    (name : String) (notValue : name ≠ "value") (notRemaining : name ≠ "remaining") :
    tailLocals locals remaining values name = locals name := by
  induction values generalizing locals remaining with
  | nil => rfl
  | cons value rest ih => simp only [tailLocals, ih, setLocal, if_neg notRemaining, if_neg notValue]

theorem tailLocals_remaining (locals : HeapCalls.Env) (remaining : Nat) (values : List Nat)
    (found : locals "remaining" = some (number remaining)) :
    tailLocals locals remaining values "remaining" = some (number (remaining + values.length)) := by
  induction values generalizing locals remaining with
  | nil => exact found
  | cons value rest ih =>
      simpa only [tailLocals, List.length_cons, Nat.add_assoc, Nat.add_comm 1] using
        ih (setLocal (setLocal locals "value" (number value))
          "remaining" (number (remaining + 1))) (remaining + 1) (by simp [setLocal])

def cycleFill (entries : store.Entries) (length : Nat) : List Nat → store.Entries
  | [] => entries
  | value :: rest => cycleFill (store.insert entries value length) length rest

def cycleLocals (locals : HeapCalls.Env) : List Nat → HeapCalls.Env
  | [] => locals
  | value :: rest => cycleLocals (setLocal locals "value" (number value)) rest

theorem cycleLocals_frame (locals : HeapCalls.Env) (values : List Nat)
    (name : String) (notValue : name ≠ "value") :
    cycleLocals locals values name = locals name := by
  induction values generalizing locals with
  | nil => rfl
  | cons value rest ih => simp only [cycleLocals, ih, setLocal, if_neg notValue]

theorem cycleFill_lookup (entries : store.Entries) (length : Nat) (values : List Nat) :
    store.lookup (cycleFill entries length values) =
      ProjectEulerStatements.DigitFactorialChains.Memo.addCycle (store.lookup entries)
        values length := by
  induction values generalizing entries with
  | nil => funext key; simp [cycleFill, ProjectEulerStatements.DigitFactorialChains.Memo.addCycle]
  | cons value rest ih =>
      funext key
      simp only [cycleFill, ih, ProjectEulerStatements.DigitFactorialChains.Memo.addCycle,
        store.lookup_insert, List.mem_cons]
      by_cases here : key = value <;> by_cases later : key ∈ rest <;> simp [here, later]

theorem cycleFill_valid (entries : store.Entries) (length : Nat) (values : List Nat)
    (valid : ProjectEulerStatements.DigitFactorialChains.Memo.Valid (store.lookup entries))
    (correct : ∀ n ∈ values, ProjectEulerStatements.DigitFactorialChains.chainLen n = length)
    (closed : ∀ n ∈ values, ProjectEulerStatements.DigitFactorialChains.nextTerm n ∈ values) :
    ProjectEulerStatements.DigitFactorialChains.Memo.Valid
      (store.lookup (cycleFill entries length values)) := by
  rw [cycleFill_lookup]
  exact valid.with_cycle values length correct closed

private theorem set_same (heap : Mutable.Heap) (address : Nat) (entries : store.Entries)
    (found : store.Found heap address entries) :
    heap.set address (.dict (store.encode entries)) = heap := by
  obtain ⟨inside, selected⟩ := List.getElem?_eq_some_iff.mp found
  rw [← selected]
  exact List.set_getElem_self inside

private theorem cache_not_list (heap : Mutable.Heap) (cacheAddress listAddress : Nat)
    (entries : store.Entries) (values : List Nat)
    (foundCache : store.Found heap cacheAddress entries)
    (foundList : heap[listAddress]? = some (.list (values.map number))) :
    cacheAddress ≠ listAddress := by
  intro same
  subst listAddress
  rw [foundCache] at foundList
  cases foundList

variable {globals : Mutable.Globals}

theorem tail_step (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress value remaining : Nat) (entries : store.Entries)
    (found : store.Found heap cacheAddress entries)
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getValue : locals "value" = some (number value))
    (getRemaining : locals "remaining" = some (number remaining)) :
    BlockContinues program locals tailBody heap
      (setLocal locals "remaining" (number (remaining + 1)))
      (heap.set cacheAddress (.dict (store.encode (store.insert entries value (remaining + 1)))))
      globals := by
  have incremented : BlockContinues program locals
      [.augAssign (.name "remaining") .add (.constant (.int 1))] heap
      (setLocal locals "remaining" (number (remaining + 1))) heap globals := by
    apply BlockContinues.augAssign getRemaining Evaluates.constant
    simp [inplace, Mutable.binary, scalarBinary, number, factorials.number]
  have written := store.insert_evaluates (globals := globals) heap
    (setLocal locals "remaining" (number (remaining + 1))) "lengths" "value"
    cacheAddress value (remaining + 1) entries (.name "remaining") found
    (by simpa [setLocal] using getCache) (by simpa [setLocal] using getValue)
    (Evaluates.lookup (by simp [setLocal]))
  exact incremented.append written

/-- The private reverse cursor reads the unchanged list after every dictionary write. -/
theorem reverse_prefix (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress listAddress remaining : Nat) (entries : store.Entries)
    (front suffix : List Nat) (foundCache : store.Found heap cacheAddress entries)
    (foundList : heap[listAddress]? = some (.list ((front ++ suffix).map number)))
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getRemaining : locals "remaining" = some (number remaining)) :
    BlockContinues program locals
      [.reverseSequenceStepLoop "value" (.ref listAddress) ((front.length : Int) - 1) tailBody []]
      heap (tailLocals locals remaining front.reverse)
      (heap.set cacheAddress (.dict (store.encode (tailFill entries remaining front.reverse))))
      globals := by
  induction front using List.reverseRecOn generalizing heap locals entries remaining suffix with
  | nil =>
      simpa only [List.length_nil, List.reverse_nil, tailLocals, tailFill,
        set_same heap cacheAddress entries foundCache] using
        (BlockContinues.reverse_done (module := program) (env := locals)
          (source := .ref listAddress) (position := -1) (name := "value")
          (body := tailBody) (globals := globals) (by omega) BlockContinues.nil)
  | append_singleton front value ih =>
      let nextEntries := store.insert entries value (remaining + 1)
      let nextHeap := heap.set cacheAddress (.dict (store.encode nextEntries))
      let nextLocals := setLocal (setLocal locals "value" (number value))
        "remaining" (number (remaining + 1))
      have different := cache_not_list heap cacheAddress listAddress entries
        ((front ++ [value]) ++ suffix) foundCache foundList
      have nextFound : nextHeap[listAddress]? =
          some (.list ((front ++ (value :: suffix)).map number)) := by
        simpa only [nextHeap, List.getElem?_set_ne different, List.append_assoc,
          List.singleton_append] using foundList
      have recurse := ih nextHeap nextLocals (remaining + 1) nextEntries (value :: suffix)
        (store.found_set foundCache nextEntries) nextFound
        (by simpa [nextLocals, setLocal] using getCache) (by simp [nextLocals, setLocal])
      have executed := tail_step (globals := globals) heap
        (setLocal locals "value" (number value)) cacheAddress value remaining entries foundCache
        (by simpa [setLocal] using getCache) (by simp [setLocal])
        (by simpa [setLocal] using getRemaining)
      have cursor : ((front ++ [value]).length : Int) - 1 = front.length := by simp
      rw [cursor]
      apply BlockContinues.reverse_step (items := ((front ++ [value]) ++ suffix).map number)
        (value := number value) (by omega)
        (by simp [sequenceValue, Mutable.reversibleSequence, Mutable.sequence, foundList])
        (by simp)
      have joined := executed.append recurse
      simpa only [List.reverse_append, List.reverse_cons, List.reverse_nil, List.nil_append,
        List.singleton_append, tailLocals, tailFill, Calls.loopContinuation,
        tailBody, Calls.loopControl, Calls.loopControlStmt, Bool.false_eq_true, if_false,
        nextHeap, nextLocals, nextEntries, List.set_set] using joined

theorem reverse_result (heap middle : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress listAddress remaining : Nat) (entries : store.Entries) (values : List Nat)
    (iterable : Expr) (evaluated : Evaluates program locals iterable heap (.ref listAddress)
      middle globals) (foundCache : store.Found middle cacheAddress entries)
    (foundList : middle[listAddress]? = some (.list (values.map number)))
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getRemaining : locals "remaining" = some (number remaining)) :
    BlockContinues program locals [.forReversed "value" iterable tailBody []] heap
      (tailLocals locals remaining values.reverse)
      (middle.set cacheAddress (.dict (store.encode (tailFill entries remaining values.reverse))))
      globals := by
  apply BlockContinues.for_reversed (items := values.map number) evaluated
    (by simp [Mutable.reversibleSequence, Mutable.sequence, foundList])
  simpa only [List.length_map, sequenceSource] using
    reverse_prefix middle locals cacheAddress listAddress remaining entries values [] foundCache
      (by simpa only [List.append_nil] using foundList) getCache getRemaining

theorem cycle_step (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress value length : Nat) (entries : store.Entries)
    (found : store.Found heap cacheAddress entries)
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getValue : locals "value" = some (number value))
    (getLength : locals "cycle_length" = some (number length)) :
    BlockContinues program locals cycleBody heap locals
      (heap.set cacheAddress (.dict (store.encode (store.insert entries value length)))) globals :=
  store.insert_evaluates heap locals "lengths" "value" cacheAddress value length entries
    (.name "cycle_length") found getCache getValue (Evaluates.lookup getLength)

/-- Forward cycle iteration writes a constant length and preserves the sliced list. -/
theorem cycle_loop (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress listAddress length : Nat) (entries : store.Entries)
    (front remaining : List Nat) (foundCache : store.Found heap cacheAddress entries)
    (foundList : heap[listAddress]? = some (.list ((front ++ remaining).map number)))
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getLength : locals "cycle_length" = some (number length)) :
    BlockContinues program locals
      [.sequenceStepLoop "value" (.ref listAddress) front.length cycleBody []] heap
      (cycleLocals locals remaining)
      (heap.set cacheAddress (.dict (store.encode (cycleFill entries length remaining))))
      globals := by
  induction remaining generalizing heap locals entries front with
  | nil =>
      have found : heap[listAddress]? = some (.list (front.map number)) := by
        simpa only [List.append_nil] using foundList
      simpa only [cycleLocals, cycleFill, set_same heap cacheAddress entries foundCache] using
        (BlockContinues.sequence_done (module := program) (env := locals)
          (source := .ref listAddress) (position := front.length) (name := "value")
          (body := cycleBody) (globals := globals) (items := front.map number)
          (by simp [sequenceValue, Mutable.sequence, found])
          (by simp) BlockContinues.nil)
  | cons value rest ih =>
      let nextEntries := store.insert entries value length
      let nextHeap := heap.set cacheAddress (.dict (store.encode nextEntries))
      let nextLocals := setLocal locals "value" (number value)
      have different := cache_not_list heap cacheAddress listAddress entries
        (front ++ value :: rest) foundCache foundList
      have nextFound : nextHeap[listAddress]? =
          some (.list (((front ++ [value]) ++ rest).map number)) := by
        simpa only [nextHeap, List.getElem?_set_ne different, List.append_assoc,
          List.singleton_append] using foundList
      have recurse := ih nextHeap nextLocals nextEntries (front ++ [value])
        (store.found_set foundCache nextEntries) nextFound
        (by simpa [nextLocals, setLocal] using getCache)
        (by simpa [nextLocals, setLocal] using getLength)
      have executed := cycle_step (globals := globals) heap nextLocals
        cacheAddress value length entries foundCache
        (by simpa [nextLocals, setLocal] using getCache) (by simp [nextLocals, setLocal])
        (by simpa [nextLocals, setLocal] using getLength)
      apply BlockContinues.sequence_step (items := (front ++ value :: rest).map number)
        (value := number value) (by simp [sequenceValue, Mutable.sequence, foundList]) (by simp)
      simpa only [List.length_append, List.length_singleton, cycleLocals, cycleFill,
        Calls.loopContinuation, cycleBody, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_eq_true, if_false, nextHeap, nextLocals, nextEntries, List.set_set] using
        executed.append recurse

theorem cycle_result (heap middle : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress listAddress length : Nat) (entries : store.Entries) (values : List Nat)
    (iterable : Expr) (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates program locals iterable heap (.ref listAddress) middle globals)
    (foundCache : store.Found middle cacheAddress entries)
    (foundList : middle[listAddress]? = some (.list (values.map number)))
    (getCache : locals "lengths" = some (.ref cacheAddress))
    (getLength : locals "cycle_length" = some (number length)) :
    BlockContinues program locals [.forIn (.name "value") iterable cycleBody []] heap
      (cycleLocals locals values)
      (middle.set cacheAddress (.dict (store.encode (cycleFill entries length values))))
      globals := by
  apply BlockContinues.for_sequence (items := values.map number) ordinary evaluated
    (by simp [Mutable.sequence, foundList])
  exact cycle_loop middle locals cacheAddress listAddress length entries [] values foundCache
    foundList getCache getLength

end solvers.«74».write
