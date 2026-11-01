import solvers.«92.py.lean.d».Walk

namespace solvers.«92».write
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := store.number
abbrev body : List Stmt := [.assign [.subscript (.name "known") (.name "value")] (.name "end")]

def finalLocals (locals : HeapCalls.Env) : List Nat → HeapCalls.Env
  | [] => locals
  | value :: rest => finalLocals (setLocal locals "value" (number value)) rest

theorem finalLocals_frame (locals : HeapCalls.Env) (values : List Nat)
    (name : String) (different : name ≠ "value") :
    finalLocals locals values name = locals name := by
  induction values generalizing locals with
  | nil => rfl
  | cons value rest ih => simp only [finalLocals, ih, setLocal, if_neg different]

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

theorem step (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress value endpoint : Nat) (entries : store.Entries)
    (found : store.Found heap cacheAddress entries)
    (getCache : locals "known" = some (.ref cacheAddress))
    (getValue : locals "value" = some (number value))
    (getEndpoint : locals "end" = some (number endpoint)) :
    BlockContinues program locals body heap locals
      (heap.set cacheAddress (.dict (store.encode (store.insert entries value endpoint)))) globals :=
  NaturalDictionary.insert_evaluates (program := program) heap locals "known" "value" cacheAddress value endpoint entries
    (.name "end") found getCache getValue (Evaluates.lookup getEndpoint)

/-- The live path iterator retains its original list while every visited term receives the cached endpoint. -/
theorem loop (heap : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress listAddress endpoint : Nat) (entries : store.Entries)
    (front remaining : List Nat) (foundCache : store.Found heap cacheAddress entries)
    (foundList : heap[listAddress]? = some (.list ((front ++ remaining).map number)))
    (getCache : locals "known" = some (.ref cacheAddress))
    (getEndpoint : locals "end" = some (number endpoint)) :
    BlockContinues program locals
      [.sequenceStepLoop "value" (.ref listAddress) front.length body []] heap
      (finalLocals locals remaining)
      (heap.set cacheAddress (.dict (store.encode (store.fill entries endpoint remaining))))
      globals := by
  induction remaining generalizing heap locals entries front with
  | nil =>
      have found : heap[listAddress]? = some (.list (front.map number)) := by
        simpa only [List.append_nil] using foundList
      simpa only [finalLocals, store.fill, set_same heap cacheAddress entries foundCache] using
        (BlockContinues.sequence_done (module := program) (env := locals)
          (source := .ref listAddress) (position := front.length) (name := "value")
          (body := body) (globals := globals) (items := front.map number)
          (by simp [sequenceValue, Mutable.sequence, found])
          (by simp) BlockContinues.nil)
  | cons value rest ih =>
      let nextEntries := store.insert entries value endpoint
      let nextHeap := heap.set cacheAddress (.dict (store.encode nextEntries))
      let nextLocals := setLocal locals "value" (number value)
      have different := cache_not_list heap cacheAddress listAddress entries
        (front ++ value :: rest) foundCache foundList
      have nextFound : nextHeap[listAddress]? =
          some (.list (((front ++ [value]) ++ rest).map number)) := by
        simpa only [nextHeap, List.getElem?_set_ne different, List.append_assoc,
          List.singleton_append] using foundList
      have recurse := ih nextHeap nextLocals nextEntries (front ++ [value])
        (NaturalDictionary.found_set foundCache nextEntries) nextFound
        (by simpa [nextLocals, setLocal] using getCache)
        (by simpa [nextLocals, setLocal] using getEndpoint)
      have executed := step (globals := globals) heap nextLocals
        cacheAddress value endpoint entries foundCache
        (by simpa [nextLocals, setLocal] using getCache) (by simp [nextLocals, setLocal])
        (by simpa [nextLocals, setLocal] using getEndpoint)
      apply BlockContinues.sequence_step (items := (front ++ value :: rest).map number)
        (value := number value) (by simp [sequenceValue, Mutable.sequence, foundList]) (by simp)
      simpa only [List.length_append, List.length_singleton, finalLocals, store.fill,
        Calls.loopContinuation, body, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_eq_true, if_false, nextHeap, nextLocals, nextEntries, List.set_set] using
        executed.append recurse

theorem result (heap middle : Mutable.Heap) (locals : HeapCalls.Env)
    (cacheAddress listAddress endpoint : Nat) (entries : store.Entries) (values : List Nat)
    (iterable : Expr) (ordinary : ∀ expressions, iterable ≠ .call (.name "range") expressions)
    (evaluated : Evaluates program locals iterable heap (.ref listAddress) middle globals)
    (foundCache : store.Found middle cacheAddress entries)
    (foundList : middle[listAddress]? = some (.list (values.map number)))
    (getCache : locals "known" = some (.ref cacheAddress))
    (getEndpoint : locals "end" = some (number endpoint)) :
    BlockContinues program locals [.forIn (.name "value") iterable body []] heap
      (finalLocals locals values)
      (middle.set cacheAddress (.dict (store.encode (store.fill entries endpoint values))))
      globals := by
  apply BlockContinues.for_sequence (items := values.map number) ordinary evaluated
    (by simp [Mutable.sequence, foundList])
  exact loop middle locals cacheAddress listAddress endpoint entries [] values foundCache
    foundList getCache getEndpoint

end solvers.«92».write
