import solvers.«92.py.lean.d».Write

namespace solvers.«92».terminal
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := store.number
abbrev tail : List Stmt :=
  [.assign [.name "end"] (.call (.attribute (.name "known") "get") [.name "current", .constant (.int 0)]),
   .forIn (.name "value") (.name "path") write.body [],
   .return (.name "end")]

theorem tail_source :
    (executableBody ((findFunctionIn program.body "terminal").getD ([], [])).2).drop 4 = tail := rfl

theorem finish (base : Mutable.Heap) (cache last value : Nat) (entries : store.Entries)
    (path : List Nat) (garbage : Mutable.Heap) (locals : HeapCalls.Env) (globals : Mutable.Globals)
    (found : store.Found base cache entries) (captured : walk.Captured locals base cache last)
    (selected : (store.lookup entries last).getD 0 = value) :
    BlockReturns program locals tail (heap.work base path path garbage) (number value)
      (heap.work (base.set cache (.dict (store.encode (store.fill entries value path)))) path path garbage) globals := by
  have foundCache : store.Found (heap.work base path path garbage) cache entries :=
    heap.old_found base path path garbage cache _ found
  apply BlockReturns.assign (value := number value)
  · have retrieved := NaturalDictionary.get_evaluates (program := program) (globals := globals)
      (heap.work base path path garbage) locals "known" "current" cache last 0 entries
      foundCache captured.known captured.current
    simpa only [selected] using retrieved
  · have written := write.result (globals := globals) (heap.work base path path garbage)
      (heap.work base path path garbage) (setLocal locals "end" (number value))
      cache base.length value entries path (.name "path") (by intro expressions; simp)
      (Evaluates.lookup (by simpa [setLocal] using captured.path)) foundCache
      (heap.path_found base path path garbage)
      (by simpa [setLocal] using captured.known) (by simp [setLocal])
    rw [heap.set_old base path path garbage cache _ (NaturalDictionary.found_inside found)] at written
    apply written.returns
    apply BlockReturns.return_value
    apply Evaluates.lookup
    simp [write.finalLocals_frame, setLocal]

/-- The source follows a complete abstract trace and then performs the original forward cache writes. -/
theorem result_of_trace (start cache : Nat) (entries : store.Entries) (base : Mutable.Heap)
    (globals : Mutable.Globals) (found : store.Found base cache entries)
    (last value : Nat) (path : List Nat) (trace : Memo.Trace (store.lookup entries) start [] last path value) :
    ∃ extra, ReturnsValue program "terminal" [number start, .ref cache] base (number value)
      (base.set cache (.dict (store.encode (store.fill entries value path))) ++ extra) globals := by
  let args := ((findFunctionIn program.body "terminal").getD ([], [])).1
  let source := ((findFunctionIn program.body "terminal").getD ([], [])).2
  let entry := functionEnv args source [number start, .ref cache] globals
  let initialized := setLocal (setLocal (setLocal entry "path" (.ref base.length))
    "positions" (.ref (base.length + 1))) "current" (number start)
  have captured : walk.Captured initialized base cache start := by
    constructor
    · cases globals <;> rfl
    · simp [initialized, setLocal]
    · simp [initialized, setLocal]
    · simp [initialized, setLocal]
  obtain ⟨after, garbage, walked, finalCaptured⟩ := walk.run trace base cache found [] initialized captured globals
  refine ⟨.list (heap.numbers path) :: .set (heap.constants path) :: garbage, ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.assign (Evaluates.list Evaluates.Arguments.nil)
  apply BlockReturns.assign (value := .ref (base.length + 1)) (middle := heap.work base [] [] [])
  · apply Evaluates.builtin .nil
    simp [primitive, Mutable.allocateSet, heap.work, Except.map]
  · apply BlockReturns.assign (value := number start)
      (Evaluates.lookup (by cases globals <;> rfl))
    have finished := finish base cache last value entries path garbage after globals found finalCaptured trace.selected
    have executed := walked.returns (by simpa only [List.nil_append] using finished)
    simpa only [heap.work, List.nil_append, walk.loop, walk.body, tail] using executed

/-- The observable label is correct and the shared cache remains correct after all path updates. -/
def Verified (start cache : Nat) (entries : store.Entries) (base : Mutable.Heap) (globals : Mutable.Globals) : Prop :=
  ∃ value path extra,
    ReturnsValue program "terminal" [number start, .ref cache] base (number value)
      (base.set cache (.dict (store.encode (store.fill entries value path))) ++ extra) globals ∧
    decide (value = 89) = endsAt89 start ∧ Memo.Valid (store.lookup (store.fill entries value path))

theorem verified (start cache : Nat) (entries : store.Entries) (base : Mutable.Heap) (globals : Mutable.Globals)
    (found : store.Found base cache entries) (valid : Memo.Valid (store.lookup entries)) :
    Verified start cache entries base globals := by
  obtain ⟨last, path, value, trace, classified, preserved⟩ := Memo.from_start (store.lookup entries) valid start
  obtain ⟨extra, executed⟩ := result_of_trace start cache entries base globals found last value path trace
  exact ⟨value, path, extra, executed, classified, by simpa only [store.fill_lookup] using preserved⟩

end solvers.«92».terminal
