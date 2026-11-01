import solvers.«92.py.lean.d».Heap

namespace solvers.«92».walk
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.DigitSquareChains
set_option Elab.async false

abbrev program := Solvers.P92Ast.module
abbrev number := store.number
abbrev condition : Expr := .boolOp .and
  [.membership (.name "current") (.name "known") true,
   .membership (.name "current") (.name "positions") true]
abbrev body : List Stmt :=
  [.expr (.call (.attribute (.name "positions") "add") [.name "current"]),
   .expr (.call (.attribute (.name "path") "append") [.name "current"]),
   .assign [.name "current"] (.call (.name "next_term") [.name "current"])]
abbrev loop : Stmt := .whileDo condition body []
def active (entries : store.Entries) (current : Nat) (path : List Nat) : Bool :=
  !(store.lookup entries current).isSome && !decide (current ∈ path)

theorem source :
    (executableBody ((findFunctionIn program.body "terminal").getD ([], [])).2)[3]? = some loop := rfl

structure Captured (locals : HeapCalls.Env) (base : Mutable.Heap) (cache current : Nat) : Prop where
  known : locals "known" = some (.ref cache)
  path : locals "path" = some (.ref base.length)
  positions : locals "positions" = some (.ref (base.length + 1))
  current : locals "current" = some (number current)

theorem Captured.advance {locals : HeapCalls.Env} {base : Mutable.Heap} {cache current : Nat}
    (captured : Captured locals base cache current) :
    Captured (setLocal locals "current" (number (nextTerm current))) base cache (nextTerm current) := by
  constructor
  · simpa [setLocal] using captured.known
  · simpa [setLocal] using captured.path
  · simpa [setLocal] using captured.positions
  · simp [setLocal]

theorem guard_result (base : Mutable.Heap) (cache current : Nat) (entries : store.Entries)
    (found : store.Found base cache entries) (path : List Nat) (garbage : Mutable.Heap)
    (locals : HeapCalls.Env) (captured : Captured locals base cache current) (globals : Mutable.Globals) :
    Evaluates program locals condition (heap.work base path path garbage)
      (.scalar (.bool (active entries current path))) (heap.work base path path garbage) globals := by
  have first := NaturalDictionary.membership_evaluates (program := program) (globals := globals)
    (heap.work base path path garbage) locals "known" "current" cache current entries
    (heap.old_found base path path garbage cache _ found) captured.known captured.current true
  have second := heap.position_test (globals := globals) base path path garbage locals current
    captured.positions captured.current
  cases cached : (store.lookup entries current).isSome with
  | true =>
      simp only [cached, Bool.not_true] at first
      simpa only [condition, active, cached, Bool.not_true, Bool.false_and] using
        (Evaluates.boolean_stop (op := .and)
          (rest := [.membership (.name "current") (.name "positions") true]) first rfl rfl)
  | false =>
      simp only [cached, Bool.not_false] at first
      simpa only [condition, active, cached, Bool.not_false, Bool.true_and] using
        (Evaluates.boolean_next (op := .and) first rfl rfl (Evaluates.boolean_last second rfl))

/-- The forward traversal leaves the caller's cache untouched and allocates only private path data. -/
theorem run {entries : store.Entries} {current last value : Nat} {path result : List Nat}
    (trace : Memo.Trace (store.lookup entries) current path last result value)
    (base : Mutable.Heap) (cache : Nat) (found : store.Found base cache entries)
    (garbage : Mutable.Heap) (locals : HeapCalls.Env) (captured : Captured locals base cache current)
    (globals : Mutable.Globals) :
    ∃ after extra, BlockContinues program locals [loop] (heap.work base path path garbage)
      after (heap.work base result result (garbage ++ extra)) globals ∧ Captured after base cache last := by
  induction trace generalizing garbage locals with
  | cached selected =>
      refine ⟨locals, [], ?_, captured⟩
      simp only [List.append_nil]
      apply BlockContinues.while_loop (flag := false)
        (guard_result base cache _ entries found _ garbage locals captured globals)
        (by simp [active, selected, Mutable.truth, scalarTruth])
      exact .nil
  | repeated missing present =>
      refine ⟨locals, [], ?_, captured⟩
      simp only [List.append_nil]
      apply BlockContinues.while_loop (flag := false)
        (guard_result base cache _ entries found _ garbage locals captured globals)
        (by simp [active, missing, present, Mutable.truth, scalarTruth])
      exact .nil
  | @step current path last result value missing fresh trace ih =>
      obtain ⟨digitExtra, converted⟩ := next_term.result current
        (heap.work base (path ++ [current]) (path ++ [current]) garbage) globals
      rw [heap.work_append] at converted
      obtain ⟨after, extra, rest, finalCaptured⟩ := ih (garbage ++ digitExtra)
        (setLocal locals "current" (number (nextTerm current))) captured.advance
      refine ⟨after, digitExtra ++ extra, ?_, finalCaptured⟩
      apply BlockContinues.while_loop (flag := true)
        (guard_result base cache current entries found path garbage locals captured globals)
        (by simp [active, missing, fresh, Mutable.truth, scalarTruth])
      have added := BlockContinues.discard (heap.position_add (globals := globals) base path path garbage
        locals current fresh captured.positions captured.current)
      have appended := BlockContinues.discard (heap.path_append (globals := globals) base path (path ++ [current])
        garbage locals current captured.path captured.current)
      have advanced := BlockContinues.assign
        (Evaluates.call (.cons (Evaluates.lookup captured.current) .nil) converted)
        (name := "current")
      simpa only [loop, body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_or, Bool.false_eq_true, if_false, List.append_assoc] using
        (added.append (appended.append advanced)).append rest

end solvers.«92».walk
