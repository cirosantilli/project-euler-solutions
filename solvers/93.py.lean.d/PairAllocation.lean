import solvers.«93.py.lean.d».PairExpressions

namespace solvers.«93».pair_results
open FormalSW.PyAst HeapCalls ProjectEulerSolutions.RationalPair
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

abbrev firstObjects (a b : Pair) := (initialChoices a b).map object
abbrev firstRefs (base : Mutable.Heap) : List Mutable.Value :=
  [.ref base.length, .ref (base.length + 1), .ref (base.length + 2), .ref (base.length + 3)]
abbrev listAddress (base : Mutable.Heap) := base.length + 4
def initialHeap (base : Mutable.Heap) (a b : Pair) : Mutable.Heap :=
  working (base ++ firstObjects a b) (firstRefs base) []

theorem initial (a b : Pair) (positiveA : 0 < a.2) (positiveB : 0 < b.2)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals) (ready : Ready env a b) :
    Evaluates program env (.list [addExpr, subExpr, reverseSubExpr, mulExpr]) heap (.ref (listAddress heap))
      (initialHeap heap a b) globals := by
  have first := addition (heap := heap) (globals := globals) a b positiveA positiveB ready
  have second := subtraction (heap := heap ++ [object (canonical (add a b))]) (globals := globals)
    a b positiveA positiveB ready
  have third := reverse_subtraction
    (heap := (heap ++ [object (canonical (add a b))]) ++ [object (canonical (sub a b))])
    (globals := globals) a b positiveA positiveB ready
  have fourth := multiplication
    (heap := ((heap ++ [object (canonical (add a b))]) ++ [object (canonical (sub a b))]) ++
      [object (canonical (sub b a))]) (globals := globals) a b positiveA positiveB ready
  have built := Evaluates.list (.cons first (.cons second (.cons third (.cons fourth .nil))))
  simpa [initialHeap, working, initialChoices, List.append_assoc] using built

theorem initial_rows (base : Mutable.Heap) (a b : Pair) :
    Heap.Rows (initialHeap base a b) (firstRefs base) (initialChoices a b) := by
  refine .cons ?_ (.cons ?_ (.cons ?_ (.cons ?_ .nil)))
  · exact ⟨base.length, rfl, by simp [initialHeap, working, initialChoices]⟩
  · exact ⟨base.length + 1, rfl, by simp [initialHeap, working, initialChoices]⟩
  · exact ⟨base.length + 2, rfl, by simp [initialHeap, working, initialChoices]⟩
  · exact ⟨base.length + 3, rfl, by simp [initialHeap, working, initialChoices]⟩

abbrev appendBody (expression : Expr) : List Stmt :=
  [.expr (.call (.attribute (.name "results") "append") [expression])]

/-- Appending a newly normalized tuple changes only the private result list. -/
theorem append_result (base : Mutable.Heap) (old : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List Pair) (pair : Pair) (env : HeapCalls.Env) (globals : Mutable.Globals) (expression : Expr)
    (rows : Heap.Rows (working base old garbage) old pairs)
    (getOut : env "results" = some (.ref base.length))
    (evaluated : Evaluates program env expression (working base old garbage)
      (.ref (working base old garbage).length) (working base old garbage ++ [object pair]) globals) :
    BlockContinues program env (appendBody expression) (working base old garbage) env
      (working base (old ++ [.ref (working base old garbage).length]) (garbage ++ [object pair])) globals ∧
    Heap.Rows (working base (old ++ [.ref (working base old garbage).length]) (garbage ++ [object pair]))
      (old ++ [.ref (working base old garbage).length]) (pairs ++ [pair]) := by
  have afterList : (working base old garbage ++ [object pair])[base.length]? = some (.list old) :=
    prefix_found _ _ _ _ (found base old garbage)
  have appended := Evaluates.append (Evaluates.lookup getOut) (found base old garbage) evaluated afterList
  constructor
  · simpa only [append, store] using BlockContinues.discard appended
  · have row : Heap.Row (working base old (garbage ++ [object pair]))
        (.ref (working base old garbage).length) pair := by
      refine ⟨_, rfl, ?_⟩
      rw [← append]
      simp
    have last := row.collector (old ++ [.ref (working base old garbage).length]) []
    simp only [List.append_nil] at last
    exact List.rel_append (rows.collector _ [object pair]) (.cons last .nil)

/-- The truth test omits zero divisors before evaluating the normalization call. -/
theorem conditional_append (base : Mutable.Heap) (old : List Mutable.Value) (garbage : Mutable.Heap)
    (pairs : List Pair) (pair : Pair) (n : Int) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (test expression : Expr) (rows : Heap.Rows (working base old garbage) old pairs)
    (getOut : env "results" = some (.ref base.length))
    (tested : Evaluates program env test (working base old garbage) (number n) (working base old garbage) globals)
    (evaluated : n ≠ 0 → Evaluates program env expression (working base old garbage)
      (.ref (working base old garbage).length) (working base old garbage ++ [object pair]) globals) :
    ∃ added extra,
      BlockContinues program env [.ifThen test (appendBody expression) []] (working base old garbage) env
        (working base (old ++ added) (garbage ++ extra)) globals ∧
      Heap.Rows (working base (old ++ added) (garbage ++ extra)) (old ++ added)
        (pairs ++ if n = 0 then [] else [pair]) := by
  by_cases zero : n = 0
  · refine ⟨[], [], ?_, ?_⟩
    · simp only [List.append_nil]
      exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, zero]) .nil
    · simpa [zero] using rows
  · obtain ⟨appended, collected⟩ := append_result base old garbage pairs pair env globals expression
      rows getOut (evaluated zero)
    refine ⟨[.ref (working base old garbage).length], [object pair], ?_, ?_⟩
    · exact BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, zero]) appended
    · simpa [zero] using collected

end solvers.«93».pair_results
