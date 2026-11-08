import solvers.«23.py.lean.d».SieveFunction
import lang.py.FormalSW.HeapRuntimeCostComprehensions
import Mathlib.Data.List.GetD

/-! The source comprehension costs one indexed table read, comparison and
possible amortized append per candidate. Its budget needs only table length. -/
namespace solvers.«23».runtimeCost.Abundant
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
abbrev program := Solvers.P23Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
def encode (items : List Nat) : List Mutable.Value := items.map number

def test : Expr := .compare (.subscript (.name "sums") (.name "n")) [(.gt, .name "n")]
def body (address : Nat) : List Stmt :=
  [.ifThen test [.comprehensionAppend address (.name "n")] []]
def expression : Expr := .listComp (.name "n")
  [("n", .call (.name "range") [.constant (.int 12),
    .binOp (.name "limit") .add (.constant (.int 1))], [test])]

def selected (table : List Int) (n : Nat) : Bool := decide ((n : Int) < table.getD n 0)
def values (table : List Int) (limit : Nat) : List Nat :=
  (List.range' 12 (limit - 11)).filter (selected table)

theorem length_bound (table : List Int) (limit : Nat) : (values table limit).length ≤ limit := by
  have h := List.length_filter_le (selected table) (List.range' 12 (limit - 11))
  simp only [List.length_range'] at h
  exact h.trans (by omega)

theorem tested (table : List Int) (n address : Nat) (inside : n < table.length)
    (heap : Mutable.Heap) (found : heap[address]? = some (.list (Sieve.encode table)))
    (env : HeapCalls.Env) (getSums : env "sums" = some (.ref address))
    (getN : env "n" = some (number n)) :
    EvaluatesIn .word program env test heap 16 (.scalar (.bool (selected table n))) heap := by
  have read : EvaluatesIn .word program env (.subscript (.name "sums") (.name "n")) heap 10
      (.scalar (.int (table.getD n 0))) heap := by
    rw [List.getD_eq_getElem table 0 inside]
    apply EvaluatesIn.list_subscript (by intro a b c equal; cases equal)
      (EvaluatesIn.lookup getSums) (EvaluatesIn.lookup getN) found
    simpa only [Sieve.encode, List.getElem_map, Sieve.number] using
      Mutable.getItem_list heap address n (Sieve.encode table) found (by simpa using inside)
  apply EvaluatesIn.compare (work := 2) read (EvaluatesIn.lookup getN)
  · simp [HeapCalls.compare, scalarCompare, evalCmpOp, selected]
  · rfl

theorem loop (table : List Int) (limit count start address : Nat)
    (size : table.length = limit + 1) (stop : start + count = limit + 1)
    (heap : Mutable.Heap) (found : heap[address]? = some (.list (Sieve.encode table)))
    (collected : List Nat) (env : HeapCalls.Env) (getSums : env "sums" = some (.ref address)) :
    ∃ finalEnv, BlockContinuesIn .word program env
      [.rangeStepLoop "n" start (limit + 1) 1 (body heap.length) []]
      (heap ++ [.list (encode collected)]) (30 * count + 4) finalEnv
      (heap ++ [.list (encode (collected ++ (List.range' start count).filter (selected table)))]) := by
  have addressBound := (List.getElem?_eq_some_iff.mp found).1
  induction count generalizing start collected env with
  | zero =>
      refine ⟨env, ?_⟩
      have same : start = limit + 1 := by omega
      subst start
      simpa using BlockContinuesIn.range_done (module := program) (body := body heap.length)
        (by simp [Calls.rangeContinues]) BlockContinuesIn.nil
  | succ count ih =>
      let boundEnv := setLocal env "n" (number start)
      have saved : boundEnv "sums" = some (.ref address) := by
        simpa [boundEnv, setLocal] using getSums
      have evaluated := tested table start address (by omega) (heap ++ [.list (encode collected)])
        (by simpa only [List.getElem?_append_left addressBound] using found) boundEnv saved rfl
      let nextItems := if selected table start then collected ++ [start] else collected
      have executed : BlockContinuesIn .word program boundEnv (body heap.length)
          (heap ++ [.list (encode collected)]) 26 boundEnv (heap ++ [.list (encode nextItems)]) := by
        cases chosen : selected table start with
        | true =>
            apply BlockContinuesIn.branch (cc := 16) (cb := 8) (flag := true)
              (by simpa [chosen] using evaluated) rfl
            have emitted := BlockContinuesIn.comprehension_append (globals := [])
              (EvaluatesIn.lookup (arithmetic := .word) (module := program)
                (env := boundEnv) (name := "n") rfl)
              (by simp : (heap ++ [.list (encode collected)])[heap.length]? =
                some (.list (encode collected)))
            simpa [nextItems, chosen, encode, List.set_append_right] using emitted
        | false =>
            have skipped := BlockContinuesIn.branch (yes := [.comprehensionAppend heap.length (.name "n")])
              (no := []) (flag := false) (by simpa [chosen] using evaluated) rfl BlockContinuesIn.nil
            simpa [body, nextItems, chosen] using skipped.mono (large := 26) (by omega)
      obtain ⟨finalEnv, rest⟩ := ih (start + 1) (by omega) nextItems boundEnv saved
      refine ⟨finalEnv, ?_⟩
      have combined : BlockContinuesIn .word program boundEnv
          (body heap.length ++ [Calls.loopContinuation (body heap.length)
            (.rangeStepLoop "n" ((start : Int) + 1) (limit + 1) 1 (body heap.length) [])])
          (heap ++ [.list (encode collected)]) (26 + (30 * count + 4)) finalEnv
          (heap ++ [.list (encode (collected ++
            (List.range' start (count + 1)).filter (selected table)))]) := by
        cases chosen : selected table start <;>
          simpa [body, Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
            nextItems, chosen, List.range'_succ, List.append_assoc] using executed.append rest
      have run := BlockContinuesIn.range_step (env := env) (name := "n")
        (next := (start : Int)) (stop := (limit + 1 : Nat)) (stride := 1)
        (body := body heap.length) (otherwise := [])
        (by simp [Calls.rangeContinues]; omega) combined
      exact run.mono (by omega)

/-- The exact source expression returns a fresh list of at most `limit` naturals. -/
theorem expression_run (table : List Int) (limit address : Nat) (size : table.length = limit + 1)
    (heap : Mutable.Heap) (found : heap[address]? = some (.list (Sieve.encode table)))
    (env : HeapCalls.Env) (getLimit : env "limit" = some (number limit))
    (getSums : env "sums" = some (.ref address)) :
    EvaluatesIn .word program env expression heap (30 * limit + 24) (.ref heap.length)
      (heap ++ [.list (encode (values table limit))]) := by
  let localEnv := comprehensionEnv env [("n", .call (.name "range")
    [.constant (.int 12), .binOp (.name "limit") .add (.constant (.int 1))], [test])]
  have upper : EvaluatesIn .word program env
      (.binOp (.name "limit") .add (.constant (.int 1))) heap 6 (number (limit + 1)) heap := by
    apply EvaluatesIn.binary (work := 1) (EvaluatesIn.lookup getLimit) EvaluatesIn.constant
    · simp [Mutable.binary, scalarBinary, number]
    · rfl
  have arguments : ArgumentsIn .word program env 0
      [.constant (.int 12), .binOp (.name "limit") .add (.constant (.int 1))]
      heap 10 [number 12, number (limit + 1)] heap :=
    ArgumentsIn.cons EvaluatesIn.constant (ArgumentsIn.cons upper ArgumentsIn.nil)
  by_cases large : 12 ≤ limit
  · obtain ⟨finalEnv, executed⟩ := loop table limit (limit - 11) 12 address size (by omega)
      heap found [] localEnv (by simpa [localEnv, comprehensionEnv, setLocal] using getSums)
    have run := EvaluatesIn.list_comp_range (element := .name "n") (name := "n")
      (filters := [test]) (generators := []) (work := 4) arguments rfl rfl
      (by simpa [localEnv, comprehensionFilters, comprehensionBody, body, values] using executed)
    exact run.mono (by omega)
  · have count : limit - 11 = 0 := by omega
    have done : BlockContinuesIn .word program localEnv
        [.rangeStepLoop "n" 12 (limit + 1) 1 (body heap.length) []]
        (heap ++ [.list []]) 4 localEnv (heap ++ [.list []]) :=
      BlockContinuesIn.range_done (by simp [Calls.rangeContinues]; omega) BlockContinuesIn.nil
    have run := EvaluatesIn.list_comp_range (element := .name "n") (name := "n")
      (filters := [test]) (generators := []) (work := 4) arguments rfl rfl
      (by simpa [localEnv, comprehensionFilters, comprehensionBody, body] using done)
    simpa [expression, values, count, encode] using run.mono (large := 30 * limit + 24) (by omega)

end solvers.«23».runtimeCost.Abundant
