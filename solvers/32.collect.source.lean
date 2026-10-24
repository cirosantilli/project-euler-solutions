import solvers.«32.product.source»
import lang.py.FormalSW.SetSumCalls
import lang.py.FormalSW.ScalarSetLemmas

set_option Elab.async false

namespace solvers.«32».solve
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.PandigitalProductSearch
private abbrev program := Solvers.P32Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev flag (b : Bool) : Mutable.Value := .scalar (.bool b)

def working (base : Mutable.Heap) (values : List Constant) : Mutable.Heap := base ++ [.set values]

@[simp] theorem working_found (base : Mutable.Heap) (values : List Constant) :
    (working base values)[base.length]? = some (.set values) := by simp [working]

theorem working_set (base : Mutable.Heap) (values replacement : List Constant) :
    (working base values).set base.length (.set replacement) = working base replacement := by
  simp [working, List.set_append_right]

def inserted (before : List Constant) (items : List Nat) : List Constant :=
  (items.map fun (n : Nat) => Constant.int n).foldl ScalarSet.insert before

theorem inserted_append (before : List Constant) (first second : List Nat) :
    inserted before (first ++ second) = inserted (inserted before first) second := by
  simp [inserted, List.map_append, List.foldl_append]

def selected (width a b : Nat) : List Nat := if accepted width a b then [a * b] else []

def rowSlice (width a first count : Nat) : List Nat :=
  (List.range' first count).flatMap (selected width a)

theorem rowSlice_full (width a : Nat) :
    rowSlice width a (firstB width a) (lastB width a + 1 - firstB width a) = row width a := by
  have same (items : List Nat) : items.flatMap (selected width a) =
      (items.filter (accepted width a)).map (a * ·) := by
    induction items with
    | nil => rfl
    | cons b rest ih => cases yes : accepted width a b <;> simp [selected, yes, ih]
  exact same _

def writes : List String := ["product", "b", "first_b", "a"]
def Stable (before after : HeapCalls.Env) : Prop :=
  ∀ query, query ∉ writes → after query = before query

def candidateBody : List Stmt :=
  [.assign [.name "product"] (.binOp (.name "a") .mult (.name "b")),
   .ifThen productTest
     [.expr (.call (.attribute (.name "products") "add") [.name "product"])] []]

def sourceCandidateBody : List Stmt :=
  match (((executableBody ((findFunctionIn program.body "solve").getD ([], [])).2)[6]?) : Option Stmt) with
  | some (.forIn _ _ outer _) =>
      match (outer[1]? : Option Stmt) with
      | some (.forIn _ _ inner _) => inner
      | _ => []
  | _ => []

theorem candidate_source : sourceCandidateBody = candidateBody := by rfl

theorem candidate_result (width a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b)
    (base : Mutable.Heap) (before : List Constant) (env : HeapCalls.Env)
    (getA : env "a" = some (number a)) (getB : env "b" = some (number b))
    (getWidth : env "width" = some (number width))
    (getProducts : env "products" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env candidateBody (working base before) finalEnv
      (working base (inserted before (selected width a b))) ∧
      finalEnv "a" = some (number a) ∧ Stable env finalEnv := by
  let ready := setLocal env "product" (number (a * b))
  have assigned : BlockContinues program env [candidateBody[0]'(by decide)]
      (working base before) ready (working base before) :=
    BlockContinues.assign (Evaluates.binary (av := number a) (bv := number b)
      (middle := working base before) (afterRight := working base before)
      (Evaluates.lookup getA) (Evaluates.lookup getB) (by simp [Mutable.binary, scalarBinary]))
  have tested := product_test width a b aPositive bPositive (working base before) ready
    (by simpa [ready, setLocal] using getA) (by simpa [ready, setLocal] using getB)
    (by simp [ready, setLocal]) (by simpa [ready, setLocal] using getWidth)
  refine ⟨ready, ?_, by simpa [ready, setLocal] using getA, ?_⟩
  · apply assigned.append
    by_cases yes : accepted width a b = true
    · simp only [selected, yes, if_true, inserted]
      apply BlockContinues.branch (middle := working base before) (flag := true) (value := flag true)
        (by simpa only [yes] using tested) rfl
      apply BlockContinues.discard
      simpa only [working_set] using Evaluates.add (value := .int (a * b))
        (Evaluates.lookup (module := program) (env := ready)
          (by simpa [ready, setLocal] using getProducts)) (working_found base before)
        (Evaluates.lookup (by simp [ready, setLocal])) (working_found base before)
    · have no : accepted width a b = false := Bool.eq_false_iff.mpr yes
      simp only [selected, no, inserted]
      apply BlockContinues.branch (middle := working base before) (flag := false) (value := flag false)
        (by simpa only [no] using tested) rfl
      exact .nil
  · intro query unchanged
    simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
    simp [ready, setLocal, unchanged.1]

theorem inner_loop (width a first count stop : Nat) (aPositive : 0 < a) (firstPositive : 0 < first)
    (length : stop - first = count) (base : Mutable.Heap) (before : List Constant) (env : HeapCalls.Env)
    (getA : env "a" = some (number a)) (getWidth : env "width" = some (number width))
    (getProducts : env "products" = some (.ref base.length)) :
    ∃ finalEnv, BlockContinues program env [.rangeStepLoop "b" first stop 1 candidateBody []]
      (working base before) finalEnv
      (working base (inserted before (rowSlice width a first count))) ∧ Stable env finalEnv := by
  induction count generalizing first before env with
  | zero =>
      refine ⟨env, ?_, fun _ _ => rfl⟩
      simpa only [rowSlice, List.range'_zero, List.flatMap_nil, inserted, List.map_nil,
        List.foldl_nil] using
        (BlockContinues.range_done (module := program) (env := env) (name := "b")
          (next := first) (stop := stop) (stride := 1) (body := candidateBody) (otherwise := [])
          (heap := working base before) (by simp [Calls.rangeContinues]; omega) .nil)
  | succ count ih =>
      let bound := setLocal env "b" (number first)
      obtain ⟨middle, executed, middleA, middleStable⟩ :=
        candidate_result width a first aPositive firstPositive base before bound
          (by simpa [bound, setLocal] using getA) (by simp [bound, setLocal])
          (by simpa [bound, setLocal] using getWidth)
          (by simpa [bound, setLocal] using getProducts)
      have middleWidth : middle "width" = some (number width) :=
        (middleStable "width" (by decide)).trans (by simpa [bound, setLocal] using getWidth)
      have middleProducts : middle "products" = some (.ref base.length) :=
        (middleStable "products" (by decide)).trans (by simpa [bound, setLocal] using getProducts)
      obtain ⟨finalEnv, completed, finalStable⟩ := ih (first + 1) (by omega) (by omega)
        (inserted before (selected width a first)) middle middleA middleWidth middleProducts
      refine ⟨finalEnv, ?_, ?_⟩
      · apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
        simpa [rowSlice, List.range'_succ, inserted_append, candidateBody,
          Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt] using executed.append completed
      · intro query unchanged
        have different : query ≠ "b" := by
          simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at unchanged
          exact unchanged.2.1
        exact (finalStable query unchanged).trans ((middleStable query unchanged).trans
          (by simp [bound, setLocal, different]))

end solvers.«32».solve
