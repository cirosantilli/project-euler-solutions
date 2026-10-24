import «.formalsw».solvers.«72.py.ast»
import ProjectEulerSolutions.Lib.TotientSieveCalls
import ProjectEulerSolutions.P72
import lang.py.FormalSW.SumCalls

/-! The Python solver sums a fresh suffix of its proved complete totient table. -/
namespace solvers.«72»
open FormalSW.PyAst HeapCalls ProjectEulerStatements ProjectEulerSolutions
set_option Elab.async false

abbrev program := Solvers.P72Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev integer (n : Int) : Mutable.Value := .scalar (.int n)
abbrev flag (value : Bool) : Mutable.Value := .scalar (.bool value)

theorem sieve_implementation : TotientSieveCalls.Implements program "totients_up_to" :=
  ⟨((findFunctionIn program.body "totients_up_to").getD ([], [])).2, rfl, rfl⟩

theorem totients_up_to.result (limit : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "totients_up_to" [integer limit] heap
      (.ref (heap.length + 1)) finalHeap ∧
      finalHeap[heap.length + 1]? = some
        (.list (TotientSieveCalls.entries (limit + 1).toNat (fun n => number n.totient))) :=
  TotientSieveCalls.result "totients_up_to" sieve_implementation limit heap

abbrev tailValues (size : Nat) : List Nat := ((List.range size).drop 2).map Nat.totient
abbrev suffixExpression : Expr :=
  .subscript (.name "phi") (.slice (some (.constant (.int 2))) none none)
abbrev sumExpression : Expr := .sumSequence suffixExpression (.constant (.int 0))

theorem sum_cast (values : List Nat) (initial : Nat) :
    (values.map (fun (n : Nat) => (n : Int))).foldl Int.add initial =
      ((initial + values.sum : Nat) : Int) := by
  induction values generalizing initial with
  | nil => rfl
  | cons head tail ih =>
      rw [List.map_cons, List.foldl_cons]
      have add : Int.add (initial : Int) (head : Int) = ((initial + head : Nat) : Int) := by
        exact (Int.natCast_add initial head).symm
      rw [add, ih]
      simp [Nat.add_assoc]

/-- Slicing allocates once, and summation leaves every pre-existing object unchanged. -/
theorem slice_sum_result (size address : Nat) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getPhi : env "phi" = some (.ref address))
    (found : heap[address]? = some
      (.list (TotientSieveCalls.entries size (fun n => number n.totient)))) :
    Evaluates program env sumExpression heap (number (tailValues size).sum)
      (heap ++ [.list ((tailValues size).map number)]) := by
  let items := TotientSieveCalls.entries size (fun n => number n.totient)
  have tailEq : items.drop 2 = (tailValues size).map number := by
    simp only [items, TotientSieveCalls.entries, tailValues, List.map_drop, List.map_map]
    rfl
  have sliced : Evaluates program env suffixExpression heap (.ref heap.length)
      (heap ++ [.list ((tailValues size).map number)]) := by
    apply Evaluates.slice
      (.cons (Evaluates.lookup getPhi) (.cons Evaluates.constant
        (.cons Evaluates.constant (.cons Evaluates.constant .nil))))
    simpa only [tailEq] using Mutable.getSlice_list_suffix heap address items 2 found
  have encoded : ((tailValues size).map (fun (n : Nat) => (n : Int))).map
      (fun n => Mutable.Value.scalar (.int n)) = (tailValues size).map number := by
    simp only [List.map_map]
    rfl
  have summed := Evaluates.sum_integers (module := program)
    (values := (tailValues size).map (fun (n : Nat) => (n : Int)))
    (initial := 0) (.cons sliced (.cons Evaluates.constant .nil))
    (by rw [encoded]; simp [Mutable.sequence])
  have total : ((tailValues size).map (fun (n : Nat) => (n : Int))).foldl Int.add 0 =
      ((tailValues size).sum : Int) := by
    simpa only [Nat.zero_add, Int.natCast_zero] using sum_cast (tailValues size) 0
  rw [total] at summed
  exact summed

namespace solve
abbrev arguments : List Arg := [⟨"limit", some (.name "int"), none⟩]
abbrev guard : Expr := .compare (.name "limit") [(.lt, .constant (.int 2))]
abbrev body : List Stmt :=
  [.ifThen guard [.return (.constant (.int 0))] [],
   .assign [.name "phi"] (.call (.name "totients_up_to") [.name "limit"]),
   .return sumExpression]

theorem code : executableBody ((findFunctionIn program.body "solve").getD ([], [])).2 = body := rfl

theorem body_result (limit : Int) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getLimit : env "limit" = some (integer limit)) :
    ∃ finalHeap, BlockReturns program env body heap
      (number (ReducedFractionCount.result limit)) finalHeap := by
  have tested : Evaluates program env guard heap (flag (decide (limit < 2))) heap :=
    Evaluates.compare (middle := heap) (afterRight := heap)
      (Evaluates.lookup getLimit) Evaluates.constant
      (by simp [HeapCalls.compare, scalarCompare, evalCmpOp])
  by_cases small : limit < 2
  · refine ⟨heap, ?_⟩
    rw [ReducedFractionCount.result_small limit small]
    apply BlockReturns.branch (flag := true)
      (by simpa only [small, decide_true] using tested) rfl
    exact BlockReturns.return_value Evaluates.constant
  · obtain ⟨tableHeap, returned, found⟩ := totients_up_to.result limit heap
    let ready := setLocal env "phi" (.ref (heap.length + 1))
    have summed := slice_sum_result (limit + 1).toNat (heap.length + 1) tableHeap ready
      (by simp [ready, setLocal]) found
    have value : (tailValues (limit + 1).toNat).sum = ReducedFractionCount.result limit :=
      ReducedFractionCount.tail_sum (limit + 1).toNat
    rw [value] at summed
    refine ⟨tableHeap ++ [.list ((tailValues (limit + 1).toNat).map number)], ?_⟩
    apply BlockReturns.branch (flag := false)
      (by simpa only [small, decide_false] using tested) rfl
    apply BlockReturns.assign
      (Evaluates.call (.cons (Evaluates.lookup getLimit) .nil) returned)
    exact BlockReturns.return_value summed

theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer limit] heap
      (number (ReducedFractionCount.result limit)) finalHeap := by
  obtain ⟨finalHeap, returned⟩ := body_result limit heap
    (HeapCalls.argumentEnv arguments [integer limit]) rfl
  refine ⟨finalHeap, ?_⟩
  apply BlockReturns.function rfl rfl
  exact returned

/-- The source and the runnable array model have the same result at every signed bound. -/
theorem model_result (limit : Int) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [integer limit] heap
      (number (ProjectEulerSolutions.P72.solve limit)) finalHeap := by
  rw [ProjectEulerSolutions.P72.solve_eq_result]
  exact result limit heap

end solve
end solvers.«72»
