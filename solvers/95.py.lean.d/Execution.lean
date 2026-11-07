import solvers.«95.py.lean.d».Search

/-! Initialize the actual divisor and processed tables, then execute the complete optimized search. -/
namespace solvers.«95»
open FormalSW.PyAst HeapCalls ProjectEulerSolutions
open BoundedCycles BoundedCycleWalk
set_option Elab.async false

private abbrev program := Solvers.P95Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev divisors := ProjectEulerStatements.P21.d

namespace amicable_chain_best

private abbrev definition := (findFunctionIn program.body "amicable_chain_best").getD ([], [])

theorem result (limit : Nat) (heap : Mutable.Heap) :
    ∃ after, ReturnsValue program "amicable_chain_best" [number limit] heap (.ref after.length)
      (after ++ [.tuple [number (BoundedCycleWalk.solve divisors limit).1,
        number (BoundedCycleWalk.solve divisors limit).2]]) := by
  let entry := functionEnv definition.1 definition.2 [number limit] []
  let table := heap.length + 1
  let sieveHeap := heap ++ [.list [number 0], .list (Sieve.table (limit + 1) divisors)]
  let withTable := setLocal entry "s" (.ref table)
  let processedAddress := heap.length + 3
  let allocated := sieveHeap ++ [.list [.scalar (.bool false)], .list (processed.items limit ∅)]
  let withProcessed := setLocal withTable "processed" (.ref processedAddress)
  let ready := setLocal (setLocal withProcessed "best_len" (number 0)) "best_min" (number 0)
  have allocatedTable : search.Tables allocated limit table processedAddress divisors ∅ := by
    constructor
    · simp [allocated, sieveHeap, table]
    · simp [allocated, sieveHeap, processedAddress, show heap.length + 3 = heap.length + (2 + 1) by omega]
  have readyLocals : search.Locals ready limit table processedAddress (0, 0) := ⟨rfl, rfl, rfl, rfl, rfl⟩
  have one : divisors 1 = 0 := by decide
  have invariant : search.Invariant divisors limit 2 ∅ (0, 0) := by
    refine ⟨Or.inl rfl, ?_, ?_⟩
    · intro n member; simp at member
    · intro n before
      have cases : n = 0 ∨ n = 1 := by omega
      rcases cases with rfl | rfl
      · exact safe_zero _ _ _
      · exact safe_predecessor (by simpa only [one] using safe_zero divisors limit (0, 0))
  obtain ⟨after, finalEnv, finalKnown, best, scanned, finalTables, finalLocals, optimal⟩ :=
    search.range_result (globals := []) allocated ready limit table processedAddress 2 divisors ∅ (0, 0)
      (by dsimp [table, processedAddress]; omega) allocatedTable readyLocals invariant
  have same := optimal.unique (BoundedCycleWalk.solve_optimal divisors limit one)
  subst best
  refine ⟨after, ?_⟩
  have allocation : Evaluates program withTable (.binOp (.list [.constant (.bool false)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) sieveHeap (.ref processedAddress) allocated := by
    apply Evaluates.binary (av := .ref sieveHeap.length) (bv := number (limit + 1))
      (middle := sieveHeap ++ [.list [.scalar (.bool false)]])
      (afterRight := sieveHeap ++ [.list [.scalar (.bool false)]])
    · exact Evaluates.list (.cons Evaluates.constant .nil)
    · apply Evaluates.binary (Evaluates.lookup (show withTable "limit" = some (number limit) by rfl)) Evaluates.constant
      simp [Mutable.binary, scalarBinary, number]
    · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
        Mutable.array, Mutable.allocate, processed.items, allocated, sieveHeap, processedAddress, Except.map, number]
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockReturns.assign (value := .ref table)
    (Evaluates.call (.cons (Evaluates.lookup (show entry "limit" = some (number limit) by rfl)) .nil) (Sieve.result limit heap))
  apply BlockReturns.assign allocation
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockContinues.returns (body := [.forIn (.name "start") (.call (.name "range")
    [.constant (.int 2), .binOp (.name "limit") .add (.constant (.int 1))]) search.body []])
    (middleEnv := finalEnv) (middleHeap := after)
  · apply BlockContinues.for_range (values := [number 2, number (limit + 1)]) (middle := allocated)
    · apply Evaluates.Arguments.cons (middle := allocated) Evaluates.constant
      apply Evaluates.Arguments.cons (middle := allocated)
      · apply Evaluates.binary (middle := allocated) (afterRight := allocated)
          (Evaluates.lookup readyLocals.getLimit) Evaluates.constant
        simp [Mutable.binary, scalarBinary, number]
      · exact .nil
    · rfl
    · simpa only [number, Nat.cast_add, Nat.cast_one] using scanned
  · apply BlockReturns.return_value
    exact Evaluates.tuple (.cons (Evaluates.lookup finalLocals.getLength)
      (.cons (Evaluates.lookup finalLocals.getMinimum) .nil)) (by simp)

/-- Negative bounds allocate empty tables and execute no outer iteration. -/
theorem negative_result (limit : Int) (negative : limit < 0) (heap : Mutable.Heap) :
    ∃ after, ReturnsValue program "amicable_chain_best" [.scalar (.int limit)] heap (.ref after.length)
      (after ++ [.tuple [number 0, number 0]]) := by
  let entry := functionEnv definition.1 definition.2 [.scalar (.int limit)] []
  let sieveHeap := heap ++ [.list [number 0], .list []]
  let withTable := setLocal entry "s" (.ref (heap.length + 1))
  let after := sieveHeap ++ [.list [.scalar (.bool false)], .list []]
  have empty : (limit + 1).toNat = 0 := by omega
  have sieve := Sieve.small_result limit (by omega) heap
  simp only [empty, List.replicate_zero] at sieve
  have allocation : Evaluates program withTable (.binOp (.list [.constant (.bool false)]) .mult
      (.binOp (.name "limit") .add (.constant (.int 1)))) sieveHeap (.ref (sieveHeap.length + 1)) after := by
    apply Evaluates.binary (av := .ref sieveHeap.length) (bv := .scalar (.int (limit + 1)))
      (middle := sieveHeap ++ [.list [.scalar (.bool false)]])
      (afterRight := sieveHeap ++ [.list [.scalar (.bool false)]])
    · exact Evaluates.list (.cons Evaluates.constant .nil)
    · apply Evaluates.binary (Evaluates.lookup (show withTable "limit" = some (.scalar (.int limit)) by rfl)) Evaluates.constant
      simp [Mutable.binary, scalarBinary]
    · simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
        Mutable.array, Mutable.allocate, empty, after, Except.map]
  refine ⟨after, ?_⟩
  apply BlockReturns.function (args := definition.1) (body := definition.2) rfl rfl
  apply BlockReturns.assign (value := .ref (heap.length + 1))
    (Evaluates.call (.cons (Evaluates.lookup (show entry "limit" = some (.scalar (.int limit)) by rfl)) .nil) sieve)
  apply BlockReturns.assign allocation
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.assign (value := number 0) Evaluates.constant
  apply BlockReturns.for_range (values := [.scalar (.int 2), .scalar (.int (limit + 1))]) (middle := after)
  · apply Evaluates.Arguments.cons (middle := after) Evaluates.constant
    apply Evaluates.Arguments.cons (middle := after)
    · apply Evaluates.binary (middle := after) (afterRight := after)
        (Evaluates.lookup rfl) Evaluates.constant
      simp [Mutable.binary, scalarBinary]
    · exact .nil
  · rfl
  · apply BlockReturns.range_done (by simp [Calls.rangeContinues]; omega)
    exact BlockReturns.return_value (Evaluates.tuple
      (.cons (Evaluates.lookup rfl) (.cons (Evaluates.lookup rfl) .nil)) (by simp))

end amicable_chain_best

namespace solve

theorem result_nat (limit : Nat) (heap : Mutable.Heap) :
    ∃ after, ReturnsValue program "solve" [number limit] heap
      (number (BoundedCycleWalk.solve divisors limit).2) after := by
  obtain ⟨after, executed⟩ := amicable_chain_best.result limit heap
  refine ⟨after ++ [.tuple [number (BoundedCycleWalk.solve divisors limit).1,
    number (BoundedCycleWalk.solve divisors limit).2]], ?_⟩
  apply BlockReturns.function rfl rfl
  apply BlockReturns.return_value
  apply Evaluates.subscript (by intros; simp)
    (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) executed) Evaluates.constant
  simp [Mutable.getItem, Mutable.index, Except.map]

def answer (limit : Int) : Nat := (BoundedCycleWalk.solve divisors limit.toNat).2

theorem result (limit : Int) (heap : Mutable.Heap) :
    ∃ after, ReturnsValue program "solve" [.scalar (.int limit)] heap (number (answer limit)) after := by
  by_cases nonnegative : 0 ≤ limit
  · lift limit to Nat using nonnegative
    simpa only [answer, Int.toNat_natCast] using result_nat limit heap
  · obtain ⟨after, executed⟩ := amicable_chain_best.negative_result limit (by omega) heap
    have empty : answer limit = 0 := by
      simp [answer, Int.toNat_of_nonpos (by omega : limit ≤ 0), BoundedCycleWalk.solve, BoundedCycleWalk.search]
    rw [empty]
    refine ⟨after ++ [.tuple [number 0, number 0]], ?_⟩
    apply BlockReturns.function rfl rfl
    apply BlockReturns.return_value
    apply Evaluates.subscript (by intros; simp)
      (Evaluates.call (.cons (Evaluates.lookup rfl) .nil) executed) Evaluates.constant
    simp [Mutable.getItem, Mutable.index, Except.map]

end solve
end solvers.«95»
