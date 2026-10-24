import solvers.«37.suffix.source»
import ProjectEulerSolutions.Lib.TruncatablePrimeSearch

set_option Elab.async false

namespace solvers.«37».solve
open FormalSW.PyAst HeapCalls ProjectEulerStatements.TruncatablePrimeBase
open ProjectEulerSolutions.TruncatablePrimeCalls
private abbrev program := Solvers.P37Ast.module

private theorem predicate_result (radix : Nat) (valid : 2 ≤ radix)
    (n : Nat) (_large : radix ≤ n) (_right : RightPrime radix n) (heap : Mutable.Heap) :
    ∃ extra, ReturnsValue program "is_left_truncatable" [number n, number radix] heap
      (flag (leftCheck radix n)) (heap ++ extra) :=
  ⟨[], by simpa using is_left_truncatable.result radix n valid heap⟩

theorem result (radix : Nat) (valid : 2 ≤ radix) (success : Exhausts radix) (heap : Mutable.Heap) :
    ∃ finalHeap, ReturnsValue program "solve" [number radix] heap
      (number (answer radix success)) finalHeap :=
  function_result program radix "is_left_truncatable" valid is_prime.result (predicate_result radix valid)
    "solve" ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 rfl rfl rfl rfl success heap

theorem diverges (radix : Nat) (valid : 2 ≤ radix) (absent : ¬ Exhausts radix) :
    (HeapCalls.scalarFunction program "solve" [.int radix]).Diverges :=
  function_diverges program radix "is_left_truncatable" valid is_prime.result (predicate_result radix valid)
    "solve" ((findFunctionIn program.body "solve").getD ([], [])).1
    ((findFunctionIn program.body "solve").getD ([], [])).2 rfl rfl rfl rfl absent

theorem returns_iff (radix : Nat) (valid : 2 ≤ radix) :
    (HeapCalls.scalarFunction program "solve" [.int radix]).Returns ↔ Exhausts radix := by
  constructor
  · rintro ⟨value, returned⟩
    by_contra absent
    exact Execution.diverges_not_halts (diverges radix valid absent) ⟨.ok value, returned⟩
  · intro success
    obtain ⟨_, returned⟩ := result radix valid success []
    exact returned.scalar_returns (values := [.int radix])

end solvers.«37».solve
