import solvers.«12.py.lean.d».FactorMethodRuntime

namespace solvers.«12».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

/-- One prime bound is retained across all divisor-count calls. -/
def BoundedCache (heap : Heap) (address storage bound : Nat) : Prop :=
  ∃ ps last,
    heap[address]? = some (.instance origin "PrimeTable" [("primes", .ref storage)]) ∧
    heap[storage]? = some (.list (primeValues ps)) ∧ ps.getLast? = some last ∧ last ≤ bound ∧
    ps.Pairwise (· < ·) ∧ ProjectEulerStatements.PrimeTrialDivision.CompleteBelow ps (last+1)

theorem initial_cache (heap : Heap) (bound : Nat) (large : 2 ≤ bound) :
    BoundedCache (initialized heap) heap.length (heap.length+1) bound := by
  rw [initialized_eq]
  exact ⟨[2], 2, by simp, by simp, rfl, large, by simp, ProjectEulerStatements.PrimeTrialDivision.initial⟩

/-- Each factorization operand is bounded by the outer search's final index. -/
abbrev boundedFactorBudget (bound limit : Nat) : Nat :=
  ensureBudget bound + 80*(bound+1) + 41*limit + 121

theorem factor_budget_le (bound m limit : Nat) (small : m ≤ limit) :
    factorBudget bound m ≤ boundedFactorBudget bound limit := by
  have width := Nat.log2_le_self m
  simp only [factorBudget, boundedFactorBudget, PrimitiveWork.bitLength, Int.natAbs_natCast]
  omega

/-- Method binding is charged before the argument, and the resulting cache is
available to the next call without allocating another object. -/
theorem factor_call (m limit bound : Nat) (positive : 0 < m) (small : m ≤ limit)
    (primeBound : Nat.Prime bound) (enough : Nat.sqrt limit+1 ≤ bound)
    (heap : Heap) (address storage : Nat) (cache : BoundedCache heap address storage bound)
    (env : HeapCalls.Env) (receiver : String) (expression : Expr) (cost : Nat)
    (lookup : env receiver = some (.ref address))
    (argument : EvaluatesIn .word program env expression heap cost (.scalar (.int m)) heap) :
    ∃ finalHeap, BoundedCache finalHeap address storage bound ∧
      EvaluatesIn .word program env (.call (.attribute (.name receiver) "divisor_count") [expression])
        heap (cost+boundedFactorBudget bound limit+25) (.scalar (.int m.divisors.card)) finalHeap := by
  obtain ⟨ps, last, object, found, lastEntry, lastBound, ordered, complete⟩ := cache
  have defined : program.findClass "PrimeTable" =
      some (origin, (program.findClassBody "PrimeTable").getD []) := rfl
  have method : findFunctionIn ((program.findClassBody "PrimeTable").getD []) "divisor_count" = some factorMethod := rfl
  have boundMethod : bindMethod program (.ref address) "divisor_count" heap =
      some (.ok (.method (.ref address) factorMethod.1 factorMethod.2), heap) := by
    simp [bindMethod, object, instanceClass, defined, method]
    rfl
  have priced : RecordWork.method program heap (.ref address) "divisor_count" = some 21 := by
    simp [RecordWork.method, object, boundMethod, RecordWork.lookup]
    rfl
  have rootSmall := Nat.sqrt_le_sqrt small
  obtain ⟨nextPrimes, nextLast, nextEntry, nextBound, nextOrder, nextComplete, execution⟩ :=
    factor_method_cost m last bound positive primeBound (by omega) lastBound heap address storage ps
      object found lastEntry ordered complete
  let finalHeap := heap.set storage (.list (primeValues nextPrimes))
  have distinct : storage ≠ address := by rintro rfl; rw [object] at found; cases found
  refine ⟨finalHeap, ⟨nextPrimes, nextLast, ?_, ?_, nextEntry, nextBound, nextOrder, nextComplete⟩, ?_⟩
  · simpa [finalHeap, distinct] using object
  · simp [finalHeap, (List.getElem?_eq_some_iff.mp found).1]
  · exact (EvaluatesIn.method lookup (by decide) boundMethod priced
      (ArgumentsIn.cons argument ArgumentsIn.nil) execution).mono
      (by have bounded := factor_budget_le bound m limit small; omega)

end solvers.«12».runtimeCost
